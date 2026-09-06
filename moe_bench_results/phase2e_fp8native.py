"""Phase 2e: FP8 N-split with NATIVE fp8 dot (no per-iter dequant).
a -> fp8 (per-token scale), b -> fp8 (per-expert scale). tl.dot(fp8,fp8)->f32 acc.
Dequant only the final accumulator: out = acc * a_scale * b_scale.
This removes the per-iteration dequant overhead that killed phase2d."""
import torch, triton, triton.language as tl

DEV = "cuda"
NUM_XCD = 8
E = 256
TOPK = 8
K = 6144
N = 2048
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
GROUP_M = 8
N_SLICE = N // NUM_XCD
NUM_NB_PER_SLICE = N_SLICE // BLOCK_N


@triton.jit
def _moe_fp8n_kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, sti_ptr, eid_ptr, rowstart_ptr, npp_ptr, nvt,
    N, K, EM,
    sam, sak, sbe, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, MODE: tl.constexpr, NX: tl.constexpr, TOPK: tl.constexpr,
    N_NB_PER_SLICE: tl.constexpr,
):
    pid = tl.program_id(0)
    npp = tl.load(npp_ptr)
    npm = tl.cdiv(npp, BM)
    npn = tl.cdiv(N, BN)
    GRID = npn * npm
    if pid >= GRID and MODE != 2:
        return
    if MODE == 2:
        xcd = pid % NX
        local = pid // NX
        per_expert = npm * N_NB_PER_SLICE
        exp_seq = local // per_expert
        rem = local % per_expert
        tb = rem // N_NB_PER_SLICE
        nb_in_slice = rem % N_NB_PER_SLICE
        pid_n = xcd * N_NB_PER_SLICE + nb_in_slice
        row_start = tl.load(rowstart_ptr + exp_seq.to(tl.int64))
        pid_m = row_start + tb
        if pid_m >= npm:
            return
    elif MODE == 1:
        tile_id = tl.load(rowstart_ptr + pid.to(tl.int64))
        pid_m = tile_id // npn
        pid_n = tile_id % npn
    else:
        pid_m = pid // npn
        pid_n = pid % npn

    oti = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    ot = tl.load(sti_ptr + oti)
    tm = ot < nvt
    ot = tl.where(tm, ot, 0)
    oe = tl.load(eid_ptr + pid_m.to(tl.int64)).to(tl.int64)
    obn = (pid_n * BN + tl.arange(0, BN).to(tl.int64)) % N
    ok = tl.arange(0, BK)
    ap = a_ptr + (ot[:, None] // TOPK * sam + ok[None, :] * sak)
    bp = b_ptr + (oe * sbe + ok[:, None] * sbk + obn[None, :] * sbn)
    # per-token a scale (M_tokens,) and per-expert b scale (E,)
    a_tok = ot // TOPK
    a_scale = tl.load(a_scale_ptr + a_tok).to(tl.float32)        # (BM,) -> broadcast
    b_scale = tl.load(b_scale_ptr + oe).to(tl.float32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a_fp8 = tl.load(ap, mask=tm[:, None])           # fp8 (BM, BK)
        b_fp8 = tl.load(bp)                              # fp8 (BK, BN)
        acc = tl.dot(a_fp8, b_fp8, acc=acc)             # native fp8 dot
        ap += BK * sak
        bp += BK * sbk
    # dequant accumulator: out = acc * a_scale * b_scale
    acc = acc * a_scale[:, None] * b_scale
    c = acc.to(tl.bfloat16)
    ocm = pid_m * BM + tl.arange(0, BM).to(tl.int64)
    ocn = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    cp = c_ptr + ocm[:, None] * scm + ocn[None, :] * scn
    tl.store(cp, c, mask=tm[:, None])


def build_routing_balanced(M_tokens, E, topk, device):
    g = torch.Generator(device="cpu").manual_seed(7)
    idx = torch.empty(M_tokens, topk, dtype=torch.int32)
    for i in range(M_tokens):
        idx[i] = torch.tensor([torch.randperm(32, generator=g)[0].item() + xcd * 32 for xcd in range(NUM_XCD)], dtype=torch.int32)
    flat_idx = idx.reshape(-1)
    flat_token = torch.arange(M_tokens * topk, dtype=torch.int32)
    order = torch.argsort(flat_idx, stable=True)
    sorted_expert = flat_idx[order].to(torch.int32)
    sorted_token = flat_token[order].to(torch.int32).to(device)
    npm = (M_tokens * topk + BLOCK_M - 1) // BLOCK_M
    eid = torch.empty(npm, dtype=torch.int32, device=device)
    for pm in range(npm):
        pos = pm * BLOCK_M
        eid[pm] = sorted_expert[pos] if pos < len(sorted_expert) else 0
    pad = npm * BLOCK_M - len(sorted_token)
    if pad > 0:
        sorted_token = torch.cat([sorted_token, torch.full((pad,), M_tokens * topk, dtype=torch.int32, device=device)])
    starts = []; cur = -1
    for pm in range(npm):
        e = int(eid[pm].item())
        if e != cur:
            starts.append(pm); cur = e
    return sorted_token, eid, npm, M_tokens * topk, torch.tensor(starts, dtype=torch.int32)


def make_fp8_ab(M_tokens, E, K, N, device):
    a_bf16 = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(E, K, N, dtype=torch.bfloat16, device=device) * 0.02
    amax = a_bf16.abs().reshape(M_tokens, -1).max(dim=1).values.float()
    a_scale = (amax / 448.0).clamp(min=1e-12)
    a_fp8 = (a_bf16.float() / a_scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    bmax = b_bf16.abs().reshape(E, -1).max(dim=1).values.float()
    b_scale = (bmax / 448.0).clamp(min=1e-12)
    b_fp8 = (b_bf16.float() / b_scale[:, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return a_fp8, a_scale.to(torch.float32), b_fp8, b_scale.to(torch.float32)


def bench(M_tokens, mode, warmup=15, iters=40):
    a_fp8, a_scale, b_fp8, b_scale = make_fp8_ab(M_tokens, E, K, N, DEV)
    sti, eid, npm, npp, rowstarts = build_routing_balanced(M_tokens, E, TOPK, DEV)
    npn = (N + BLOCK_N - 1) // BLOCK_N
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV)
    nvt = M_tokens * TOPK
    rowstarts = rowstarts.to(DEV)

    if mode == 2:
        num_distinct = rowstarts.numel()
        max_tb = 0
        for i in range(num_distinct):
            s = int(rowstarts[i].item()); e = int(eid[s].item()); cnt = 1
            while s + cnt < npm and int(eid[s + cnt].item()) == e:
                cnt += 1
            max_tb = max(max_tb, cnt)
        grid = (NUM_XCD * num_distinct * max_tb * NUM_NB_PER_SLICE,)
        pid_remap = rowstarts
    else:
        grid = (npm * npn,)
        if mode == 1:
            GRID = npm * npn
            pids_per_xcd = (GRID + NUM_XCD - 1) // NUM_XCD
            groups = [[] for _ in range(NUM_XCD)]
            for raw in range(GRID):
                pm = raw // npn
                e = int(eid[pm].item()) if pm < npm else 0
                groups[e % NUM_XCD].append(raw)
            pid_remap = torch.full((GRID,), -1, dtype=torch.int32)
            overflow = list(range(GRID))
            for xcd in range(NUM_XCD):
                for k, raw in enumerate(groups[xcd]):
                    if k < pids_per_xcd:
                        pid_remap[xcd + k * NUM_XCD] = raw
                        overflow.remove(raw)
            free = [i for i in range(GRID) if pid_remap[i] < 0]
            for raw in overflow:
                if free:
                    pid_remap[free.pop()] = raw
            pid_remap = pid_remap.to(DEV)
        else:
            pid_remap = torch.zeros(1, dtype=torch.int32, device=DEV)

    def run():
        c = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
        args = (a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, pid_remap, npp_t, nvt,
                N, K, E, a_fp8.stride(0), a_fp8.stride(1), b_fp8.stride(0), b_fp8.stride(1), b_fp8.stride(2),
                c.stride(0), c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, mode, NUM_XCD, TOPK,
                NUM_NB_PER_SLICE)
        for _ in range(warmup):
            _moe_fp8n_kernel[grid](*args)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _moe_fp8n_kernel[grid](*args)
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / iters
    return run()


def main():
    print(f"=== Phase 2e: FP8 NATIVE-dot N-split (GLM-5.2) ===")
    print(f"K={K} N={N} E={E} topk={TOPK} NUM_XCD={NUM_XCD}")
    print(f"per-expert per-XCD fp8 weight = {K*N_SLICE*1/1e6:.3f} MB (fits 4MB L2)\n")
    print(f"{'M':>7} {'M(topk)':>9} {'rr us':>9} {'affine us':>10} {'nsplit us':>10} {'rr/aff':>7} {'rr/nsplit':>9}")
    for M in [2048, 4096, 8192, 16384]:
        t_rr = bench(M, 0)
        t_aff = bench(M, 1)
        t_ns = bench(M, 2)
        print(f"{M:>7} {M*TOPK:>9} {t_rr:>9.1f} {t_aff:>10.1f} {t_ns:>10.1f} {t_rr/t_aff:>6.2f}x {t_rr/t_ns:>8.2f}x")


if __name__ == "__main__":
    main()
