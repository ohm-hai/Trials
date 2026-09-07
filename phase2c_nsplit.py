"""Phase 2c: N-split decomposition for GLM-5.2 MoE GEMM.

Decomposition: split the N dimension across XCDs (N/NUM_XCD = 256 cols per XCD).
  per-expert per-XCD weight = K * (N/8) * 2 = 6144*256*2 = 3.14 MB  < 4 MB L2/XCD  -> FITS.
Within each XCD, dispatch blocks in EXPERT-SEQUENTIAL order so the L2 holds one
expert's N-slice (3.14 MB) while all its token-blocks reuse it, then moves on.

Three modes compared (same kernel, REMAP_MODE constexpr):
  0 = round-robin (aiter original): tile_id = pid
  1 = expert-affine full-N:  all tiles of expert e -> XCD e%8 (25 MB expert, doesn't fit)
  2 = N-split + expert-seq:  XCD = n_slice; within-XCD order = expert-sequential
Config: GLM-5.2 MoE stage-1, E=256 topk=8 K=6144 N=2048 bf16, NUM_XCD=8.
"""
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
N_SLICE = N // NUM_XCD          # 256 cols per XCD
NUM_NB_PER_SLICE = N_SLICE // BLOCK_N   # 2 N-blocks per slice


@triton.jit
def _moe_kernel(
    a_ptr, b_ptr, c_ptr, sti_ptr, eid_ptr, pid_remap_ptr, npp_ptr, nvt,
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
    if pid >= GRID:
        return
    if MODE == 2:
        # N-split + expert-sequential: pid -> (n_slice=XCD, local) ; local -> (expert_seq, tb, nb_in_slice)
        xcd = pid % NX
        local = pid // NX
        nb_per_slice = N_NB_PER_SLICE
        per_expert = npm * nb_per_slice
        exp_seq = local // per_expert
        rem = local % per_expert
        tb = rem // nb_per_slice
        nb_in_slice = rem % nb_per_slice
        pid_n = xcd * nb_per_slice + nb_in_slice
        # expert-sequential: block-rows are already sorted by expert in sti/eid,
        # so expert e occupies block-rows [e*tokens_per_expert/BM ... ]. Map exp_seq
        # to the actual block-row range via eid lookup: pid_m = tb + (start of exp_seq's rows).
        # Simplest: pid_m = tb  (tb already global block-row index within the expert's run)
        # We need pid_m to be the global block-row. Use a per-expert row-start table.
        pid_m = tl.load(pid_remap_ptr + (exp_seq * 2 + 0).to(tl.int64)) + tb  # row_start[exp_seq]
        # validate pid_m in range
        tile_id = pid_m * npn + pid_n
    elif MODE == 1:
        tile_id = tl.load(pid_remap_ptr + pid.to(tl.int64))
        pid_m = tile_id // npn
        pid_n = tile_id % npn
    else:
        tile_id = pid
        pid_m = tile_id // npn
        pid_n = tile_id % npn

    oti = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    ot = tl.load(sti_ptr + oti)
    tm = ot < nvt
    ot = tl.where(tm, ot, 0)
    oe = tl.load(eid_ptr + pid_m.to(tl.int64)).to(tl.int64)
    obn = (pid_n * BN + tl.arange(0, BN).to(tl.int64)) % N
    ok = tl.arange(0, BK)
    ap = a_ptr + (ot[:, None] // TOPK * sam + ok[None, :] * sak)
    bp = b_ptr + (oe * sbe + ok[:, None] * sbk + obn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=tm[:, None], other=0.0)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc)
        ap += BK * sak
        bp += BK * sbk
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
    return sorted_token, eid, npm, M_tokens * topk


def build_expert_rowmap(eid, npm):
    """For MODE 2: row_start[exp_seq], num_tb[exp_seq] for each distinct expert in sorted order.
    Returns a table of shape (num_distinct_experts, 2): [row_start, num_tb]."""
    starts = []
    counts = []
    cur = -1
    cnt = 0
    for pm in range(npm):
        e = int(eid[pm].item())
        if e != cur:
            if cur >= 0:
                starts.append(prev_start); counts.append(cnt)
            cur = e; prev_start = pm; cnt = 1
        else:
            cnt += 1
    if cur >= 0:
        starts.append(prev_start); counts.append(cnt)
    return torch.tensor(starts, dtype=torch.int32), torch.tensor(counts, dtype=torch.int32)


def build_mode2_remap(eid, npm, npn, NUM_XCD, NUM_NB_PER_SLICE):
    """pid_remap for MODE 2: row_start[exp_seq] table (length num_distinct_experts).
    Grid = NX * num_distinct_experts * npm_per_expert * NUM_NB_PER_SLICE (approx).
    We pack row_start[exp_seq] into pid_remap (just the starts); tb indexes within expert."""
    starts, counts = build_expert_rowmap(eid, npm)
    return starts.to(DEV), counts.to(DEV)


def bench(M_tokens, mode, warmup=15, iters=40):
    a = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(E, K, N, dtype=torch.bfloat16, device=DEV) * 0.02
    sti, eid, npm, npp = build_routing_balanced(M_tokens, E, TOPK, DEV)
    npn = (N + BLOCK_N - 1) // BLOCK_N
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV)
    nvt = M_tokens * TOPK

    if mode == 2:
        starts, counts = build_mode2_remap(eid, npm, npn, NUM_XCD, NUM_NB_PER_SLICE)
        # grid: NX * num_distinct_experts * max_tb_per_expert * NUM_NB_PER_SLICE
        # use counts to size; pad to max count
        max_tb = int(counts.max().item())
        num_distinct = starts.numel()
        grid = (NUM_XCD * num_distinct * max_tb * NUM_NB_PER_SLICE,)
        pid_remap = starts  # row_start[exp_seq]
    else:
        # round-robin (mode 0) or expert-affine (mode 1): grid = npm*npn
        grid = (npm * npn,)
        if mode == 1:
            # expert-affine remap: group raw tiles by eid%NX
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
                if free: pid_remap[free.pop()] = raw
            pid_remap = pid_remap.to(DEV)
        else:
            pid_remap = torch.zeros(1, dtype=torch.int32, device=DEV)

    def run():
        c = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
        args = (a, b, c, sti, eid, pid_remap, npp_t, nvt,
                N, K, E, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
                c.stride(0), c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, mode, NUM_XCD, TOPK,
                NUM_NB_PER_SLICE)
        for _ in range(warmup):
            _moe_kernel[grid](*args)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _moe_kernel[grid](*args)
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / iters
    return run()


def main():
    print(f"=== Phase 2c: N-split decomposition (GLM-5.2 N=2048, N/8=256 per XCD, 3.14MB/expert < 4MB L2) ===")
    print(f"K={K} N={N} E={E} topk={TOPK} NUM_XCD={NUM_XCD} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    print(f"per-expert per-XCD weight = {K*N_SLICE*2/1e6:.2f} MB (fits 4MB L2: {K*N_SLICE*2 < 4e6})\n")
    print(f"{'M':>7} {'M(topk)':>9} {'rr us':>9} {'affine us':>10} {'nsplit us':>10} {'rr/aff':>7} {'rr/nsplit':>9}")
    for M in [2048, 4096, 8192, 16384]:
        t_rr = bench(M, 0)
        t_aff = bench(M, 1)
        t_ns = bench(M, 2)
        print(f"{M:>7} {M*TOPK:>9} {t_rr:>9.1f} {t_aff:>10.1f} {t_ns:>10.1f} {t_rr/t_aff:>6.2f}x {t_rr/t_ns:>8.2f}x")


if __name__ == "__main__":
    main()
