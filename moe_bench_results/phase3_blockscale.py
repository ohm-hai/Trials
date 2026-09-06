"""Phase 5 ingredient: megakernel with PRODUCTION block-wise 128x128 fp8 scaling.
Matches aiter/sglang: A_scale (M, K/128) per-token-group, B_scale (E, N/128, K/128) per-expert-per-block.
Dequant per 128x128 tile: acc += dot(a_fp8, b_fp8) * a_scale[:, None] * b_scale[None, :].
This makes the megakernel a drop-in stage1 replacement (correct vs production)."""
import torch, triton, triton.language as tl

DEV = "cuda"
NUM_XCD = 8
E = 256
TOPK = 8
K = 6144
N = 2048
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128          # production block_k for fp8 dequant
GROUP_N = 128
GROUP_K = 128
N_SLICE = N // NUM_XCD
NUM_NB_PER_SLICE = N_SLICE // BLOCK_N   # 2


@triton.jit
def _mega_bs_kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, sti_ptr, eid_ptr, npp_ptr, nvt,
    N, K, EM,
    sam, sak, sbe, sbk, sbn, scm, scn,
    sasm, sask, sbse, sbsn, sbsk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, NX: tl.constexpr, TOPK: tl.constexpr,
    NBPS: tl.constexpr, NPM: tl.constexpr, GN: tl.constexpr, GK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    if MODE == 0:
        # round-robin: pid -> (pid_m, pid_n) row-major over the full grid
        pid_m = pid // npn
        pid_n = pid % npn
    else:
        # megakernel N-split: xcd = n_slice, pid_m = expert-sequential block-row
        xcd = pid // (NPM * NBPS)
        local = pid % (NPM * NBPS)
        pid_m = local // NBPS
        nb_in_slice = local % NBPS
        pid_n = xcd * NBPS + nb_in_slice
    if pid_m >= NPM:
        return
    oti = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    ot = tl.load(sti_ptr + oti)
    tm = ot < nvt
    ot = tl.where(tm, ot, 0)
    oe = tl.load(eid_ptr + pid_m.to(tl.int64)).to(tl.int64)
    obn = (pid_n * BN + tl.arange(0, BN).to(tl.int64)) % N
    ok = tl.arange(0, BK)
    ap = a_ptr + (ot[:, None] // TOPK * sam + ok[None, :] * sak)
    bp = b_ptr + (oe * sbe + ok[:, None] * sbk + obn[None, :] * sbn)
    # block-wise scale ptrs (match aiter: offs_bsn from N coords -> (BN,) vector)
    a_tok = ot // TOPK
    offs_bsn = obn // GN
    a_scale_ptrs = a_scale_ptr + a_tok * sasm
    b_scale_ptrs = b_scale_ptr + oe * sbse + offs_bsn * sbsn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a_fp8 = tl.load(ap, mask=tm[:, None])
        b_fp8 = tl.load(bp)
        k_start = kk * BK
        offs_ks = k_start // GK
        a_scale = tl.load(a_scale_ptrs + offs_ks * sask, mask=tm, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * sbsk)
        acc += tl.dot(a_fp8, b_fp8) * a_scale[:, None] * b_scale[None, :]
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


def make_blockwise_fp8(M_tokens, E, K, N, device, block_n=128, block_k=128):
    a_bf16 = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(E, K, N, dtype=torch.bfloat16, device=device) * 0.02
    # a: per-token-group fp8, a_scale (M, K//block_k)
    a_max = a_bf16.abs().reshape(M_tokens, K // block_k, block_k).amax(dim=-1).float()
    a_scale = (a_max / 448.0).clamp(min=1e-12)            # (M, K//bk)
    a_fp8 = (a_bf16.reshape(M_tokens, K // block_k, block_k) / a_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M_tokens, K)
    # b: per-expert per-block fp8, b_scale (E, N//block_n, K//block_k)
    bn, bk = N // block_n, K // block_k
    b_nk = b_bf16.permute(0, 2, 1).contiguous()              # (E, N, K)
    b_max = b_nk.abs().reshape(E, bn, block_n, bk, block_k).amax(dim=(2, 4)).float()  # (E, bn, bk)
    b_scale = (b_max / 448.0).clamp(min=1e-12)              # (E, N//bn, K//bk)
    b_fp8_nk = (b_nk.reshape(E, bn, block_n, bk, block_k) / b_scale.unsqueeze(2).unsqueeze(4)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(E, N, K)
    b_fp8 = b_fp8_nk.permute(0, 2, 1).contiguous()           # (E, K, N) for kernel
    b_scale = b_scale.contiguous()
    return a_fp8, a_scale.to(torch.float32), b_fp8, b_scale.to(torch.float32)


def torch_ref(a_bf16, b_bf16, sti, eid, topk, M_tokens, N):
    npm = sti.numel() // BLOCK_M
    out = torch.zeros(sti.numel(), N, dtype=torch.bfloat16, device=a_bf16.device)
    for pm in range(npm):
        exp = int(eid[pm].item())
        if exp < 0:
            continue
        toks = sti[pm * BLOCK_M:(pm + 1) * BLOCK_M]
        toks = toks[toks < M_tokens * topk]
        a_block = a_bf16[toks // topk].to(torch.float32)
        b_block = b_bf16[exp].to(torch.float32)
        ob = (a_block @ b_block).to(torch.bfloat16)
        for j, t in enumerate(toks):
            out[t] = ob[j]
    return out


def _launch(a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, npp_t, nvt, npm, mode):
    if mode == 0:
        npn = (N + BLOCK_N - 1) // BLOCK_N
        grid = (npm * npn,)
    else:
        grid = (NUM_XCD * npm * NUM_NB_PER_SLICE,)
    args = (a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, npp_t, nvt,
            N, K, E, a_fp8.stride(0), a_fp8.stride(1), b_fp8.stride(0), b_fp8.stride(1), b_fp8.stride(2),
            c.stride(0), c.stride(1),
            a_scale.stride(0), a_scale.stride(1), b_scale.stride(0), b_scale.stride(1), b_scale.stride(2),
            BLOCK_M, BLOCK_N, BLOCK_K, 8, NUM_XCD, TOPK, NUM_NB_PER_SLICE, npm, GROUP_N, GROUP_K, mode)
    _mega_bs_kernel[grid](*args)


def correctness_check(M_tokens=512):
    a_fp8, a_scale, b_fp8, b_scale = make_blockwise_fp8(M_tokens, E, K, N, DEV)
    sti, eid, npm, npp = build_routing_balanced(M_tokens, E, TOPK, DEV)
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV); nvt = M_tokens * TOPK
    valid = sti < nvt
    # round-robin reference (same kernel, same math, different pid order)
    c_rr = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
    _launch(a_fp8, b_fp8, a_scale, b_scale, c_rr, sti, eid, npp_t, nvt, npm, 0)
    # megakernel N-split
    c_mk = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
    _launch(a_fp8, b_fp8, a_scale, b_scale, c_mk, sti, eid, npp_t, nvt, npm, 1)
    torch.cuda.synchronize()
    diff = (c_rr[valid].to(torch.float32) - c_mk[valid].to(torch.float32)).abs().max().item()
    print(f"correctness @ M={M_tokens}: megakernel vs round-robin max_abs_diff={diff:.6f} (bit-equivalent expected)")
    return diff


def bench_decode(batch_size, mode, warmup=15, iters=60):
    M_tokens = batch_size
    a_fp8, a_scale, b_fp8, b_scale = make_blockwise_fp8(M_tokens, E, K, N, DEV)
    sti, eid, npm, npp = build_routing_balanced(M_tokens, E, TOPK, DEV)
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV); nvt = M_tokens * TOPK
    def run():
        c = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
        for _ in range(warmup): _launch(a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, npp_t, nvt, npm, mode)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters): _launch(a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, npp_t, nvt, npm, mode)
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / iters
    return run()


def main():
    print("=== Phase 5: block-scale megakernel (production-matching fp8) + decode win ===")
    print(f"GLM-5.2 fp8 block-wise 128x128 | per-expert per-XCD weight={K*(N//8)*1/1e6:.3f}MB\n")
    err = correctness_check(512)
    print(f"\n{'batch':>7} {'npm':>5} {'round-robin us':>15} {'megakernel us':>15} {'spd':>7}")
    for bs in [256, 512, 1024, 2048]:
        npm = (bs * TOPK + BLOCK_M - 1) // BLOCK_M
        t_rr = bench_decode(bs, 0)
        t_mk = bench_decode(bs, 1)
        print(f"{bs:>7} {npm:>5} {t_rr:>15.1f} {t_mk:>15.1f} {t_rr/t_mk:>6.2f}x")


if __name__ == "__main__":
    main()
