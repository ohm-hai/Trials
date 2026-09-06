"""Phase 2h: arithmetic megakernel (no pid_remap table, no wasted blocks).
Mode 3: grid = NUM_XCD * npm * NUM_NB_PER_SLICE. pid -> (xcd, pid_m, nb_in_slice) by arithmetic.
  xcd = pid // (npm*NBPS); local = pid % (npm*NBPS);
  pid_m = local // NBPS; nb_in_slice = local % NBPS; pid_n = xcd*NBPS + nb_in_slice.
This is the segmented-launch megakernel: expert-sequential (pid_m ascending = expert-sorted)
and N-split (xcd = n_slice), with zero dispatch overhead (no table lookup)."""
import sys, torch
sys.path.insert(0, "/root/moe_bench_results")
import triton, triton.language as tl
from phase2f_compact import (_moe_fp8n_kernel, build_routing_balanced, make_fp8_ab,
    NUM_XCD, E, TOPK, K, N, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_NB_PER_SLICE, DEV)



@triton.jit
def _moe_megakernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, sti_ptr, eid_ptr, npp_ptr, nvt,
    N, K, EM,
    sam, sak, sbe, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, MODE: tl.constexpr, NX: tl.constexpr, TOPK: tl.constexpr,
    NBPS: tl.constexpr, NPM: tl.constexpr,
):
    pid = tl.program_id(0)
    # arithmetic decode: xcd = n_slice, pid_m = block-row (expert-sequential), nb_in_slice
    xcd = pid // (NPM * NBPS)
    local = pid % (NPM * NBPS)
    pid_m = local // NBPS
    nb_in_slice = local % NBPS
    pid_n = xcd * NBPS + nb_in_slice
    npn = tl.cdiv(N, BN)
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
    a_tok = ot // TOPK
    a_scale = tl.load(a_scale_ptr + a_tok).to(tl.float32)
    b_scale = tl.load(b_scale_ptr + oe).to(tl.float32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a_fp8 = tl.load(ap, mask=tm[:, None])
        b_fp8 = tl.load(bp)
        acc = tl.dot(a_fp8, b_fp8, acc=acc)
        ap += BK * sak
        bp += BK * sbk
    acc = acc * a_scale[:, None] * b_scale
    c = acc.to(tl.bfloat16)
    ocm = pid_m * BM + tl.arange(0, BM).to(tl.int64)
    ocn = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    cp = c_ptr + ocm[:, None] * scm + ocn[None, :] * scn
    tl.store(cp, c, mask=tm[:, None])


def bench_decode(batch_size, mode, warmup=15, iters=60):
    M_tokens = batch_size
    a_fp8, a_scale, b_fp8, b_scale = make_fp8_ab(M_tokens, E, K, N, DEV)
    sti, eid, npm, npp = build_routing_balanced(M_tokens, E, TOPK, DEV)
    npn = (N + BLOCK_N - 1) // BLOCK_N
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV); nvt = M_tokens * TOPK
    if mode == 3:
        grid = (NUM_XCD * npm * NUM_NB_PER_SLICE,)
        def run():
            c = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
            args3 = (a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, npp_t, nvt,
                     N, K, E, a_fp8.stride(0), a_fp8.stride(1), b_fp8.stride(0), b_fp8.stride(1), b_fp8.stride(2),
                     c.stride(0), c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, 3, NUM_XCD, TOPK,
                     NUM_NB_PER_SLICE, npm)
            for _ in range(warmup): _moe_megakernel[grid](*args3)
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters): _moe_megakernel[grid](*args3)
            t1.record(); torch.cuda.synchronize()
            return t0.elapsed_time(t1) * 1e3 / iters
        return run()
    # reuse phase2f for modes 0/2
    from phase2f_compact import build_mode2_pid_remap
    grid = (npm * npn,)
    if mode == 2:
        pid_remap = build_mode2_pid_remap(npm, npn, NUM_NB_PER_SLICE, NUM_XCD).to(DEV)
    elif mode == 1:
        GRID = npm * npn; pids_per_xcd = (GRID + NUM_XCD - 1) // NUM_XCD
        groups = [[] for _ in range(NUM_XCD)]
        for raw in range(GRID):
            pm = raw // npn; e = int(eid[pm].item()) if pm < npm else 0
            groups[e % NUM_XCD].append(raw)
        pid_remap = torch.full((GRID,), -1, dtype=torch.int32); overflow = list(range(GRID))
        for xcd in range(NUM_XCD):
            for k, raw in enumerate(groups[xcd]):
                if k < pids_per_xcd: pid_remap[xcd + k * NUM_XCD] = raw; overflow.remove(raw)
        free = [i for i in range(GRID) if pid_remap[i] < 0]
        for raw in overflow:
            if free: pid_remap[free.pop()] = raw
        pid_remap = pid_remap.to(DEV)
    else:
        pid_remap = torch.zeros(1, dtype=torch.int32, device=DEV)
    def run():
        c = torch.empty(npm * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
        args = (a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, pid_remap, npp_t, nvt,
                N, K, E, a_fp8.stride(0), a_fp8.stride(1), b_fp8.stride(0), b_fp8.stride(1), b_fp8.stride(2),
                c.stride(0), c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, mode, NUM_XCD, TOPK, NUM_NB_PER_SLICE)
        for _ in range(warmup): _moe_fp8n_kernel[grid](*args)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters): _moe_fp8n_kernel[grid](*args)
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / iters
    return run()

def main():
    print(f"=== Phase 2h: arithmetic megakernel (no table, no waste) vs round-robin/N-split, DECODE regime ===")
    print(f"GLM-5.2 fp8 | per-expert per-XCD weight={K*(N//8)*1/1e6:.3f}MB\n")
    print(f"{'batch':>7} {'npm':>5} {'rr us':>8} {'nsplit us':>10} {'megakernel us':>13} {'rr/nsplit':>9} {'rr/mega':>9}")
    for bs in [64, 128, 256, 512, 1024, 2048]:
        npm = (bs * TOPK + BLOCK_M - 1) // BLOCK_M
        t_rr = bench_decode(bs, 0)
        t_ns = bench_decode(bs, 2)
        t_mk = bench_decode(bs, 3)
        print(f"{bs:>7} {npm:>5} {t_rr:>8.1f} {t_ns:>10.1f} {t_mk:>13.1f} {t_rr/t_ns:>8.2f}x {t_rr/t_mk:>8.2f}x")

if __name__ == "__main__":
    main()
