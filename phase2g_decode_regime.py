"""Phase 2g: test the N-split lever in the DECODE regime (small tokens-per-expert).
Decode: batch_size tokens, topk=8 -> M(topk)=batch*8 routed rows over E=256 experts.
  tokens_per_expert = batch*8/256 = batch/32. For batch<=256, tokens_per_expert<=8.
Hypothesis: with small tokens_per_expert, per-expert ACTIVATIONS (tokens_per_expert*K*2) also fit
  in the 4MB L2 alongside the 3.14MB weight slice -> 8x replication is L2 hits (no HBM) -> lever wins.
Reuses the compact-grid fp8 native-dot kernel from phase2f."""
import sys, torch
sys.path.insert(0, "/root/moe_bench_results")
from phase2f_compact import (_moe_fp8n_kernel, build_routing_balanced, make_fp8_ab,
    build_mode2_pid_remap, NUM_XCD, E, TOPK, K, N, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    NUM_NB_PER_SLICE, DEV)

def bench_decode(batch_size, mode, warmup=15, iters=60):
    M_tokens = batch_size
    a_fp8, a_scale, b_fp8, b_scale = make_fp8_ab(M_tokens, E, K, N, DEV)
    # decode routing: each token picks topk=8 experts (one per XCD, balanced)
    sti, eid, npm, npp = build_routing_balanced(M_tokens, E, TOPK, DEV)
    npn = (N + BLOCK_N - 1) // BLOCK_N
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV); nvt = M_tokens * TOPK
    grid = (npm * npn,)
    if mode == 2:
        pid_remap = build_mode2_pid_remap(npm, npn, NUM_NB_PER_SLICE, NUM_XCD).to(DEV)
    elif mode == 1:
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
    print(f"=== Phase 2g: N-split lever in DECODE regime (small tokens/expert) ===")
    print(f"GLM-5.2: E=256 topk=8 K={K} N={N} fp8 | per-expert per-XCD weight={K*(N//8)*1/1e6:.3f}MB\n")
    print(f"{'batch':>7} {'tok/exp':>8} {'npm':>5} {'rr us':>9} {'nsplit us':>10} {'rr/nsplit':>9}  act_fitL2?")
    for bs in [4, 8, 16, 32, 64, 128, 256, 512]:
        tpe = bs * TOPK / E  # tokens per expert (routed)
        npm = (bs * TOPK + BLOCK_M - 1) // BLOCK_M
        t_rr = bench_decode(bs, 0)
        t_ns = bench_decode(bs, 2)
        # activation per expert = tpe * K * 2 bytes (bf16) ; per-XCD weight slice = K*(N/8)*1 (fp8)
        act_per_exp = tpe * K * 2 / 1e6
        wtslice = K * (N // NUM_XCD) * 1 / 1e6
        fit = "YES" if (act_per_exp + wtslice) < 4.0 else "no"
        print(f"{bs:>7} {tpe:>8.2f} {npm:>5} {t_rr:>9.1f} {t_ns:>10.1f} {t_rr/t_ns:>8.2f}x   {fit:>10} {act_per_exp:.2f}+{wtslice:.2f}MB")

if __name__ == "__main__":
    main()
