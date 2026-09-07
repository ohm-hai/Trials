"""Phase 2b: characterize WHY expert-affine XCD pinning is slower.
Hypothesis: expert weights (K*N*2 bytes) must fit in per-XCD L2 (4 MB) for pinning to
create residency. GLM-5.2 N=2048 -> 25 MB expert >> 4 MB L2 -> no residency -> only overhead.
Test: sweep N (expert size) and routing skew; find the regime where expert-affine helps."""
import sys, torch, triton
sys.path.insert(0, "/root/moe_bench_results")
from phase2_expert_affine_bench import (
    _fused_moe_stage1_kernel, build_pid_remap, NUM_XCD, TOPK, K as KDIM, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, DEV,
)
import triton.language as tl


def build_routing_balanced(M_tokens, E, topk, device, n_hot=None, hot_frac=None):
    """balanced: each token picks one expert per XCD. If n_hot set, route hot_frac of tokens
    to n_hot experts (skewed) and the rest balanced."""
    g = torch.Generator(device="cpu").manual_seed(7)
    idx = torch.empty(M_tokens, topk, dtype=torch.int32)
    if n_hot is None:
        for i in range(M_tokens):
            idx[i] = torch.tensor([torch.randperm(32, generator=g)[0] + xcd * 32 for xcd in range(NUM_XCD)], dtype=torch.int32)
    else:
        hot = list(range(n_hot))
        for i in range(M_tokens):
            if i < int(M_tokens * hot_frac):
                picks = [hot[(torch.randperm(n_hot, generator=g)[0]).item()] for _ in range(topk)]
            else:
                picks = [torch.randperm(32, generator=g)[0].item() + xcd * 32 for xcd in range(NUM_XCD)]
            idx[i] = torch.tensor(picks, dtype=torch.int32)
    flat_idx = idx.reshape(-1)
    flat_token = torch.arange(M_tokens * topk, dtype=torch.int32)
    order = torch.argsort(flat_idx, stable=True)
    sorted_expert = flat_idx[order].to(torch.int32)
    sorted_token = flat_token[order].to(torch.int32).to(device)
    num_pid_m = (M_tokens * topk + BLOCK_M - 1) // BLOCK_M
    expert_ids = torch.empty(num_pid_m, dtype=torch.int32, device=device)
    for pm in range(num_pid_m):
        pos = pm * BLOCK_M
        expert_ids[pm] = sorted_expert[pos] if pos < len(sorted_expert) else 0
    pad = num_pid_m * BLOCK_M - len(sorted_token)
    if pad > 0:
        sorted_token = torch.cat([sorted_token, torch.full((pad,), M_tokens * topk, dtype=torch.int32, device=device)])
    return sorted_token, expert_ids, num_pid_m, M_tokens * topk


def bench_config(M_tokens, N, n_hot=None, hot_frac=None, warmup=15, iters=40):
    E = 256
    a = torch.randn(M_tokens, KDIM, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(E, KDIM, N, dtype=torch.bfloat16, device=DEV) * 0.02
    sorted_token, expert_ids, num_pid_m, npp = build_routing_balanced(M_tokens, E, TOPK, DEV, n_hot, hot_frac)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_remap = build_pid_remap(expert_ids, num_pid_m, num_pid_n, NUM_XCD).to(DEV)
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV)
    nvt = M_tokens * TOPK
    grid = (num_pid_m * num_pid_n,)
    exp_bytes = KDIM * N * 2 / 1e6
    def run(mode):
        c = torch.empty(num_pid_m * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
        args = (a, b, c, sorted_token, expert_ids, pid_remap, npp_t, nvt,
                N, KDIM, E, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
                c.stride(0), c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, mode, NUM_XCD, TOPK)
        for _ in range(warmup):
            _fused_moe_stage1_kernel[grid](*args)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _fused_moe_stage1_kernel[grid](*args)
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / iters  # us
    t_rr = run(0)
    t_ea = run(1)
    return t_rr, t_ea, exp_bytes


def main():
    print(f"=== Phase 2b: expert-affine lever regime characterization (gfx950, 8 XCD, 4MB L2/XCD) ===")
    print(f"K={KDIM} TOPK={TOPK} E=256 BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    print(f"per-expert weight = K*N*2 bytes; per-XCD L2 = 4 MB\n")
    print(f"{'M':>7} {'N':>6} {'expMB':>7} {'route':>10} {'rr us':>9} {'ea us':>9} {'spd':>7}  fitL2?")
    configs = [
        # (M_tokens, N, n_hot, hot_frac, label)
        (2048, 2048, None, None, "balanced"),
        (8192, 2048, None, None, "balanced"),
        (16384, 2048, None, None, "balanced"),
        (8192, 2048, 8, 0.9, "skew8-90"),
        (16384, 2048, 8, 0.9, "skew8-90"),
        # small N: expert weight fits in L2
        (2048, 128, None, None, "balanced"),
        (8192, 128, None, None, "balanced"),
        (16384, 128, None, None, "balanced"),
        (8192, 128, 8, 0.9, "skew8-90"),
        (16384, 128, 8, 0.9, "skew8-90"),
        (16384, 64, None, None, "balanced"),   # 0.75MB expert, fits easily
        (16384, 32, None, None, "balanced"),   # 0.375MB expert
    ]
    for M, N, nh, hf, label in configs:
        t_rr, t_ea, expMB = bench_config(M, N, nh, hf)
        sp = t_rr / t_ea if t_ea > 0 else 0
        fit = "YES" if expMB < 4.0 else "no"
        print(f"{M:>7} {N:>6} {expMB:>7.2f} {label:>10} {t_rr:>9.1f} {t_ea:>9.1f} {sp:>6.2f}x  {fit}")


if __name__ == "__main__":
    main()
