"""Phase 7 tiny: megakernel-only timing at 1,2,4,8,12,16,32,64 (fast, no prod JIT)."""
import torch, sys
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import (per_block_quant_fp8,
    build_routing_from_topk, E, TOPK, K_HIDDEN, N_INTER, N2, DEV)
from sglang.srt.layers.moe.moe_runner.aiter import _moe_decode_megakernel_full


def bench_fn(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def main():
    torch.manual_seed(0)
    print("=== Phase 7 tiny: megakernel-only timing ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER}\n")
    print(f"{'batch':>7} {'npm':>5} {'megakernel us':>14}")
    for bs in [1, 2, 4, 8, 12, 16, 32, 64]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        st, ei, npm = build_routing_from_topk(ti, DEV)
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        w2f, w2s = per_block_quant_fp8(w2_bf16)
        t_mk = bench_fn(lambda: _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw.to(torch.float32), ti))
        print(f"{bs:>7} {npm:>5} {t_mk:>14.1f}")


if __name__ == "__main__":
    main()
