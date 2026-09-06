"""Phase 7 end-to-end: megakernel-full decode MoE vs production aiter.fused_moe
(opaque hsaco) at serving-level decode batches. Same weights fed to both paths.
This is the real serving-loop comparison."""
import torch, sys
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import (moe_decode_full, per_block_quant_fp8,
    build_routing_from_topk, E, TOPK, K_HIDDEN, N_INTER, N2, DEV)
from aiter.fused_moe import fused_moe, QuantType, ActivationType


def bench_fn(fn, warmup=10, iters=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def main():
    torch.manual_seed(0)
    print("=== Phase 7 E2E: megakernel-full vs production aiter.fused_moe ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER}\n")
    print(f"{'batch':>7} {'npm':>5} {'megakernel us':>14} {'prod fused_moe us':>18} {'spd':>7}")
    for bs in [256, 512, 1024, 2048]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        st, ei, npm = build_routing_from_topk(ti, DEV)
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        w2f, w2s = per_block_quant_fp8(w2_bf16)
        # megakernel-full
        t_mk = bench_fn(lambda: moe_decode_full(h, w13f, w13s, w2f, w2s, tw, st, ei, npm))
        # production aiter.fused_moe (opaque hsaco) - same weights
        def prod():
            return fused_moe(hidden_states=h, w1=w13f, w2=w2f,
                             topk_weight=tw.to(torch.float32), topk_ids=ti,
                             quant_type=QuantType.per_128x128,
                             activation=ActivationType.Silu,
                             w1_scale=w13s, w2_scale=w2s, a1_scale=None, a2_scale=None)
        try:
            t_prod = bench_fn(prod)
            spd = t_prod / t_mk
            print(f"{bs:>7} {npm:>5} {t_mk:>14.1f} {t_prod:>18.1f} {spd:>6.2f}x")
        except Exception as ex:
            print(f"{bs:>7} {npm:>5} {t_mk:>14.1f} {'ERR: '+str(ex)[:40]:>18}")


if __name__ == "__main__":
    main()
