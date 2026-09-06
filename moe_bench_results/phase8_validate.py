"""Phase 8 validate: modified orchestrator (tight routing + adaptive BM)
vs production aiter.fused_moe at bs=1,2,4,8,12,16,32,64,128. Correctness + speedup."""
import torch, sys
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import (per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV)
from sglang.srt.layers.moe.moe_runner.aiter import _moe_decode_megakernel_full
from aiter.fused_moe import fused_moe, QuantType, ActivationType


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
    print("=== Phase 8: tight-routing + adaptive-BM megakernel vs production ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER}\n")
    print(f"{'bs':>5} {'BM':>4} {'npm':>5} {'mk us':>9} {'prod us':>9} {'spd':>7} {'corr':>7}")
    for bs in [1, 2, 4, 8, 12, 16, 32, 64, 128]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        w2f, w2s = per_block_quant_fp8(w2_bf16)
        # megakernel (tight routing + adaptive BM)
        out_mk = _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw.to(torch.float32), ti)
        t_mk = bench_fn(lambda: _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw.to(torch.float32), ti))
        # production aiter.fused_moe (opaque hsaco) - same weights
        def prod():
            return fused_moe(hidden_states=h, w1=w13f, w2=w2f,
                             topk_weight=tw.to(torch.float32), topk_ids=ti,
                             quant_type=QuantType.per_128x128,
                             activation=ActivationType.Silu,
                             w1_scale=w13s, w2_scale=w2s, a1_scale=None, a2_scale=None)
        try:
            out_prod = prod()
            t_prod = bench_fn(prod)
            # correctness: compare mk vs prod
            err = (out_mk.to(torch.float32) - out_prod.to(torch.float32)).abs().max().item()
            spd = t_prod / t_mk
            print(f"{bs:>5} {'?':>4} {'?':>5} {t_mk:>9.1f} {t_prod:>9.1f} {spd:>6.2f}x err={err:.3f}")
        except Exception as ex:
            print(f"{bs:>5} {'?':>4} {'?':>5} {t_mk:>9.1f} {'ERR':>9} {str(ex)[:40]}")


if __name__ == "__main__":
    main()
