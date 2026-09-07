"""Phase 9b-quick: single-launch decode megakernel orchestrator at PREFILL M sizes
vs prod hsaco, across (bs x seq_len) grid. Confirms single-launch structure works
for prefill (expert-sequential via sorted ei -> LLC resident)."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import (per_block_quant_fp8, moe_decode_full, build_routing_from_topk,
                                 E, TOPK, K_HIDDEN, N_INTER, N2, DEV, BLOCK_M)
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
    w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
    w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
    w13f, w13s = per_block_quant_fp8(w13_bf16)
    w2f, w2s = per_block_quant_fp8(w2_bf16)
    print("=== Phase 9b-quick: single-launch megakernel @ prefill M vs prod hsaco ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER} BM={BLOCK_M}\n")
    print(f"{'bs':>5} {'seq':>5} {'M':>7} {'mk us':>8} {'prod us':>9} {'spd':>7} {'err':>8}")
    grid = [(1,2048),(1,4096),(1,8192),(2,2048),(4,1024),(4,2048),
             (8,512),(8,1024),(8,2048),(16,256),(16,512),(16,1024),
             (32,128),(32,256),(32,512),(64,64),(64,128),(64,256)]
    for bs, seq in grid:
        M = bs * seq
        h = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        st, eid, npm = build_routing_from_topk(ti, DEV)
        def mk(): return moe_decode_full(h, w13f, w13s, w2f, w2s, tw, st, eid, npm)
        def prod():
            return fused_moe(hidden_states=h, w1=w13f, w2=w2f, topk_weight=tw.to(torch.float32),
                               topk_ids=ti, quant_type=QuantType.per_128x128,
                               activation=ActivationType.Silu,
                               w1_scale=w13s, w2_scale=w2s, a1_scale=None, a2_scale=None)
        try:
            out_mk = mk(); out_p = prod()
            err = (out_mk.to(torch.float32) - out_p.to(torch.float32)).abs().max().item()
            t_mk = bench_fn(mk); t_p = bench_fn(prod)
            print(f"{bs:>5} {seq:>5} {M:>7} {t_mk:>8.1f} {t_p:>9.1f} {t_p/t_mk:>6.2f}x {err:>8.3f}")
        except Exception as ex:
            print(f"{bs:>5} {seq:>5} {M:>7} {'ERR':>8} {str(ex)[:30]:>9}")


if __name__ == "__main__":
    main()
