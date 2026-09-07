"""Phase 9c: prefill via FUSED megakernel (stage1 fused_silu, stage2 fused
unpermute/atomic-add into final (M,K) output). Eliminates the 3x unpermute
overhead (no (T,K) buffer, no index_add_). Microbench vs prod hsaco, (bs x seq)."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel_fused import (
    moe_decode_megakernel_stage1_fused, moe_decode_megakernel_stage2_fused)
from aiter.fused_moe import fused_moe, QuantType, ActivationType

BM = 128; BK = 128


def moe_prefill_fused(h, w13f, w13s, w2f, w2s, tw, ti):
    """Fused prefill: stage1 fused_silu -> inter; stage2 fused unpermute -> out(M,K)."""
    M, K = h.shape
    a1, a1s = per_token_group_quant_fp8(h, BK)
    st, ei, _ = moe_align_block_size(ti, BM, E)
    npm = ei.shape[0]
    inter = torch.empty(npm * BM, N_INTER, dtype=torch.bfloat16, device=DEV)
    moe_decode_megakernel_stage1_fused(a1, w13f, a1s, w13s, st, ei, inter, TOPK,
                                        block_m=BM, block_k=BK, fused_silu=True)
    a2, a2s = per_token_group_quant_fp8(inter, BK)
    out_final = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    moe_decode_megakernel_stage2_fused(a2, w2f, a2s, w2s, st, ei, out_final,
                                         tw.to(torch.float32), TOPK, block_m=BM, block_k=BK)
    return out_final.to(torch.bfloat16)


def bench_fn(fn, warmup=10, iters=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def bf16_ref(h, w13, w2, ti, tw):
    M, K = h.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    for t in range(M):
        for k in range(TOPK):
            e = int(ti[t, k].item())
            g = w13[e, 0::2].to(torch.float32); u = w13[e, 1::2].to(torch.float32)
            inter = torch.nn.functional.silu(h[t].to(torch.float32) @ g.T) * (h[t].to(torch.float32) @ u.T)
            out[t] += (inter @ w2[e].to(torch.float32).T) * tw[t, k].to(torch.float32)
    return out.to(torch.bfloat16)


def main():
    torch.manual_seed(0)
    w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
    w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
    w13f, w13s = per_block_quant_fp8(w13_bf16)
    w2f, w2s = per_block_quant_fp8(w2_bf16)
    # correctness on small M
    print("=== Phase 9c: fused prefill megakernel vs prod hsaco ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER} BM={BM}\n")
    Mc = 128
    h = torch.randn(Mc, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
    ti = torch.randint(0, E, (Mc, TOPK), dtype=torch.int32, device=DEV)
    tw = torch.rand(Mc, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
    out_f = moe_prefill_fused(h, w13f, w13s, w2f, w2s, tw, ti)
    out_r = bf16_ref(h, w13_bf16, w2_bf16, ti, tw)
    err = (out_f.float() - out_r.float()).abs().max().item()
    print(f"correctness M={Mc}: max_abs_err={err:.4f} out_mag={out_r.float().abs().max().item():.4f} rel={err/(out_r.float().abs().max().item()+1e-6):.3f}\n")
    print(f"{'bs':>5} {'seq':>5} {'M':>7} {'fused us':>9} {'prod us':>9} {'spd':>7}")
    grid = [(1,2048),(1,4096),(1,8192),(1,16384),(2,2048),(4,1024),(4,2048),
             (8,512),(8,1024),(8,2048),(16,256),(16,512),(16,1024),(16,2048),
             (32,128),(32,256),(32,512),(32,1024),
             (64,64),(64,128),(64,256),(64,512),
             (128,64),(128,128),(128,256),(128,512)]
    for bs, seq in grid:
        M = bs * seq
        h = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        def mk(): return moe_prefill_fused(h, w13f, w13s, w2f, w2s, tw, ti)
        def prod():
            return fused_moe(hidden_states=h, w1=w13f, w2=w2f, topk_weight=tw.to(torch.float32),
                               topk_ids=ti, quant_type=QuantType.per_128x128,
                               activation=ActivationType.Silu,
                               w1_scale=w13s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                               gate_mode="interleave")
        try:
            t_mk = bench_fn(mk); t_p = bench_fn(prod)
            print(f"{bs:>5} {seq:>5} {M:>7} {t_mk:>9.1f} {t_p:>9.1f} {t_p/t_mk:>6.2f}x")
        except Exception as ex:
            print(f"{bs:>5} {seq:>5} {M:>7} {'ERR':>9} {str(ex)[:30]:>9}")


if __name__ == "__main__":
    main()
