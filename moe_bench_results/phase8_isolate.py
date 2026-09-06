"""Phase 8 isolate: measure ONLY the stage1 megakernel GEMM (no orchestrator
overhead) at tiny bs, to find the GEMM floor vs orchestrator overhead."""
import torch, sys
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import (per_block_quant_fp8, build_routing_from_topk,
    E, TOPK, K_HIDDEN, N_INTER, N2, DEV)
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel import moe_decode_megakernel_stage1
from sglang.srt.layers.moe.moe_runner.aiter import _moe_decode_megakernel_full


def bench_fn(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def main():
    torch.manual_seed(0)
    print("=== Phase 8 isolate: GEMM floor vs orchestrator overhead ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER}\n")
    print(f"{'bs':>5} {'BM':>4} {'npm':>5} {'s1 us':>7} {'s1+s2 us':>9} {'orch us':>9} {'overhead':>9}")
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        if bs <= 32: BM = 16
        elif bs <= 128: BM = 32
        else: BM = 64
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        st, ei, ntp = moe_align_block_size(ti, BM, E)
        npm = ei.shape[0]
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        w2f, w2s = per_block_quant_fp8(w2_bf16)
        a1f, a1s = per_token_quant_fp8_local(h)
        g1 = torch.zeros(npm*BM, N2, dtype=torch.bfloat16, device=DEV)
        g2 = torch.zeros(npm*BM, K_HIDDEN, dtype=torch.bfloat16, device=DEV)
        # stage1 only
        def s1(): moe_decode_megakernel_stage1(a1f, w13f, a1s, w13s, st, ei, g1, TOPK, block_m=BM, block_n=128, block_k=128)
        t_s1 = bench_fn(s1)
        # stage1 + stage2 (no silu/quant/unpermute)
        def s12():
            moe_decode_megakernel_stage1(a1f, w13f, a1s, w13s, st, ei, g1, TOPK, block_m=BM, block_n=128, block_k=128)
            a2f, a2s = per_token_quant_fp8_local(g1)
            moe_decode_megakernel_stage1(a2f, w2f, a2s, w2s, st, ei, g2, TOPK, block_m=BM, block_n=128, block_k=128)
        t_s12 = bench_fn(s12)
        # full orchestrator
        t_orch = bench_fn(lambda: _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw.to(torch.float32), ti))
        print(f"{bs:>5} {BM:>4} {npm:>5} {t_s1:>7.1f} {t_s12:>9.1f} {t_orch:>9.1f} {t_orch-t_s12:>9.1f}")


def per_token_quant_fp8_local(x):
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
    return per_token_group_quant_fp8(x, 128)


if __name__ == "__main__":
    main()
