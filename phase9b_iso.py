"""Phase 9b-iso: isolate megakernel GEMM time vs orchestrator overhead
(zeros/quant/unpermute) at prefill M, to find the real bottleneck."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, silu_gate_up, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel import moe_decode_megakernel_stage1

BM = 128; BK = 128

def bench(fn, warmup=10, iters=40):
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
    print("=== Phase 9b-iso: megakernel GEMM vs overhead (prefill M) ===\n")
    print(f"{'M':>7} {'npm':>5} {'gemm us':>8} {'quant+silu':>11} {'unpermute':>10} {'total':>8}")
    for M in [2048, 4096, 8192, 16384, 32768, 65536]:
        h = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        st, ei, _ = moe_align_block_size(ti, BM, E); npm = ei.shape[0]
        a1, a1s = per_token_group_quant_fp8(h, BK)
        g1 = torch.zeros(npm*BM, N2, dtype=torch.bfloat16, device=DEV)
        # GEMM-only (both stages)
        def gemms():
            moe_decode_megakernel_stage1(a1, w13f, a1s, w13s, st, ei, g1, TOPK, block_m=BM, block_k=BK)
            inter = silu_gate_up(g1, N_INTER)
            a2, a2s = per_token_group_quant_fp8(inter, BK)
            g2 = torch.zeros(npm*BM, K_HIDDEN, dtype=torch.bfloat16, device=DEV)
            moe_decode_megakernel_stage1(a2, w2f, a2s, w2s, st, ei, g2, TOPK, block_m=BM, block_k=BK, a_by_sorted=True)
            return g2, inter
        t_gemm = bench(gemms)
        # quant+silu only
        def qs():
            inter = silu_gate_up(g1, N_INTER)
            a2, a2s = per_token_group_quant_fp8(inter, BK)
            return a2
        t_qs = bench(qs)
        # unpermute only
        g2, inter = gemms()
        T = M*TOPK
        def unp():
            idx = st.clamp(max=T-1)
            vw = tw.reshape(-1)[idx].to(torch.bfloat16)
            out_flat = torch.zeros(T, K_HIDDEN, dtype=torch.float32, device=DEV)
            out_flat.index_add_(0, idx, g2[idx].to(torch.float32) * vw.unsqueeze(1).to(torch.float32))
            return out_flat.view(M, TOPK, K_HIDDEN).sum(1).to(torch.bfloat16)
        t_unp = bench(unp)
        print(f"{M:>7} {npm:>5} {t_gemm:>8.1f} {t_qs:>11.1f} {t_unp:>10.1f} {t_gemm+t_unp:>8.1f}")

if __name__ == "__main__":
    main()
