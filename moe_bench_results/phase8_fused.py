"""Phase 8 fused: validate fused-silu + fused-unpermute megakernel orchestrator
vs a clean bf16 reference. Correctness + timing at bs=1,2,4,8,16,32,64,128."""
import torch, sys
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import (per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel_fused import (
    moe_decode_megakernel_stage1_fused, moe_decode_megakernel_stage2_fused)


def bench_fn(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def bf16_ref(h, w13, w2, ti, tw):
    """Clean bf16 reference MoE: for each token, gather its experts, do
    gate-up @ w1, silu, @ w2, weight, sum over topk."""
    M, K = h.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=h.device)
    for t in range(M):
        acc = torch.zeros(K, dtype=torch.float32, device=h.device)
        for j in range(TOPK):
            e = int(ti[t, j])
            g = h[t] @ w13[e].t()          # (N2,)
            gate = g[0::2]; up = g[1::2]
            inter = torch.nn.functional.silu(gate) * up   # (N_inter,)
            o = inter @ w2[e].t()           # (K,)
            acc += o * float(tw[t, j])
        out[t] = acc
    return out.to(torch.bfloat16)


def moe_fused(h_bf16, w13f, w13s, w2f, w2s, tw, ti, BM):
    M, K = h_bf16.shape
    st, ei, ntp = moe_align_block_size(ti, BM, E)
    npm = ei.shape[0]
    a1f, a1s = per_token_group_quant_fp8(h_bf16, 128)
    inter = torch.zeros(npm * BM, N_INTER, dtype=torch.bfloat16, device=DEV)
    moe_decode_megakernel_stage1_fused(
        a1f, w13f, a1s, w13s, st, ei, inter, TOPK,
        block_m=BM, block_n=128, block_k=128, fused_silu=True)
    a2f, a2s = per_token_group_quant_fp8(inter, 128)
    out = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    moe_decode_megakernel_stage2_fused(
        a2f, w2f, a2s, w2s, st, ei, out, tw.to(torch.float32).reshape(-1), TOPK,
        block_m=BM, block_n=128, block_k=128)
    return out.to(torch.bfloat16)


def main():
    torch.manual_seed(0)
    print("=== Phase 8 fused: silu+unpermute fusion vs bf16 ref ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER}\n")
    print(f"{'bs':>5} {'BM':>4} {'npm':>5} {'fused us':>9} {'ref err':>9}")
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        w2f, w2s = per_block_quant_fp8(w2_bf16)
        BM = 16 if bs <= 32 else (32 if bs <= 128 else 64)
        out = moe_fused(h, w13f, w13s, w2f, w2s, tw, ti, BM)
        ref = bf16_ref(h, w13_bf16, w2_bf16, ti, tw)
        err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
        t = bench_fn(lambda: moe_fused(h, w13f, w13s, w2f, w2s, tw, ti, BM))
        print(f"{bs:>5} {BM:>4} {'?':>5} {t:>9.1f} {err:>9.3f}")


if __name__ == "__main__":
    main()
