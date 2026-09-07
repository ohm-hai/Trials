"""Phase 8e: verify runtime-NPM -> no per-npm JIT. Run two different npm (bs=64
then bs=128 -> different npm) and time the FIRST call of the second shape; if
NPM is runtime, the second shape reuses the cached kernel (first call fast)."""
import torch, sys, time
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel import moe_decode_megakernel_stage1

def first_call_us(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    fn(); torch.cuda.synchronize()
    return (time.time() - t0) * 1e6

def run_stage1(h, w13f, w13s, ti, BM):
    M, K = h.shape
    st, ei, ntp = moe_align_block_size(ti, BM, E)
    npm = ei.shape[0]
    a1f, a1s = per_token_group_quant_fp8(h, 128)
    out = torch.zeros(npm * BM, N2, dtype=torch.bfloat16, device=DEV)
    moe_decode_megakernel_stage1(a1f, w13f, a1s, w13s, st, ei, out, TOPK, block_m=BM, block_n=128, block_k=128)
    return npm

def main():
    torch.manual_seed(0)
    w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
    w13f, w13s = per_block_quant_fp8(w13_bf16)
    BM = 128
    print("=== Phase 8e: runtime-NPM no per-npm JIT ===")
    # shape A: bs=64
    h64 = torch.randn(64, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
    ti64 = torch.randint(0, E, (64, TOPK), dtype=torch.int32, device=DEV)
    npm_a = run_stage1(h64, w13f, w13s, ti64, BM)
    tA = first_call_us(lambda: run_stage1(h64, w13f, w13s, ti64, BM))
    # shape B: bs=128 -> different npm
    h128 = torch.randn(128, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
    ti128 = torch.randint(0, E, (128, TOPK), dtype=torch.int32, device=DEV)
    npm_b = run_stage1(h128, w13f, w13s, ti128, BM)
    tB_first = first_call_us(lambda: run_stage1(h128, w13f, w13s, ti128, BM))
    print(f"shape A bs=64  npm={npm_a} first-call(JIT)={tA/1e3:.0f}ms")
    print(f"shape B bs=128 npm={npm_b} first-call={tB_first/1e3:.1f}ms  (should be <50ms if no per-npm JIT)")
    verdict = "PASS (runtime NPM)" if tB_first < 50000 else "FAIL (still per-npm JIT)"
    print(f"verdict: {verdict}")

if __name__ == "__main__": main()
