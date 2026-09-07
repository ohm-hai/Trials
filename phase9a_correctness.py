"""Phase 9a: prefill stage-1 kernel correctness — MODE=0 (round-robin) vs
MODE=1 (expert-sequential M-split) must be bit-identical. Per-expert launch.
Uses production moe_align_block_size for routing, then splits by expert."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_prefill_megakernel import moe_prefill_stage1_expert

BM = 128; BN = 128; BK = 128

def split_by_expert(sorted_token_ids, expert_ids, M, topk):
    """Return list of (expert_id, sti_e, MB_e) for each expert that has >=1 block.
    expert_ids is sorted ascending; each expert occupies a contiguous block range.
    moe_align_block_size may leave padding-block expert_ids uninitialized (garbage);
    clamp to E-1 (padding rows are masked in the kernel, so grouping them under
    expert E-1 is safe -- their output is never read)."""
    out = []
    npm = expert_ids.shape[0]
    EM = E - 1
    eids = expert_ids.clamp(min=0, max=EM).tolist()
    e = eids[0]; start = 0
    for pm in range(1, npm):
        ee = eids[pm]
        if ee != e:
            out.append((e, sorted_token_ids[start*BM:pm*BM], pm - start))
            e = ee; start = pm
    out.append((e, sorted_token_ids[start*BM:npm*BM], npm - start))
    return out

def main():
    torch.manual_seed(0)
    print("=== Phase 9a: prefill stage-1 MODE 0 vs MODE 1 bit-equiv ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N2={N2} BM={BM}\n")
    for M in [2048, 4096, 8192]:
        h = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
        a1, a1s = per_token_group_quant_fp8(h, BK)
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        st, ei, ntp = moe_align_block_size(ti, BM, E)
        experts = split_by_expert(st, ei, M, TOPK)
        max_err = 0.0
        n_checked = 0
        for e_id, sti_e, MB_e in experts:
            if MB_e == 0: continue
            sti_e = sti_e.contiguous()
            b_e = w13f[e_id]; bs_e = w13s[e_id]
            out0 = torch.zeros(MB_e * BM, N2, dtype=torch.bfloat16, device=DEV)
            out1 = torch.zeros(MB_e * BM, N2, dtype=torch.bfloat16, device=DEV)
            moe_prefill_stage1_expert(a1, b_e, a1s, bs_e, sti_e, out0, TOPK, block_m=BM, block_n=BN, block_k=BK, mode=0)
            moe_prefill_stage1_expert(a1, b_e, a1s, bs_e, sti_e, out1, TOPK, block_m=BM, block_n=BN, block_k=BK, mode=1)
            err = (out0.to(torch.float32) - out1.to(torch.float32)).abs().max().item()
            max_err = max(max_err, err); n_checked += 1
        print(f"M={M:>6} experts_checked={n_checked:>3} max_abs_diff={max_err:.6e}  {'PASS' if max_err == 0.0 else 'FAIL'}")

if __name__ == "__main__":
    main()
