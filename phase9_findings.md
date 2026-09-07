# Phase 9: Prefill Megakernel — Build, Correctness, Microbench

## Goal
Build the prefill MoE megakernel (expert-sequential + N-split, 2D locality for
LLC-resident weights) and microbench it vs prod `aiter.fused_moe` hsaco across a
(batch_size × seq_len) grid → M = batch_size × seq_len.

## What was built
1. `moe_prefill_megakernel.py` — per-expert prefill GEMM kernel skeleton with
   MODE=0 (round-robin) and MODE=1 (expert-sequential M-split within expert),
   plus `A_BY_SORTED` for stage-2 (inter is in sorted order).
2. Reused the **single-launch** decode megakernel (`moe_decode_megakernel_stage1`)
   via the verified integrated orchestrator (`_moe_decode_megakernel_full`) at
   prefill M. The single-launch structure (expert-sequential via sorted `ei` +
   N-split across XCDs) already implements the 2D-locality lever for prefill.
   Per-expert launch was confirmed catastrophically slow (0.04-0.15x) due to 256
   Python-loop launches — a single launch is mandatory.

## Critical bug found & fixed (Phase 9bf)
The **base** decode megakernel indexed activation `a` by `ot // TOPK` (original
token) for BOTH stages. Stage-1 is correct (hidden is in original-token order),
but stage-2's `inter` is in **sorted** order → stage-2 read the wrong rows,
producing garbage (rel_err = 1.0 vs bf16 ref, i.e. uncorrelated output).

**Fix**: added `A_BY_SORTED` constexpr to `_moe_decode_mega_kernel` and
`moe_decode_megakernel_stage1(a_by_sorted=...)`. The integrated orchestrator
now passes `a_by_sorted=True` for stage-2. Verified: rel_err 0.055-0.074 vs
bf16 ref (proper FP8 noise). This fix applies to **both decode and prefill**.

(The earlier "verified" decode orchestrator must have used the fused kernel,
which already had this fix; the base-kernel orchestrator was silently wrong.)

## Correctness (Phase 9a)
- MODE 0 (round-robin) vs MODE 1 (expert-sequential M-split): bit-identical
  (max_abs_diff = 0.0) at M=2048, 4096, 8192. Per-expert prefill kernel is correct.
- Integrated orchestrator vs bf16 ref (after fix): rel_err 0.055-0.074.

## Microbench (Phase 9b) — 26 (bs, seq_len) combinations
M = 2048 → 65536. Megakernel orchestrator vs prod hsaco:

| bs | seq | M | mk us | prod us | spd | err |
|----|-----|------|-------|----------|-----|-----|
| 1 | 2048 | 2048 | 9392 | 2709 | 0.29x | 0.23 |
| 1 | 16384 | 16384 | 50550 | 15207 | 0.30x | 0.26 |
| 16 | 2048 | 32768 | 96556 | 30488 | 0.32x | 0.28 |
| 128 | 512 | 65536 | 192015 | 56051 | 0.29x | 0.27 |

(Full 26-row table in phase9b_prefill.py output.)

**Result**: megakernel orchestrator is **0.26-0.32x** of prod at prefill — slower.
The err vs prod (~0.22) is a gate/up layout convention difference (megakernel
matches bf16 ref, so it is correct; prod uses a different convention in this
synthetic test — does not affect the performance comparison).

## Root cause: unpermute dominates (Phase 9b-iso)
Isolated timing (M = 2048..65536):

| M | npm | gemm us | quant+silu us | unpermute us | total us |
|------|------|--------|--------------|------------|---------|
| 2048 | 383 | 4878 | 416 | 11976 | 16853 |
| 8192 | 767 | 9775 | 823 | 24433 | 34207 |
| 32768 | 2303 | 28771 | 2622 | 74020 | 102791 |
| 65536 | 4351 | 54451 | 5029 | 143726 | 198177 |

- **Unpermute (zeros(T,K) + index_add_) is 3x the GEMM time** — the bottleneck.
  At M=65536, unpermute = 144ms vs GEMM = 54ms. The `zeros(T=M*TOPK, K)` buffer
  is huge (12.9 GB at M=65536) and the index_add_ scatter is unfused.
- **GEMMs alone are competitive at large M**: 54ms vs prod 56ms at M=65536
  (~0.96x). At small prefill M (2048), GEMMs are 4.9ms vs prod 2.7ms (~1.8x).
- quant+silu is small (0.4-5ms).

## Next lever (Phase 9c): fuse unpermute into stage-2
Eliminate the `(T, K)` buffer and the separate scatter by having stage-2 write
directly to the final `(M, K)` output via atomic-add (each (token, topk_idx) is
unique → scatter-store; topk=8 experts sum per token → atomic-add). This is the
same fusion the decode fused kernel uses. Expected to remove the 3x overhead
and make the prefill megakernel competitive (GEMMs are already ~1x at large M).

### Phase 9c result (fused prefill)
Built `moe_prefill_fused` using `moe_decode_megakernel_stage1_fused(fused_silu=True)`
+ `moe_decode_megakernel_stage2_fused` (atomic-add into `(M,K)` output).
Correctness: rel_err 0.072 vs bf16 ref (correct).

| bs | seq | M | fused us | prod us | spd |
|----|-----|------|---------|---------|-----|
| 1 | 2048 | 2048 | 11696 | 2719 | 0.23x |
| 1 | 8192 | 8192 | 20197 | 7583 | 0.38x |
| 1 | 16384 | 16384 | 28755 | 15193 | 0.53x |
| 16 | 1024 | 16384 | 29931 | 15190 | 0.51x |

Fusing the unpermute **did help scaling** (0.23→0.53x as M grows 2048→16384,
vs 0.26→0.32x non-fused) — the atomic-add amortizes over more GEMM work. But it
is **still slower than prod hsaco everywhere** (best 0.53x). The fused kernel
also hit a memory-access fault at M>=32768 (atomic-add / over-padding grid).
The atomic-add to `(M,K)` fp32 is itself costly (8M row-atomics at M=16384).

## Verdict: prefill megakernel does NOT beat prod hsaco
The decode win (1.26-3.60x via L2 residency) does **not** transfer to prefill:
1. At prefill M (>=2048) the working set is large → L2/LLC locality benefit is
   small relative to raw GEMM throughput.
2. Prod hsaco (FlyDSL/ASM) is already highly tuned for large prefill GEMMs.
3. Megakernel overhead (atomic-add unpermute, moe_align over-padding, Python
   orchestrator dispatch) is proportionally larger and not absorbed.
4. GEMM-only is competitive only at very large M (~0.96x at M=65536), but the
   full orchestrator (with unpermute) never catches up.

**Recommendation**: two-band router — **megakernel for decode (M <= 1024)**,
**prod hsaco for prefill (M >= 2048)**. This is already wired via
`SGLANG_MOE_DECODE_MEGAKERNEL` + `_MOE_DECODE_MEGAKERNEL_MAX_TOKENS=1024` in
`aiter.py`. No prefill kernel change is warranted; the lever is decode-only.

### Bugfix note (important)
The `A_BY_SORTED` stage-2 fix applies to the **decode** orchestrator too — the
base-kernel decode path was silently producing wrong stage-2 results before
this fix. The fused decode kernel already had the fix; the base-kernel
orchestrator did not. Now both are correct.

