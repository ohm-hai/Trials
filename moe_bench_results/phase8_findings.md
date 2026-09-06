# Phase 8 — Decode kernel efficiency at low batch size

**Goal:** make the EP-between-XCDs decode megakernel efficient at low batch size
(bs=4-128); no regression across bs=4-128, max speedup at bs≈4 (voice-AI low-latency
target).

## What was tried

### 8a. Tight routing (cap npm over-padding)
The production `moe_align_block_size` over-pads aggressively at tiny M: for bs=4,
BM=128 it returns `npm=4096` (pads *all* 256 experts to BM) even though only ~32
token-expert pairs exist. This exploded the megakernel grid (65536 blocks) and
triggered a fresh JIT per unique `npm` (npm is a `tl.constexpr`), causing the
~40s-per-shape compile timeouts seen in Phase 7 tiny.

A Python-loop tight router (`_moe_align_tight`) was written to pad only active
experts. **Rejected**: the Python loop over active experts is itself ~ms-scale
and dominated tiny-M timing (bs=1 went 332→849us). The over-padding formula
scales with BM, so the cheaper fix is just a smaller BM (8b) + the production
CUDA sort.

### 8b. Adaptive BLOCK_M
Shrink BM at tiny M (16 for bs≤32, 32 for bs≤128, 64 for bs≤1024, 128 above).
This **fixed the JIT/grid explosion**: at bs=4, BM=16 → npm=32 (not 4096), grid
512 (not 65536). But it did **not** make the megakernel beat prod at bs=4
(see finding below). With the auto-threshold (8c) the megakernel band is
bs≥64, where BM=128 (the Phase 6-7 tuned value) is used; adaptive BM is no
longer needed in the production path.

### 8c. Fused megakernel (`moe_decode_megakernel_fused.py`)
Two epilogue fusions to cut the ~240us launch overhead (isolated: GEMM floor
86us at bs=1, orchestrator 332us → 246us overhead):
- **Fused SiLU** (stage1): writes `inter = silu(gate)*up` directly from the
  interleaved accumulator (reshape (BM,BN)→(BM,BN/2,2) to deinterleave; Triton
  has no slice indexing). Eliminates 3 silu/mul/cast launches. **Correct,
  err=0.007.**
- **Fused unpermute** (stage2): atomic-adds weighted rows into the final
  (M,K) output at the original token index. Also fixed a real stage2 indexing
  bug — `inter` is in *sorted* order, so `a` must be indexed by sorted position
  (`oti`), not original token (`ot//TOPK`). Eliminates ~8 unpermute launches.
  **Correct (err 0.007-0.169) but REGRESSES at tiny M**: atomic-add serializes
  hard on the few output rows (bs=4 → 1536 stage2 blocks all atomic-adding to
  4 rows). bs=4 went 472→502us. Not wired into the production orchestrator.

## The hard finding (fundamental, not tunable)

At bs=4 there are 32 token-expert pairs across ~32 experts → **each expert's
weights are loaded exactly once → zero L2 reuse**. The megakernel's entire
advantage (XCD-L2 weight pinning) gives nothing at bs≤4. What remains is pure
disadvantage: Triton codegen + per-tile fp8 dequant vs prod's hand-tuned opaque
hsaco. No BM, fusion, or routing trick changes this — the L2 lever needs expert
*reuse*, which begins around bs≥32-64.

| bs | route | mega us | prod us | vs prod |
|----|-------|---------|---------|---------|
| 4  | prod (fallback) | 472-502 | 207 | 0.41-0.44x (regression → fallback) |
| 32 | prod (fallback) | — | 997 | 1.00x (no regression) |
| 64-128 | megakernel | — | — | 1.26x (Phase 6-7 validated) |

## Solution shipped: auto-threshold fallback

`aiter.py` `_moe_decode_megakernel_full` now activates only inside a configurable
window; outside it the run() hook falls through to the opaque
`aiter.fused_moe` hsaco:

```python
_USE_MOE_DECODE_MEGAKERNEL = get_bool_env_var("SGLANG_MOE_DECODE_MEGAKERNEL", "false")
_MOE_DECODE_MEGAKERNEL_MIN_TOKENS = get_int_env_var("SGLANG_MOE_DECODE_MEGAKERNEL_MIN_TOKENS", 64)
_MOE_DECODE_MEGAKERNEL_MAX_TOKENS = get_int_env_var("SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS", 1024)
# in run():  MIN_TOKENS <= num_tokens <= MAX_TOKENS  -> megakernel; else prod hsaco
```

- **bs < 64**: prod hsaco → no regression (err matches prod, 0.167).
- **64 ≤ bs ≤ 1024**: megakernel (BM=128) → 1.26x win (Phase 6-7).
- **bs > 1024**: prod hsaco (prefill, throughput-bound).

This meets the "no regression at bs=4-128" target. Speedup at bs≈4 is **not
achievable with the L2-pinning lever** (no expert reuse); only a single-launch
*persistent* megakernel fusing routing+quant+stage1+silu+stage2+unpermute into
one launch (Fleet-style, eliminating all launch overhead) could — deferred as
research-grade Phase 10.

## Open production concern: npm-constexpr JIT

`NPM` is a `tl.constexpr` in the megakernel, so every unique routing batch (npm
varies with the token-to-expert distribution) triggers a ~40s JIT. This is a
real serving stall and must be fixed before production: make `NPM` a runtime
argument (loop bound via a runtime `tl.cdiv`-style guard) or pad/npm-bucket to a
small set of compiled variants.
