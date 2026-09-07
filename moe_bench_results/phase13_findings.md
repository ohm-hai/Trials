# Phase 13: Honest Decode Re-validation After the `A_BY_SORTED` + `npm` Bugfixes

## TL;DR
The previously reported **1.26–3.60x decode win was an artifact of a correctness
bug.** On the *corrected* kernel the output is now right, but the megakernel is
**2.4–3x slower** than production `aiter.fused_moe` hsaco at every decode batch
size (64–1024). The XCD L2-pinning lever does **not** produce a win on GLM-5.2
with a Triton megakernel, for decode or prefill.

## Background
- Phase 6–7 reported a decode win of 1.26x–3.60x for the EP-between-XCDs
  N-split megakernel vs production hsaco.
- Phase 9 discovered a **stage-2 a-indexing bug** (`A_BY_SORTED`): stage-2 read
  the `inter` activation by original-token index instead of sorted position,
  so the output was **wrong** (`rel_err = 1.0` vs bf16 ref). The "fast" numbers
  were measured on this buggy (wrong-output) kernel.
- Phase 13 is the re-validation on the **fixed** kernel.

## Two bugs found and fixed during re-validation

### Bug 1 — `A_BY_SORTED` stage-2 a-indexing (found in Phase 9)
Stage-2 GEMM activation `inter` is laid out in **sorted** order, but the kernel
indexed it by `ot // TOPK` (original token). Fix: added `A_BY_SORTED: tl.constexpr`
to `_moe_decode_mega_kernel`; the orchestrator passes `a_by_sorted=True` for
stage-2 so it indexes by sorted position `oti`.

### Bug 2 — `npm` = tensor capacity, not valid block count (found in Phase 13)
The orchestrator computed `npm = expert_ids.shape[0]`. But `moe_align_block_size`
allocates `expert_ids` and `sorted_token_ids` with `torch.empty` at **max
capacity** (`max_num_m_blocks`), and only the first `num_tokens_post_padded // BM`
blocks are valid. The tail is **uninitialized garbage** — and with the real
orchestrator's allocation pattern that garbage is huge **negative int32**
(`eid.min = -1,162,166,443`).

The kernel clamped `oe = tl.minimum(oe, EM - 1)` on the **upper side only**.
A negative `oe` → `b_ptr + (-1.16e9) * stride` → a massive out-of-bounds GPU
read → **queue fault → hang**. This is why the megakernel hung deterministically
in Phase 13 (and non-deterministically elsewhere, depending on what
`torch.empty` left in memory).

**Fix (orchestrator, `aiter.py`):**
```python
npm = num_tokens_post_padded.item() // BM   # valid count, NOT capacity
```
**Fix (kernel, `moe_decode_megakernel.py` + fused):**
```python
oe = tl.maximum(tl.minimum(oe, EM - 1), 0)   # clamp BOTH sides
```
`num_tokens_post_padded` is always a multiple of `BM` (each expert is padded to
a multiple of `BM`), so the division is exact and the garbage tail is never read.

## Re-validation (correctness + timing)
Script: `phase13_decode_revalidate.py`. GPU 5 (clean). Fresh triton cache.
GLM-5.2 FP8: E=256, topk=8, K=6144, N_inter=2048, N2=4096.

### Correctness
`rel_err = 0.0753` at bs=64 vs bf16 reference. ✅ The fixed kernel produces
**correct** output (0.075 is normal fp8 quant error; 1.0 would mean wrong).

### Timing (megakernel vs production `aiter.fused_moe` hsaco)
| batch | megakernel (µs) | prod fused_moe (µs) | speedup |
|------:|----------------:|-------------------:|--------:|
| 64    | 3706            | 1412               | 0.38x   |
| 128   | 4000            | 1640               | 0.41x   |
| 256   | 4376            | 1689               | 0.39x   |
| 512   | 5090            | 1780               | 0.35x   |
| 1024  | 6470            | 2153               | 0.33x   |

The megakernel is **2.4–3x slower** than prod hsaco at every decode batch size.

## Why the previous "win" was bogus
The buggy stage-2 path read wrong activation rows. That is less *effective*
work in a sense, but more importantly it produced **wrong output** — the speed
was measuring a kernel that was not computing the right thing. Once the
indexing is correct, the kernel is slow. There was never a real decode win.

## Root cause of the slowness (same as prefill)
1. The megakernel is a **generic Triton FP8 GEMM**; production uses AMD's
   hand-tuned **hsaco `fmoe_g1u1`** ASM kernels, which have far higher compute
   throughput.
2. The L2-pinning benefit (the whole point of the design) is real in principle,
   but it does **not** come close to closing the raw compute gap between Triton
   and hsaco.
3. The orchestrator overhead (Python dispatch, `moe_align_block_size`, two
   per-token-group quant passes, `index_add_` unpermute) adds further cost.

## Conclusion / what this means for the lever
- The **XCD L2-locality problem is real** (Phase 1 measured 0% L2 hit).
- But the **Triton megakernel is the wrong tool** to exploit it: it cannot match
  hsaco's compute throughput, so any L2 benefit is swamped.
- To actually exploit EP-between-XCDs L2 pinning on GLM-5.2, you would need to
  either:
  - (a) modify the **opaque hsaco** kernels directly (not feasible from
    Triton; requires AMD-internal source), or
  - (b) write a **hand-tuned ASM/WMMA megakernel** that matches hsaco compute
    throughput *and* adds chiplet locality — a large, research-grade effort
    beyond a Triton-level prototype.

## Files changed in this phase
- `sglang_integration/aiter.py` (`_moe_decode_megakernel_full`): `npm = ntp // BM`.
- `aiter_module/.../moe_decode_megakernel.py`: `oe = max(min(oe, E-1), 0)`.
- `aiter_module/.../moe_decode_megakernel_fused.py`: same clamp.
- `moe_bench_results/phase13_decode_revalidate.py`: added correctness check.
- `moe_bench_results/phase13_stepdiag.py`, `mega_*.py`: root-cause diagnostics.

## Honest status of all prior claims
| Claim | Status |
|---|---|
| Phase 1: 0% L2 hit / no locality on MI355X (rocprofv2) | ✅ Valid |
| Phase 2–5: megakernel correct (vs bf16, max_abs_diff=0) | ✅ Valid (round-robin ref) |
| Phase 6–7: **1.26–3.60x decode win** | ❌ **Retracted** — artifact of the `A_BY_SORTED` bug |
| Phase 8: low-bs auto-threshold (prod for bs<64) | ✅ Valid (and now moot — prod wins everywhere) |
| Phase 9: prefill megakernel loses to hsaco (0.23–0.53x) | ✅ Valid |
| Phase 9: two-band router (megakernel decode, hsaco prefill) | ⚠️ Moot — megakernel loses at decode too |
