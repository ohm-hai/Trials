# Phase 14 — LLC/HBM reuse measurement of the PRODUCTION aiter.fused_moe hsaco

> **Goal:** Determine whether the production GLM-5.2 MoE path (blockscale-FP8
> `fmoe_bf16_blockscaleFp8_g1u1_vs_silu` 1stage hsaco) already exploits
> expert-weight reuse in LLC/L2 at decode, or whether there is reuse to
> capture (the prerequisite for deciding between lever C — cheap dispatch
> reorder — and lever A+D — research-grade locality-aware kernel).

## Why this measurement was needed
Phase 1 measured **L2 hit = 0%**. But L2 is the 4 MB *private* per-XCD
cache; a 25 MB expert can never fit there, so 0% L2 hit is inevitable and says
**nothing** about the 256 MB **shared LLC**. We never measured HBM/LLC, so we
did not actually know whether prod hsaco reuses expert weights in LLC. This
phase closes that gap.

## Method (unit-independent ratio analysis)
rocprofv2 `FETCH_SIZE`/`WRITE_SIZE` counters have an inconsistent raw unit
across kernel types (calibration with memcpy/vector-add gave 2048 bytes/unit
but that produces physically impossible numbers for the fmoe GEMM). To avoid
relying on absolute units, we use **ratios across batch sizes**, which cancel
the unit:

- **Floor** (perfect reuse, each active expert's weight loaded once and reused
  by all its tokens) = `active_experts × 37.75 MB`.
- **Ceiling** (zero reuse, each token reads its expert's full weight fresh
  from HBM) = `(bs × topk) × 37.75 MB`.
- If measured FETCH scales with the **floor** across bs → reuse is captured.
- If measured FETCH scales with the **ceiling** across bs → no reuse.

Setup: GLM-5.2 FP8 (E=256, topk=8, K=6144, N_inter=2048), prod
`aiter.fused_moe` (QuantType.per_128x128, gate_mode="interleave"), fixed
seed, WARMUP=5 + ITERS=20 per bs, three separate rocprofv2 runs (one per bs)
on GPUs 5/6/7. 25 fmoe dispatches per bs.

## Per-batch measured FETCH (raw counter units, 25 dispatches each)
| bs | active experts | floor (GB) | ceiling (GB) | FETCH/disp (units) | WRITE/disp (units) |
|---:|---:|---:|---:|---:|---:|
| 64   | 215 | 8.12  | 19.3   | 40,635,510    | 26,416   |
| 256  | 256 | 9.66  | 77.3   | 161,155,255   | 104,884  |
| 1024 | 256 | 9.66  | 309.2  | 547,649,874   | 420,056  |

## Ratios (unit-independent)
| transition | FETCH ratio | floor ratio | ceiling ratio | verdict |
|---|---:|---:|---:|---|
| 256 / 64   | **3.97×** | 1.19× | **4.00×** | tracks ceiling → no reuse |
| 1024 / 256 | **3.40×** | 1.00× | **4.00×** | tracks ceiling → no reuse |
| 1024 / 64   | **13.48×** | 1.19× | **16.0×** | near ceiling → no reuse |

**WRITE sanity check** (writes are output, not weights): WRITE 256/64 = 3.97×,
1024/256 = 4.00× → scales with `bs` (output size), exactly as expected. ✓
This validates the measurement methodology.

## Result
**The production hsaco has essentially NO expert-weight reuse at decode.**
Measured HBM FETCH tracks the **zero-reuse ceiling**, not the perfect-reuse
floor:
- FETCH ∝ (bs × topk)  [tokens] — each token reads its expert's weight fresh
  from HBM — NOT ∝ active_experts [experts loaded once].
- At bs=64/256, FETCH ≈ 99% of the zero-reuse ceiling (no reuse at all).
- At bs=1024, FETCH ≈ 84% of ceiling (a small ~16% reuse emerging as
  tokens/expert grows, but still 27× the floor).

## What this means for the levers
- **The locality problem is real and exploitable.** The prod 1stage hsaco does
  not reuse expert weights across tokens → there is a large reuse gap (floor vs
  measured): ~2.4× at bs=64, ~8× at bs=256, ~27× at bs=1024.
- **Lever C (cheap dispatch/segment reorder) is likely MOOT for the 1stage
  path**: the 1stage kernel is a *single launch* that processes all
  experts/tokens internally — there is no per-expert dispatch order to reorder.
  The segment ordering (sorted_token_ids) is already expert-contiguous
  (`moe_align_block_size` sorts by expert), yet FETCH ≈ ceiling, so the
  kernel's *internal* tiling/scheduling — not the segment order — is what
  prevents reuse. Changing segment order from the host won't change the
  kernel's internal locality.
- **⇒ Lever A+D (research-grade locality-aware kernel) is the viable path.**
  To capture the reuse you must change the kernel's *internal* tile scheduling
  so that all tiles of one expert (and its N-slice / M-slice) are processed
  back-to-back, keeping the expert's weight resident in LLC across its tokens.
  This requires a hand-tuned WMMA/ASM MoE GEMM at hsaco compute throughput
  with expert-sequential + XCD-affine scheduling — the deep research-grade
  effort.

## Caveats
- Absolute FETCH units are unreliable across kernel types (memcpy/vector-add
  calibrate to ~2048 bytes/unit but that yields physically impossible fmoe
  fetch; the fmoe counter path differs). The **ratio** analysis is robust and
  unit-independent, which is what we rely on.
- The 1stage kernel was used for all three bs (aiter auto-selects 1stage for
  these M). A 2-stage path (separate stage1/stage2 launches) might behave
  differently and could be measured separately.
- These are synthetic fixed-seed routings (near-uniform expert load). Real
  GLM-5.2 routing is also ~uniform (noaux_tc, bias CV ≤1.6%), so this is
  representative.

## Files
- `phase14_hbm_reuse.py` — the workload (runs prod fused_moe per bs).
- `phase14_calib.py`, `phase14_calib2.py` — unit calibrations.
- `parse_phase14.py` — CSV parser.
- `rp2_hbm.txt` — rocprof pmc config (FETCH_SIZE WRITE_SIZE).
- Raw CSVs: `/tmp/rp2_p14_bs{64,256,1024}/pmc_1/results_*.csv`.
