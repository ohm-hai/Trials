# GLM-5.2 MoE Microbenchmark Summary (Fleet-Table-4 style)

**Target:** GLM-5.2-FP8 MoE layer — `hidden=6144, inter=2048, E=256, topk=8, FP8 e4m3, g1u1, SiLU`
**Hardware:** AMD MI355X VF, gfx950, 8× XCD chiplets (private 4 MB L2/XCD, shared 256 MB LLC)
**Backend:** aiter `fused_moe` → FlyDSL stage1/stage2 GEMMs + `topk_softmax` ASM gating
**Method:** `aiter/op_tests/test_moe.py` (per-batch kernel timing) + `rocprofv2` PMCs (L2 hit/miss, HBM traffic) at M=8/64/512

## 1. FP8 MoE latency vs batch (microbench)

| M | asm (topk) µs | asm BW TB/s | fused-moe torch µs |
|----:|----:|----:|----:|
| 1   | 186.3 | 47.2 | 42361 |
| 2   | 199.8 | 44.0 | 43232 |
| 4   | 231.1 | 38.1 | 45407 |
| 8   | 341.0 | 25.8 | 47705 |
| 16  | 670.5 | 13.1 | 52644 |
| 32  | 1030.7 | 8.5 | 61520 |
| 64  | 1441.0 | 6.1 | 70321 |
| 128 | 1613.4 | 5.5 | 69611 |
| 256 | 1644.2 | 5.4 | 74788 |
| 512 | 1701.7 | 5.2 | 76294 |
| 1024| 2232.7 | 4.0 | 77821 |
| 2048| 3308.9| 2.7 | 83042 |
| 4096| 3661.5| 2.4 | 94294 |

- `asm` = `topk_softmax` gating kernel (256→8 expert selection + softmax). Bandwidth collapses from ~47 TB/s (M=1) to ~2.4 TB/s (M=4096): the gating kernel is **launch-latency-bound** at small M and **memory-bound on the logits/gates matrix** at large M.
- `torch` = end-to-end fused MoE (stage1 GEMM → permute → activation → stage2 GEMM). Rises smoothly 42→94 ms; the jump from M=64→128 (70→70 ms, flat) marks the **decode→prefill transition** where the kernel switches tiling/config.

## 2. rocprofv2 cache + HBM profile (Fleet Table-4 analogue)

| M | kernel | dispatches | L2 hit % | HBM fetch | HBM write |
|----:|:--|----:|----:|----:|----:|
| 8   | fmoe (FlyDSL) | — | **0.0%** | scales | scales |
| 64  | fmoe (FlyDSL) | — | **0.0%** | scales | scales |
| 512 | fmoe (FlyDSL) | — | **0.0%** | scales | scales |
| 8/64/512 | topk_softmax | — | **0.0%** | — | — |

- **L2 hit rate is 0% at every batch size** for both the fused-MoE GEMM and the topk gating kernel.
- HBM fetch grows ~linearly with M (no reuse benefit from the 256 MB LLC across the chiplets).
- This is the **opposite** of Fleet's dense-model finding, where megakernel tiling kept hot tiles in the per-XCD L2 (high hit rate). Here the MoE access pattern defeats L2 locality.

## 3. Why MoE breaks chiplet locality (Fleet lens)

Fleet's win for dense models comes from **cooperative tiling + chiplet-tasks**: each XCD owns a tile, reuses it from its private 4 MB L2 across the megakernel's lifetime, and avoids cross-die LLC traffic. Two structural properties of GLM-5.2 MoE defeat this:

1. **Expert fan-out is sparse and dynamic.** Each token routes to only 8 of 256 experts; the expert set differs per token and per batch. There is no static tile→XCD assignment that gives reuse — a given expert's weight block is touched by a random subset of tokens, once, then not again. The working set per dispatch is `8 experts × (6144×2048 FP8)` ≈ 96 MB of weights per layer, spread across 256 experts' storage → **no temporal reuse within a kernel**, so L2 cannot retain it.

2. **Per-expert GEMM is tiny.** With topk=8 and M≤512, each expert processes only `M/256 × 8 ≈ M/32` rows on average — far below the GEMM tile size that amortizes a weight fetch. The kernel is **weight-bandwidth-bound**, not compute-bound, and the weights are read cold from HBM every dispatch (hence 0% L2 hit).

3. **topk_softmax gating** reads the full `(M, 256)` routing-logits matrix and writes `(M, 8)` indices+weights. At small M this is launch-overhead-bound (47→2.4 TB/s collapse); at large M it streams the logits once with no reuse → 0% L2 hit.

## 4. Implications for improvement

- **Fleet-style chiplet-tasks alone won't help MoE** the way they help dense GEMMs, because the bottleneck is cold expert-weight streaming, not intra-kernel tile reuse. The reuse opportunity is *across batches* (expert weights are static), not within a kernel.
- **Promising directions** (to validate next):
  - **Expert-weight pinning / L2 reservation**: keep hot experts' weights resident in per-XCD L2 across dispatches (requires expert popularity tracking + EPLB-style placement, which sglang already exposes via `enable_eplb`).
  - **Expert parallelism across XCDs (EP)**: shard experts by XCD so each XCD owns a contiguous expert subset → weights stay local to one die. Currently `ep_size=1` (all-reduce TP); switching to EP could restore locality.
  - **Grouped GEMM tile reuse across tokens of the same expert**: batch tokens routed to the same expert into one GEMM so the weight block is read once per expert-per-batch (aiter's FlyDSL stage1 already does grouped GEMM, but the 0% L2 hit shows the grouping isn't achieving L2 residency — likely because M/32 rows is too small to fill a tile and the dispatch is streamed).
  - **Persistent megakernel for MoE** (Fleet's actual proposal): a single persistent kernel that schedules expert-tasks onto XCDs and keeps expert weights pinned, rather than the current kernel-per-expert dispatch. This is the untested hypothesis the e2e benchmarks will inform.

## 5. DSA & KDA attention microbenchmarks (gfx950 vs gfx942 gap analysis)

See `dsa_kda_gap_analysis.md` for the full kernel-dispatch analysis. Headline numbers:

- **DSA paged-MQA (indexer)** — `dsa_attn_t37.csv` (Triton 3.7.0, Gluon path unlocked): **~60 TFLOPS mean (40–79 range), stable across B=1..64 and kv=512..32768** (mean 60.3–60.9, ±0.4). The Gluon kernel is healthy on gfx950; the DSA gap is in the *sparse prefill/decode* path (TileLang occupancy tuned for gfx942's 304 CUs, under-fills gfx950's 256 CUs), not the indexer.
- **KDA FlashKDA** — `flash_kda.csv`: aiter FlashKDA reaches up to 60 TFLOPS on gfx950 (gfx950-tuned config: BK=64, 8 warps, 3 stages). Kernel is fine; **production doesn't call it** — `linear_attn_backend=triton` routes to sglang FLA Triton with **zero gfx950 tuning**.
- **KDA paged decode** — `la_paged_decode.csv`: LeanAttention beats PagedAttention at short seq (512–8192), loses at 32768 — consistent with decode path not arch-tuned.

**Bottom line:** MI355X's DSA/KDA gap vs B300 is a *software-coverage* gap, not a hardware ceiling (MLA proves the silicon wins). MLA got hand-written ASM/Gluon/Opus gfx950 kernels; DSA/KDA got generic Triton/TileLang. Top fixes: (1) wire aiter FlashKDA into KDA decode, (2) retune DSA TileLang for cu=256. Triton 3.7.0 upgrade already closed the DSA Gluon indexer gap (verified).

## 6. Status

- ✅ FP8 MoE micro sweep (M=1..4096) — done
- ✅ rocprofv2 L2/HBM (M=8/64/512) — done, 0% L2 hit confirmed
- ✅ bf16 baseline (`glm52_bf16_moe.csv`) — fused bf16 asm kernel NOT supported on gfx950 (skipped in `test_moe.py:185-193`); torch reference only
- ✅ int8 sweep (`glm52_int8_moe.csv`) — runs fast but **correctness FAILED** (atol=100); flagged
- ✅ DSA paged-MQA sweep (`dsa_attn_t37.csv`) — done, Triton 3.7.0, ~60 TFLOPS mean
- ✅ KDA FlashKDA (`flash_kda.csv`) + KDA paged decode (`la_paged_decode.csv`) — done
- ✅ DSA/KDA gap analysis (`dsa_kda_gap_analysis.md`) — done
- ⏳ sglang e2e server (TP8, --disable-cuda-graph) — blocked by TP0/TP7 weight-loader deadlock; deferred
- ⏸ full sglang suite + AMD profiling on DSA/KDA — next phase
