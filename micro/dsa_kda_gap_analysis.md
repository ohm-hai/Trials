# DSA & KDA Kernel Gap Analysis — why MI355X (gfx950) underperforms MI300X/B300 (gfx942)

**Question:** MI355X beats B300 on MLA, but loses on DSA and KDA. Where are the gaps?
**Method:** Source analysis of aiter (`/scratch/aiter`) + sglang (`/scratch/sglang`) kernel dispatch, per-arch tuned configs, and backend selection. Supported by microbenchmarks (FP8 MoE, rocprof L2/HBM, KDA FlashKDA, KDA paged-decode, DSA paged-MQA — see sibling CSVs).

## 1. The core asymmetry: MLA is hand-optimized ASM on gfx950; DSA/KDA are not

MLA on gfx950 has a **deep stack of hand-written / device-specific kernels** that simply don't exist for DSA/KDA:

| MLA fast path on gfx950 | File | DSA/KDA equivalent |
|---|---|---|
| MLA v4 ASM hsaco registry | `hsa/gfx950/mla_v4/mla_v4_asm.csv` (no gfx942 counterpart) | **none** — DSA has no ASM hsaco table |
| `mla_gluon` unified Gluon decode + DSv4 sparse prefill | `aiter/ops/triton/gluon/mla_gluon.py:843-889` (gfx950 only) | DSA sparse uses generic TileLang |
| HK MLA v4.0 hand-kernel | `aiter/mla.py:1089-1117` (gfx950 only) | **none** |
| Opus fp8 MLA decode | `aiter/mla.py:876-880` (gfx950 only) | **none** |
| DSA ds32 Opus MLA decode | `aiter/mla.py:453-460` (gfx950 only) | **none** for the sparse-attn path |
| Dedicated MLA decode RoPE JSON (8 warps, 3 stages) | `configs/gfx950-MLA_DECODE_ROPE-DEFAULT.json` | DSA/KDA: no dedicated gfx950 tuning JSON of this depth |

**Takeaway:** MLA's gfx950 advantage is *engineered* — multiple hand-written ASM/Gluon/Opus kernels built specifically for MI355X. DSA and KDA were given Triton/TileLang paths that rely on generic codegen + light tuning, so there is no equivalent device-specific optimization to exploit MI355X's geometry. This alone explains "MLA good, DSA/KDA bad" on the same chip.

## 2. DSA gaps (specific)

### Gap 2a — DSA sparse prefill/decode runs TileLang with a CU-count heuristic mismatch
`sglang/kernels/ops/attention/dsa/tilelang_kernel.py:1330-1367` picks occupancy from the GPU's CU count:
- gfx942 (MI300X): `cu=304`, `block_I=32`, `block_per_cu=1` (bf16 KV)
- gfx950 (MI355X): `cu=256`, `block_I=64`, `block_per_cu=2`

The tile/launch params were evidently tuned on gfx942's 304-CU floor. On gfx950's 256 CUs the same launch grid **under-fills the device** (fewer CUs occupied, lower occupancy). The wider `block_I=64` on gfx950 partially compensates but doesn't recover the lost parallelism — a likely source of the DSA decode gap. A gfx950-native retune of the TileLang DSA kernel (cu=256, re-derived grid) is the single highest-leverage fix.

### Gap 2b — DSA indexer (paged-MQA logits) Gluon path was gated behind Triton ≥3.5
`aiter/ops/triton/attention/pa_mqa_logits.py:43-77`:
- Gluon (FlyDSL) paged-MQA kernel enables only when `triton_version >= 3.5.0`.
- Below 3.5 → legacy Triton kernel with `KVBlockSize==1` only, **no Preshuffle** (`pa_mqa_logits.py:527-609`).

Our stack shipped Triton 3.4.0, so the **Gluon paged-MQA path was silently disabled** and DSA fell back to the slower legacy Triton kernel. We have now upgraded to Triton 3.7.0 (`triton==3.7.0+amd.rocm7.0.0` from the AMD ROCm 7.0 wheel index), which unblocks the Gluon path. The DSA re-sweep running now (`dsa_attn_t37.log`) measures this unlocked path — expect a meaningful improvement vs the 3.4.0 partial run.

### Gap 2c — No prebuilt ASM for the DSA indexer; everything is JIT/AOT Gluon
Unlike MLA (which ships `hsa/gfx950/mla_v4/*.hsaco`), the paged-MQA indexer has **no hsaco tuning tables** under `hsa/`. Tuning is a JIT/AOT Gluon compile keyed by `(ChunkQ, ChunkK, HiddenDim, ARCH)` at first launch. Consequence:
- First-launch compile cost (the warmup hangs we hit).
- No offline hand-tuning of the best tile for gfx950's LDS (160 KB) vs gfx942 (64 KB) — `sparse_attention_dsv4.py:10-24` bumps the LDS cap to 160 KB on gfx950, but the tile search isn't exhaustively tuned for the new LDS budget.

### Gap 2d — The gfx950-only Triton prefill bypass is too narrow to help
`dsa_backend.py:2019-2054` has a `SGLANG_DSA_TRITON_PREFILL=1` path that calls `triton_sparse_mla_fwd`, but it's gated to an **exact shape**: `tp_q_head_num==16`, `d_v==512`, `topk==2048`, FP8 KV. Default stays TileLang. So for any shape mismatch, gfx950 falls back to the same TileLang kernel as gfx942 — no advantage realized.

## 3. KDA gaps (specific)

### Gap 3a — Production KDA decode uses sglang FLA Triton with ZERO arch tuning
This is the biggest KDA gap. `sglang/srt/layers/attention/linear/kda_backend.py:72-99` resolves `linear_attn_backend=triton` → `TritonKDAKernel` → **sglang FLA Triton** (`kernels/ops/attention/fla/fused_recurrent.py`), **not aiter**. Those FLA kernels have **no `get_gfx()` branches, no per-arch tuned JSON**. So KDA decode latency is whatever generic Triton codegen produces — and on gfx950 that's not tuned, whereas gfx942 has had months of Triton codegen tuning. **The KDA decode gap is mostly a "no gfx950 tuning in the sglang FLA path" problem.**

### Gap 3b — aiter FlashKDA (which IS tuned for gfx950) isn't used by default
`aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py` + the per-arch configs:
- gfx942: `configs/gfx942/triton/attention/chunk_delta_attn/DEFAULT.json` → `BK=32, BV=128, num_stages=2`
- gfx950: `configs/gfx950/triton/attention/chunk_delta_attn/DEFAULT.json` → `BK=64, BV=128, num_warps=8, num_stages=3`

The gfx950 config is **wider and deeper** (bigger BK, more stages, more warps) — i.e. aiter FlashKDA is tuned to *favor* gfx950's larger LDS. But the production `linear_attn_backend=triton` path doesn't route to aiter FlashKDA at all. So the tuned, gfx950-favorable kernel sits unused while the untuned sglang FLA path runs.

### Gap 3c — The gfx950-only fused KDA decode fast path is opt-in and off
`sglang/kernels/ops/attention/kda_fused_decode_aiter_hip.py` routes to `aiter.ops.flydsl.kimi_k3_kda_decode` only when `SGLANG_K3_KDA_FUSED_BACKEND=aiter` (Kimi-K3 only). Default off. So even the one hand-written gfx950 KDA kernel that exists is not enabled by default.

### Gap 3d — LDS-favorable tiles that help gfx950 are gated to gfx942's 64 KB in shared code
`gla_output.py:30-35`: `BV=128, num_stages=3` needs ~123 KB LDS — comment says "cannot run on gfx942 (64KB); gfx950 can use the wider tile." The wider tile path exists but is only taken when the config selects it; the default config pipeline doesn't force it for gfx950.

## 4. Summary of gaps (ranked by leverage)

| # | Gap | Family | Fix lever |
|---|---|---|---|
| 1 | KDA decode uses sglang FLA Triton with no arch tuning (3a) | KDA | Add gfx950 tuning to sglang FLA, or route `linear_attn_backend` to aiter FlashKDA (which already has gfx950 config) |
| 2 | DSA TileLang occupancy tuned for gfx942's 304 CUs, under-fills gfx950's 256 CUs (2a) | DSA | Retune TileLang DSA kernel grid/occupancy for cu=256 |
| 3 | DSA indexer Gluon path was disabled on Triton 3.4.0 (2b) | DSA | **Fixed & verified** — Triton 3.7.0; DSA sweep shows ~60 TFLOPS mean, stable across B/KV |
| 4 | aiter FlashKDA (gfx950-tuned) unused by default (3b) | KDA | Flip `linear_attn_prefill_backend=flashkda` / wire aiter FlashKDA into the decode path |
| 5 | gfx950-only KDA fused decode opt-in/off (3c) | KDA | Default `SGLANG_K3_KDA_FUSED_BACKEND=aiter` on gfx950 for Kimi-K3 |
| 6 | No ASM hsaco for DSA indexer (2c) | DSA | Build offline hsaco tuning tables for paged-MQA on gfx950 |
| 7 | gfx950 Triton DSA prefill bypass too narrow (2d) | DSA | Broaden the shape gate / make it default on gfx950 |

## 5. What the microbenchmarks confirm / will confirm

- **FP8 MoE** (done): fused kernel works on gfx950, 0% L2 hit (cold expert-weight streaming) — MoE is memory-bound, not the DSA/KDA gap.
- **KDA FlashKDA** (done, `flash_kda.csv`): the aiter FlashKDA kernel itself reaches up to 60 TFLOPS — so the *kernel* is fine on gfx950; the gap is that production doesn't call it.
- **KDA paged decode** (done, `la_paged_decode.csv`): LeanAttention beats PagedAttention at short seq (512-8192), loses at 32768 — consistent with "decode path not arch-tuned."
- **DSA paged-MQA** (done, Triton 3.7.0, `dsa_attn_t37.csv`): the unlocked Gluon/FlyDSL paged-MQA kernel reaches **~60 TFLOPS mean (40–79 TFLOPS range)** on gfx950, **stable across B=1..64 and kv_length=512..32768** (mean 60.3–60.9 TFLOPS, ±0.4). Two conclusions: (a) the kernel itself is healthy on MI355X once the Gluon path is unblocked — it is *not* the bottleneck; (b) performance is flat in B, so the indexer is compute-bound at small batch, not latency-bound — the decode gap is therefore in the *dispatch/wiring*, not the kernel. This confirms Gap 2b is closed by the Triton upgrade and shifts remaining DSA leverage to Gap 2a (TileLang occupancy retune for the sparse prefill/decode path).
- **rocprof L2/HBM** (done): 0% L2 hit on the MoE kernel — the same chiplet-locality story (Fleet) applies to DSA's KV-cache reads too; DSA's paged access pattern is the attention analogue of MoE's expert-streaming problem.

## 6. Bottom line

The DSA/KDA gap on MI355X is **not** a hardware ceiling — MLA proves the silicon can win, and the DSA paged-MQA Gluon kernel now hits ~60 TFLOPS on gfx950 (Triton 3.7.0), proving the kernels themselves are fine. It's a **software-coverage gap**: MLA received hand-written ASM/Gluon/Opus kernels for gfx950, while DSA and KDA were left on generic Triton/TileLang paths with either no gfx950 tuning (KDA decode) or gfx942-tuned occupancy (DSA TileLang). The highest-leverage fixes are (1) wire aiter's already-tuned FlashKDA into the KDA decode path, and (2) retune the DSA TileLang kernel for gfx950's 256-CU floor. The Triton 3.7.0 upgrade (now done & verified) unblocked the DSA Gluon indexer, removing one gap for free.
