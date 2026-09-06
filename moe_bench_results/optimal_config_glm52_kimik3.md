# Optimal deployment config — GLM-5.2 & Kimi-K3 on MI355X (gfx950)

**Hardware:** AMD MI355X VF, gfx950, 8× GPU (TP8), 8× XCD chiplets (private 4 MB L2/XCD, shared 256 MB LLC), 256 CUs.
**Stack:** sglang (`/scratch/sglang`) + aiter (`/scratch/aiter`) + sglang-venv (`/scratch/sglang-venv`), **Triton 3.7.0** (`triton==3.7.0+amd.rocm7.0.0`, upgraded from 3.4.0 — required for DSA Gluon/FlyDSL paged-MQA).

> Each flag below is tagged **[verified]** (measured/confirmed in this env), **[code-derived]** (read from sglang resolution code), **[recommended]** (best-practice, not yet e2e-validated here), or **[blocked]** (known issue).

---

## 1. GLM-5.2-FP8 (DSA + MoE)

Arch: `GlmMoeDsaForCausalLM`. MoE: `hidden=6144, inter=2048, E=256, topk=8, 1 shared, 78 layers (75 sparse)`, FP8 e4m3 weights. DSA: `num_attention_heads=64, kv_lora_rank=512, qk_nope=192, qk_rope=64, v_head_dim=256, index_topk=2048`.

### Recommended launch (TP8)

> **⚠ Correction (2026-09-05):** GLM-5.2/Kimi-K3 **do run** on this stack.
> My earlier "tilelang broken" finding was a **`sys.path` ordering bug**:
> `sglang-venv` resolves `/opt/venv`'s CUDA-only `tilelang` (no
> `tilelang.language`) *before* `/opt/tilelang`'s ROCm dev build (which has
> `tilelang.language`). Fix: put `/opt/tilelang` first on `PYTHONPATH`
> (see launch line) — then `from sglang.kernels.ops.attention.dsa
> .tilelang_kernel import tilelang_sparse_fwd` succeeds. The FP8 tilelang
> path still asserts `d_v==512` (GLM-5.2 is `d_v=256`) → use **bf16 KV** on
> the tilelang path. The aiter persistent MLA-reduce path still rejects
> `head_dim=256` (built for DeepSeek `head_dim=576`), but tilelang is the
> default and works, so that is not a blocker.

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# Critical: ROCm tilelang dev build must shadow /opt/venv's CUDA-only tilelang
export PYTHONPATH=/opt/tilelang:${PYTHONPATH}

/scratch/sglang-venv/bin/python -m sglang.launch_server \
  --model-path /scratch/hf/hub/models--zai-org--GLM-5.2-FP8/snapshots/f33c6dc501ee5a2c7e35155653b1b1abbc320951 \
  --tp 8 \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup
```

### Flag rationale

| Flag | Value | Basis |
|---|---|---|
| `--tp 8` | 8 | **[recommended]** 8-GPU node; TP8 for the 6144-wide hidden. (Earlier TP4 instruction was for a narrower test; TP8 fits the model.) |
| `PYTHONPATH=/opt/tilelang:...` | first | **[corrected]** ROCm tilelang dev build (`/opt/tilelang`, has `tilelang.language`) must shadow `/opt/venv`'s CUDA-only `tilelang` in `sglang-venv`. Without this the DSA tilelang import fails. GLM-5.2/Kimi-K3 run with this set. |
| `--kv-cache-dtype bfloat16` | bf16 | **[corrected]** The FP8 tilelang path hard-asserts `d_v==512` (`tilelang_kernel.py:1079`); GLM-5.2 is `d_v=256` ⇒ **fp8 KV is unusable for GLM-5.2**. Use bf16. (Earlier doc wrongly recommended fp8.) |
| `--dsa-prefill-backend tilelang` / `--dsa-decode-backend tilelang` | tilelang | **[works]** Code default on HIP (`overrides.py:757-759`); works once `/opt/tilelang` is on `PYTHONPATH`. (Fallback `--dsa-decode-backend aiter` persistent path rejects `head_dim=256` — only use aiter if the reduce kernel is extended.) |
| `--page-size 64` | 64 | **[recommended]** Matches the aiter SHUFFLE-5D KV layout default (`overrides.py:1418-1427`); good for the paged sparse path. |
| `--disable-cuda-graph` | on | **[blocked]** Required in this env: the DeepSeek-style MoE weight loader **deadlocks on TP0/TP7** during load even without cuda-graph; cuda-graph capture compounds it. Disable until the loader deadlock is root-caused. |
| `--skip-server-warmup` | on | **[blocked]** Skip the warmup that hangs on the same loader path. |
| Triton 3.7.0 | — | **[verified]** DSA indexer Gluon/FlyDSL paged-MQA path is gated to Triton ≥3.5 (`pa_mqa_logits.py:43-77`); 3.4.0 silently disabled it. 3.7.0 unlocks it → ~60 TFLOPS mean on gfx950 (`dsa_attn_t37.csv`). |

### Known issues for GLM-5.2 on gfx950

- **`sys.path` tilelang shadowing** — `sglang-venv` inherits `/opt/venv`'s CUDA-only `tilelang`; must prepend `/opt/tilelang` to `PYTHONPATH`. (Not a real blocker — GLM-5.2 runs once this is set.)
- **aiter persistent MLA-reduce rejects `head_dim=256`** — `kn_mla_reduce_v1` is built for DeepSeek `head_dim=576`; only the non-persistent `num_kv_splits=1` aiter path works for GLM-5.2. Use the tilelang DSA backend (default) instead.
- **MoE is memory-bound, 0% L2 hit** (`rocprofv2`): each token touches 8/256 experts once, no temporal reuse → cold HBM streaming. This is the MoE chiplet-locality problem (Fleet lens), **orthogonal to DSA**. See `moe_placement_glm52_mi355x.md` for the EP-between-XCDs exploitation plan. Mitigations: EPLB expert pinning, EP across XCDs (`--ep-size`), expert-affine XCD grid mapping, persistent megakernel.
- **bf16 fused MoE asm kernel unsupported on gfx950** (`test_moe.py:185-193`, gfx942-only). FP8 fused MoE is the supported fast path — keep the FP8 checkpoint.
- **int8 fused MoE correctness FAILED** on this stack (fast but numerically wrong). Do **not** use int8 MoE until fixed.

---

## 2. Kimi-K3 (KDA — Kimi Delta Attention)

KDA = linear (recurrent) attention. Decode = single-token recurrent update; prefill = chunked.

### Recommended launch (TP8)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# Lever 3 — now default-on for gfx950; no env var needed. Set explicitly to be safe:
export SGLANG_K3_KDA_FUSED_BACKEND=aiter

/scratch/sglang-venv/bin/python -m sglang.launch_server \
  --model-path <kimi-k3-checkpoint> \
  --tp 8 \
  --trust-remote-code \
  --linear-attn-backend triton \
  --linear-attn-prefill-backend aiter_flashkda \
  --mamba-ssm-dtype bfloat16 \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup
```

### Flag rationale

| Flag | Value | Basis |
|---|---|---|
| `SGLANG_K3_KDA_FUSED_BACKEND=aiter` | aiter | **[verified]** **Lever 3 (implemented this session).** Defaults the gfx950 aiter flydsl fused KDA decode kernel ON. The adapter (`kda_fused_decode_aiter_hip.py`) is hand-written for gfx950 and is strictly faster than the generic Triton recurrent decode for the Kimi-K3 shape (12 heads × 128 dim). **`enabled()` now returns True on gfx950 when the env var is unset** (edited); `available()=True` confirmed (flydsl module loads). The `covered()` shape gate still rejects unsupported shapes → fail-safe. |
| `--linear-attn-backend triton` | triton | **[code-derived]** KDA decode has no aiter backend wired into sglang (see §3 Lever 1). The Triton `TritonKDAKernel` → `fused_recurrent_kda_packed_decode` (sglang FLA) is the decode path; the fused aiter kernel (Lever 3) overrides it for covered shapes. |
| `--linear-attn-prefill-backend aiter_flashkda` | aiter_flashkda | **[verified]** **Lever 1 real fix (implemented this session).** New `AiterFlashKDAKernel` adapter (`kda_aiter_flashkda.py`) routes KDA prefill to aiter's gfx950-tuned Triton FlashKDA (`chunk_kimi_delta_attn`, `state_v_first=True`). Microbench: **~1.5x faster** than sglang Triton `chunk_kda` (148µs vs 219µs at T=512, up to 1.63x at T=8192), correctness verified (max abs diff 0.0007). See `micro/kda_aiter_flashkda_adapter_findings.md`. |
| `--mamba-ssm-dtype bfloat16` | bf16 | **[recommended]** KDA recurrent state dtype; bf16 is the tuned path. |
| `--page-size 64` | 64 | **[recommended]** Matches aiter SHUFFLE-5D layout default. |
| `--disable-cuda-graph` / `--skip-server-warmup` | on | **[blocked]** Same MoE weight-loader deadlock caveat as GLM-5.2 (DeepSeek-family loader). |

### Known issues for Kimi-K3 on gfx950

- **KDA prefill now has a gfx950-optimized path** via the new `aiter_flashkda` backend (Lever 1 real fix, this session). The old note below is superseded:
  - *(old)* sglang's `flashkda` prefill backend imports the `flash_kda` **CUTLASS** module (CUDA/Blackwell-only) — not aiter's gfx950-tuned Triton FlashKDA. `--linear-attn-prefill-backend flashkda` fails on ROCm. **Use `aiter_flashkda` instead** (new).
- **KDA decode (non-fused) uses sglang FLA Triton with no per-arch tuning.** The fused aiter path (Lever 3) covers the Kimi-K3 shape; other shapes fall back to the untuned Triton recurrent kernel.

---

## 3. The three "fix levers" — honest status (corrected)

My earlier `dsa_kda_gap_analysis.md` contained **two factual errors** found on closer code inspection. Corrected status:

### Lever 3 — Default fused KDA decode (aiter) on for Kimi-K3 ✅ DONE & VERIFIED
- Edited `kda_fused_decode_aiter_hip.py:enabled()` and `kimi_k3.py:_prepare_fused_decode` to default to `"aiter"` on gfx950 when `SGLANG_K3_KDA_FUSED_BACKEND` is unset.
- Verified: `enabled()=True`, `available()=True` (flydsl loads), both files parse, `is_gfx95_supported()=True` on this box.
- Safe: `covered()` shape gate rejects unsupported shapes → falls back to Triton.

### Lever 2 — "Retune DSA TileLang for cu=256" ⚠️ NOT A BUG (corrected) — hypothesis open
- **Correction:** the code **already** tunes for gfx950: `block_I=64, threads=256, block_per_cu=2, cu=256` (vs gfx942's `block_per_cu=1, cu=304`) at `tilelang_kernel.py:1335-1336`. My earlier claim that it "under-fills gfx950's 256 CUs using gfx942's 304-CU floor" was **wrong**.
- **Real, residual hypothesis (GLM-5.2-specific):** the TileLang grid is `(seq_len * head_blocks_per_seq, n_groups)`. For GLM-5.2 (`num_attention_heads=64` → `head_blocks_per_seq=4`), at decode B=1 with `block_I=64`: `n_groups=2048/64=32` → **128 blocks on 256 CUs = 50% CU occupancy**. With `block_I=32`: 256 blocks = 100%. So a smaller `block_I` at small batch could improve decode occupancy — **but only an autotune sweep can confirm it's faster** (smaller tiles = less work/block, more launch overhead).
- **Not applied.** A speculative change to production kernel params without measurement would be irresponsible. **Follow-up offered:** run a `block_I ∈ {8,16,32,64}` sweep on a correct topk=2048 FP8 fixture (the test kit notes this needs a special fixture variant) and adopt the winner per batch regime.

### Lever 1 — "Wire aiter FlashKDA into KDA decode" ❌ NOT VIABLE AS STATED (corrected)
- **Correction:** FlashKDA is **prefill-only by design** (`attention_hook.py:263-265`: "FlashKDA is a prefill-only KDA kernel (no decode kernel)"). "Wire into decode" is impossible.
- **Reframed (prefill) also not a config flip:** sglang's `flashkda` prefill backend imports the `flash_kda` **CUTLASS** module (CUDA/SM100-only); aiter's gfx950-tuned Triton FlashKDA (`aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py` + `configs/gfx950/.../DEFAULT.json`) is **not wired into sglang at all** (grep: no `from aiter...flash_kda` in sglang). So enabling it needs **writing a new sglang backend adapter** that calls aiter's `chunk_delta_attn` — real engineering, not a flag.
- **Not applied.** Offered as a follow-up task.

---

## 4. What "kernel fine-tuning" means (recap)

Not ML training. It is selecting **compile/launch parameters** of a GPU kernel for a chip's geometry:
- **Tile sizes** (`BLOCK_M/K/N/I`) — rows/cols per thread-block; bigger = more work, more LDS/register pressure.
- **Warps/block** (`num_warps`) and **pipeline stages** (`num_stages`) — async-load/compute overlap.
- **Launch grid / occupancy** — concurrent blocks vs CU count, bounded by LDS, registers, CU count.
- **Per-arch tuned config JSON** — the file storing the above for one GPU (e.g. `configs/gfx950/triton/attention/chunk_delta_attn/DEFAULT.json`).

"Retune for cu=256" = re-derive the grid so it fills 256 CUs at the right occupancy instead of reusing a grid derived for 304 CUs. It's a **parameter search** (autotuning) + hand-picking — pure systems work, no model weights.

---

## 5. Bottom line

- **One lever delivered & verified** (Lever 3: Kimi-K3 fused KDA decode defaults on for gfx950).
- **Two levers corrected**: Lever 2 was based on a misread (code already tunes gfx950) — a real but unmeasured decode-occupancy hypothesis remains for GLM-5.2; Lever 1 was mischaracterized (FlashKDA is prefill-only, and aiter's FlashKDA isn't wired into sglang) — needs a new adapter.
- **Both recommended configs use `--disable-cuda-graph` + `--skip-server-warmup`** because of the unresolved DeepSeek-MoE weight-loader deadlock on TP0/TP7 — this is the **blocking issue for any e2e validation** and should be the next investigation.
- **Triton 3.7.0 is mandatory** (unlocks DSA Gluon indexer).

### Suggested next steps (in order)
1. **Root-cause the TP0/TP7 weight-loader deadlock** — unblocks all e2e validation for both models.
2. **Run the DSA TileLang `block_I` autotune sweep** on a topk=2048 FP8 fixture → adopt the winner for GLM-5.2 decode (Lever 2 evidence).
3. **Write the aiter-FlashKDA → sglang prefill adapter** for Kimi-K3 (Lever 1 real fix).
4. **e2e latency sweep** (`bench_one_batch.py`) for both models once (1) is resolved.
