# Phase A — DSA decode path for GLM-5.2 on gfx950 (MI355X): findings

> **⚠ CORRECTION (2026-09-05, later):** GLM-5.2/Kimi-K3 **do run** on this
> stack. The "tilelang is broken" finding below was a **`sys.path` ordering
> bug**, not a real support gap. `sglang-venv` inherits `/opt/venv`'s
> CUDA-only `tilelang` (no `tilelang.language`) and resolves it *before*
> `/opt/tilelang`'s ROCm dev build (which has `tilelang.language`).
> Prepending `/opt/tilelang` to `PYTHONPATH` makes
> `from sglang.kernels.ops.attention.dsa.tilelang_kernel import
> tilelang_sparse_fwd` succeed. So:
> - **Finding 1 is downgraded to a `sys.path` fix** (set
>   `PYTHONPATH=/opt/tilelang:...`), not a missing ROCm tilelang.
> - The block_I sweep (original Phase A task) is now **unblocked** and remains
>   a valid next step on the working bf16 tilelang path.
> - Findings 2 (FP8 path asserts `d_v==512` → use bf16 KV) and 3 (aiter
>   persistent reduce rejects `head_dim=256` → use tilelang, not aiter) still
>   hold and are the real config guidance.
> - Finding 4 (num_kv_splits=1 fallback crashes at B≥32) is moot — tilelang
>   is the production path, not the aiter fallback.
>
> The text below is the original (now-superseded) investigation record.


> Lever 2 ("retune DSA TileLang occupancy for gfx950 cu=256") was the original
> Phase A task. Investigation produced a **different, more important result**:
> the DSA decode backend situation for GLM-5.2 on gfx950 is a support gap, not
> an occupancy-tuning problem. The block_I sweep is moot.

## GLM-5.2 attention shape (from config.json)

- `num_attention_heads = 64`, `qk_head_dim = 256` (qk_nope 192 + qk_rope 64),
  `v_head_dim = 256`, `kv_lora_rank = 512`, `q_lora_rank = 2048`.
- DSA index `topk = 2048` pages per request, `page_size = 1`.
- So the sparse-attention kernel sees: `num_heads=64, dim=d_v=256, tail_dim=0, topk=2048`.

## Finding 1 — the `tilelang` DSA backend is non-functional on this stack

- sglang's default DSA backend on HIP is `tilelang` (`overrides.py`).
- The installed `tilelang` is a **CUDA-only build**: `0.1.7.post3+cuda.gita55a8230`,
  with no `tilelang.language` submodule (`ModuleNotFoundError: No module named
  'tilelang.language'` at `tilelang_kernel.py:6`). The AMD PyPI index also only
  serves this CUDA build; the latest `0.1.14` needs `apache-tvm-ffi` which is
  unavailable on the index.
- ⇒ `from sglang.kernels.ops.attention.dsa.tilelang_kernel import ...` fails.
  GLM-5.2 DSA decode **cannot use the default `tilelang` path** on this gfx950 stack.

## Finding 2 — the FP8 tilelang path is unusable for GLM-5.2 regardless

- `sparse_mla_fwd_decode_partial_fp8` hard-asserts `d_v == 512`
  (`tilelang_kernel.py:1079`). GLM-5.2 has `v_head_dim = 256`.
- ⇒ Even with a working tilelang, the FP8 KV path is unavailable for GLM-5.2.
  GLM-5.2 DSA must use a **bf16 KV cache** on the tilelang path.

## Finding 3 — the `aiter` DSA decode backend's persistent path does NOT support GLM-5.2

- `--dsa-decode-backend=aiter` → `_forward_aiter` → `aiter.mla.mla_decode_fwd`
  (the gfx950-optimized MLA decode kernel, the same one that makes MLA win on
  MI355X). This is the *correct* idea — but it fails for GLM-5.2's shape:
  ```
  [AITER] reduce.cu:1267 kn_mla_reduce_v1 doesn't support the specified
  settings: #heads: 64, head dimension: 256.
  ```
- The persistent/split MLA decode path (`mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps`,
  auto `num_kv_splits` via `get_mla_metadata_v1`) needs the **reduce kernel**
  `kn_mla_reduce_v1`, which was built for DeepSeek's `head_dim = 576`
  (`kv_lora_rank 512 + rope 64`) and does **not** support `head_dim = 256`.
- ⇒ The production-recommended persistent path errors out for GLM-5.2.

## Finding 4 — only a fragile non-persistent fallback works (and it crashes at B≥32)

- Forcing `num_kv_splits = 1` (no reduce kernel) runs for small batch:
  - B=1: 116.7 µs, B=2: 117.2, B=4: 116.5, B=8: 118.0, B=16: 119.9 (median, GPU0).
  - **B ≥ 32 → memory access fault** (`qseqlen1` fixed-shape kernel has a batch limit).
  - Latency is suspiciously **flat** across B=1..16 (~117 µs), i.e. the kernel is
    not scaling with batch — it is a fixed `qseqlen1`/`gqaratio64` dispatch, not a
    proper varlen decode. This is not a production-grade path.

## Conclusion / recommendation for GLM-5.2 DSA decode on gfx950

GLM-5.2 DSA decode on MI355X is in a **real support gap**:
1. `tilelang` (default) — broken on this stack (CUDA-only tilelang).
2. `aiter` persistent (recommended gfx950 path) — reduce kernel rejects
   `head_dim=256` (built for DeepSeek `head_dim=576`).
3. `aiter` non-persistent `num_kv_splits=1` — works only for B<32, flat/unscaled
   latency, crashes at B≥32. Not production-grade.

**Actionable items (ranked):**
1. **Install a ROCm/HIP build of tilelang** (or build from source) to restore the
   default DSA path. This is the highest-leverage fix — it unblocks the path the
   code already targets, and the bf16 partial+combine kernels support
   `d_v=256` (no `d_v==512` assertion). The `block_I` occupancy hypothesis
   (block_I=64 → 128 blocks vs 256 CUs at B=1) can then be tested on the
   working bf16 path.
2. **Extend `kn_mla_reduce_v1` (gfx950) to support `head_dim=256`** so the
   aiter persistent MLA decode path works for GLM-5.2 — this would make DSA
   decode reuse the same gfx950-optimized kernel that wins on MLA.
3. Until either lands, GLM-5.2 DSA decode on gfx950 is **not deployable** at
   production batch sizes. Use `--kv-cache-dtype bfloat16` (NOT fp8) on any
   tilelang path that does come up.

## Evidence

- `/root/moe_bench_results/micro/dsa_aiter_decode.csv` — aiter non-persistent
  latency, B=1..16 (B≥32 crashes).
- This session's shell logs: tilelang import error; `kn_mla_reduce_v1` rejection;
  B≥32 memory-access fault.

## Note on the original block_I hypothesis

The hypothesis (block_I=64 → 50% CU utilization at B=1 for 64 heads) is
**untestable here** because the tilelang path is broken. It remains a valid
hypothesis to test *after* a ROCm tilelang is installed. It is **not** the
binding constraint on GLM-5.2 DSA decode on gfx950 today — the binding
constraints are Findings 1 and 3.
