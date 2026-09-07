# Phase 9 — Prefill Kernel: Exact Plan & Design

## 0. Goal

Exploit EP-between-XCDs locality for the **prefill** regime (large M, throughput-bound) on
MI355X (gfx950) for GLM-5.2 FP8 MoE, as the counterpart to the decode megakernel
(Phase 1-8). Deliverable: a correctness-verified, fused, sglang-integrated prefill
kernel that beats prod `aiter.fused_moe` hsaco at prefill batch sizes.

## 1. Why the decode lever does NOT transfer to prefill

Decode lever = N-split across XCDs (each XCD owns a fixed N-slice of every expert) +
expert-sequential in M. Win came from: per-XCD weight slice `K*(N/8) fp8` fits the
**4 MB private L2** → high residency at small tokens-per-expert.

At prefill (large M) this breaks for two reasons:
1. **Activation replication**: N-split makes all 8 XCDs read the SAME activation A
   (each XCD needs full A to compute its N-slice). At M=16384 this is 8x A replication
   = 6.3 MB extra HBM/token-block > 3.14 MB L2 weight saving → net regression
   (measured in Phase 2f: 0.71x at M=16384).
2. **Weight doesn't fit private L2 anyway**: a full expert w13 = N2*K = 4096*6144 fp8
   = 25.17 MB >> 4 MB private L2. So pinning a full expert to one XCD's private L2 is
   impossible at any M.

Conclusion: prefill cannot use the private-L2 weight-pinning lever. It must use a
**different** locality: the **256 MB shared LLC (Infinity Cache)**.

## 2. The prefill locality lever: LLC-resident weights via expert-sequential scheduling

### 2.1 The cost model

Per MoE layer, HBM traffic floor (each weight & activation read exactly once):
- W: 256 experts * (25.17 + 12.58) MB = **9.67 GB** weight traffic
- A: M * topk * K * 1B (fp8) = M * 8 * 6144 = M * 49 KB. At M=8192 → 0.4 GB;
  M=131072 → 6.4 GB
- Intermediate (stage1 out / stage2 in): M * topk * N_inter * 2B (bf16) =
  M * 8 * 2048 * 2 = M * 32 KB. At M=8192 → 0.26 GB; M=131072 → 4.2 GB

Total floor HBM traffic at M=8192 ≈ 9.67 + 0.4 + 0.26*2 ≈ **10.6 GB/layer**.
At 5.3 TB/s HBM BW (MI355X) → ~2.0 ms/layer floor. Prod hsaco is the bar.

### 2.2 The lever

LLC = 256 MB shared across all 8 XCDs. It holds ~10 w13 experts (10*25 MB) or ~20
w2 experts. The lever: **expert-sequential scheduling** — process ALL tiles of
expert e (all its tokens * all N-tiles) before moving to e+1, so W_e is loaded
from HBM into LLC exactly once and reused by every tile of e, then evicted. This
achieves the weight-traffic floor (1 load/expert) AND lets LLC absorb the reuse.

Round-robin tile scheduling (prod's likely default) interleaves experts: if it
touches >10 experts before returning to e, W_e is evicted and reloaded from HBM →
**>1 load/expert**, inflating the 9.67 GB weight term. Expert-sequential removes
that inflation. The win magnitude = (prod's reload factor - 1) * 9.67 GB.

### 2.3 Why activation replication is NOT a problem here

Unlike decode's N-split (8x A replication), prefill uses **M-split across XCDs
within an expert**: each XCD owns a token-slice of the current expert. A is
partitioned (no replication); each XCD reads only its token-slice. W_e is shared
via LLC (loaded once, all XCDs read from LLC). So:
- W: 1 HBM load (first XCD) + 7 LLC reads (shared cache) ≈ optimal
- A: 0 replication (partitioned by token-slice)

This is the mirror image of decode: decode pinned W to private L2 (N-split,
accepting A replication because A is tiny); prefill pins W to shared LLC
(M-split, no A replication because A is large).

## 3. Architecture: "Expert-Sequential M-tiled, LLC-resident megakernel"

### 3.1 Tile decomposition

Per expert e, the stage-1 GEMM is (tokens_e, K) @ (K, N2) → (tokens_e, N2).
- BLOCK_M = 128 (tokens per M-tile; tuned, same as decode)
- BLOCK_N = 128 (N2 = 4096 → 32 N-tiles)
- BLOCK_K = 128 (K = 6144 → 48 K-iters)
- Grid per expert: (M_blocks_e * N_blocks) where M_blocks_e = ceil(tokens_e/128),
  N_blocks = 32. Total tiles per expert = M_blocks_e * 32.

### 3.2 XCD assignment (M-split within expert, expert-sequential across experts)

- **Across experts**: strictly sequential, e = 0, 1, ..., 255. One expert's
  entire grid completes before the next launches (LLC residency).
- **Within an expert**: M-split across XCDs. M-tiles of expert e are partitioned
  into 8 contiguous ranges; XCD x owns M-tiles [x*M_blocks_e/8 : (x+1)*M_blocks_e/8).
  Each XCD reads its A token-slice (private L2 if it fits) and the shared W_e (LLC).
- N-tiles: round-robin across the XCDs' CUs (standard 2D tile sweep within the
  XCD's M-range). No N-split (avoids A replication).

Grid (single launch per expert, or fused single launch over all experts — see §5):
```
for e in 0..E-1:                      # sequential, LLC-resident
  grid_e = (NUM_XCD * M_blocks_e * N_blocks,)
  xcd = pid // (M_blocks_e_per_xcd * N_blocks)
  m_local = (pid % (M_blocks_e_per_xcd * N_blocks)) // N_blocks
  pid_m = xcd * M_blocks_e_per_xcd + m_local
  pid_n = (pid % N_blocks)            # round-robin N within XCD's M-range
```

### 3.3 The two regimes (A_e fit)

- **Moderate prefill (M=2k-8k)**: tokens_e = M/32 = 64-256 → A_e = 64-256 * 6144
  = 0.4-1.6 MB **fits private L2 (4 MB)**. Each XCD pins its A token-slice in
  private L2, reuses across all 32 N-tiles. Best case: A_e L2-resident too.
- **Large prefill (M>8k)**: A_e > 4 MB, spills to LLC/HBM. A reuse drops but W
  reuse (LLC) still holds. The win narrows toward the HBM floor; this is where we
  must beat prod purely on scheduling + fusion, not cache magic.

## 4. Correctness plan (bit-equivalence to round-robin)

Same strategy as decode (Phase 3): add a `MODE` constexpr to the kernel.
- MODE=0: round-robin (pid_m, pid_n) over the flat (M_blocks * N_blocks) grid —
  the reference, identical math, different scheduling.
- MODE=1: expert-sequential M-split (§3.2).

Both compute the same per-tile GEMM with the same fp8 block-dequant; only the
(pid_m, pid_n) → (expert, token, N-tile) mapping differs. Since MoE GEMM is a
per-(expert,token,N-tile) independent accumulation with no cross-tile
interaction, any tile→XCD assignment that covers each tile exactly once produces
a bit-identical result. Gate: `max_abs_diff == 0.0` vs MODE=0 (same as decode
Phase 3 verification).

## 5. Fusion plan (no extra dispatches/copies)

Mirror the decode fused megakernel (Phase 8c) but adapted for large M:
- **Stage-1 fused SiLU epilogue**: write `inter = silu(gate)*up` directly (no
  gemm1 intermediate, no separate silu/mul/cast launches). Reuse the reshape
  (BM, BN)→(BM, BN/2, 2) deinterleave from `moe_decode_megakernel_fused.py`.
- **Stage-2 fused unpermute epilogue**: for prefill, M is large so the
  atomic-add contention that killed decode (few output rows) is NOT a problem
  (M=8192 → 8192 output rows, low contention). Use the fused atomic-add
  unpermute from `moe_decode_megakernel_fused.py` directly — it should finally
  WIN here, unlike decode. This eliminates the ~8 unpermute launches.
- **Per-token-group fp8 quant fused into prologue**: each K-iter loads A_bf16,
  computes per-row amax over BK=128 (= 1 group), scales, quants to fp8 in-kernel.
  Eliminates the 2 separate `per_token_group_quant_fp8` launches.
- **Single launch per stage**: one grid over all experts (expert-sequential via
  the pid→expert mapping, not 256 separate launches). The grid is
  `(NUM_XCD * sum_e M_blocks_e * N_blocks)` with a runtime `expert_offsets`
  table (cumulative M_blocks per expert) so pid→(e, pid_m, pid_n) is a single
  lookup. This is the "arithmetic megakernel" form proven in decode Phase 2h.

Net: 2 launches total (stage1 fused, stage2 fused) vs prod's hsaco (also ~2
hsaco). The win is HBM traffic (expert-sequential) + zero intermediate copies.

## 6. Integration plan (sglang prefill path)

Hook site: `AiterRunnerCore.run` in `aiter.py` — extend the existing auto-threshold
window to a TWO-band router:
```
if _USE_MOE_PREFILL_MEGAKERNEL and num_tokens > _MOE_PREFILL_MEGAKERNEL_MIN_TOKENS:
    out_hs = _moe_prefill_megakernel_full(...)   # NEW
elif _USE_MOE_DECODE_MEGAKERNEL and MIN <= num_tokens <= MAX:
    out_hs = _moe_decode_megakernel_full(...)
else:
    out_hs = fused_moe(...)  # prod hsaco fallback
```
- `SGLANG_MOE_PREFILL_MEGAKERNEL` (bool), `..._MIN_TOKENS` (default 2048).
- Same quant gate as decode (PER_128X128, no bias, no swiglu_limit, gate-up
  interleaved) so it targets GLM-5.2/Kimi-K3.
- Overlap band with decode at MIN_TOKENS=1024 (decode max) vs prefill min=2048:
  gap 1024-2048 stays prod hsaco (safe) until prefill is proven.

## 7. Validation & benchmark plan (scaled, serving-level)

1. **Correctness** (Phase 9a): MODE=1 vs MODE=0 bit-equiv at M=2048,4096,8192.
2. **Microbench** (Phase 9b): stage1+stage2 fused vs prod hsaco at
   M=2048,4096,8192,16384,32768,65536,131072. Target: ≥1.0x (no regression)
   and >1x where expert-sequential beats prod's reload factor.
3. **rocprofv2** (Phase 9c): measure L2/LLC hit rate + HBM traffic for
   expert-sequential vs prod at M=8192. Prove the lever: LLC hit rate up, HBM
   weight traffic down toward the 1-load/expert floor.
4. **Serving-level** (Phase 9d): full sglang server (post-TP0/TP7 deadlock fix),
   prefill throughput (tokens/s) on 4k-32k prompts, decode+prefill mixed
   workload. This is the "scaled beyond microbenchmark" gate from the goal.

## 8. Risks & fallbacks

- **Risk**: prod hsaco may already be near the HBM floor (good scheduling) → no win.
  *Mitigation*: the fusion (zero intermediate copies, 2 launches) still gives a
  small win; if even that ties, keep prod hsaco for prefill (the decode win
  already satisfies the goal's decode deliverable).
- **Risk**: expert-sequential underutilizes XCDs when an expert has few tokens
  (skewed routing) → some XCDs idle.
  *Mitigation*: M-split within expert balances across XCDs by token count; if
  tokens_e < 8*BM, fall back to round-robin for that expert (hybrid schedule).
- **Risk**: single-launch megakernel over all 256 experts has a huge grid
  (M=8192 → ~256*64*32 = 524k blocks) → launch/scheduling overhead.
  *Mitigation*: per-expert launch (256 launches) is acceptable at prefill
  (launch overhead amortized by large GEMM); only fuse to single-launch if it
  measures faster.
- **Risk**: large-M A_e spills private L2 → A reuse drops.
  *Mitigation*: this is expected; the win is W-LLC reuse, not A-L2. rocprofv2
  (Phase 9c) confirms whether W-LLC hit actually rises.

## 9. Concrete first steps (execution order)

1. Write `moe_prefill_megakernel.py` with the MODE=0/MODE=1 kernel (M-split,
   expert-sequential) + launcher. Reuse the fp8 block-dequant + fused-silu
   machinery from `moe_decode_megakernel_fused.py`.
2. Phase 9a correctness: bit-equiv vs MODE=0 at M=2048,8192.
3. Phase 9b microbench vs prod hsaco.
4. Phase 9c rocprofv2 LLC/HBM evidence.
5. Wire into `aiter.py` two-band router (§6).
6. Phase 9d serving-level (requires TP0/TP7 deadlock fix first).

## 10. Numbers cheat-sheet (GLM-5.2 FP8)

| quantity | value |
|---|---|
| E (routed experts) | 256 |
| topk | 8 |
| K_hidden | 6144 |
| N_inter | 2048 |
| N2 (gate-up) | 4096 |
| MoE layers | 75 |
| w13 / expert (fp8) | 25.17 MB |
| w2 / expert (fp8) | 12.58 MB |
| total W / layer | 9.67 GB |
| private L2 / XCD | 4 MB |
| LLC (shared) | 256 MB |
| XCDs | 8 |
| CUs | 304 |
| HBM BW | ~5.3 TB/s |
| tokens/expert @ M=8192 | 256 |
| A_e @ M=8192 (fp8) | 1.5 MB (fits L2) |
| A_e @ M=131072 (fp8) | 25 MB (spills) |
