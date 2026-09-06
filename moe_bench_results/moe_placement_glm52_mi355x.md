# Phase C — MoE expert placement on AMD chiplet GPUs (MI355X / gfx950)

> **Question:** For MoE on AMD chiplet GPUs, how do the four placements differ —
> TP-between-GPUs, TP-between-XCDs, EP-between-GPUs, EP-between-XCDs — and how
> do Fleet's findings (more tokens / bigger batch, or more speculative-decoding
> verification) change the picture? What is the best way to place experts on this
> architecture?
>
> **Anchor finding (this stack, rocprofv2):** FP8 MoE shows **0% L2 hit at every
> batch size** — each token touches 8 of 256 experts once; no temporal reuse within
> a kernel → cold expert-weight streaming defeats chiplet locality. This is the
> Fleet megakernel hypothesis: expert-weight pinning / EP across XCDs /
> persistent megakernel could restore locality, but it is untested and orthogonal
> to the DSA/KDA gap.

## Hardware (measured)

- **MI355X VF, gfx950**: 256 CUs = **8 XCDs × 32 CUs/XCD**; **4 MB private L2
  per XCD**; **256 MB shared LLC (L3 / Infinity Cache)** across all XCDs
  (`rocminfo`: `L2: 4096 KB`, `L3: 262144 KB`; `multi_processor_count=256`).
- 8 GPUs/node (only 1 VF visible to this container via `HIP_VISIBLE_DEVICES`).
- HBM-bandwidth-bound for MoE: expert weights (w13/w2) dominate traffic; the
  0% L2 hit means every expert-weight byte is fetched from HBM, not L2/LLC.

## The four placements (matrix)

Notation: a token routes to `topk=8` of `E=256` experts. Each expert is a
2-GEMM (w13 gate/up, w2 down), `intermediate=2048`, `hidden=6144`, FP8 e4m3.

| # | Placement | What is sharded | Experts per GPU | Cross-XCD | Cross-GPU | L2 locality (current) | Comms |
|---|---|---|---|---|---|---|---|
| 1 | **TP between GPUs** (`ep=1`,`tp=8`) | `intermediate` dim of **every** expert on every GPU (w13 col-parallel, w2 row-parallel); router replicated | **all 256** (sharded intermediate) | n/a | post-MoE **all-reduce** | **0% L2 hit** (measured) — each GPU streams all 256 experts' sharded weights once per token | all-reduce/layer (cheap, on-node RCCL) |
| 2 | **TP between XCDs** (intra-GPU; `moe_tp` shards one expert across the 8 XCDs) | `intermediate` dim of one expert across XCDs | all 256 | **XCD swizzle** (`aiter` `remap_xcd`, `moe_op.py:189`) | n/a | GEMM-tile locality only — **does not fix expert-weight reuse** (still 0% L2) | none (intra-GPU) |
| 3 | **EP between GPUs** (`--ep-size 8`,`moe_tp=1`) | **experts** split across GPUs: `num_local=256/8=32`/GPU; router replicated; A2A dispatch (Mori on ROCm) | **32** (8× less weight/GPU) | n/a (EP rank=GPU) | **A2A token dispatch** (Mori `EpDispatchCombineOp`, `moriep.py:756-796`) | better: each GPU streams only 32 experts → 8× less cold HBM; reuse/expert rises with tokens/expert | A2A all-to-all/layer (heavier than all-reduce; needs Mori/RCCL) |
| 4 | **EP between XCDs** (intra-GPU EP — pin experts to XCDs, Fleet megakernel) | experts **pinned to XCDs** (32 local / 8 XCDs = 4 experts/XCD); persistent megakernel streams each expert's weight **once** into its XCD's 4 MB L2, reuses across tokens routed to it | 32/GPU, 4/XCD | **the lever**: XCD-local expert-weight residency in L2 | none (intra-GPU) | **restores L2 hit** (untested) — weights stay hot in XCD L2 across tokens | none (intra-GPU) |

### Reading the matrix

- **#1 TP-between-GPUs** is the *current measured baseline* (all experts
  replicated, sharded intermediate). It maximizes cold HBM streaming → the
  0% L2 hit. Its only win is a cheap post-MoE all-reduce. For GLM-5.2's 256
  experts this is the **worst** placement for memory bandwidth.
- **#2 TP-between-XCDs** is what aiter **already does** (`remap_xcd` in the
  MoE GEMM kernel, `aiter/aiter/ops/triton/_triton_kernels/moe/moe_op.py:189`).
  It schedules GEMM tiles across XCDs for tile-level L2 locality, but each
  expert weight is still consumed once per token → the 0% L2 hit is unchanged.
  Necessary baseline, **insufficient on its own**.
- **#3 EP-between-GPUs** is supported today (`--ep-size`, `moriep.py`,
  `deepseek_v2.py`). It cuts per-GPU expert weight 8× (32 vs 256) → 8× less HBM
  traffic — the single biggest lever available *without* new kernels. Cost is
  A2A token dispatch (Mori on ROCm). EPLB (`--enable-eplb`,
  `eplb_algorithms/deepseek.py`) replicates hot experts and packs expert groups
  to nodes then GPUs (`expert_location.py:699-737`).
- **#4 EP-between-XCDs** is the **Fleet megakernel hypothesis** — *not
  implemented* in sglang/aiter (no expert-to-XCD placement exists; XCD awareness
  is only kernel-grid remap, #2). It is the only placement that targets the
  *root cause* of the 0% L2 hit: pin expert weights into a XCD's private 4 MB
  L2 and reuse them across tokens via a persistent megakernel.

## How Fleet's findings change the picture (bs / speculative verification)

The 0% L2 hit is a **low tokens-per-expert** problem: at decode `B=1`, each
expert is touched by ~1 token, so the weight is streamed once and discarded — no
reuse to capture. Fleet's central finding is that **megakernels restore locality
when work is reused across many tokens pinned to a chiplet**. Two trends raise
tokens-per-expert and therefore amplify the EP/pinning benefit:

1. **Bigger batch / more tokens.** As `bs` grows (large decode batch, long
   prefill chunk, aggregated serving batch), tokens routed to each expert grow
   ~`bs·topk/E`. Each expert's weight is then reused across many tokens → L2
   hit rises *if* the weight is resident on the XCD that computes it. So:
   - #1 (TP-GPU) and #2 (TP-XCD) improve only marginally (weights still streamed
     per token; reuse is across the sharded slice, partial).
   - #3 (EP-GPU) improves more (fewer experts/GPU → higher tokens/expert →
     higher reuse, plus 8× less HBM traffic to begin with).
   - #4 (EP-XCD megakernel) improves **most** — exactly the regime Fleet
     designed for: a persistent kernel keeps expert weights hot in L2 across a
     large token batch. The bigger the batch, the bigger the win.
2. **Speculative-decoding verification (MTP / draft tokens).** `target_verify`
   adds `draft_token_num` (e.g. +1…+8) tokens per request per forward → more
   tokens per expert per forward → same effect as `bs` increase: higher
   tokens-per-expert → higher L2 reuse potential. Two caveats specific to verify:
   - Verify must stay **rollback-able** (the KDA `is_spec_decode` fallback in
     the FlashKDA adapter exists for exactly this reason — kernels that commit
     state in-place, like a Fleet megakernel that pins expert weights, must not
     run for verify). So #4's megakernel must support a rollback/checkpoint
     contract for the verify round, or fall back to #3/#2 for verify.
   - Verify raises token count without raising unique-expert count → pure win
     for reuse, but only if the megakernel is verify-safe.

**Net Fleet effect:** both trends push the optimum toward **#3 + #4** (EP across
GPUs, then EP across XCDs via a verify-safe megakernel). They do not help #1/#2
because those never make expert weights resident across tokens.

## Recommendation for GLM-5.2 (E=256, topk=8) on MI355X 8-GPU

Ranked by leverage on the 0% L2 hit, with implementation cost:

1. **EP between GPUs (`--ep-size 8`) + EPLB** — **primary, available now.**
   - 32 experts/GPU → 8× less per-GPU HBM streaming; higher tokens/expert →
     rising L2 reuse as bs grows.
   - `--enable-eplb` (DeepSeek hierarchical pack: node→GPU, `eplb_algorithms/
     deepseek.py:126-150`) replicates hot experts and keeps expert groups
     node-local → cuts cross-node A2A for hot experts.
   - A2A via Mori on ROCm (`moriep.py:756-796`); EP all-reduce
     (`communication_op.py:136-138`).
   - **Caveat:** does not avoid the DeepSeek MoE weight-loader deadlock
     (`deepseek_weight_loader.py:507-509`, `as_completed` on TP0/TP7) — that
     is in the load path, orthogonal to EP vs TP. Both use the same FusedMoE loader.
2. **EP between XCDs (Fleet megakernel, expert-weight pinning)** — **highest
   leverage, not yet implemented.** Pin 32 local experts across 8 XCDs
   (4 experts/XCD); a persistent megakernel streams each expert's FP8 weight
   (w13+w2 ≈ 2·2048·6144·2 bytes ≈ 50 MB/expert sharded, ~6 MB/XCD for 4 experts
   — fits the 4 MB L2 with tiling) **once** into its XCD's L2 and reuses it
   across all tokens routed to that XCD. This is the only lever that converts the
   0% L2 hit into a high L2 hit. Requires: a Fleet-style persistent megakernel
   with XCD-affine launch, expert→XCD mapping, and a verify-safe rollback
   contract (fall back to #3 for `target_verify`).
3. **TP between XCDs (aiter `remap_xcd`)** — **keep as the baseline kernel
   optimization.** Already in aiter; helps GEMM-tile locality but does not fix
   expert-weight reuse. Necessary, not sufficient.
4. **TP between GPUs (`ep=1`)** — **avoid for GLM-5.2 MoE.** Replicates all 256
   experts on every GPU → max cold streaming → the measured 0% L2 hit. Use only
   if EP/A2A is unavailable.

### Concrete target config (once the loader deadlock is fixed)

```bash
--tp 8 --ep-size 8 --moe-a2a-backend mori --enable-eplb
# + (future) Fleet megakernel for EP-between-XCDs expert pinning
```

- `--ep-size 8`: 32 experts/GPU (vs 256 replicated under TP).
- `--moe-a2a-backend mori`: ROCm EP dispatch/combine (`moriep.py`).
- `--enable-eplb`: hot-expert replication + node-local packing.
- Keep aiter `remap_xcd` MoE GEMM (TP-between-XCDs, #2) as the per-kernel base.
- Future: a verify-safe Fleet megakernel that pins the 32 local experts to the
  8 XCDs (4/XCD) and reuses their weights across tokens — the lever that turns
  the 0% L2 hit into locality.

## Evidence

- 0% L2 hit: `/root/moe_bench_results/rocprof_moe.txt` + parsed
  (`micro/summary.md`).
- Hardware: `rocminfo` (this session) — 8 XCDs, 4 MB L2/XCD, 256 MB LLC.
- TP MoE sharding: `fused_moe_triton/layer.py:360-378,690-745,800-833,1568-1569`.
- EP split + A2A: `fused_moe_triton/layer.py:173-191`, `moriep.py:756-796`,
  `parallel_state.py:2643-2671`, `server_args.py:2369-2409`.
- EPLB: `eplb/expert_location.py:62-71,699-737`,
  `eplb_algorithms/deepseek.py:126-150`.
- XCD awareness (kernel only): `aiter/aiter/ops/triton/_triton_kernels/moe/
  moe_op.py:189` (`remap_xcd`); `aiter/aiter/ops/triton/utils/device_info.py:25-27`.
- DeepSeek loader (deadlock path): `deepseek_weight_loader.py:507-509`.

## Bottom line

The 0% L2 hit is a **memory-bandwidth** problem, not a DSA/KDA problem. On
MI355X the best *available* lever is **EP between GPUs (#3) + EPLB** — it cuts
per-GPU expert-weight streaming 8× and raises tokens-per-expert as batch grows.
The **highest-leverage** lever is **EP between XCDs (#4)** — a Fleet persistent
megakernel that pins expert weights into the 4 MB per-XCD L2 and reuses them
across tokens — but it is unimplemented and must be made verify-safe. TP between
GPUs (#1) is the worst for GLM-5.2 and should be avoided; TP between XCDs (#2)
is already done in aiter and is the necessary kernel baseline but does not fix
reuse. Bigger batch and speculative-decoding verification both push the optimum
toward #3 + #4 by raising tokens-per-expert.

