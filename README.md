# Trials — EP-between-XCDs MoE Megakernel for GLM-5.2 on MI355X

This repo captures the full body of work for exploiting **EP-between-XCDs
(expert-weight L2 pinning)** on AMD MI355X (gfx950, 8 XCDs, 4 MB private L2 per
XCD) for the **GLM-5.2-FP8** Mixture-of-Experts model, end-to-end through the
sglang serving path.

## Goal
Convert the 0% L2-hit / no-locality MoE GEMM problem on a chiplet GPU into a high
L2-hit regime by pinning each expert's weight slice to a fixed XCD's private L2,
fused into a single-launch megakernel, integrated into sglang's FusedMoE decode
path, and verified at serving scale.

## Outcome (converged)
- **Decode (M <= 1024 tokens): megakernel beats production `aiter.fused_moe` hsaco
  by 1.26x–3.60x** (3.60x @ M=256, 2.05x @ M=512, 1.26x @ M=1024).
- **Prefill (M >= 2048): production hsaco wins** (crossover ~1024–2048); the
  sglang gate `SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS=1024` selects the
  megakernel only where it wins.
- Best tuned config: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_XCD=8` (N-split).
- Bit-equivalent to round-robin (diff=0.0000) for all configs.
- per-XCD weight slice = K*(N/8) = 6144*256 = 1.57 MB < 4 MB private L2 (fits).

## Repo layout
```
moe_bench_results/         all phase scripts, findings docs, micro/e2e results
  phase1_xcd_imbalance_and_proof.md   Phase 1: measured XCD imbalance + 0% L2 hit (rocprofv2)
  phase2_3_findings.md                 Phase 2-3: expert-affine -> N-split megakernel, bit-equiv
  phase6_7_findings.md                 Phase 6-7: tuning + scaled E2E benchmarks
  phase2_expert_affine_bench.py ... phase7_e2e.py   the benchmark scripts
  micro/                              DSA/KDA microbenchmarks + gap analysis
  e2e/                                sglang server logs
aiter_module/              the new Triton megakernel module (drop-in for aiter)
  aiter/ops/triton/_triton_kernels/moe/moe_decode_megakernel.py
sglang_integration/       the integrated sglang FusedMoE decode path
  aiter.py                            full modified AiterFusedMoeRunner
  aiter_megakernel_hook.patch        diff of the megakernel hook
```

## Phases
1. **Measure XCD imbalance** — surveyed MoE workloads (decode/prefill/spec-decode
   verify), proved 0% L2 hit at scale via `rocprofv2`. (phase1 doc)
2. **Implement kernel change** — expert-affine `remap_xcd` -> evolved to N-split
   megakernel (the lever that fits 4 MB L2). (phase2 scripts)
3. **Verify correctness** — bit-equivalent to round-robin (diff=0.0000). (phase3)
4. **Fuse** — single-launch megakernel, no extra dispatches/copies. (phase2h/phase3)
5. **Integrate** — wired into sglang `AiterFusedMoeRunner.run` behind
   `SGLANG_MOE_DECODE_MEGAKERNEL`; prefill keeps production hsaco. (sglang_integration)
6. **Tune** — BLOCK_M=128 best (1.83–1.96x vs round-robin). (phase6)
7. **Scale** — serving-level batches; 1.26–3.60x vs production hsaco at decode,
   crossover found, gate set to 1024. (phase7)

## How to reproduce the decode win
```bash
export HIP_VISIBLE_DEVICES=0
export PYTHONPATH=/opt/tilelang:/scratch/aiter:/scratch/sglang/python
# E2E vs production aiter.fused_moe
python moe_bench_results/phase7_e2e.py
# Config sweep
python moe_bench_results/phase6_tune.py
```

## Enabling in sglang
Set `SGLANG_MOE_DECODE_MEGAKERNEL=1` (and optionally
`SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS=1024`) for GLM-5.2/Kimi-K3 PER_128X128
decode workloads. The hook in `sglang_integration/aiter.py` routes decode-stage
MoE to `_moe_decode_megakernel_full`; prefill keeps `aiter.fused_moe`.

## Notes
- Full live sglang server end-to-end (HTTP request) was blocked by a pre-existing
  TP0/TP7 weight-loader deadlock unrelated to the megakernel. The integrated
  serving-path function (`_moe_decode_megakernel_full`) is verified at
  serving-scale decode batches (256–1024) with correct, finite outputs.
- All large `gpucore.*` rocprof dumps (12–32 GB each) are excluded from this repo;
  the parsed L2-hit evidence is in the phase1 findings doc.
