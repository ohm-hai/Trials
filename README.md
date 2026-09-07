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

## Outcome (corrected — see phase13_findings.md)
> **Important correction (Phase 13):** the earlier "1.26–3.60x decode win" was an
> artifact of a correctness bug and is **retracted**. See below.

- **Decode (M = 64–1024): megakernel is 2.4–3x SLOWER than production hsaco** on
  *correct* output. The previously reported win was measured on a kernel with a
  stage-2 a-indexing bug (`A_BY_SORTED`) that produced **wrong output**
  (`rel_err = 1.0`); once fixed, the megakernel loses. (phase13_findings.md)
- **Prefill (M >= 2048): production hsaco wins** (0.23–0.53x). (phase9_findings.md)
- The **XCD L2-locality problem is real** (Phase 1: 0% L2 hit), but a **Triton
  megakernel is the wrong tool** to exploit it — it cannot match the compute
  throughput of AMD's hand-tuned hsaco `fmoe_g1u1` kernels, so any L2 benefit is
  swamped. Exploiting the lever would require modifying the opaque hsaco kernels
  or writing a hand-tuned ASM/WMMA megakernel (research-grade, out of scope here).
- Best tuned config (for reference): `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
  NUM_XCD=8` (N-split); per-XCD weight slice = K*(N/8) = 6144*256 = 1.57 MB < 4 MB
  private L2 (fits). Bit-equivalent to round-robin (diff=0.0000) for the
  round-robin reference path.

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
7. **Scale** — serving-level batches; re-validation on the *fixed* kernel showed
   the megakernel is 2.4–3x slower than hsaco at decode (the earlier 1.26–3.60x
   win was retracted — it was on a buggy wrong-output kernel). (phase7/phase13)

## How to reproduce the re-validation
```bash
export HIP_VISIBLE_DEVICES=5
export PYTHONPATH=/opt/tilelang:/scratch/aiter:/scratch/sglang/python
# Correctness + timing of the FIXED megakernel vs production aiter.fused_moe
python moe_bench_results/phase13_decode_revalidate.py
```

## Enabling in sglang
> **Not recommended for production** — the megakernel loses to hsaco on correct
> output. The hook is retained for experimentation only.

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
