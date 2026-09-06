#!/usr/bin/env bash
# bf16 no-quant MoE baseline sweep (g1u0) — compare vs FP8. Same GLM-5.2 shape.
set -u
cd /scratch/aiter
export PYTHONPATH=/scratch/aiter
export HIP_VISIBLE_DEVICES=0
PY=/scratch/sglang-venv/bin/python
OUT=/root/moe_bench_results/micro
BATCHES="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
echo "=== GLM-5.2 bf16 no-quant MoE (g1u0) sweep ===" > "$OUT/glm52_bf16_moe.log"
echo "hidden=6144 inter=2048 E=256 topk=8 No-quant g1u0 silu" >> "$OUT/glm52_bf16_moe.log"
echo "started $(date -Is)" >> "$OUT/glm52_bf16_moe.log"
for m in $BATCHES; do
  echo "----- M=$m -----" >> "$OUT/glm52_bf16_moe.log"
  $PY op_tests/test_moe.py -t test_fmoe_16_bit -d bf16 -m "$m" -hd 6144 -id 2048 -e 256 -k 8 -a silu >> "$OUT/glm52_bf16_moe.log" 2>&1
done
echo "finished $(date -Is)" >> "$OUT/glm52_bf16_moe.log"
echo "DONE_GLM52_BF16_SWEEP" >> "$OUT/glm52_bf16_moe.log"
