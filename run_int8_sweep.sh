#!/usr/bin/env bash
# int8quant MoE asm sweep — FP8-vs-int8 comparison (bf16 asm skipped on gfx950, int8 runs).
set -u
cd /scratch/aiter
export PYTHONPATH=/scratch/aiter
export HIP_VISIBLE_DEVICES=0
PY=/scratch/sglang-venv/bin/python
OUT=/root/moe_bench_results/micro
BATCHES="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
echo "=== GLM-5.2 int8quant MoE (g1u1) sweep ===" > "$OUT/glm52_int8_moe.log"
echo "hidden=6144 inter=2048 E=256 topk=8 int8quant g1u1 silu" >> "$OUT/glm52_int8_moe.log"
echo "started $(date -Is)" >> "$OUT/glm52_int8_moe.log"
for m in $BATCHES; do
  echo "----- M=$m -----" >> "$OUT/glm52_int8_moe.log"
  $PY op_tests/test_moe.py -t g1u1_int8quant -d bf16 -m "$m" -hd 6144 -id 2048 -e 256 -k 8 -a silu >> "$OUT/glm52_int8_moe.log" 2>&1
done
echo "finished $(date -Is)" >> "$OUT/glm52_int8_moe.log"
echo "DONE_GLM52_INT8_SWEEP" >> "$OUT/glm52_int8_moe.log"
