#!/usr/bin/env bash
# rocprof L2/HBM profiling of the FP8 MoE kernel at representative batch sizes.
# Mirrors Fleet Table 4 (L2 hit %, HBM read/write) but per-XCD on MI355X.
set -u
cd /scratch/aiter
export PYTHONPATH=/scratch/aiter
export HIP_VISIBLE_DEVICES=2
PY=/scratch/sglang-venv/bin/python
OUT=/root/moe_bench_results/micro
INPUT=/root/moe_bench_results/rocprof_moe.txt
echo "=== rocprof FP8 MoE L2/HBM ===" > "$OUT/rocprof_moe.log"
echo "started $(date -Is)" >> "$OUT/rocprof_moe.log"
for M in 8 64 512; do
  echo "===== M=$M =====" >> "$OUT/rocprof_moe.log"
  rm -f /tmp/rf_moe_*.csv
  rocprof -i "$INPUT" -o "/tmp/rf_moe_M${M}.csv" -- \
    $PY op_tests/test_moe.py -t g1u1_fp8quant -d bf16 -m "$M" -hd 6144 -id 2048 -e 256 -k 8 -a silu \
    >> "$OUT/rocprof_moe.log" 2>&1
  echo "--- results CSV M=$M ---" >> "$OUT/rocprof_moe.log"
  cat "/tmp/rf_moe_M${M}.csv" 2>/dev/null | head -1 >> "$OUT/rocprof_moe.log"
  # show rows for the fused moe kernel (fmoe) and topk
  grep -iE "fmoe|topk|moe" "/tmp/rf_moe_M${M}.csv" 2>/dev/null >> "$OUT/rocprof_moe.log"
  cp "/tmp/rf_moe_M${M}.csv" "$OUT/rocprof_moe_M${M}.csv" 2>/dev/null
done
echo "finished $(date -Is)" >> "$OUT/rocprof_moe.log"
echo "DONE_ROCPROF_MOE" >> "$OUT/rocprof_moe.log"
