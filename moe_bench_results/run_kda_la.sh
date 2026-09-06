#!/usr/bin/env bash
# FlashKDA (Kimi Delta Attention) + linear-attention paged decode microbench.
set -u
cd /scratch/aiter/op_tests/op_benchmarks/triton
export PYTHONPATH=/scratch/aiter
PY=/scratch/sglang-venv/bin/python
OUT=/root/moe_bench_results/micro

# FlashKDA — default sweep (writes CSV with -o)
echo "=== FlashKDA default sweep ===" > "$OUT/flash_kda.log"
echo "started $(date -Is)" >> "$OUT/flash_kda.log"
HIP_VISIBLE_DEVICES=2 $PY bench_flash_kda.py -o --warmup-ms 200 --rep-ms 400 >> "$OUT/flash_kda.log" 2>&1
echo "finished $(date -Is)" >> "$OUT/flash_kda.log"
echo "DONE_FLASH_KDA" >> "$OUT/flash_kda.log"

# Linear-attention paged decode — a few shapes (batch, q heads, kv heads, seq)
echo "=== LA paged decode ===" > "$OUT/la_paged.log"
echo "started $(date -Is)" >> "$OUT/la_paged.log"
for B in 1 8 64; do
  for SQ in 512 8192; do
    echo "----- B=$B hq=32 hk=8 sq=$SQ bf16 -----" >> "$OUT/la_paged.log"
    HIP_VISIBLE_DEVICES=2 $PY bench_la_paged_decode.py -b "$B" -hq 32 -hk 8 -sq "$SQ" -dtype bf16 -kv_cache_dtype bf16 -compute_type bf16 -output_type bf16 -o >> "$OUT/la_paged.log" 2>&1
  done
done
echo "finished $(date -Is)" >> "$OUT/la_paged.log"
echo "DONE_LA_PAGED" >> "$OUT/la_paged.log"
