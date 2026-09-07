#!/usr/bin/env bash
# rocprofv2 L2/HBM profiling of FP8 MoE at SCALE (M=1024,4096,16384) to confirm
# the 0% L2 hit persists at serving scale (round-robin baseline).
set -u
cd /scratch/aiter
export PYTHONPATH=/scratch/aiter HIP_VISIBLE_DEVICES=0
PY=/scratch/sglang-venv/bin/python
cat > /tmp/rp2_scale.txt <<'EOF'
pmc : TCC_HIT_sum TCC_MISS_sum SQ_WAVES
pmc : FETCH_SIZE WRITE_SIZE
EOF
OUT=/root/moe_bench_results/micro
LOG="$OUT/rocprof_moe_scale.log"
echo "=== rocprofv2 FP8 MoE SCALE sweep M=1024,4096,16384 ===" > "$LOG"
echo "started $(date -Is)" >> "$LOG"
for M in 1024 4096 16384; do
  rm -rf /tmp/rp2s_m${M}; mkdir -p /tmp/rp2s_m${M}
  echo "===== M=$M =====" >> "$LOG"
  timeout 400 rocprofv2 -i /tmp/rp2_scale.txt -d /tmp/rp2s_m${M} \
    $PY op_tests/test_moe.py -t g1u1_fp8quant -d bf16 -m $M -hd 6144 -id 2048 -e 256 -k 8 -a silu \
    >> "$LOG" 2>&1
  echo "--- M=$M parsed ---" >> "$LOG"
  $PY /root/moe_bench_results/parse_rocprof.py $M >> "$LOG" 2>&1
done
echo "finished $(date -Is)" >> "$LOG"
echo "DONE_ROCPROF_SCALE" >> "$LOG"
