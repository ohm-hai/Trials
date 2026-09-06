#!/usr/bin/env bash
# rocprofv2 L2/HBM profiling of FP8 MoE for M=64 and M=512 (M=8 already done).
set -u
cd /scratch/aiter
export PYTHONPATH=/scratch/aiter HIP_VISIBLE_DEVICES=2
PY=/scratch/sglang-venv/bin/python
cat > /tmp/rp2.txt <<'EOF'
pmc : TCC_HIT_sum TCC_MISS_sum SQ_WAVES
pmc : FETCH_SIZE WRITE_SIZE
EOF
OUT=/root/moe_bench_results/micro
echo "=== rocprofv2 FP8 MoE M=64,512 ===" > "$OUT/rocprof_moe_v2.log"
echo "started $(date -Is)" >> "$OUT/rocprof_moe_v2.log"
for M in 64 512; do
  rm -rf /tmp/rp2_m${M}; mkdir -p /tmp/rp2_m${M}
  echo "===== M=$M =====" >> "$OUT/rocprof_moe_v2.log"
  timeout 200 rocprofv2 -i /tmp/rp2.txt -d /tmp/rp2_m${M} \
    $PY op_tests/test_moe.py -t g1u1_fp8quant -d bf16 -m $M -hd 6144 -id 2048 -e 256 -k 8 -a silu \
    >> "$OUT/rocprof_moe_v2.log" 2>&1
  echo "--- M=$M parsed ---" >> "$OUT/rocprof_moe_v2.log"
  $PY /root/moe_bench_results/parse_rocprof.py $M >> "$OUT/rocprof_moe_v2.log" 2>&1
done
echo "finished $(date -Is)" >> "$OUT/rocprof_moe_v2.log"
echo "DONE_ROCPROF_V2" >> "$OUT/rocprof_moe_v2.log"
