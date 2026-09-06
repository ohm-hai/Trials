#!/usr/bin/env bash
# DSA / MLA paged-MQA logits attention microbench (the GLM-5.2 DSA indexer path).
# bench_deepgemm_attention: -B batch -hq heads --index_dim head_dim -kv_length seq
# GLM-5.2 indexer: index_n_heads=32, index_head_dim=128, index_topk=2048.
set -u
cd /scratch/aiter/op_tests/op_benchmarks/triton
export PYTHONPATH=/scratch/aiter
export HIP_VISIBLE_DEVICES=1
PY=/scratch/sglang-venv/bin/python
OUT=/root/moe_bench_results/micro
echo "=== DSA paged-MQA logits (deepgemm_attention) ===" > "$OUT/dsa_attn.log"
echo "hq=64 index_dim=128 (GLM-5.2 indexer shape)" >> "$OUT/dsa_attn.log"
echo "started $(date -Is)" >> "$OUT/dsa_attn.log"
for B in 1 4 8 32 64; do
  for KV in 512 2048 8192 32768; do
    echo "----- B=$B kv_length=$KV -----" >> "$OUT/dsa_attn.log"
    $PY bench_deepgemm_attention.py -B "$B" -hq 64 --index_dim 128 -kv_length "$KV" --perf >> "$OUT/dsa_attn.log" 2>&1
  done
done
echo "finished $(date -Is)" >> "$OUT/dsa_attn.log"
echo "DONE_DSA_ATTN" >> "$OUT/dsa_attn.log"
