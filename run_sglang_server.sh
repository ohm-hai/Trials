#!/usr/bin/env bash
# sglang e2e server for GLM-5.2 FP8 — eager (no cuda graph) to avoid the earlier warmup hang.
set -u
export PYTHONPATH=/opt/tilelang:/scratch/aiter:/sgl-workspace/mori
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# Workaround: async ThreadPoolExecutor weight-load deadlocks on TP0/TP7 (boundary ranks)
# for GLM-5.2-FP8 on 8x MI355X. Force synchronous loading (one-time cost, reliable).
export SGLANG_DISABLE_ASYNC_WEIGHT_LOAD=1
# Enable the EP-between-XCDs decode megakernel for MoE (decode band M<=1024).
export SGLANG_MOE_DECODE_MEGAKERNEL=true
export SGLANG_MOE_DECODE_MEGAKERNEL_MIN_TOKENS=64
export SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS=1024
PY=/scratch/sglang-venv/bin/python
OUT=/root/moe_bench_results/e2e
mkdir -p "$OUT"
MODEL=/scratch/hf/hub/models--zai-org--GLM-5.2-FP8/snapshots/f33c6dc501ee5a2c7e35155653b1b1abbc320951
LOG=$OUT/server_mega.log
echo "started $(date -Is)" > "$LOG"
$PY -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port 30000 \
  --tp 8 --trust-remote-code \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --chunked-prefill-size 131072 --mem-fraction-static 0.80 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --watchdog-timeout 1800 \
  --enable-metrics --log-level info \
  >> "$LOG" 2>&1
echo "server exited rc=$? $(date -Is)" >> "$LOG"
