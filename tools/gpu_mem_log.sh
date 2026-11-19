#!/usr/bin/env bash
# Whole-GPU memory sampler for the model-service comparison: writes "t_s,mem_used_mib" every PERIOD seconds until
# killed. Reads nvidia-smi only (no host settings changed).
#   tools/gpu_mem_log.sh out.csv [period_s=0.2]
OUT=${1:?out.csv}; P=${2:-0.2}
echo "t_s,mem_used_mib" > "$OUT"
T0=$(date +%s.%N)
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$(echo "$(date +%s.%N) - $T0" | bc),$m" >> "$OUT"
  sleep "$P"
done
