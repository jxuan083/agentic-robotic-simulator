#!/usr/bin/env bash
# Model service for the chat layer: Qwen3.5-9B behind vLLM's OpenAI-compatible API, localhost only.
#   docker/vllm/serve.sh up | down [ID] | status | logs | info
# Pinned: image vllm/vllm-openai:v0.17.1 (Qwen3.5 needs vLLM >= 0.17.0), model Qwen/Qwen3.5-9B (Apache-2.0, BF16).
# Weights are cached in ~/docker/vllm/hf-cache (about 20 GB on first start). GPU memory is capped so the simulator
# keeps room on the same GPU (LLM_GPU_UTIL, default 0.72 of the card). Text only, thinking off by default.
# One chat user: at most 4 sequences, and the CUDA graph capture size kept at 4 (the default 512 is larger than the
# hybrid model's state cache here: 'assert num_cache_lines >= batch', vllm#34571).
# up creates the container only when none with this name runs (a lost name race counts as already running) and, with
# ARS_CREATED_FILE set, docker writes the ID of the container it created there; down <ID> stops exactly that container.
# info prints name=, url=, model= (tools/lab.py reads them, so LLM_CONTAINER_NAME / LLM_PORT / LLM_MODEL apply there).
set -euo pipefail
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.17.1}
MODEL=${LLM_MODEL:-Qwen/Qwen3.5-9B}
NAME=${LLM_CONTAINER_NAME:-ars-llm}
PORT=${LLM_PORT:-8000}
CACHE=${LLM_CACHE_DIR:-$HOME/docker/vllm/hf-cache}
UTIL=${LLM_GPU_UTIL:-0.72}
MAXLEN=${LLM_MAX_MODEL_LEN:-8192}
running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }
case "${1:-status}" in
  up)
    if running; then echo "already running"; exit 0; fi
    mkdir -p "$CACHE"
    CIDFILE=(); [ -z "${ARS_CREATED_FILE:-}" ] || CIDFILE=(--cidfile "$ARS_CREATED_FILE")  # ID of what we create
    if ! OUT=$(docker run -d --rm --name "$NAME" "${CIDFILE[@]}" --gpus all --ipc=host -p 127.0.0.1:$PORT:8000 \
      -v "$CACHE:/root/.cache/huggingface" \
      "$IMAGE" --model "$MODEL" --served-model-name "$MODEL" \
      --language-model-only --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
      --max-num-seqs 4 --max-cudagraph-capture-size 4 \
      --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' 2>&1); then
      if running; then echo "already running (started by another command)"; exit 0; fi  # lost the name race
      echo "$OUT" >&2; exit 1
    fi
    echo -n "waiting for the model service"
    for _ in $(seq 1 600); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo; echo "ready: http://127.0.0.1:$PORT/v1 ($MODEL)"; exit 0; fi
      if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo; echo "container exited; see: docker logs $NAME (removed with --rm)"; exit 1; fi
      echo -n "."; sleep 2
    done
    echo; echo "not ready after 20 min; docker logs $NAME"; exit 1 ;;
  down) docker stop "${2:-$NAME}" > /dev/null 2>&1 && echo stopped || echo "not running" ;;   # down [container ID]
  info) echo "name=$NAME"; echo "url=http://127.0.0.1:$PORT/v1"; echo "model=$MODEL" ;;
  status) curl -sf "http://127.0.0.1:$PORT/v1/models" && echo || echo "not answering" ;;
  logs) docker logs --tail 80 "$NAME" ;;
  *) echo "usage: $0 up|down|status|logs|info" >&2; exit 2 ;;
esac
