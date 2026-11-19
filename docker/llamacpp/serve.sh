#!/usr/bin/env bash
# Quantized model service for the chat layer: Qwen3.5-9B Q4_K_M (GGUF) behind llama.cpp's OpenAI-compatible server,
# localhost only. Same API as docker/vllm/serve.sh, so the chat layer only needs base_url / model changed.
#   docker/llamacpp/serve.sh fetch | up | down | status | logs
# Pinned: image ghcr.io/ggml-org/llama.cpp:server-cuda12-b12033 (CUDA 12 build, also for Turing GPUs), model file
# unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf at revision 3885219 (Apache-2.0, sha256 03b74727...).
# The file is kept in ~/docker/llamacpp/models (5.7 GB). All layers on the GPU, one request at a time, text only.
# Thinking is turned off per request by the chat layer (chat_template_kwargs, needs --jinja).
set -euo pipefail
IMAGE=${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda12-b12033}
REPO=unsloth/Qwen3.5-9B-GGUF
REV=3885219b6810b007914f3a7950a8d1b469d598a5
FILE=Qwen3.5-9B-Q4_K_M.gguf
SHA256=03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8
ALIAS=${LLM_MODEL:-Qwen3.5-9B-Q4_K_M}
NAME=${LLM_CONTAINER_NAME:-ars-llm-q4}
PORT=${LLM_PORT:-8001}
DIR=${LLM_MODEL_DIR:-$HOME/docker/llamacpp/models}
CTX=${LLM_MAX_MODEL_LEN:-8192}
fetch() {
  mkdir -p "$DIR"
  if [ ! -f "$DIR/$FILE" ]; then
    echo "downloading $REPO/$FILE (5.7 GB)"
    curl -fL -o "$DIR/$FILE.part" "https://huggingface.co/$REPO/resolve/$REV/$FILE" && mv "$DIR/$FILE.part" "$DIR/$FILE"
  fi
  echo "$SHA256  $DIR/$FILE" | sha256sum -c --quiet || { echo "checksum mismatch: $DIR/$FILE" >&2; exit 1; }
}
case "${1:-status}" in
  fetch) fetch ;;
  up)
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "already running"; exit 0; fi
    fetch
    docker run -d --rm --name "$NAME" --gpus all -p 127.0.0.1:$PORT:8080 -v "$DIR:/models:ro" \
      "$IMAGE" -m "/models/$FILE" --alias "$ALIAS" --host 0.0.0.0 --port 8080 \
      -ngl 999 -c "$CTX" -np 1 --jinja > /dev/null
    echo -n "waiting for the model service"
    for _ in $(seq 1 300); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        echo; echo "ready: http://127.0.0.1:$PORT/v1 ($ALIAS)"; exit 0; fi
      if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo; echo "container exited (removed with --rm)"; exit 1; fi
      echo -n "."; sleep 2
    done
    echo; echo "not ready after 10 min; docker logs $NAME"; exit 1 ;;
  down) docker stop "$NAME" > /dev/null 2>&1 && echo stopped || echo "not running" ;;
  status) curl -sf "http://127.0.0.1:$PORT/v1/models" && echo || echo "not answering" ;;
  logs) docker logs --tail 80 "$NAME" ;;
  *) echo "usage: $0 fetch|up|down|status|logs" >&2; exit 2 ;;
esac
