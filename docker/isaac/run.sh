#!/usr/bin/env bash
# Launch the pinned Isaac Lab 2.3.2 (Isaac Sim 5.1.0) container for the simulator role.
# Usage: docker/isaac/run.sh [command ...]   (default: interactive bash)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ISAAC_LAB_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}"
WORKSHOP_DIR="${SO101_WORKSHOP_DIR:-$HOME/ref/Sim-to-Real-SO-101-Workshop}"
CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/docker/isaac-lab-2.3.2}"
NAME="${ISAAC_CONTAINER_NAME:-ars-isaac}"

if [[ ! -f "$WORKSHOP_DIR/source/sim_to_real_so101/assets/usd/SO-ARM101-USD-NO-CAMERA.usd" ]]; then
  echo "SO-101 workshop assets not found at $WORKSHOP_DIR; run docker/isaac/fetch_assets.sh" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"/{kit,ov,pip,glcache,computecache,logs,data,documents} "$REPO_DIR/outputs"

X11_ARGS=()
if [[ -n "${DISPLAY:-}" ]]; then
  X11_ARGS=(-e DISPLAY -v "${ISAAC_X11_DIR:-/tmp/.X11-unix}:/tmp/.X11-unix:rw")
  [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]] && X11_ARGS+=(-e XAUTHORITY=/root/.Xauthority -v "$XAUTHORITY:/root/.Xauthority:ro")
fi

if [[ $# -eq 0 ]]; then set -- bash; fi
CMD="$(printf '%q ' "$@")"

TTY_ARGS=(-i); [[ -t 0 ]] && TTY_ARGS=(-it)

exec docker run --rm "${TTY_ARGS[@]}" --name "$NAME" --gpus all --network host --shm-size 8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  "${X11_ARGS[@]}" \
  -v "$CACHE_DIR/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE_DIR/ov:/root/.cache/ov:rw" \
  -v "$CACHE_DIR/pip:/root/.cache/pip:rw" \
  -v "$CACHE_DIR/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE_DIR/computecache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE_DIR/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE_DIR/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE_DIR/documents:/root/Documents:rw" \
  -v "$WORKSHOP_DIR:/workspace/so101_workshop:ro" \
  -v "$REPO_DIR:/workspace/ars:rw" \
  -e SO101_WORKSHOP_DIR=/workspace/so101_workshop \
  -e PYTHONPATH=/workspace/ars/src -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workspace/ars \
  --entrypoint bash "$IMAGE" \
  -c "$CMD; rc=\$?; find /workspace/ars -user 0 -exec chown $(id -u):$(id -g) {} + 2>/dev/null; exit \$rc"
