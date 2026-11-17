#!/usr/bin/env bash
# Persistent Isaac Sim server (issue #3): Kit + assets load once, clients connect from any terminal.
#   docker/isaac/server.sh up       start in the background, wait until it answers (about 30 s cold)
#   docker/isaac/server.sh down     stop and remove the container
#   docker/isaac/server.sh status   container state + server status
#   docker/isaac/server.sh logs     follow the server log
# Env: DISPLAY (required for up), XAUTHORITY, ISAAC_SERVER_EXPERIMENT (default minimal_pick_place),
#      SO101_WORKSHOP_DIR, ISAAC_CACHE_DIR, ISAAC_LAB_IMAGE
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE=(docker compose -f "$REPO_DIR/docker/isaac/compose.yaml")
export SO101_WORKSHOP_DIR="${SO101_WORKSHOP_DIR:-$HOME/ref/Sim-to-Real-SO-101-Workshop}"
export ISAAC_CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/docker/isaac-lab-2.3.2}"
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export DISPLAY="${DISPLAY:-}"

PY=(python3)  # needs PyYAML; the repo's uv environment has it
command -v uv >/dev/null 2>&1 && PY=(uv run --quiet --project "$REPO_DIR" python)

probe() {  # server status JSON, or nothing when it does not answer
  "${PY[@]}" - "$REPO_DIR" <<'PY' 2>/dev/null
import json, sys
sys.path.insert(0, sys.argv[1] + "/src")
from agentic_sim.server.client import probe
info = probe("isaac", timeout=1.0)
print(json.dumps(info) if info else "")
PY
}

case "${1:-}" in
  up)
    [ -n "$DISPLAY" ] || { echo "set DISPLAY (e.g. export DISPLAY=:1)" >&2; exit 1; }
    [ -f "$XAUTHORITY" ] || { echo "XAUTHORITY not found: $XAUTHORITY" >&2; exit 1; }
    [ -f "$SO101_WORKSHOP_DIR/source/sim_to_real_so101/assets/usd/SO-ARM101-USD-NO-CAMERA.usd" ] || {
      echo "SO-101 workshop assets not found at $SO101_WORKSHOP_DIR; run docker/isaac/fetch_assets.sh" >&2; exit 1; }
    if docker ps --format '{{.Names}}' | grep -qx ars-isaac; then
      echo "warning: a one-off Isaac container (ars-isaac) is also running" >&2
    fi
    mkdir -p "$ISAAC_CACHE_DIR"/{kit,ov,pip,glcache,computecache,logs,data,documents} "$REPO_DIR/outputs"
    "${COMPOSE[@]}" up -d
    echo -n "waiting for the server"
    for _ in $(seq 1 180); do
      if [ -n "$(probe)" ]; then echo; echo "ready: $(probe)"; exit 0; fi
      if ! docker ps --format '{{.Names}}' | grep -qx "${ISAAC_SERVER_CONTAINER:-ars-isaac-server}"; then
        echo; echo "the server container exited; last log lines:" >&2
        "${COMPOSE[@]}" logs --tail 30 >&2 || true
        exit 1
      fi
      echo -n "."; sleep 1
    done
    echo; echo "no answer after 180 s; see: docker/isaac/server.sh logs" >&2; exit 1 ;;
  down)
    "${COMPOSE[@]}" down ;;
  status)
    "${COMPOSE[@]}" ps
    s="$(probe)"; [ -n "$s" ] && echo "$s" || echo "server not answering" ;;
  logs)
    "${COMPOSE[@]}" logs -f ;;
  *)
    sed -n '2,8p' "$0"; exit 2 ;;
esac
