#!/usr/bin/env bash
# Persistent MuJoCo server (issue #3): a host process (uv environment), clients connect from any terminal.
#   tools/mujoco_server.sh up       start in the background with its window, wait until it answers (about 1 s)
#   tools/mujoco_server.sh down [PID]  ask it to shut down (kill after 10 s); with PID only when it is that process
#   tools/mujoco_server.sh status   process + server status
#   tools/mujoco_server.sh logs     follow the server log
# Env: DISPLAY (window; MUJOCO_SERVER_HEADLESS=1 for none), XAUTHORITY, MUJOCO_SERVER_EXPERIMENT
#      (default minimal_pick_place), MENAGERIE_DIR, ARS_CREATED_FILE (up writes the PID of a server it started there;
#      the check and the start are one step under a lock, so two concurrent ups start one server)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_DIR/outputs/mujoco"
PID_FILE="$RUN_DIR/server.pid"
LOG="$RUN_DIR/server.log"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export DISPLAY="${DISPLAY:-}"
PY=(uv run --quiet --project "$REPO_DIR" python)

probe() {  # server status JSON, or nothing when it does not answer
  "${PY[@]}" - "$REPO_DIR" <<'PY' 2>/dev/null
import json, sys
sys.path.insert(0, sys.argv[1] + "/src")
from agentic_sim.server.client import probe
info = probe("mujoco", timeout=1.0)
print(json.dumps(info) if info else "")
PY
}

alive() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-}" in
  up)
    mkdir -p "$RUN_DIR"
    exec 9> "$RUN_DIR/up.lock"; flock 9          # check + start: one up at a time
    if alive; then echo "already running (pid $(cat "$PID_FILE")): $(probe)"; exit 0; fi
    ARGS=(--experiment "${MUJOCO_SERVER_EXPERIMENT:-minimal_pick_place}")
    if [ "${MUJOCO_SERVER_HEADLESS:-0}" = 1 ]; then ARGS+=(--headless)
    else [ -n "$DISPLAY" ] || { echo "set DISPLAY (e.g. export DISPLAY=:1) or MUJOCO_SERVER_HEADLESS=1" >&2; exit 1; }; fi
    cd "$REPO_DIR"
    nohup uv run --quiet scripts/mujoco_sim_server.py "${ARGS[@]}" > "$LOG" 2>&1 9>&- &
    echo $! > "$PID_FILE"
    [ -z "${ARS_CREATED_FILE:-}" ] || echo $! > "$ARS_CREATED_FILE"
    exec 9>&-                                     # started and recorded: release the lock
    echo -n "waiting for the server"
    for _ in $(seq 1 60); do
      if [ -n "$(probe)" ]; then echo; echo "ready: $(probe)"; exit 0; fi
      if ! alive; then echo; echo "the server exited; last log lines:" >&2; tail -n 30 "$LOG" >&2; exit 1; fi
      echo -n "."; sleep 0.5
    done
    echo; echo "no answer after 30 s; see: tools/mujoco_server.sh logs" >&2; exit 1 ;;
  down)
    if ! alive; then echo "not running"; rm -f "$PID_FILE"; exit 0; fi
    PID="$(cat "$PID_FILE")"
    if [ -n "${2:-}" ] && [ "$2" != "$PID" ]; then echo "not stopped: the running server (pid $PID) is not $2"; exit 0; fi
    "${PY[@]}" - "$REPO_DIR" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from agentic_sim.server.client import SimClient
with SimClient.for_backend("mujoco", timeout=2.0, name="mujoco_server.sh") as c:
    c.call("shutdown")
PY
    for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$PID" 2>/dev/null && { echo "no clean exit, killing $PID" >&2; kill "$PID"; }
    rm -f "$PID_FILE"; echo "stopped" ;;
  status)
    if alive; then echo "process: running (pid $(cat "$PID_FILE"))"; else echo "process: not running"; fi
    s="$(probe)"; [ -n "$s" ] && echo "$s" || echo "server not answering" ;;
  logs)
    tail -n 50 -f "$LOG" ;;
  *)
    sed -n '2,8p' "$0"; exit 2 ;;
esac
