#!/usr/bin/env bash
# Task executor acceptance (issue #5): run one scenario (configs/tasks/<name>.yaml) on one engine, in its GUI window.
#   tools/run_task.sh isaac|mujoco <scenario> [run_id] [--no-video]
# Outputs: outputs/task/<scenario>/<engine>_<run_id>.{csv,json,jsonl,log,mp4}; the video records only the simulator
# window. The executor is the only control source and runs on simulated time, so window focus does not matter.
set -uo pipefail
ENGINE=${1:?isaac|mujoco}
NAME=${2:?scenario name (configs/tasks/<name>.yaml)}
RUN=${3:-1}
VIDEO=1; [ "${4:-}" = "--no-video" ] && VIDEO=0
cd "$(dirname "$0")/.."
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
OUT=outputs/task/$NAME
BASE=$OUT/${ENGINE}_$RUN
mkdir -p "$OUT"
rm -f "$BASE".csv "$BASE".json "$BASE".jsonl "$BASE".log "$BASE".mp4
ARGS=(--task "configs/tasks/$NAME.yaml" --task_log "$BASE.jsonl" --log "$BASE.csv" --result "$BASE.json"
      --exit_after_script --max_time 600)
case "$ENGINE" in
  isaac)
    docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py "${ARGS[@]}" > "$BASE.log" 2>&1 &
    TITLE="Isaac Sim 5.1.0" ;;
  mujoco)
    uv run --quiet scripts/mujoco_ee_teleop.py "${ARGS[@]}" > "$BASE.log" 2>&1 &
    TITLE=$(python3 -c "import re;print(re.search(r'title: *(.+)', open('configs/mujoco_so101.yaml').read()).group(1).strip())") ;;
  *) echo "unknown engine $ENGINE" >&2; exit 2 ;;
esac
APP=$!
if [ "$VIDEO" = 1 ]; then
  for _ in $(seq 1 300); do grep -q "reset: session" "$BASE.log" 2>/dev/null && break; sleep 0.5; done
  ISAAC_WINDOW_NAME="$TITLE" tools/record_isaac_window.sh video "$BASE.mp4" 600 &
  REC=$!
fi
wait "$APP"; RC=$?
[ "$VIDEO" = 1 ] && wait "$REC"
grep -E "wrote .*json|Traceback|Error" "$BASE.log" | tail -n 5
exit $RC
