#!/usr/bin/env bash
# Cross-engine comparison run (issue #3 P4): feed one scripted EE trajectory to one engine, in its GUI window.
#   tools/run_trajectory.sh isaac|mujoco pick_place [run_id] [--no-video]
# Outputs: outputs/compare/<trajectory>/<engine>_<run_id>.{csv,json,log,mp4}; the video records only the simulator
# window. The trajectory replaces all keyboard / mouse input and is sampled at simulated time, so window focus does
# not matter.
set -uo pipefail
ENGINE=${1:?isaac|mujoco}
TRAJ=${2:?trajectory name (configs/trajectories/<name>.yaml)}
RUN=${3:-1}
VIDEO=1; [ "${4:-}" = "--no-video" ] && VIDEO=0
cd "$(dirname "$0")/.."
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
OUT=outputs/compare/$TRAJ
BASE=$OUT/${ENGINE}_$RUN
mkdir -p "$OUT"
rm -f "$BASE".{csv,json,log,mp4}
ARGS=(--trajectory "configs/trajectories/$TRAJ.yaml" --log "$BASE.csv" --result "$BASE.json" --exit_after_script
      --max_time 300)
case "$ENGINE" in
  isaac)
    docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py "${ARGS[@]}" > "$BASE.log" 2>&1 &
    TITLE="Isaac Sim 5.1.0"; READY="reset: session" ;;
  mujoco)
    uv run --quiet scripts/mujoco_ee_teleop.py "${ARGS[@]}" > "$BASE.log" 2>&1 &
    TITLE=$(python3 -c "import re;print(re.search(r'title: *(.+)', open('configs/mujoco_so101.yaml').read()).group(1).strip())")
    READY="reset: session" ;;
  *) echo "unknown engine $ENGINE" >&2; exit 2 ;;
esac
APP=$!
if [ "$VIDEO" = 1 ]; then
  for _ in $(seq 1 300); do grep -q "$READY" "$BASE.log" 2>/dev/null && break; sleep 0.5; done
  ISAAC_WINDOW_NAME="$TITLE" tools/record_isaac_window.sh video "$BASE.mp4" 300 &
  REC=$!
fi
wait "$APP"; RC=$?
[ "$VIDEO" = 1 ] && wait "$REC"
grep -E "wrote .*json|Traceback|Error" "$BASE.log" | tail -n 5
exit $RC
