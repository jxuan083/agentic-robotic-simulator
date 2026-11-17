#!/usr/bin/env bash
# Run one scripted acceptance scenario on the MuJoCo backend with its window, recording only that window.
#   tools/run_acceptance_mujoco.sh keyboard_pick_place [--no-video]
#   MUJOCO_ARGS='--blocks 8' tools/run_acceptance_mujoco.sh multi_block_pick_place
# Outputs: $ACCEPT_OUT/<name>.{log,csv,json,mp4} (default outputs/acceptance/mujoco)
set -uo pipefail
NAME=${1:?scenario name (configs/acceptance/<name>.yaml)}
VIDEO=1; [ "${2:-}" = "--no-video" ] && VIDEO=0
cd "$(dirname "$0")/.."
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
OUT=${ACCEPT_OUT:-outputs/acceptance/mujoco}
TITLE=$(python3 -c "import re;print(re.search(r'title: *(.+)', open('configs/mujoco_so101.yaml').read()).group(1).strip())")
mkdir -p "$OUT"
rm -f "$OUT/$NAME".{log,csv,json,mp4}

uv run --quiet scripts/mujoco_ee_teleop.py \
  --script "configs/acceptance/$NAME.yaml" --log "$OUT/$NAME.csv" --result "$OUT/$NAME.json" \
  --exit_after_script --max_time 600 ${MUJOCO_ARGS:-} > "$OUT/$NAME.log" 2>&1 &
APP=$!
if [ "$VIDEO" = 1 ]; then
  for _ in $(seq 1 100); do grep -q "script step 0" "$OUT/$NAME.log" 2>/dev/null && break; sleep 0.2; done
  ISAAC_WINDOW_NAME="$TITLE" tools/record_isaac_window.sh video "$OUT/$NAME.mp4" 600 &
  REC=$!
fi
wait "$APP"; RC=$?
[ "$VIDEO" = 1 ] && wait "$REC"
grep -E "CHECK|wrote|Traceback|Error" "$OUT/$NAME.log" | tail -n 30
if [ -f "$OUT/$NAME.csv" ]; then
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="os_focus") c=i; next} c && $c=="0"{n++} END{if(n) printf "WARNING: MuJoCo window lacked OS focus for %d frames; input was dropped by design, rerun with the window focused\n", n}' "$OUT/$NAME.csv"
fi
exit $RC
