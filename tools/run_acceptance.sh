#!/usr/bin/env bash
# Run one scripted acceptance scenario in the GUI and record only the Isaac window.
#   tools/run_acceptance.sh keyboard_pick_place [--no-video]
#   ISAAC_ARGS='--device cuda:0' tools/run_acceptance.sh ...   extra app arguments (e.g. physics device)
# Outputs: outputs/acceptance/<name>.{log,csv,json,mp4}
set -uo pipefail
NAME=${1:?scenario name (configs/acceptance/<name>.yaml)}
VIDEO=1; [ "${2:-}" = "--no-video" ] && VIDEO=0
cd "$(dirname "$0")/.."
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
OUT=${ACCEPT_OUT:-outputs/acceptance}
mkdir -p "$OUT"
rm -f "$OUT/$NAME".{log,csv,json,mp4}

docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py \
  --script "configs/acceptance/$NAME.yaml" --log "$OUT/$NAME.csv" --result "$OUT/$NAME.json" \
  --exit_after_script --profile --max_time 600 ${ISAAC_ARGS:-} > "$OUT/$NAME.log" 2>&1 &
APP=$!

if [ "$VIDEO" = 1 ]; then
  # start recording once the scene is live (first script step), stop when the window closes
  for _ in $(seq 1 300); do grep -q "script step 0" "$OUT/$NAME.log" 2>/dev/null && break; sleep 1; done
  tools/record_isaac_window.sh video "$OUT/$NAME.mp4" 600 &
  REC=$!
fi
wait "$APP"; RC=$?
[ "$VIDEO" = 1 ] && wait "$REC"
grep -E "CHECK|wrote|profile|Traceback|Error" "$OUT/$NAME.log" | grep -v "Performance warning" | tail -n 30
# real (OS) focus loss drops all input by design, so a run without focus is not a valid acceptance run
if [ -f "$OUT/$NAME.csv" ]; then
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="os_focus") c=i; next} c && $c=="0"{n++} END{if(n) printf "WARNING: Isaac window lacked OS focus for %d frames; input was dropped by design, rerun with the window focused\n", n}' "$OUT/$NAME.csv"
fi
exit $RC
