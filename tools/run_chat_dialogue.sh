#!/usr/bin/env bash
# Chat layer acceptance: run a scripted dialogue (configs/chat/dialogues/<name>.yaml) against the running model service
# and simulator server, recording only the simulator window.
#   docker/vllm/serve.sh up; tools/mujoco_server.sh up        (or docker/isaac/server.sh up)
#   tools/run_chat_dialogue.sh mujoco|isaac hold_return [run_id] [--no-video]
#   CHAT_ARGS="--base-url http://127.0.0.1:8001/v1 --model Qwen3.5-9B-Q4_K_M" tools/run_chat_dialogue.sh ...  # other model
# Outputs: outputs/chat/<name>/<engine>_<run_id>/{transcript.jsonl,result.json,chat.log,video.mp4}
set -uo pipefail
ENGINE=${1:?isaac|mujoco}
NAME=${2:?dialogue name}
RUN=${3:-1}
VIDEO=1; [ "${4:-}" = "--no-video" ] && VIDEO=0
cd "$(dirname "$0")/.."
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
OUT=outputs/chat/$NAME/${ENGINE}_$RUN
mkdir -p "$OUT"
rm -f "$OUT"/transcript.jsonl "$OUT"/result.json "$OUT"/chat.log "$OUT"/video.mp4
case "$ENGINE" in
  isaac) TITLE="Isaac Sim 5.1.0" ;;
  mujoco) TITLE=$(python3 -c "import re;print(re.search(r'title: *(.+)', open('configs/mujoco_so101.yaml').read()).group(1).strip())") ;;
  *) echo "unknown engine $ENGINE" >&2; exit 2 ;;
esac
if [ "$VIDEO" = 1 ]; then
  ISAAC_WINDOW_NAME="$TITLE" setsid tools/record_isaac_window.sh video "$OUT/video.mp4" 900 &   # own group
  REC=$!
  sleep 2
fi
uv run --quiet tools/chat.py --backend "$ENGINE" --script "configs/chat/dialogues/$NAME.yaml" --out "$OUT" ${CHAT_ARGS:-} 2>&1 | tee "$OUT/chat.log"
RC=${PIPESTATUS[0]}
if [ "$VIDEO" = 1 ]; then
  kill -INT -- "-$REC" 2>/dev/null   # SIGINT to the recorder's group: ffmpeg finalizes the file
  wait "$REC" 2>/dev/null
fi
exit $RC
