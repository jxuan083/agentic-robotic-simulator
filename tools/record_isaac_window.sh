#!/usr/bin/env bash
# Capture ONLY the Isaac Sim window by X window id (never the whole desktop, never overlapping windows).
#   tools/record_isaac_window.sh shot  outputs/shot.png
#   tools/record_isaac_window.sh video outputs/run.mp4 [max_seconds]   # stops when the window closes
set -euo pipefail
MODE=${1:?shot|video}
OUT=${2:?output path}
MAX=${3:-600}
export DISPLAY=${DISPLAY:-:1}
NAME=${ISAAC_WINDOW_NAME:-Isaac Sim 5.1.0}

geom() {
  local info
  info=$(xwininfo -display "$DISPLAY" -name "$NAME" 2>/dev/null) || return 1
  WID=$(awk '/Window id:/{print $4}' <<<"$info")
  W=$(awk '/Width:/{print $2}' <<<"$info")
  H=$(awk '/Height:/{print $2}' <<<"$info")
  W=$((W / 2 * 2)); H=$((H / 2 * 2))
  [ "$W" -gt 0 ] && [ "$H" -gt 0 ]
}

for _ in $(seq 1 120); do geom && break; sleep 1; done
geom || { echo "window '$NAME' not found" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
echo "capturing window '$NAME' id=$WID ${W}x${H}" >&2

if [ "$MODE" = shot ]; then
  ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -video_size "${W}x${H}" -window_id "$((WID))" -i "$DISPLAY" -frames:v 1 "$OUT"
else
  ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -framerate 30 -video_size "${W}x${H}" -window_id "$((WID))" -i "$DISPLAY" \
    -t "$MAX" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$OUT" &
  FF=$!
  while kill -0 "$FF" 2>/dev/null; do
    if ! xwininfo -display "$DISPLAY" -name "$NAME" >/dev/null 2>&1; then
      kill -INT "$FF" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "$FF" || true
fi
echo "wrote $OUT" >&2
