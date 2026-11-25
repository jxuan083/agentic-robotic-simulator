#!/usr/bin/env bash
# Capture ONLY one application window by X window id (never the whole desktop, never overlapping windows).
# Isaac Sim by default; ISAAC_WINDOW_NAME=<window title> for another (e.g. the MuJoCo window), or
# ISAAC_WINDOW_ID=<X window id> for exactly one window (the simulator's right-column "Record screen" passes its own).
#   tools/record_isaac_window.sh shot  outputs/shot.png
#   tools/record_isaac_window.sh video outputs/run.mp4 [max_seconds]   # stops when the window closes, at max_seconds
#                                                                       # or on SIGINT / SIGTERM (the file is still
#                                                                       # written); a resized window (fullscreen)
#                                                                       # continues as a new segment
set -euo pipefail
MODE=${1:?shot|video}
OUT=${2:?output path}
MAX=${3:-600}
export DISPLAY=${DISPLAY:-:1}
NAME=${ISAAC_WINDOW_NAME:-Isaac Sim 5.1.0}
ID=${ISAAC_WINDOW_ID:-}
[ -n "$ID" ] && NAME="id $ID"

geom() {
  local info
  if [ -n "$ID" ]; then
    info=$(xwininfo -display "$DISPLAY" -id "$ID" 2>/dev/null) || return 1
    grep -q "Map State: IsViewable" <<<"$info" || return 1
  else
    info=$(xwininfo -display "$DISPLAY" -name "$NAME" 2>/dev/null) || return 1
  fi
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
  # The window can change size while recording (e.g. the operator workspace going fullscreen and back): each size is
  # recorded as its own segment, then the segments are scaled / padded to the largest size and joined.
  STOP=0
  trap 'STOP=1' INT TERM                # stop requested: end the current segment, still write the file
  START=$(date +%s); I=0; SEGS=()
  while [ "$STOP" = 0 ] && geom; do
    LEFT=$((MAX - ($(date +%s) - START))); [ "$LEFT" -gt 0 ] || break
    SEG="${OUT%.*}.part${I}.mkv"
    ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -framerate 30 -video_size "${W}x${H}" -window_id "$((WID))" \
      -i "$DISPLAY" -t "$LEFT" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$SEG" &
    FF=$!; W0=$W; H0=$H; GONE=0
    while kill -0 "$FF" 2>/dev/null; do
      if [ "$STOP" = 1 ]; then kill -INT "$FF" 2>/dev/null || true; break; fi
      if ! geom; then GONE=1; kill -INT "$FF" 2>/dev/null || true; break; fi
      if [ "$W" != "$W0" ] || [ "$H" != "$H0" ]; then kill -INT "$FF" 2>/dev/null || true; break; fi
      sleep 0.3
    done
    while kill -0 "$FF" 2>/dev/null; do wait "$FF" || true; done   # a trapped signal interrupts wait
    if ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$SEG" 2>/dev/null | grep -q .; then
      SEGS+=("$SEG")
    else
      rm -f "$SEG"                      # nothing readable (the window was being mapped / resized)
    fi
    I=$((I + 1))
    [ "$GONE" = 0 ] || break
  done
  trap - INT TERM
  if [ "${#SEGS[@]}" -eq 0 ]; then echo "nothing recorded" >&2; exit 1; fi
  if [ "${#SEGS[@]}" -eq 1 ]; then
    ffmpeg -loglevel error -y -i "${SEGS[0]}" -c copy "$OUT"
  else
    MW=0; MH=0; IN=(); F=""
    for S in "${SEGS[@]}"; do
      IFS=, read -r SW SH < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$S")
      [ "$SW" -gt "$MW" ] && MW=$SW; [ "$SH" -gt "$MH" ] && MH=$SH
    done
    K=0
    for S in "${SEGS[@]}"; do
      IN+=(-i "$S")
      F+="[$K:v]scale=${MW}:${MH}:force_original_aspect_ratio=decrease,pad=${MW}:${MH}:(ow-iw)/2:(oh-ih)/2,setsar=1[v$K];"
      K=$((K + 1))
    done
    for ((J = 0; J < K; J++)); do F+="[v$J]"; done
    ffmpeg -loglevel error -y "${IN[@]}" -filter_complex "${F}concat=n=${K}:v=1:a=0[out]" -map "[out]" \
      -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$OUT"
  fi
  rm -f "${SEGS[@]}"
fi
echo "wrote $OUT" >&2
