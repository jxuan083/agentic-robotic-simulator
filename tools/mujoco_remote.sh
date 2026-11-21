#!/usr/bin/env bash
# Run MuJoCo on the host against the browser desktop X server (not inside Isaac).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${ISAAC_DESKTOP_CONTAINER:-ars-isaac-desktop}"
SOCKET_DIR="${ISAAC_DESKTOP_DIR:-$HOME/docker/ars-isaac-desktop}/socket"
PYTHON="${MUJOCO_PYTHON:-$ROOT/.venv-mujoco/bin/python}"
DISPLAY=":99"

if [[ ! -x "$PYTHON" ]]; then
  cat >&2 <<EOF
MuJoCo's isolated host Python was not found: $PYTHON
Create it without touching Isaac or the robot environment:
  python3 -m venv "$ROOT/.venv-mujoco"
  "$ROOT/.venv-mujoco/bin/pip" install -r "$ROOT/requirements-mujoco.txt"
EOF
  exit 2
fi
MENAGERIE_DIR="${MENAGERIE_DIR:-$HOME/ref/mujoco_menagerie}"
if [[ ! -f "$MENAGERIE_DIR/robotstudio_so101/scene.xml" && ! -d "$MENAGERIE_DIR/robotstudio_so101" ]]; then
  echo "SO-101 MuJoCo assets missing under $MENAGERIE_DIR." >&2
  echo "Fetch the repo-pinned Menagerie assets with: MENAGERIE_DIR='$MENAGERIE_DIR' tools/fetch_mujoco_assets.sh" >&2
  exit 2
fi
"$ROOT/docker/isaac/remote.sh" up >/dev/null
AUTH_SRC="$SOCKET_DIR/Xauthority"
[[ -d "$SOCKET_DIR" ]] || { echo "Desktop X socket directory is unavailable: $SOCKET_DIR" >&2; exit 2; }
[[ -S "$SOCKET_DIR/X99" ]] || { echo "Desktop X socket is missing: $SOCKET_DIR/X99" >&2; exit 2; }
[[ -f "$AUTH_SRC" ]] || { echo "Desktop Xauthority is missing: $AUTH_SRC" >&2; exit 2; }
HOST_SOCKET_DIR="/tmp/.X11-unix"
HOST_SOCKET="$HOST_SOCKET_DIR/X99"
mkdir -p "$HOST_SOCKET_DIR"
if [[ -L "$HOST_SOCKET" ]]; then
  [[ "$(readlink -f "$HOST_SOCKET")" == "$(readlink -f "$SOCKET_DIR/X99")" ]] || {
    echo "Refusing unexpected X99 symlink: $HOST_SOCKET" >&2; exit 2;
  }
elif [[ -e "$HOST_SOCKET" ]]; then
  echo "Refusing to replace existing non-symlink X99 entry: $HOST_SOCKET" >&2
  exit 2
else
  ln -s "$SOCKET_DIR/X99" "$HOST_SOCKET"
fi
AUTH_COPY="$(mktemp "${TMPDIR:-/tmp}/ars-mujoco-xauth.XXXXXX")"
trap 'rm -f "$AUTH_COPY"' EXIT
umask 077
docker exec "$NAME" cat /tmp/.X11-unix/Xauthority > "$AUTH_COPY"
chmod 600 "$AUTH_COPY"
export DISPLAY XAUTHORITY="$AUTH_COPY" MUJOCO_GL="${MUJOCO_GL:-glfw}"
export MENAGERIE_DIR
cd "$ROOT"
if [[ "${1:-}" == "--explore-wrist" ]]; then
  shift
  RUN_TAG="$(date +%Y%m%d-%H%M%S)-$$"
  "$PYTHON" tools/search_target_color.py --episodes 1 --gui \
    --output "outputs/perception-target-search/gui-$RUN_TAG" "$@"
  exit $?
fi
if [[ "${1:-}" == "--full-scan" ]]; then
  shift
  "$PYTHON" tools/evaluate_perception_exploration.py --episodes 1 --gui \
    --output "outputs/perception-exploration/gui-full-scan-$(date +%Y%m%d-%H%M%S)-$$" "$@"
  exit $?
fi
if [[ $# -eq 0 ]]; then
  set -- scripts/mujoco_ee_teleop.py \
    --scene configs/scenes/lab_20261120.yaml --random-blocks \
    --live-perception --perception-camera both --perception-hz 2 \
    --perception-json outputs/perception-live/latest.json
  "$PYTHON" "$@"
  exit $?
fi
"$PYTHON" scripts/mujoco_ee_teleop.py "$@"
