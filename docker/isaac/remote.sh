#!/usr/bin/env bash
# Browser desktop for VS Code Remote. Only the noVNC port is published, on loopback.
# remote.sh up | status | logs | down | run [Isaac command and arguments...]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ISAAC_DESKTOP_IMAGE:-ars-isaac-desktop:ubuntu24.04}"
NAME="${ISAAC_DESKTOP_CONTAINER:-ars-isaac-desktop}"
DESKTOP_DIR="${ISAAC_DESKTOP_DIR:-$HOME/docker/ars-isaac-desktop}"
PORT="${ISAAC_DESKTOP_PORT:-6080}"
SOCKET_DIR="$DESKTOP_DIR/socket"
url() { echo "http://localhost:$PORT/vnc.html?autoconnect=1&resize=scale"; }
up() {
  mkdir -p "$SOCKET_DIR"
  exec 9> "$DESKTOP_DIR/start.lock"
  flock 9
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" "$REPO_DIR/docker/isaac/desktop" 9>&-
  fi
  if ! docker container inspect "$NAME" >/dev/null 2>&1; then
    docker run -d --name "$NAME" --label ars.role=browser-desktop \
      --shm-size 256m -p "127.0.0.1:$PORT:6080" \
      -v "$SOCKET_DIR:/tmp/.X11-unix:rw" "$IMAGE" 9>&-
  else
    ROLE="$(docker inspect "$NAME" --format '{{index .Config.Labels "ars.role"}}')"
    [[ "$ROLE" == browser-desktop ]] || { echo "Container $NAME is not an ARS browser desktop" >&2; exit 1; }
    docker start "$NAME" >/dev/null 9>&-
  fi
  for _ in {1..45}; do
    HEALTH="$(docker inspect "$NAME" --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}')"
    if [[ "$HEALTH" == 'running healthy' ]]; then
      echo "VS Code: forward port $PORT, then open:"
      url
      return
    fi
    [[ "$HEALTH" == running* ]] || break
    sleep 1
  done
  docker logs --tail 25 "$NAME" >&2
  echo 'Browser desktop did not become ready.' >&2
  return 1
}
case "${1:-up}" in
  up) up ;;
  run)
    shift
    up
    exec 9>&-
    export DISPLAY=:99 XAUTHORITY="$SOCKET_DIR/Xauthority" ISAAC_X11_DIR="$SOCKET_DIR"
    if [[ $# -eq 0 ]]; then
      set -- /isaac-sim/python.sh scripts/isaac_ee_teleop.py \
        --scene configs/scenes/lab_20261120.yaml --random-blocks --live-perception
    fi
    exec "$REPO_DIR/docker/isaac/run.sh" "$@"
    ;;
  status)
    docker inspect "$NAME" --format '{{.Name}}: {{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}'
    url
    ;;
  logs) exec docker logs -f --tail 40 "$NAME" ;;
  down)
    if docker container inspect "$NAME" >/dev/null 2>&1; then
      [[ "$(docker inspect "$NAME" --format '{{index .Config.Labels "ars.role"}}')" == browser-desktop ]] || exit 1
      docker rm -f "$NAME"
    fi
    ;;
  *) echo 'Usage: remote.sh up|run [command...]|status|logs|down' >&2; exit 2 ;;
esac
