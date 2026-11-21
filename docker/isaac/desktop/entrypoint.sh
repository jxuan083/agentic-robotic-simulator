#!/usr/bin/env bash
set -euo pipefail
export DISPLAY=:99 XAUTHORITY=/tmp/.X11-unix/Xauthority
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 "$XAUTHORITY" "$XAUTHORITY-c" "$XAUTHORITY-l"
touch "$XAUTHORITY"
chmod 600 "$XAUTHORITY"
xauth add "$DISPLAY" . "$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
# The simulator and desktop have different hostnames. Share this cookie using FamilyWild.
# Read first: simultaneous xauth readers/writers on this file can deadlock.
xauth nlist "$DISPLAY" > /tmp/ars-xauth-record
sed 's/^..../ffff/' /tmp/ars-xauth-record | xauth nmerge -
rm -f /tmp/ars-xauth-record

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 0' TERM INT
# Disable MIT-SHM: host GUI clients run outside this container's IPC namespace, so
# shared-memory segment IDs passed by Xlib/Mesa are not valid inside this X server.
Xvfb "$DISPLAY" -screen 0 1600x1000x24 -nolisten tcp -noreset -extension MIT-SHM -auth "$XAUTHORITY" &
pids+=("$!")
for _ in {1..100}; do
  xdpyinfo >/dev/null 2>&1 && break
  sleep 0.1
done
xdpyinfo >/dev/null
python3 /usr/local/bin/ars-configure-openbox.py
openbox &
pids+=("$!")
# Only the browser endpoint is published, bound to host loopback by remote.sh.
x11vnc -display "$DISPLAY" -auth "$XAUTHORITY" -localhost -rfbport 5900 \
  -forever -shared -nopw -repeat -xkb -noshm &
pids+=("$!")
websockify --web /usr/share/novnc 6080 127.0.0.1:5900 &
pids+=("$!")
wait -n "${pids[@]}"
exit 1  # A required service exited; let Docker report the desktop as stopped.
