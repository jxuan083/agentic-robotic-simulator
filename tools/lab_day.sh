#!/usr/bin/env bash
# Right-arm collection day, one step per command. Run from anywhere; paths resolve to this worktree.
#
#   tools/lab_day.sh <step> [args]          steps in order:
#     env          show the day's settings, commit and control rate
#     check        read-only device/calibration check (power on, torque off)
#     overlay      D455 live overlay against the 11/25 tray reference (s saves, q quits)
#     route        teach park -> channel with the leader, reverse to park   (needs SUPPORT="...")
#     route-test   replay park -> channel without the leader, r returns home
#     drop-beige   teach channel -> drop pose over the beige tray
#     drop-pink    teach channel -> drop pose over the pink tray
#     try          park -> channel, then t = leader test grasps from the channel (reach check for the cards)
#     cards        create today's card positions from the template (once) and show them
#     cards-drop   remove unreachable positions, e.g.  cards-drop pink_far:3,5  |  cards-drop --max-x pink=200
#     setup        build today's collection setup from the routes and cards
#     collect      open the collection desk (empty-desk check first, then cards)
#     recover      follower left holding with torque on: follow the route home and release
#     torque       read-only per-joint torque readback of the follower (end of day: all must be 0)
#     status       route summaries and collection progress
#     pack         ZIP the round into AllenHZL/ and archive the handoff dir (NOTES="path ..." optional)
#
# Settings (environment): LAB_DATE (default today, YYMMDD), SUPPORT (park support description, for `route`),
# PY (default the conda lerobot python), REPLACE=1 (re-teach an existing route), FAKE=1 (rehearse route steps
# with simulated arms), RATE (route playback speed factor, default 1.0), RELEASE (drop gripper %, default 60).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/miniconda3/envs/lerobot/bin/python}"
DATE="${LAB_DATE:-$(date +%y%m%d)}"
CONFIG=outputs/single.local.yaml
LAYOUT="lab-right-$DATE"
ROUTES=outputs/routes
ROUTE="$ROUTES/right-park-channel-$DATE.json"
DROP_BEIGE="$ROUTES/right-drop-beige-$DATE.json"
DROP_PINK="$ROUTES/right-drop-pink-$DATE.json"
CARDS="outputs/$DATE-cards.json"
SETUP="outputs/$DATE-right.setup.json"
SESSION="outputs/$DATE-right"
REFERENCE=outputs/261125-camera/global/rgb.png
D455=038122250316

if [[ -n "${FAKE:-}" ]]; then MOTION=(--fake); else MOTION=(--config "$CONFIG" --allow-motion); fi
REPLACE_FLAG=(); [[ -n "${REPLACE:-}" ]] && REPLACE_FLAG=(--replace)
need() { for f in "$@"; do [[ -e "$f" ]] || { echo "missing $f — run the earlier step first" >&2; exit 1; }; done; }
say() { printf '\n== %s\n' "$*"; }

step="${1:-env}"; shift || true
case "$step" in
  env)
    say "date $DATE  layout $LAYOUT"
    git log --oneline -1
    git status --short --untracked-files=no | head
    grep -E '^control_hz' "$CONFIG" || true
    "$PY" -c "import lerobot, sys; print('python', sys.version.split()[0], 'lerobot', lerobot.__version__)"
    echo "routes: $ROUTE  $DROP_BEIGE  $DROP_PINK"
    echo "cards: $CARDS  setup: $SETUP  session: $SESSION"
    ;;
  check)
    "$PY" tools/real_teleop.py check --config "$CONFIG" && echo "check OK" ;;
  overlay)
    need "$REFERENCE"
    say "move the trays until their rims overlap the reference; s saves, q quits"
    "$PY" tools/tray_overlay.py "$REFERENCE" --serial "$D455" --output "outputs/$DATE-camera/tray-overlay.png" ;;
  route)
    [[ -n "${SUPPORT:-}" ]] || { echo 'set SUPPORT="<park support description>"' >&2; exit 1; }
    mkdir -p "$ROUTES"
    say "leader: leave the support, clear the folded area, then turn the wrist; stop at a channel pose that does"
    say "not hide the pink tray from the D455; c saves, r reverses home, y releases"
    "$PY" tools/real_route.py record "${MOTION[@]}" --route "$ROUTE" --layout "$LAYOUT" --support "$SUPPORT" \
      "${REPLACE_FLAG[@]}" ;;
  route-test)
    need "$ROUTE"
    "$PY" tools/real_route.py run "${MOTION[@]}" --route "$ROUTE" --no-leader --rate "${RATE:-1.0}" ;;
  drop-beige|drop-pink)
    need "$ROUTE"
    tray="${step#drop-}"; out="$DROP_BEIGE"; [[ "$tray" == pink ]] && out="$DROP_PINK"
    say "leader from the channel to ~3-5 cm above the $tray tray centre without touching the rim; c saves"
    "$PY" tools/real_route.py record-drop "${MOTION[@]}" --route "$ROUTE" --tray "$tray" --out "$out" \
      --release "${RELEASE:-60}" "${REPLACE_FLAG[@]}" ;;
  try)
    need "$ROUTE"
    say "at the channel press t, test grasps on the far/edge/60 deg card positions, c returns to the channel"
    "$PY" tools/real_route.py run "${MOTION[@]}" --route "$ROUTE" --rate "${RATE:-1.0}" ;;
  cards)
    [[ -e "$CARDS" ]] || cp configs/real_collection.cards.template.json "$CARDS"
    "$PY" tools/real_cards.py show "$CARDS" ;;
  cards-drop)
    need "$CARDS"
    "$PY" tools/real_cards.py drop "$CARDS" "$@" ;;
  setup)
    need "$ROUTE" "$DROP_BEIGE" "$DROP_PINK" "$CARDS"
    "$PY" tools/real_route.py setup --route "$ROUTE" --drop "$DROP_BEIGE" --drop "$DROP_PINK" --cards "$CARDS" \
      --version "right-$DATE-v1" --workspace "empty desk verified $DATE" --out "$SETUP" ;;
  collect)
    need "$SETUP"
    say "first press the empty-desk check; first card: record, review the handoff frame, then real replay"
    "$PY" tools/real_collection.py --config "$CONFIG" --setup "$SETUP" --session "$SESSION" --allow-motion ;;
  recover)
    need "$ROUTE"
    "$PY" tools/real_route.py recover "${MOTION[@]}" --route "$ROUTE" ;;
  torque)
    "$PY" - "$CONFIG" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from agentic_sim.real_hardware import check_devices, load_profile, open_follower
profile = load_profile(sys.argv[1])
check_devices(profile)
(arm, pair), = profile["arms"].items()
# managed + resume: connect even if torque is on, write nothing, never release.
with open_follower(arm, pair["follower"], managed=True, resume=True) as follower:
    states = follower.torque_states()
print(arm, "follower torque:", states)
off = all(v == 0 for v in states.values())
print("torque OFF on all joints" if off else "TORQUE STILL ON OR UNREADABLE — support the arm, then tools/lab_day.sh recover")
sys.exit(0 if off else 1)
PYEOF
    ;;
  status)
    for f in "$ROUTE" "$DROP_BEIGE" "$DROP_PINK"; do [[ -e "$f" ]] && echo "$f ok"; done
    [[ -e "$ROUTE" ]] && "$PY" tools/real_route.py show --route "$ROUTE" | grep -E '"(waypoints|recorded_moving_seconds|reverse_to_park|forward_to_channel)"'
    if [[ -e "$SESSION/session.json" ]]; then
      "$PY" -c "import json,collections,sys; s=json.load(open(sys.argv[1])); c=collections.Counter(a['status'] for a in s['attempts']); print('session', dict(c), 'replays', len(s['replays']))" "$SESSION/session.json"
    fi ;;
  pack)
    items=()
    for f in "$SESSION" "$ROUTES" "$SETUP" "$CARDS" "outputs/$DATE-camera"; do [[ -e "$f" ]] && items+=("$f"); done
    read -r -a notes <<< "${NOTES:-}"
    "$PY" tools/pack_handoff.py --date "$DATE" --include "${items[@]}" --notes "${notes[@]}" ;;
  *)
    sed -n '2,25p' "$0"; exit 1 ;;
esac
