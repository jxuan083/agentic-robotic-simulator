#!/usr/bin/env bash
# Right-arm collection day, one step per command. Run from anywhere; paths resolve to this worktree.
#
#   tools/lab_day.sh <step> [args]          steps in order:
#     env          show the day's settings, commit and control rate
#     check        read-only device/calibration check (power on, torque off)
#     overlay      D455 live overlay against the current tray reference (s saves, q quits)
#     reference    after moving the trays: save a new D455 reference (markers 40/41/42) that overlay then uses
#     trays        fix the D455 tray corners from the clean reference (once per tray layout)
#     guide        live D455 placement guide (tray grid + block mm; --target TRAY X Y YAW COLOR or --reach-plan), s saves
#     locate       D455 snapshot -> block positions in tray-local mm (+ annotated image)
#     route        teach park -> channel with the leader, reverse to park   (needs SUPPORT="...")
#     route-tune   remove taught pauses/jitter, fix the gripper opening: GRIP=<pct>, or hold the leader
#                  gripper at the cube-ready opening while it reads (read-only); then route-test again.
#                  KEEP_GRIP=1 keeps the tuned opening; --straight-from N goes straight from waypoint N to the channel
#     route-test   replay park -> channel without the leader, r returns home
#     drop-beige   teach channel -> drop pose over the beige tray
#     drop-pink    teach channel -> drop pose over the pink tray
#     drop-test-beige / drop-test-pink   replay park -> channel -> drop pose -> release -> back (no leader)
#     grasp        park-start pick-and-place rounds (random target, slots fill up), session outputs/<date>-park
#     datacheck    every 5 rounds: training selection and desk issues of the newest LAST (20) episodes
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
# Settings (environment): LAB_DATE (default today, YYMMDD; session folders), RIG_DATE (day whose routes and tray
# corners to reuse, default LAB_DATE), SUPPORT (park support description, for `route`),
# PY (default the conda lerobot python), REPLACE=1 (re-teach an existing route), FAKE=1 (rehearse route steps
# with simulated arms), MANUAL_ALIGN=1 (align the leader by hand), GRIP (route-tune gripper %), RATE (route playback speed factor, default 1.0), RELEASE (drop gripper %, default 35).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/miniconda3/envs/lerobot/bin/python}"
DATE="${LAB_DATE:-$(date +%y%m%d)}"
RIG="${RIG_DATE:-$DATE}"   # routes, tray corners and setup taught on this day (reuse them on a later day)
CONFIG=outputs/single.local.yaml
LAYOUT="lab-right-$RIG"
ROUTES=outputs/routes
ROUTE="$ROUTES/right-park-channel-$RIG.json"
DROP_BEIGE="$ROUTES/right-drop-beige-$RIG.json"
DROP_PINK="$ROUTES/right-drop-pink-$RIG.json"
CARDS="outputs/$DATE-cards.json"
TRAYS="outputs/$RIG-trays/trays.json"   # fixed D455 tray corners from the clean reference (step `trays`)
SETUP="outputs/$RIG-right.setup.json"
SESSION="outputs/$DATE-right"
# D455 tray reference for `overlay`: the latest `reference` capture, else the 11/25 one (REFERENCE=... overrides)
REFERENCE="${REFERENCE:-$(cat outputs/current-reference.txt 2>/dev/null || echo outputs/261125-camera/global/rgb.png)}"
CAMPY="${CAMPY:-$HOME/Lehome/lehome_solution/.venv/bin/python}"  # GUI + ArUco + RealSense, as on 11/24-25
D455=038122250316

if [[ -n "${FAKE:-}" ]]; then MOTION=(--fake); else MOTION=(--config "$CONFIG" --allow-motion); fi
[[ -n "${MANUAL_ALIGN:-}" ]] && MOTION+=(--manual-align)
REPLACE_FLAG=(); [[ -n "${REPLACE:-}" ]] && REPLACE_FLAG=(--replace)
need() { for f in "$@"; do [[ -e "$f" ]] || { echo "missing $f — run the earlier step first" >&2; exit 1; }; done; }
say() { printf '\n== %s\n' "$*"; }

step="${1:-env}"; shift || true
case "$step" in
  env)
    say "date $DATE  rig $RIG  layout $LAYOUT"
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
    "$CAMPY" tools/tray_overlay.py "$REFERENCE" --serial "$D455" --output "outputs/$DATE-camera/tray-overlay.png" ;;
  reference)
    # After moving the trays: new D455 RGB-D reference with markers 40/41/42, compared with the previous one.
    name="${1:-trays-$(date +%H%M)}"; root="outputs/$DATE-camera"; mkdir -p "$root"
    [[ -e "$root/measurements.json" ]] || cp outputs/261125-camera/measurements.json "$root/"  # same printed markers
    say "markers 40/41/42 and both trays fully in view (beige right edge too); s saves, q quits"
    "$CAMPY" tools/calibrate_real_cameras.py --root "$root" global --config "$CONFIG" --name "$name" \
      --compare "$(dirname "$REFERENCE")"
    if [[ -e "$root/$name/rgb.png" ]]; then
      echo "$root/$name/rgb.png" > outputs/current-reference.txt
      echo "overlay now uses $root/$name/rgb.png; re-teach both drop routes (REPLACE=1) after a tray move"
    fi ;;
  trays)
    # Fix the tray corners once from the clean reference (no hands or arm over the trays).
    [[ -e "$TRAYS" ]] && { echo "$TRAYS exists; remove it to refit"; exit 1; }
    "$PY" tools/real_locate.py --image "$REFERENCE" --save-trays --out "$(dirname "$TRAYS")" ;;
  guide)
    # Live D455 placement guide: tray grids, live block mm, optional target  (guide --target pink 120 80 30 red)
    TRAY_ARG=(); [[ -e "$TRAYS" ]] && TRAY_ARG=(--trays "$TRAYS")
    "$CAMPY" tools/real_place_guide.py --serial "$D455" --out "outputs/$DATE-camera/guide" "${TRAY_ARG[@]}" "$@" ;;
  locate)
    # D455 coarse block positions in tray-local mm (red/blue/yellow); annotated image to check against the table.
    out="outputs/$DATE-camera/locate-$(date +%H%M%S)"
    TRAY_ARG=(); [[ -e "$TRAYS" ]] && TRAY_ARG=(--trays "$TRAYS")
    "$PY" tools/real_locate.py --serial "$D455" --out "$out" "${TRAY_ARG[@]}" ;;
  route)
    [[ -n "${SUPPORT:-}" ]] || { echo 'set SUPPORT="<park support description>"' >&2; exit 1; }
    mkdir -p "$ROUTES"
    say "leader: leave the support, clear the folded area, then turn the wrist; stop at a channel pose that does"
    say "not hide the pink tray from the D455; c saves, r reverses home, y releases"
    "$PY" tools/real_route.py record "${MOTION[@]}" --route "$ROUTE" --layout "$LAYOUT" --support "$SUPPORT" \
      "${REPLACE_FLAG[@]}" ;;
  route-tune)
    need "$ROUTE"
    if [[ -n "${GRIP:-}" ]]; then src=(--grip "$GRIP"); elif [[ -n "${KEEP_GRIP:-}" ]]; then src=(); else
      say "hold the leader gripper at the cube-ready opening, keep it still"; src=(--config "$CONFIG"); fi
    "$PY" tools/real_route.py tune --route "$ROUTE" "${src[@]}" "$@" ;;
  route-test)
    need "$ROUTE"
    "$PY" tools/real_route.py run "${MOTION[@]}" --route "$ROUTE" --no-leader --rate "${RATE:-1.0}" ;;
  drop-beige|drop-pink)
    need "$ROUTE"
    tray="${step#drop-}"; out="$DROP_BEIGE"; [[ "$tray" == pink ]] && out="$DROP_PINK"
    say "leader from the channel to ~3-5 cm above the $tray tray centre without touching the rim; c saves"
    "$PY" tools/real_route.py record-drop "${MOTION[@]}" --route "$ROUTE" --tray "$tray" --out "$out" \
      --release "${RELEASE:-35}" "${REPLACE_FLAG[@]}" ;;
  drop-test-beige|drop-test-pink)
    tray="${step#drop-test-}"; drop="$DROP_BEIGE"; [[ "$tray" == pink ]] && drop="$DROP_PINK"
    need "$ROUTE" "$drop"
    say "no leader: park -> channel -> $tray drop pose -> open -> back; g empty gripper, b hold a block"
    "$PY" tools/real_route.py drop-test "${MOTION[@]}" --route "$ROUTE" --drop "$drop" ;;
  grasp)
    # Park-start pick-and-place rounds: finals set in pink, D455 picks a random block each episode, beige slots fill
    # up (4 -> 1 blocks left), re-lay when full. PICK=manual for the colour-key flow.
    need "$ROUTE" "$DROP_BEIGE" "$DROP_PINK" "$TRAYS"
    "$PY" tools/real_route.py grasp "${MOTION[@]}" --route "$ROUTE" --drop "$DROP_BEIGE" --drop "$DROP_PINK" \
      --trays "$TRAYS" --session "outputs/$DATE-park" --pick "${PICK:-random}" ;;
  datacheck)
    # Between collection blocks: would training take the newest episodes, and what to fix at the desk (read-only).
    train="${TRAIN_TREE:-../agentic-robotic-simulator-real-training}"
    need "$train/tools/check_park_session.py" "outputs/$DATE-park"
    "$PY" "$train/tools/check_park_session.py" "outputs/$DATE-park" --last "${LAST:-20}" ;;
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
    need "$ROUTE" "$DROP_BEIGE" "$DROP_PINK" "$CARDS" "$TRAYS"
    "$PY" tools/real_route.py setup --route "$ROUTE" --drop "$DROP_BEIGE" --drop "$DROP_PINK" --cards "$CARDS" --trays "$TRAYS" \
      --version "right-$DATE-v1" --workspace "empty desk verified $DATE" --out "$SETUP" ;;
  collect)
    need "$SETUP"
    say "first press the empty-desk check; first card: record from the start pose to a stable lift, review start/close/lift, then real replay"
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
    for f in "$SESSION" "outputs/$DATE-park" "outputs/$DATE-park-rejected" "$ROUTES" "$SETUP" "$CARDS" "$(dirname "$TRAYS")" "outputs/$DATE-camera"; do [[ -e "$f" ]] && items+=("$f"); done
    read -r -a notes <<< "${NOTES:-}"
    "$PY" tools/pack_handoff.py --date "$DATE" --include "${items[@]}" --notes "${notes[@]}" ;;
  *)
    sed -n '2,34p' "$0"; exit 1 ;;
esac
