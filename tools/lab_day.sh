#!/usr/bin/env bash
# Right-arm collection day, one step per command. Run from anywhere; paths resolve to this worktree.
#
#   tools/lab_day.sh <step> [args]          steps in order:
#     env          show the day's settings, commit and control rate
#     check        read-only device/calibration check (power on, torque off)
#     overlay      D455 live overlay against the current tray reference (s saves, q quits); uses overlay-zone.png if made
#     zone-image   draw the safe grasp zone + beige slots on the tray reference -> <reference dir>/overlay-zone.png
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
#     wristcam     fix the wrist camera's white balance (WB, default 4600 K) and frame rate; once per day
#     datacheck    every 5 rounds: training selection and desk issues of the newest LAST (20) episodes
#     try          park -> channel, then t = leader test grasps from the channel (reach check for the cards)
#     cards        create today's card positions from the template (once) and show them
#     cards-drop   remove unreachable positions, e.g.  cards-drop pink_far:3,5  |  cards-drop --max-x pink=200
#     setup        build today's collection setup from the routes and cards
#     collect      open the collection desk (empty-desk check first, then cards)
#     recover      follower left holding with torque on: follow the route home and release
#     d455         finals: live D455 blocks (colour, mount-frame mm, yaw, wrist turn); s saves, q quits
#     reach        finals: d455 + arm over a chosen block (park -> start -> hover, nudge, k logs the aim)
#     d455-capture finals: torque off, follower moved by hand to ~8 poses -> D455 depth + readings (camera pose)
#     d455-fit     fit the D455 pose from the capture (uv / MuJoCo) -> outputs/d455/pose_<date>.json; use POSE=...
#     joints       read-only live joint readout; Enter + label saves a sample
#     torque       read-only per-joint torque readback of the follower (end of day: all must be 0)
#     status       route summaries and collection progress
#     pack         ZIP the round into AllenHZL/ and archive the handoff dir (NOTES="path ..." optional)
#
# Finals: POSE (D455 pose file, default outputs/d455/pose_261125.json), CORR (reach sag correction: radial mm, rotation deg, z mm),
# SPEED (reach joint cap deg/s, default 75).
# Settings (environment): LAB_DATE (default today, YYMMDD; session folders), RIG_DATE (day whose routes and tray
# corners to reuse, default LAB_DATE), SUPPORT (park support description, for `route`),
# PY (default the conda lerobot python), FOLLOWER_PORT + FOLLOWER_ID (joints / torque with the follower only), REPLACE=1 (re-teach an existing route), FAKE=1 (rehearse route steps
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
    # The zone-annotated reference (step zone-image: safe grasp zone + placement squares) when it exists.
    ZONE_REF="$(dirname "$REFERENCE")/overlay-zone.png"; OVER_REF="$REFERENCE"; [[ -e "$ZONE_REF" ]] && OVER_REF="$ZONE_REF"
    say "move the trays until their rims overlap the reference ($OVER_REF); lay blocks inside the orange arc; s saves, q quits"
    "$CAMPY" tools/tray_overlay.py "$OVER_REF" --serial "$D455" --output "outputs/$DATE-camera/tray-overlay.png" ;;
  zone-image)
    # Reference frame + tray outlines + safe grasp zone (orange) + beige slots P1-P4 -> <reference dir>/overlay-zone.png
    need "$REFERENCE"
    TRAY_ARG=(); [[ -e "$TRAYS" ]] && TRAY_ARG=(--trays "$TRAYS")
    "$PY" tools/finals_reach.py zone-image --reference "$REFERENCE" --image-out "$(dirname "$REFERENCE")/overlay-zone.png" \
        "${TRAY_ARG[@]}" ${POSE:+--pose "$POSE"} ${CORR:+--corr "$CORR"} "$@" ;;
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
  wristcam)
    # Fixed white balance and frame rate for the wrist camera: auto white balance turns the white desk cyan when
    # the pink tray fills the view, and dynamic frame rate drops frames in dim light. Lost on replug: run each day.
    dev="${WRIST_DEV:-/dev/v4l/by-path/pci-0000:09:00.0-usb-0:1:1.0-video-index0}"
    v4l2-ctl -d "$dev" -c white_balance_automatic=0
    v4l2-ctl -d "$dev" -c white_balance_temperature="${WB:-4600}" -c exposure_dynamic_framerate=0
    v4l2-ctl -d "$dev" --list-ctrls | grep -E "white_balance|exposure_dynamic|auto_exposure" ;;
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
import os
if os.environ.get("FOLLOWER_PORT"):  # follower only (station bring-up: no leader / global camera in the profile)
    arm, pair = "right", {"follower": {"id": os.environ.get("FOLLOWER_ID", "wowrobo_follower"),
                                       "port": os.environ["FOLLOWER_PORT"]}}
else:
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
  d455)
    TRAY_ARG=(); [[ -e "$TRAYS" ]] && TRAY_ARG=(--trays "$TRAYS")
    "$PY" tools/finals_reach.py preview --serial "$D455" "${TRAY_ARG[@]}" ${POSE:+--pose "$POSE"} ${CORR:+--corr "$CORR"} "$@" ;;
  reach)
    need "$ROUTE"
    TRAY_ARG=(); [[ -e "$TRAYS" ]] && TRAY_ARG=(--trays "$TRAYS")
    "$PY" tools/finals_reach.py reach --serial "$D455" --route "$ROUTE" --config "$CONFIG" "${TRAY_ARG[@]}" \
        ${POSE:+--pose "$POSE"} ${CORR:+--corr "$CORR"} ${SPEED:+--speed "$SPEED"} "$@" ;;
  d455-capture)
    "$PY" tools/finals_calib.py capture --serial "$D455" --config "$CONFIG" --out "outputs/d455/capture-$DATE-$(date +%H%M)" "$@" ;;
  d455-fit)
    cap="${CAPTURE:-$(ls -d outputs/d455/capture-* | tail -1)}"
    MENAGERIE_DIR="${MENAGERIE_DIR:-$ROOT/outputs/mujoco_menagerie}" uv run -q --with opencv-python-headless \
        tools/finals_calib.py fit --capture "$cap" ${POSE:+--init "$POSE"} --out "outputs/d455/pose_$DATE.json" "$@"
    echo "use it: export POSE=outputs/d455/pose_$DATE.json" ;;
  joints)
    # Read-only joint readout for the joint-map check and touch points (writes nothing to the arm, torque stays off).
    # Each Enter saves the current reading with a typed label to outputs/<date>-joints/samples.jsonl; q quits.
    mkdir -p "outputs/$DATE-joints"
    "$PY" - "$CONFIG" "outputs/$DATE-joints/samples.jsonl" <<'PYEOF'
import json, sys, time
sys.path.insert(0, "src")
import numpy as np
from agentic_sim.config import load
from agentic_sim.kinematics import So101Chain
from agentic_sim.real_hardware import check_devices, load_profile, open_follower
from agentic_sim.real_joints import lerobot_to_shared
from agentic_sim.real_kinematics import ArmModel, find_calibration
import os
if os.environ.get("FOLLOWER_PORT"):  # follower only (station bring-up: no leader / global camera in the profile)
    arm, pair = "right", {"follower": {"id": os.environ.get("FOLLOWER_ID", "wowrobo_follower"),
                                       "port": os.environ["FOLLOWER_PORT"]}}
else:
    profile = load_profile(sys.argv[1])
    check_devices(profile)
    (arm, pair), = profile["arms"].items()
model = ArmModel.from_file(find_calibration(pair["follower"]["id"]), ".")  # calibration-file joint map (finals)
print("calibration offsets (deg):", " ".join(f"{v:+.2f}" for v in model.offsets_deg))
chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
chain.ee_offset = np.array(load("configs/teleop.yaml")["ee_offset_in_gripper_link"]["value"], float)
chain.base_pos = np.array([0.0, 0.0, -0.0301])
names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
with open_follower(arm, pair["follower"], managed=True, resume=True) as follower, open(sys.argv[2], "a") as out:
    if any(follower.torque_states().values()):
        sys.exit("torque is on: support the arm and run tools/lab_day.sh recover first")
    print("torque off. Move the arm by hand. Enter = save with a label, q + Enter = quit.")
    while True:
        r = follower.read()
        vals = [float(r[n]) for n in names]
        q, _ = lerobot_to_shared(vals)
        ee = chain.ee_pose_mount(np.clip(q, chain.lower, chain.upper))[:3, 3] * 1000
        cal = model.ee(r) * 1000
        print("LeRobot " + " ".join(f"{v:7.1f}" for v in vals) + "   EE(map) mm " + " ".join(f"{v:6.0f}" for v in ee)
              + "   EE(calib) mm " + " ".join(f"{v:6.0f}" for v in cal))
        label = input("label (Enter = next reading, text = save, q = quit): ").strip()
        if label == "q":
            break
        if label:
            out.write(json.dumps({"t": time.time(), "label": label, "lerobot": dict(zip(names, vals)),
                                  "ee_map_mm": [round(float(v), 1) for v in ee],
                                  "ee_calib_mm": [round(float(v), 1) for v in cal], "follower_id": pair["follower"]["id"]}, ensure_ascii=False) + "\n")
            out.flush()
            print("saved:", label)
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
