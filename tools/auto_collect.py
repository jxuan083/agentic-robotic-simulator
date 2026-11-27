#!/usr/bin/env python3
"""Scripted park rounds: the program grasps, records the same episode format as the human demonstrations, places,
and resets the trays itself (docs/real-dagger.md stages 2 and 5; the leader takeover is not wired yet: a failure stops
the loop for the operator).

  tools/lab_day.sh auto            # lerobot env; ROUNDS=<n> (default 5), SEED=<n>

Per episode (the follower parks in between, like the human rounds):
  1. D455 at park (aligned depth, agentic_sim.real_d455.detect, median of 6 frames): blocks in the pink tray inside the
     safe grasp zone; one picked at random.
  2. The episode recording opens; the fixed route takes the follower to the start pose (not recorded, as for people).
  3. Recorded, command_source "script", shaped like the operator's grasps (11/27, 12 demonstrations): one smooth
     motion (minimum jerk, no stop over the block) from the start pose through a point 2-4 cm back towards the start
     and 5.5-9 cm up, down to the grasp; the grasp point off the block centre by N(0, 5 mm) and the fingers off the
     block faces by N(0, 10 deg) (closing squares the block, as the operator's grasps do); gripper opened to 26-32 %
     on the way and closed to the operator's 13.5 %; 2.0-3.4 s start -> close, peak joint speed <= 85 deg/s; lift
     5.5-8.5 cm, hold (the grasp segment ACT learns) -> carry into the next beige slot (P1-P4), release, rise over
     the slot at 75 deg/s (the place segment; VLA learns the whole episode). The sampled style is in review.json.
  4. Not recorded: straight up, back to the start pose; the next scan, the landing check and the next choice are
     made there (the D455 sees both trays past the arm). Park only at the end and before the operator handles a
     failure.
  5. review.json as for the human rounds (start "start", D455 target from the opening frame, training range up to the
     lift); success = the gripper held after the lift and the D455 sees the block in the beige tray at park.
     Failures move to <session>-rejected/ and the loop waits for the operator.
When the pink tray is empty the program moves the beige blocks back into the pink tray at random free positions and
yaws inside the safe zone (not recorded; logged in <session>/reset.jsonl) and starts the next round.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from agentic_sim.real_place import (  # noqa: E402
    CARRY_Z, CLOSED_PCT, EDGE_M, GRASP_M, HELD_PCT, HOVER_M, OPEN_PCT, SLOTS_MM, Placer, beige_slots, camera_from_pose,
    hold_while, make_corr,
)

LIFT_M = 0.05            # lift of the unrecorded reset grasps
LIFT_HOLD_S = 0.5
RESET_MARGIN_M = 0.040   # reset positions this far inside the pink rim (wall + half a block + clearance)
RESET_GAP_M = 0.050      # centre distance between reset blocks (room for the open fingers)
MOVE_SPEED = 75.0        # deg/s for the unrecorded IK moves (reset) and the programmed place

# Recorded grasps shaped like the operator's (11/27 P0001-P0014 without P0006, 12 grasps; FK of the commands minus the
# sag correction against the D455 target): grasp point off the centre radial 1 +- 5 mm, tangential -1 +- 4 mm (all
# within 10 mm); fingers off the block faces sd 10 deg (max 26 deg, all held: closing squares the block); gripper
# opened to 26 % and closed to a 13.3-14.7 % command (held reading 16.7-17.7 %); start -> close 2.0-4.3 s, close ->
# lift 1.2-2.2 s; fastest joint peak 52-85 deg/s; 3 cm short of the block the hand is still ~8.5 cm above its centre;
# lift 6-8.6 cm.
STYLE_OFFSET_SD_MM, STYLE_OFFSET_MAX_MM = 5.0, 10.0
STYLE_YAW_SD_DEG, STYLE_YAW_MAX_DEG = 10.0, 20.0
STYLE_OPEN_PCT = (26.0, 32.0)
STYLE_CLOSE_PCT = 13.5        # command, like the operator; an empty grasp reads ~13.5, a held block 16.7-17.7
STYLE_HELD_PCT = 15.2         # gripper reading above this after the lift: a block is held
STYLE_GRASP_Z_MM = 3.0        # +- jitter of the grasp height
STYLE_PRE_BACK_MM = (20.0, 40.0)   # approach point: back towards the start pose ...
STYLE_PRE_UP_MM = (55.0, 90.0)     # ... and this far above the grasp point
STYLE_APPROACH_S = (2.0, 3.4)
STYLE_PEAK_DEG_S = 85.0
STYLE_CLOSE_S = (0.35, 0.6)
STYLE_LIFT_MM = (55.0, 85.0)
STYLE_LIFT_S = (0.7, 1.1)     # close -> lift of the operator: 1.2-2.2 s including the close
STYLE_HOLD_S = (0.3, 0.6)
STYLE_STEP_S = 1 / 15         # min-jerk samples (the driver interpolates at the control rate)
WRIST_TURN_DEG = 30.0         # wrist_roll travel from the start pose (operator's grasps <= 31 deg; 11/27 S0005 turned
                              # 89 deg squaring a block and adding the yaw error)


def low_z(placer, reading):
    """Grasp-centre height (sag-corrected) of a joint reading, m."""
    p = placer.model.ee(reading)
    return float((p - placer.corr_at(p))[2])


def min_jerk(t):
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def timed_path(points, seconds, step=STYLE_STEP_S):
    """Points along a piecewise-linear joint path (body joints + gripper) on a minimum-jerk clock over `seconds`:
    (points, segment seconds) for agentic_sim.real_route.Track; one smooth motion, no stop at the inner points."""
    keys = list(points[0])
    P = np.array([[p[k] for k in keys] for p in points], float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(seconds / step)))
    out = []
    for k in range(1, n + 1):
        a = min_jerk(k / n) * arc[-1]
        i = min(int(np.searchsorted(arc, a, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] < 1e-9 else (a - arc[i]) / seg[i]
        out.append({key: float(v) for key, v in zip(keys, P[i] + (P[i + 1] - P[i]) * f)})
    return out, [seconds / n] * n


def peak_seconds(points, peak=STYLE_PEAK_DEG_S):
    """Shortest minimum-jerk duration keeping every joint under `peak` deg/s (peak speed = 1.875 x mean)."""
    body = [k for k in points[0] if k != "gripper"]
    travel = max(sum(abs(b[k] - a[k]) for a, b in zip(points, points[1:])) for k in body)
    return 1.875 * travel / peak


def snapshot(read, cam, quads, frames=6):
    """Median D455 detection over a few aligned frames (positions; yaw by the 4-fold circular mean)."""
    from agentic_sim.real_d455 import detect
    seen = []
    for _ in range(frames + 3):
        bgr, depth = read()
        seen.append(detect(bgr, depth, cam, quads)[0])
    seen = seen[3:]
    out = []
    for b in seen[-1]:
        same = [min(s, key=lambda c: math.hypot(c["x"] - b["x"], c["y"] - b["y"])) for s in seen if s]
        same = [c for c in same if c["color"] == b["color"] and math.hypot(c["x"] - b["x"], c["y"] - b["y"]) < 0.01]
        m = dict(b)
        for k in ("x", "y", "z_top"):
            m[k] = float(np.median([c[k] for c in same]))
        m["yaw"] = float(np.angle(np.exp(4j * np.array([c["yaw"] for c in same])).mean()) / 4) % (math.pi / 2)
        out.append(m)
    return out


def look(serial, cam, quads):
    from agentic_sim.real_d455 import open_aligned
    read, _, stop = open_aligned(serial)
    try:
        return snapshot(read, cam, quads)
    finally:
        stop()


def tray_polygon(cam, quad, margin):
    """Tray rim corners (image quad) on the table plane in the mount frame, shrunk towards the centre by margin."""
    c = cam.on_plane(np.asarray(quad, float), 0.023)[:, :2]
    centre = c.mean(0)
    return np.array([p + (centre - p) / np.linalg.norm(centre - p) * margin * math.sqrt(2) for p in c])


def inside(poly, xy):
    x, y = xy
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def sample_free(rng, poly, zone, taken, feasible, tries=400):
    """A random (x, y, yaw) inside poly and the safe zone, RESET_GAP_M from every taken centre, that feasible() accepts."""
    from finals_reach import in_zone
    lo, hi = poly.min(0), poly.max(0)
    for _ in range(tries):
        xy = rng.uniform(lo, hi)
        if not inside(poly, xy) or not in_zone(zone, *xy):
            continue
        if any(math.hypot(*(xy - t)) < RESET_GAP_M for t in taken):
            continue
        yaw = rng.uniform(0, math.pi / 2)
        if feasible(xy, yaw):
            return xy, yaw
    return None


class Episodes:
    """Motion pieces of one scripted round on a Driver + Placer (no cameras: testable with the fake follower)."""

    def __init__(self, driver, placer, rng):
        self.driver, self.placer, self.rng = driver, placer, rng

    def _pose(self, xyz, yaw, phase, grip):
        return self.placer.pose_for(np.asarray(xyz, float), yaw, self.driver.target, phase, grip)

    def grasp(self, block, perturb=None):
        """Start pose (or anywhere above the trays) -> over the block [-> via an offset point] -> grasp -> close ->
        lift and hold. Returns (gripper reading after the lift, info) or (None, reason) when an IK fails before
        anything moved."""
        c = np.array([block["x"], block["y"], block["z_top"] - EDGE_M / 2])
        yaw = block["yaw"]
        hover = self._pose(c + [0, 0, HOVER_M], yaw, "hover", OPEN_PCT)
        low = self._pose(c + [0, 0, GRASP_M], yaw, "grasp", OPEN_PCT)
        if hover is None or low is None:
            return None, "IK 無解"
        # In from above (CARRY_Z over the block): a joint-space move from the start pose or from the other tray
        # must not dip into a rim or a neighbour; then straight down to the hover.
        above = self._pose([c[0], c[1], max(CARRY_Z, c[2] + HOVER_M + 0.03)], yaw, "above", OPEN_PCT)
        via = [above] if above is not None else []
        if perturb is not None:
            off = np.array([perturb[0], perturb[1], 0.0]) / 1000
            p = self._pose(c + off + [0, 0, HOVER_M], yaw + math.radians(perturb[2]), "hover", OPEN_PCT)
            if p is not None:
                via.append(p)
        self.placer.move(via + [hover], ("起姿", "方塊上方"))
        self.placer.move([low], ("方塊上方", "夾取高度"))
        self.placer.set_grip(CLOSED_PCT, "閉合")
        up = self._pose(c + [0, 0, GRASP_M + LIFT_M], yaw, "hover", CLOSED_PCT)
        if up is None:
            return float(self.driver.read()["gripper"]), "抬起 IK 無解"
        self.placer.move([up], ("夾取高度", "抬起"))
        held = self.hold(LIFT_HOLD_S)
        return held, None

    def style(self):
        """Random operator-like grasp parameters (constants STYLE_*), stored in review.json."""
        r = self.rng
        clip = lambda v, m: float(np.clip(v, -m, m))  # noqa: E731
        u = lambda lo_hi: float(r.uniform(*lo_hi))  # noqa: E731
        return {"offset_mm": [clip(r.normal(0, STYLE_OFFSET_SD_MM), STYLE_OFFSET_MAX_MM),
                              clip(r.normal(0, STYLE_OFFSET_SD_MM), STYLE_OFFSET_MAX_MM)],   # radial, tangential
                "yaw_err_deg": clip(r.normal(0, STYLE_YAW_SD_DEG), STYLE_YAW_MAX_DEG),
                "z_mm": float(r.uniform(-STYLE_GRASP_Z_MM, STYLE_GRASP_Z_MM)),
                "open_pct": u(STYLE_OPEN_PCT), "pre_back_mm": u(STYLE_PRE_BACK_MM), "pre_up_mm": u(STYLE_PRE_UP_MM),
                "approach_s": u(STYLE_APPROACH_S), "close_s": u(STYLE_CLOSE_S), "lift_mm": u(STYLE_LIFT_MM),
                "lift_s": u(STYLE_LIFT_S), "hold_s": u(STYLE_HOLD_S)}

    def _glide(self, points, seconds, label):
        """One smooth timed motion through points (no settle at the end); the driver's lag allowance as in
        Placer.move, the speed cap high enough that the minimum-jerk clock sets the pace."""
        from agentic_sim.real_route import LAG_PER_SPEED, Track, follow
        d = self.driver
        pts, secs = timed_path([dict(d.target), *points], seconds)
        saved = d.play_speed, d.lag_per_speed
        d.play_speed, d.lag_per_speed = 1.5 * STYLE_PEAK_DEG_S, LAG_PER_SPEED
        try:
            follow(d, Track([dict(d.target), *pts], secs), label, settle=False)
        finally:
            d.play_speed, d.lag_per_speed = saved

    def _grip_still(self, min_s=0.15, still_s=0.15, max_s=1.0):
        """Hold until the gripper reading stops (closed on the block, or at the command when empty)."""
        d = self.driver
        began, seen = d.clock(), []
        while True:
            t = d.clock()
            _, actual, _ = d.send(d.target)
            g = float(actual["gripper"])
            seen = [(ts, v) for ts, v in seen + [(t, g)] if t - ts <= still_s * 1e9]
            if (t - began >= min_s * 1e9 and t - seen[0][0] >= still_s * 0.9e9
                    and max(v for _, v in seen) - min(v for _, v in seen) < 0.6) or t - began >= max_s * 1e9:
                return g
            d.tick(t)

    def _finger_yaw(self, g, block_yaw, err, style):
        """Finger heading (rad, the 'block yaw' Placer.pose_for squares with) for `wanted`, with the wrist turned at
        most WRIST_TURN_DEG from the start pose like the operator; the rest stays as misalignment (closing squares
        the block). style["wrist_turn_deg"] / ["yaw_left_deg"] (fingers vs block faces) record what was used."""
        from agentic_sim.real_kinematics import wrap_quarter
        pl, start = self.placer, self.driver.target
        r = pl.model.radial(g[:2])
        want = wrap_quarter(block_yaw + err - r)                 # turn from radial fingers, +-45 deg

        def travel(turn):
            p = pl.pose_for(g, r + turn, start, "grasp", 30.0)
            return None if p is None else p["wrist_roll"] - start["wrist_roll"]
        f_want, f0 = travel(want), travel(0.0)
        turn = want
        if f_want is not None and f0 is not None and abs(f_want) > WRIST_TURN_DEG and abs(want) > 1e-3:
            slope = (f_want - f0) / math.degrees(want)               # deg of wrist_roll per deg of turn (~ +-1)
            if abs(slope) > 0.2:
                turn = math.radians((math.copysign(WRIST_TURN_DEG - 4.0, f_want) - f0) / slope)   # margin: the
                # approach and lift poses tilt differently and roll the wrist a few degrees more
                turn = float(np.clip(turn, min(0.0, want), max(0.0, want)))
        style["wrist_turn_deg"] = None if travel(turn) is None else round(travel(turn), 1)
        style["yaw_left_deg"] = round(math.degrees(wrap_quarter(block_yaw - r - turn)), 1)   # vs the block faces
        return r + turn

    def grasp_like_human(self, block, style):
        """Recorded grasp shaped like the operator's (see STYLE_*). Returns (gripper reading after the hold, None) or
        (None, reason) when an IK fails before anything moved."""
        d, pl = self.driver, self.placer
        c = np.array([block["x"], block["y"], block["z_top"] - EDGE_M / 2])
        rad = c[:2] / np.linalg.norm(c[:2])
        tan = np.array([-rad[1], rad[0]])
        off = (style["offset_mm"][0] * rad + style["offset_mm"][1] * tan) / 1000
        g = c + [off[0], off[1], GRASP_M + style["z_mm"] / 1000]
        yaw = self._finger_yaw(g, block["yaw"], math.radians(style["yaw_err_deg"]), style)
        here = pl.model.ee(d.target)
        here = here - pl.corr_at(here)
        back = here[:2] - g[:2]
        back = back / np.linalg.norm(back) if np.linalg.norm(back) > 1e-6 else -rad
        pre = g + [*(back * style["pre_back_mm"] / 1000), style["pre_up_mm"] / 1000]
        grip = style["open_pct"]
        seed, path = dict(d.target), []
        for s_ in (0.0, 0.35, 0.65, 0.85):
            p = pl.pose_for(pre + (g - pre) * s_, yaw, seed, "hover", grip)
            if p is None:
                return None, "接近路徑 IK 無解"
            path.append(p)
            seed = p
        low = pl.pose_for(g, yaw, seed, "grasp", grip)
        if low is None:
            return None, "夾取高度 IK 無解"
        lift_pts, seed = [], low
        top = g + [0, 0, style["lift_mm"] / 1000]
        for s_ in (0.4, 0.75, 1.0):
            p = pl.pose_for(np.array([g[0], g[1], low_z(pl, low) + (top[2] - low_z(pl, low)) * s_]), yaw, seed,
                            "hover", STYLE_CLOSE_PCT)
            if p is None:
                return None, "抬起 IK 無解"
            lift_pts.append(p)
            seed = p
        if not pl.path_ok(d.target, path[0]):      # from high up only: never dip into a rim or a neighbour
            above = pl.pose_for([pre[0], pre[1], max(CARRY_Z, pre[2] + 0.02)], yaw, d.target, "above", grip)
            if above is not None:
                path.insert(0, above)
        approach = [*path, low]
        self._glide(approach, max(style["approach_s"], peak_seconds([dict(d.target), *approach])), ("起姿", "夾取"))
        self._glide([dict(d.target) | {"gripper": STYLE_CLOSE_PCT}], style["close_s"], ("夾取", "閉合"))
        self._grip_still()
        self._glide(lift_pts, max(style["lift_s"], peak_seconds([dict(d.target), *lift_pts])), ("閉合", "抬起"))
        return self.hold(style["hold_s"]), None

    def hold(self, seconds):
        d = self.driver
        end = d.clock() + seconds * 1e9
        g = float(d.read()["gripper"])
        while d.clock() < end:
            began = d.clock()
            _, actual, _ = d.send(d.target)
            g = float(actual["gripper"])
            d.tick(began)
        return g

    def place(self, xy, yaw, label):
        return self.placer.carry_and_place(xy, yaw, yaw, label, CLOSED_PCT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "outputs/single.local.yaml"))
    ap.add_argument("--route", required=True)
    ap.add_argument("--trays", required=True)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--pose", default=str(ROOT / "outputs/d455/pose_261125.json"))
    ap.add_argument("--corr", default="21,-1.3,20")
    ap.add_argument("--session", required=True, help="episodes go to <session>/episodes/S<nnnn>")
    ap.add_argument("--rounds", type=int, default=5, help="rounds of 4 (each followed by a reset)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--follower-id", default="my_awesome_follower_arm")
    args = ap.parse_args(argv)
    return run(args)


def run(args):
    from agentic_sim.real_grasp import control_row, finalize, latest_global, lift_and_return, placed_block
    from agentic_sim.real_kinematics import ArmModel, find_calibration
    from agentic_sim.real_locate import locate, tray_homography
    from agentic_sim.real_route import Driver, Track, follow, load_route, park_and_release
    from finals_reach import safe_zone
    from real_route import Terminal, check_calibration, episode_capture, real_devices, run_dir, set_aside
    import cv2

    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)
    data, route = load_route(Path(args.route))
    park, channel, support = data["park"]["pose"], data["channel"]["command"], data["park_support"]
    quads = json.loads(Path(args.trays).read_text())["quads"]
    cam = camera_from_pose(args.pose)
    model = ArmModel.from_file(find_calibration(args.follower_id), ROOT)
    corr_at = make_corr(args.corr)
    zone = safe_zone(model, corr_at, args.follower_id, args.corr, ROOT / "outputs/d455")
    slots = beige_slots(cam, quads)
    pink = tray_polygon(cam, quads["pink"], RESET_MARGIN_M)
    H_pink = tray_homography(np.asarray(quads["pink"], np.float32), "pink")
    session = Path(args.session)
    (session / "episodes").mkdir(parents=True, exist_ok=True)
    index, reset_log = session / "index.jsonl", session / "reset.jsonl"
    from finals_reach import in_zone

    with Terminal() as term:
        term.say("自動收資料：程式夾取並錄製（起姿 → 抬起），放進米盤 P1-P4，粉盤空了自動把方塊放回粉盤。\n"
                 "開始前確認：follower 在停放位、扭力關；粉盤內 1-4 顆方塊都在橘色弧線內；米盤清空；工作區淨空。"
                 "Esc 隨時原地暫停。")
        if term.choose("y 開始（開扭力）／n 離開", {"y": "開始", "n": "離開"}) != "y":
            return 0
        with real_devices(args.config, "right", leader=False) as (_, follower, refs, hz):
            check_calibration(data, refs, leader=False)
            driver = Driver(follower, term, run_dir(Path(args.route), "auto"), hz=hz, drive_leader=False)
            driver.gripper_speed = 2 * MOVE_SPEED            # routes keep PLAY_SPEED (the fold area), like the demos
            placer = Placer(driver, model, corr_at, speed=MOVE_SPEED)
            eps = Episodes(driver, placer, rng)
            driver.arm_at(park)
            slot_k, done_rounds = 0, 0

            # The arm stays at the start pose between episodes: the D455 sees both trays from there (11/27 S0002,
            # the gripper in front of the beige rim), so the scan, the landing check and the next choice need no
            # trip through the fold area. Park only at the end and before the operator reaches into the trays.
            where = {"at": "park"}

            def to_start():
                if where["at"] == "park":
                    follow(driver, route, ("停放位", "起姿"))
                where["at"] = "start"

            def back_to_start():
                if where["at"] == "park":
                    return to_start()
                back = lift_and_return(driver.target, channel, args.follower_id)
                if back is None:        # no straight way up from here: the old way, through park
                    return home() or to_start()
                follow(driver, back, ("目前位置", "起姿"))
                driver.target = driver.target | {"gripper": channel["gripper"]}
                where["at"] = "start"

            def home():
                if where["at"] == "park":
                    return
                back = lift_and_return(driver.target, channel, args.follower_id)
                if back is not None:
                    follow(driver, back, ("目前位置", "起姿"))
                driver.target = driver.target | {"gripper": channel["gripper"]}
                follow(driver, route.reverse(), ("起姿", "停放位"))
                where["at"] = "park"

            def feasible(xy, yaw):
                c = np.array([xy[0], xy[1], EDGE_M / 2])
                return all(placer.pose_for(c + [0, 0, h], yaw, channel, ph, OPEN_PCT) is not None
                           for h, ph in ((HOVER_M, "hover"), (GRASP_M, "grasp")))

            def reset(beige_blocks, occupied=()):
                """Beige blocks back into the pink tray at random free spots (not recorded); occupied: pink blocks
                already there."""
                nonlocal slot_k
                taken = [np.array([b["x"], b["y"]]) for b in occupied]
                to_start()
                for b in beige_blocks:
                    spot = sample_free(rng, pink, zone, taken, feasible)
                    if spot is None:
                        term.say("粉盤找不到可放的位置；停在起姿。")
                        return False
                    xy, yaw = spot
                    where["at"] = "pink"
                    held, err = eps.grasp(b)
                    if err or held < HELD_PCT:
                        term.say(f"還原：{b['color']} 沒夾起（{err or f'{held:.1f}%'}）。")
                        placer.set_grip(OPEN_PCT, "張開")
                        return False
                    err = placer.carry_and_place(xy, yaw, b["yaw"], "粉盤", CLOSED_PCT)
                    if err:
                        term.say(f"還原：{err}")
                        return False
                    where["at"] = "pink"
                    taken.append(np.asarray(xy))
                    with reset_log.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"t": time.time(), "color": b["color"], "from": [b["x"], b["y"]],
                                             "to": [float(xy[0]), float(xy[1])], "yaw_deg": math.degrees(yaw)}) + "\n")
                back_to_start()
                slot_k = 0
                return True

            try:
                while done_rounds < args.rounds:
                    blocks = None
                    for attempt in range(2):     # 11/27: the D455 colour stream stopped after S0001 and the tool exited
                        try:
                            blocks = look(args.serial, cam, quads)
                            break
                        except RuntimeError as error:
                            term.say(f"D455 讀取失敗（{error}）" + ("，再試一次。" if not attempt else "。"))
                            time.sleep(2.0)
                    if blocks is None:
                        term.say("D455 讀不到：回停放位、解扭力後結束（重插 D455 或重開後再跑）。")
                        break
                    source = [b for b in blocks if b["tray"] == "pink" and in_zone(zone, b["x"], b["y"])]
                    if not source:
                        beige = [b for b in blocks if b["tray"] == "beige"]
                        done_rounds += 1 if slot_k else 0
                        if not beige or done_rounds >= args.rounds:
                            break
                        term.say(f"粉盤空了：把米盤 {len(beige)} 顆放回粉盤（不錄）。")
                        if not reset(beige):
                            home()                          # folded away before the operator reaches into the trays
                            if term.choose("還原失敗。手動整理好方塊後 c 繼續／q 結束", {"c": "繼續", "q": "結束"}) == "q":
                                break
                        continue
                    if slot_k >= len(slots):
                        term.say("米盤 4 格已滿，但粉盤還有方塊：先還原。")
                        reset([b for b in blocks if b["tray"] == "beige"], [b for b in blocks if b["tray"] == "pink"])
                        continue
                    target = pyrng.choice(source)
                    style = eps.style()
                    number = 1 + len(list((session / "episodes").glob("S*")))
                    root = session / "episodes" / f"S{number:04}"
                    u, v = target["pixel"]
                    local = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), H_pink)[0, 0]
                    term.say(f"S{number:04}：{target['color']} x {target['x'] * 1000:.0f} y {target['y'] * 1000:.0f} mm "
                             f"yaw {math.degrees(target['yaw']):.0f}°；偏心 {np.round(style['offset_mm'], 0).tolist()} mm、"
                             f"偏角 {style['yaw_err_deg']:.0f}°、{style['approach_s']:.1f} s")
                    with episode_capture(root, args.config, "right", refs, hz, data["layout_id"],
                                         tool="tools/auto_collect.py", start_pose="start", mapping="script") as (writer, check):
                        opening = latest_global(writer)
                        before = locate(opening, quads)["blocks"] if opening is not None else []
                        to_start()
                        rows, camera = [0], {}

                        def on_send(sent, actual):
                            rows[0] += 1
                            if rows[0] % 15 == 0 and not camera:
                                try:
                                    check()
                                except RuntimeError as error:   # never stop mid-move: finish, then fail the episode
                                    camera["error"] = str(error)
                            writer.add_control(control_row(writer, sent, actual) | {"command_source": "script",
                                                                                     "task_is_policy": 1})
                        driver.on_send = on_send
                        place_err = None
                        try:
                            for _ in range(3):              # an IK miss fails before anything moves: new style
                                held, err = eps.grasp_like_human(target, style)
                                if held is not None:
                                    break
                                style = eps.style()
                            demo_frames = writer.count          # start -> lift (the grasp segment)
                            ok_hold = held is not None and held >= STYLE_HELD_PCT
                            if ok_hold:                         # recorded on to the rise over the slot, like the demos
                                xy, yaw = slots[slot_k]
                                current = dict(driver.target)
                                plan = hold_while(driver, lambda: placer.plan_place(xy, yaw, yaw, CLOSED_PCT, current))
                                if isinstance(plan, str):
                                    place_err = plan
                                else:
                                    placer.execute_place(plan, f"米盤 P{slot_k + 1}")
                        finally:
                            driver.on_send = None
                        if ok_hold:
                            if place_err:   # still holding: never open on the way home
                                term.choose(f"程式放置沒有執行（{place_err}）。扶住夾爪裡的方塊後按 o 張爪",
                                            {"o": "張爪"})
                                placer.set_grip(OPEN_PCT, "張開")
                        else:
                            placer.set_grip(OPEN_PCT, "張開")
                        where["at"] = "slot"
                        back_to_start()
                        final = latest_global(writer)       # at the start pose: the D455 sees the landing
                        landed = placed_block(before, locate(final, quads)["blocks"], target["color"], "beige",
                                              {"index": slot_k + 1, "xy": SLOTS_MM[slot_k]}) if final is not None else None
                    place_err = place_err or (camera and f"相機中斷（{camera['error']}）")
                    success = bool(ok_hold and not place_err and landed is not None)
                    slot = {"index": slot_k + 1, "xy": json.loads((ROOT / "configs/real_slots.json").read_text())["beige"][slot_k]}
                    extra = {"script": {"style": style, "gripper_after_lift": held, "error": err or place_err,
                                        "target_mount_m": [target["x"], target["y"], target["z_top"]],
                                        "target_yaw_deg": math.degrees(target["yaw"]), "corr": args.corr,
                                        "seed": args.seed}}
                    report, review = finalize(root, color=target["color"], destination="beige", success=success,
                                              quads=quads, target_pose=(float(local[0]), float(local[1])), slot=slot,
                                              landed=landed, extra=extra, start="start", mode="script")
                    line = {"episode": root.name, "color": target["color"], "success": success, "ok": report["ok"],
                            "demo_frames": demo_frames, "training_ranges": review.get("training_ranges"),
                            "style": style, "reasons": report["review_reasons"]}
                    if not success:
                        home()                              # folded away before the operator reaches into the trays
                        moved = set_aside(root, session, "failed", line)
                        term.say(f"失敗（{err or place_err or ('沒夾到' if not ok_hold else 'D455 沒看到落點')}），"
                                 f"原檔 {moved.name}。")
                        if term.choose("把掉落的方塊放回粉盤弧線內、米盤該格清空後 c 繼續／q 結束",
                                       {"c": "繼續", "q": "結束"}) == "q":
                            break
                        continue
                    with index.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                    slot_k += 1
                    term.say(f"{root.name} 成功：{demo_frames} 幀，訓練段 {review.get('training_ranges')}。")
            except BaseException as error:
                # Any stop (an error, the D455, the bus): home by the fixed route first, then release.
                driver.on_send = None
                term.say(f"自動收集中止（{type(error).__name__}: {error}）：先沿路線回停放位再解扭力。")
                try:
                    home()
                except BaseException as again:  # noqa: BLE001
                    term.say(f"回停放位沒有完成（{again}）：follower 原地保持，之後用 tools/lab_day.sh recover。")
                    raise error from again
                park_and_release(driver, Track([dict(driver.target), park], [0.0]), park, support)
                raise
            home()                                      # the fixed route through the fold area, never straight
            park_and_release(driver, Track([dict(driver.target), park], [0.0]), park, support)
    return 0


if __name__ == "__main__":
    sys.exit(main())
