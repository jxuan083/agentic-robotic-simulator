#!/usr/bin/env python3
"""Finals D455 localisation and geometric reach (strategy 1): live block colour / mount-frame mm, arm over a block.

  tools/lab_day.sh d455            # preview only: blocks with colour, x / y (mm, mount frame), yaw, wrist turn
  tools/lab_day.sh reach           # preview + arm: park -> start pose -> over the chosen block (IK) -> back

Mount frame of the right follower: +X forward, +Y left, +Z up, origin on the table below the base. The D455 pose
comes from --pose (tools/finals_calib.py, arm depth + table plane), the joint map from the follower's own LeRobot
calibration file (agentic_sim.real_kinematics). The real arm sags a little under gravity, so the reach adds a
correction --corr (radial mm, rotation about the base deg, z mm: command minus target; 11/26 aim checks) and the operator can nudge the hover;
k logs each aimed block to outputs/d455/aim_checks.jsonl, the data for fitting the correction.

Keys (terminal; the preview window only shows):
  1-9  choose a block           g  go over it (hover)         w/s a/d r/f  nudge +x/-x +y/-y +z/-z by 5 mm
  l    lower to grasp height     c  close the gripper          o  open the gripper        u  up to hover
  k    log the aim check         b  back to the start pose     q  back to park, release torque, quit
  p    holding a block over its tray: carry it into the next beige slot, release, rise
  a    automatic pick and place of the chosen block (g, l, c, u, grip check, p, b); stops on a miss
  Esc  pause (route menu)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from agentic_sim.real_d455 import EDGE_M, Camera, detect, draw, open_aligned  # noqa: E402
from agentic_sim.real_kinematics import ArmModel, find_calibration  # noqa: E402

HOVER_M = 0.045      # grasp centre above the block centre while hovering (fingertips ~3 cm over the top)
TILT = {"above": 30.0, "hover": 15.0, "grasp": 8.0, "place": 25.0}   # releasing may tilt: far slots need ~20 deg   # max gripper tilt from vertical (deg) per phase
FAR_TILT = 30        # fallback tilt when the phase limit has no IK solution
GRASP_M = 0.015      # grasp centre above the block centre when lowered (11/26 aims: command top + 20..30 mm grasps)
SAFE_Z = 0.05        # lowest grasp-centre height along a move between start pose and hover
OPEN_PCT, CLOSED_PCT = 40.0, 8.0
HELD_PCT = 12.0      # gripper reading above this after closing: a block is between the jaws (held ~16.5 %)
PLACE_Z = 0.026      # grasp centre (= block centre) height when releasing in the beige tray (bottom ~6 mm above floor)
# Beige slots, tray-local mm (x from the near inner wall, y towards +Y): the 2 x 2 corner nearest the base and the pink
# tray, 55 mm apart, nearest first (11/26: the old row's first slot sat 35 mm from the far right rim, r 311 mm, and the
# block hit the rim). Mount ~ (210,-65) (210,-120) (265,-65) (265,-120) mm with the 11/25 trays.
SLOTS_MM = [[50, 190], [50, 132], [108, 190], [108, 132]]
CARRY_Z = 0.11       # grasp-centre height while carrying a block between trays (block bottom ~6 cm over the rims)
NUDGE_M = 0.005


class Live:
    """Camera thread: newest frame, detections and preview image."""

    def __init__(self, serial, pose, quads, arm_model=None):
        R, t = pose["mount_from_color"]["R"], pose["mount_from_color"]["t"]
        self.read, intr, self.stop_cam = open_aligned(serial)
        self.cam = Camera(intr, R, t)
        self.quads, self.model = quads, arm_model
        self.blocks, self.rejected, self.image, self.info, self.raw = [], [], None, {}, None
        self.selected, self.arm_xyz, self.lines = None, None, []
        self.slots, self.next_slot = [], 0      # beige placement squares drawn on the preview
        self.zone = None                        # safe grasp zone (safe_zone): blocks outside are marked "out"
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                bgr, depth = self.read()
            except RuntimeError:
                continue
            blocks, rejected, info = detect(bgr, depth, self.cam, self.quads)
            self.info = info
            if self.model is not None:
                for b in blocks:
                    b["turn"] = self.model.wrist_turn((b["x"], b["y"]), b["yaw"])
            for b in blocks:
                b["reach"] = in_zone(self.zone, b["x"], b["y"])
            lines = self.lines + (["! something over the trays: hidden / shifted blocks"] if info.get("occluded") else [])
            img = draw(bgr, self.cam, blocks, selected=self.selected, quads=self.quads, rejected=rejected,
                       arm=self.arm_xyz, lines=lines, slots=self.slots, next_slot=self.next_slot, zone=self.zone)
            with self.lock:
                self.blocks, self.rejected, self.image, self.raw = blocks, rejected, img, (bgr, depth)

    def snapshot(self, frames=5):
        """Median of the next frames' detections of each block (by nearest position)."""
        seen = []
        for _ in range(frames):
            time.sleep(0.12)
            with self.lock:
                seen.append([dict(b) for b in self.blocks])
        ref = seen[-1]
        out = []
        for b in ref:
            same = [min(s, key=lambda c: math.hypot(c["x"] - b["x"], c["y"] - b["y"])) for s in seen if s]
            same = [c for c in same if c["color"] == b["color"] and math.hypot(c["x"] - b["x"], c["y"] - b["y"]) < 0.01]
            m = dict(b)
            for k in ("x", "y", "z_top"):
                m[k] = float(np.median([c[k] for c in same]))
            z = np.exp(4j * np.array([c["yaw"] for c in same])).mean()   # yaw repeats every 90 deg
            m["yaw"] = float(np.angle(z) / 4 % (math.pi / 2))
            out.append(m)
        return out

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.stop_cam()


class Window:
    def __init__(self, title):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.Image, self.ImageTk = Image, ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.photo = None
        self.next = 0.0

    def show(self, bgr, force=False):
        now = time.monotonic()
        if bgr is None or (not force and now < self.next):
            return
        self.next = now + 0.15
        self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(bgr[..., ::-1]))
        self.label.configure(image=self.photo)
        self.root.update()


def make_corr(spec):
    """--corr "radial mm, rotation deg, z mm" -> corr_at(xyz): command minus target (gravity sag), mount m."""
    radial_c, tangent_c, z_c = (float(c) for c in spec.split(","))

    def corr_at(xyz):
        x, y = float(xyz[0]), float(xyz[1])
        r = math.hypot(x, y) or 1.0
        u, v = np.array([x, y]) / r, np.array([-y, x]) / r
        xy = u * radial_c / 1000 + v * r * math.radians(tangent_c)
        return np.array([xy[0], xy[1], z_c / 1000])
    return corr_at


ZONE_PHI = list(range(-90, 91, 10))   # directions from the base (deg, 0 = +X, + = towards +Y)
ZONE_MARGIN = 0.015                    # inside the IK limit
ZONE_MIN_R = 0.12                      # nearer than this the vertical grasp folds into the base


def safe_zone(model, corr_at, follower_id, corr_spec, out_dir):
    """Safe grasp zone: per direction the farthest block centre whose hover (top + 45 mm) and grasp both solve
    with <= FAR_TILT tilt and whose carry can start at 8 cm, minus ZONE_MARGIN. 11/26: a yellow at r 340 mm (36 deg)
    grasped but could not be carried (limit there ~345 mm). Depends on the calibration file and --corr only, so it is
    cached in out_dir/safe_zone_<follower>.json (~10 s to compute)."""
    key = {"offsets": [round(float(o), 3) for o in model.offsets_deg], "corr": corr_spec, "phi": ZONE_PHI,
           "margin": ZONE_MARGIN, "version": 1}
    path = Path(out_dir) / f"safe_zone_{follower_id}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("key") == key:
            return cached
    seed = {"shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 70.0, "wrist_roll": 0.0}
    centre = EDGE_M / 2

    def ok(x, y):
        sd = dict(seed, shoulder_pan=-math.degrees(math.atan2(y, x)))
        for z in (centre + HOVER_M, centre + GRASP_M, 0.08):
            xyz = np.array([x, y, z])
            if model.hover(xyz + corr_at(xyz), 0.0, sd, FAR_TILT)[0] is None:
                return False
        return True
    r_max = []
    for phi in ZONE_PHI:
        c, s_ = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        lo, hi = 0.15, 0.45
        for _ in range(10):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ok(mid * c, mid * s_) else (lo, mid)
        r_max.append(round(lo - ZONE_MARGIN, 4))
    zone = {"key": key, "phi_deg": ZONE_PHI, "r_safe_m": r_max, "r_min_m": ZONE_MIN_R,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zone, indent=1))
    return zone


def in_zone(zone, x, y):
    if not zone:
        return True
    r, phi = math.hypot(x, y), math.degrees(math.atan2(y, x))
    return zone["r_min_m"] <= r <= float(np.interp(phi, zone["phi_deg"], zone["r_safe_m"]))


def beige_slots(cam, quads):
    """Beige slots in the mount frame and the tray's x axis direction (rad), from the tray corners."""
    import cv2
    from agentic_sim.real_locate import tray_homography
    quad = np.asarray(quads["beige"], np.float32)
    Hinv = np.linalg.inv(tray_homography(quad, "beige"))
    pts = cv2.perspectiveTransform(np.asarray(SLOTS_MM, np.float32).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    xy = cam.on_plane(pts, 0.023)[:, :2]
    c = cam.on_plane(quad, 0.023)[:, :2]
    yaw = math.atan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0])
    return [(p, yaw) for p in xy]


def load_pose(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preview(args):
    from real_route import Terminal
    profile_quads = json.loads(Path(args.trays).read_text())["quads"] if args.trays and Path(args.trays).exists() else None
    live = Live(args.serial, load_pose(args.pose), profile_quads,
                ArmModel.from_file(find_calibration(args.follower_id), ROOT))
    if profile_quads and "beige" in profile_quads:
        live.slots = beige_slots(live.cam, profile_quads)
    print("安全夾取區（第一次約 10 秒）…", flush=True)
    live.zone = safe_zone(live.model, make_corr(args.corr), args.follower_id, args.corr, args.out)
    win = Window("D455 finals locate")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    live.lines = ["preview: s save snapshot, q quit"]
    try:
        with Terminal() as term:
            term.say("D455 定位預覽：s 存一筆（影像與座標）／q 離開")
            while True:
                with live.lock:
                    img = live.image
                win.show(img)
                key = term.key(0.05)
                if key == "q":
                    break
                if key == "s":
                    blocks = live.snapshot()
                    stamp = time.strftime("%H%M%S")
                    import cv2
                    cv2.imwrite(str(out / f"locate-{stamp}.png"), img)
                    with live.lock:
                        raw, rejected = live.raw, list(live.rejected)
                    if raw is not None:   # raw aligned frame: re-run detection offline when tuning
                        np.savez_compressed(out / f"locate-{stamp}.npz", bgr=raw[0], depth=raw[1].astype(np.float32))
                    (out / f"locate-{stamp}.json").write_text(json.dumps(
                        {"blocks": blocks, "rejected": rejected, "table_z_mm": live.info.get("table_z_mm"),
                         "over_trays_px": live.info.get("over_trays_px"), "occluded": live.info.get("occluded"),
                         "pose": args.pose, "frame": "mount, m"}, indent=1, ensure_ascii=False))
                    for k, b in enumerate(blocks, 1):
                        term.say(f"{k} {b['color']:11s} x {b['x'] * 1000:6.1f}  y {b['y'] * 1000:6.1f}  高 "
                                 f"{b['height_mm']:4.1f} mm  yaw {math.degrees(b['yaw']):4.1f}°  "
                                 f"轉腕 {math.degrees(b.get('turn', 0)):+5.1f}°  {b['tray'] or ''}  "
                                 f"HSV {b['hsv']} 色距 {b['color_distance']}")
                    term.say(f"桌面 z {live.info.get('table_z_mm')} mm；被排除：{[r['why'] for r in live.rejected]}")
                    term.say(f"已存 {out}/locate-{stamp}.json")
    finally:
        live.close()
    return 0


def reach(args):
    from agentic_sim.real_route import Paused, Track, follow, park_and_release, load_route  # noqa: F401
    from agentic_sim.real_route import Driver
    from real_route import Terminal, real_devices, run_dir, check_calibration

    data, track = load_route(Path(args.route))
    park, start, support = data["park"]["pose"], data["channel"]["command"], data["park_support"]
    model = ArmModel.from_file(find_calibration(args.follower_id), ROOT)
    quads = json.loads(Path(args.trays).read_text())["quads"] if args.trays and Path(args.trays).exists() else None
    corr_at = make_corr(args.corr)
    print("安全夾取區（第一次約 10 秒）…", flush=True)
    zone = safe_zone(model, corr_at, args.follower_id, args.corr, args.out)
    live = Live(args.serial, load_pose(args.pose), quads, model)
    live.zone = zone
    win = Window("D455 finals reach")
    log_path = Path(args.out) / "aim_checks.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"block": None, "nudge": np.zeros(3), "at": "park", "aim": None, "grip": start["gripper"], "slot": 0}
    slots = beige_slots(live.cam, quads) if quads and "beige" in quads else []
    live.slots = slots

    def refresh(*_):
        with live.lock:
            img = live.image
        win.show(img)

    def pose_for(xyz, block_yaw, seed, phase="hover"):
        for tilt in (TILT[phase], FAR_TILT):   # far blocks (r > ~0.30 m) need the gripper tilted more
            reading, info = model.hover(xyz + corr_at(xyz), block_yaw, seed, tilt)
            if reading is not None:
                break
        if reading is None:
            return None
        return reading | {"gripper": state["grip"]}

    def path_ok(a, b):
        for s in np.linspace(0, 1, 15):
            mid = {j: a[j] + (b[j] - a[j]) * s for j in a}
            if model.ee(mid)[2] < SAFE_Z - 1e-3:
                return False
        return True

    def move(points, label):
        follow(driver, Track([dict(driver.target), *points], [0.0] * len(points)), (label[0], label[1]))
        live.arm_xyz = model.ee(driver.target)

    def block_z(h):
        b = state["aim"]
        return np.array([b["x"], b["y"], b["z_top"] - EDGE_M / 2 + h])

    def go_hover():
        b = state["aim"]
        target = block_z(HOVER_M) + state["nudge"]
        state["grip"] = OPEN_PCT
        above = pose_for(target + [0, 0, 0.05], b["yaw"], driver.target, "above")
        hover = pose_for(target, b["yaw"], above or driver.target)
        if hover is None:
            return "IK 無解（超出可達範圍，夾爪要傾斜超過 15°），不移動。"
        via = [above] if above is not None and state["at"] == "start" else []
        if state["at"] == "start" and not path_ok(driver.target, (via or [hover])[0]):
            return "起姿到方塊上方的路徑太低，不移動。"
        move(via + [hover], ("目前位置", "方塊上方"))
        state["at"] = "hover"
        return None

    def go_level(h, phase, at, label):
        p = pose_for(block_z(h) + state["nudge"] * [1, 1, 0], state["aim"]["yaw"], driver.target, phase)
        if p is None:
            return "IK 無解，不移動。"
        move([p], label)
        state["at"] = at
        return None

    def set_grip(pct, label):
        state["grip"] = pct
        move([dict(driver.target) | {"gripper": pct}], ("夾爪", label))
        driver.settle(0.6)
        return float(driver.read()["gripper"])

    def place():
        """Holding a block at hover: carry it into the next free beige slot, release, rise over the slot."""
        if not slots:
            return "沒有米盤盤角（--trays），不能放。"
        if state["slot"] >= len(slots):
            return "米盤 4 格都放過了；清空後重開 reach。"
        xy, tray_yaw = slots[state["slot"]]
        top = np.array([xy[0], xy[1], PLACE_Z + HOVER_M])
        # Straight up to the carry height, across at that height (Cartesian waypoints: a joint-space move between
        # the trays dips and the block's bottom caught the beige rim), then down into the slot.
        here = model.ee(driver.target)
        here = here - corr_at(here)
        held_yaw = state["aim"]["yaw"] if state["aim"] is not None else tray_yaw
        path, seed = [], driver.target
        for k, s_ in enumerate(np.linspace(0, 1, 6)):
            p_xy = here[:2] + (xy - here[:2]) * s_
            p = None
            for z in (CARRY_Z, 0.095, 0.08, max(0.065, here[2])):   # far blocks: start lower, rise as the path comes in
                p = pose_for(np.array([p_xy[0], p_xy[1], z]), tray_yaw if k == 5 else held_yaw, seed, "above")
                if p is not None:
                    break
            if p is None:
                return f"搬運路徑第 {k + 1} 點 IK 無解，不移動。"
            path.append(p)
            seed = p
        over = pose_for(top, tray_yaw, seed, "place")
        low = pose_for(np.array([xy[0], xy[1], PLACE_Z]), tray_yaw, over or seed, "place")
        if over is None or low is None:
            return "格位 IK 無解，不移動。"
        move(path + [over, low], ("方塊上方", f"米盤第 {state['slot'] + 1} 格"))
        set_grip(OPEN_PCT, "張開")
        move([over | {"gripper": OPEN_PCT}], ("格位", "格位上方"))
        state["slot"] += 1
        live.next_slot = state["slot"] if state["slot"] < len(slots) else None
        state["at"] = "slot"
        return None

    try:
        with Terminal() as term:
            term.say("決賽定位＋手臂對準方塊。確認 follower 在停放位、扭力關、夾爪空、工作區淨空，有人隨時可按 Esc。"
                     f"\n下垂補償 {args.corr}（徑向 mm、繞基座 °、z mm；指令減目標）。")
            if term.choose("y 開始（開扭力，走固定路線到起姿）／n 離開", {"y": "開始", "n": "離開"}) != "y":
                return 0
            with real_devices(args.config, "right", leader=False) as (_, follower, refs, hz):
                check_calibration(data, refs, leader=False)
                driver = Driver(follower, term, run_dir(Path(args.route), "reach"), hz=hz, drive_leader=False)
                driver.on_send = refresh
                driver.arm_at(park)
                follow(driver, track, ("停放位", "起姿"))
                state["at"] = "start"
                live.arm_xyz = model.ee(driver.target)
                term.say(__doc__.split("Keys (terminal; the preview window only shows):")[1].split('"""')[0])
                while True:
                    blocks = live.blocks
                    live.lines = [f"at {state['at']}  block {state['block'] or '-'}  nudge mm "
                                  f"{np.round(state['nudge'] * 1000).astype(int).tolist()}  corr {args.corr}"]
                    refresh()
                    key = term.key(0.05)
                    if key is None:
                        continue
                    if key.isdigit() and key != "0":
                        k = int(key)
                        if k > len(blocks):
                            term.say(f"沒有第 {k} 顆")
                            continue
                        snap = live.snapshot()
                        if not in_zone(zone, blocks[k - 1]["x"], blocks[k - 1]["y"]):
                            term.say(f"第 {k} 顆在安全夾取區外（可能 IK 無解或搬不走）；a 不會自動做，g 可手動試")
                        if live.info.get("occluded"):
                            term.say("注意：托盤上方有東西（手或手臂），方塊位置可能偏移；移開後再選一次")
                        b = min(snap, key=lambda c: math.hypot(c["x"] - blocks[k - 1]["x"], c["y"] - blocks[k - 1]["y"]))
                        state.update(block=k, aim=b, nudge=np.zeros(3))
                        live.selected = k
                        term.say(f"選 {k}：{b['color']} x {b['x'] * 1000:.1f} y {b['y'] * 1000:.1f} mm，yaw "
                                 f"{math.degrees(b['yaw']):.1f}°，轉腕 {math.degrees(model.wrist_turn((b['x'], b['y']), b['yaw'])):+.1f}°。g 過去")
                    elif key == "g" and state["aim"] is not None and state["at"] in ("start", "hover"):
                        err = go_hover()
                        term.say(err or "已在方塊上方。看夾爪中心是否對準方塊中心；w/s a/d r/f 微調，k 記錄，l 下降。")
                    elif key in "wsadrf" and state["at"] in ("hover", "low") and state["aim"] is not None:
                        step = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0),
                                "r": (0, 0, 1), "f": (0, 0, -1)}[key]
                        state["nudge"] = state["nudge"] + NUDGE_M * np.array(step)
                        b = state["aim"]
                        base = b["z_top"] - EDGE_M / 2 + (HOVER_M if state["at"] == "hover" else GRASP_M)
                        p = pose_for(np.array([b["x"], b["y"], base]) + state["nudge"], b["yaw"], driver.target,
                                     "hover" if state["at"] == "hover" else "grasp")
                        if p is None:
                            term.say("IK 無解，不移動。")
                            continue
                        move([p], ("目前位置", "微調"))
                    elif key == "k" and state["aim"] is not None and state["at"] in ("hover", "low"):
                        rb = driver.read()
                        row = {"t": time.time(), "at": state["at"], "block": state["aim"], "nudge_m": state["nudge"].tolist(),
                               "corr_m": corr_at([b["x"], b["y"], 0]).tolist(), "corr": args.corr, "command": driver.target, "readback": rb,
                               "ee_command_m": model.ee(driver.target).tolist(), "ee_readback_m": model.ee(rb).tolist(),
                               "pose_file": args.pose, "route": args.route}
                        with log_path.open("a") as fh:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        term.say(f"已記錄：微調 {np.round(state['nudge'] * 1000).astype(int).tolist()} mm → {log_path}")
                    elif key == "l" and state["at"] == "hover" and state["aim"] is not None:
                        term.say(go_level(GRASP_M, "grasp", "low", ("方塊上方", "夾取高度")) or "已在夾取高度。c 閉爪")
                    elif key in "co" and state["at"] in ("hover", "low", "slot"):
                        got = set_grip(CLOSED_PCT if key == "c" else OPEN_PCT, "閉合" if key == "c" else "張開")
                        if key == "c":
                            term.say(f"夾爪讀數 {got:.1f}%：" + ("有夾到" if got >= HELD_PCT else "像是空爪"))
                    elif key == "u" and state["at"] == "low":
                        term.say(go_level(HOVER_M, "hover", "hover", ("夾取高度", "方塊上方")) or "已升回方塊上方。p 放進米盤")
                    elif key == "p" and state["at"] == "hover" and state["grip"] == CLOSED_PCT:
                        err = place()
                        term.say(err or "已放進米盤，在格位上方。b 回起姿。")
                    elif key == "a" and state["at"] == "start" and state["aim"] is not None:
                        if not in_zone(zone, state["aim"]["x"], state["aim"]["y"]):
                            term.say("這顆在安全夾取區外，不自動夾。換一顆，或用 g / l / c 手動試。")
                            continue
                        err = go_hover() or go_level(GRASP_M, "grasp", "low", ("方塊上方", "夾取高度"))
                        if err is None:
                            held = set_grip(CLOSED_PCT, "閉合")
                            err = go_level(HOVER_M, "hover", "hover", ("夾取高度", "方塊上方"))
                            held = float(driver.read()["gripper"])
                            if err is None and held < HELD_PCT:
                                set_grip(OPEN_PCT, "張開")
                                err = f"沒夾到（夾爪讀數 {held:.1f}% < {HELD_PCT:.0f}%），停在方塊上方。"
                        if err is None:
                            err = place()
                        if err is None:
                            term.say(f"自動夾放完成（夾爪讀數 {held:.1f}%）。b 回起姿，或選下一顆。")
                        else:
                            term.say(err)
                    elif key in "bq":
                        if state["at"] != "start":
                            if state["at"] == "low":
                                term.say("先按 u 升到方塊上方。")
                                continue
                            up = dict(driver.target)
                            ee = model.ee(up)
                            yaw = state["aim"]["yaw"] if state["aim"] is not None else model.radial(ee)
                            lift = pose_for(ee - corr_at(ee) + [0, 0, 0.05], yaw, up, "above")
                            back = [lift] if lift is not None else []
                            move(back + [start | {"gripper": start["gripper"]}], ("方塊上方", "起姿"))
                            state["at"] = "start"
                            state["grip"] = start["gripper"]
                        if key == "q":
                            if driver.target["gripper"] < 15:
                                term.say("夾爪是閉的：先確認手上沒有方塊（按 o 不在起姿時有效）。")
                            park_and_release(driver, track.reverse(), park, support)
                            break
    finally:
        live.close()
    return 0


def zone_image(args):
    """The tray reference frame annotated with the tray outlines, the safe grasp zone, the placement squares and the
    mount axes: tools/lab_day.sh overlay blends it with the live D455 image, so trays and blocks are laid inside the
    zone (and a re-mounted camera is aligned against the same lines)."""
    import cv2
    pose = load_pose(args.pose)
    K = pose.get("intrinsics_640x480") or pose["intrinsics"]
    cam = Camera(K, pose["mount_from_color"]["R"], pose["mount_from_color"]["t"])
    quads = json.loads(Path(args.trays).read_text())["quads"] if args.trays and Path(args.trays).exists() else None
    model = ArmModel.from_file(find_calibration(args.follower_id), ROOT)
    zone = safe_zone(model, make_corr(args.corr), args.follower_id, args.corr, args.out)
    ref = cv2.imread(args.reference)
    if ref is None:
        sys.exit(f"cannot read {args.reference}")
    img = draw(ref, cam, [], quads=quads, slots=beige_slots(cam, quads) if quads and "beige" in quads else [],
               next_slot=0, zone=zone, lines=["orange: safe grasp zone   P1-P4: beige slots (P1 first)"])
    Path(args.image_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.image_out, img)
    print(f"{args.image_out}  (reference {args.reference}, pose {args.pose}, zone {args.out}/safe_zone_{args.follower_id}.json)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("preview", "reach", "zone-image"))
    ap.add_argument("--serial", default=None)
    ap.add_argument("--reference", default=None, help="zone-image: the tray reference RGB frame")
    ap.add_argument("--image-out", default=None, help="zone-image: annotated output PNG")
    ap.add_argument("--pose", default=str(ROOT / "outputs/d455/pose_261125.json"))
    ap.add_argument("--trays", default=None, help="tray corners (trays.json) for tray names and outlines")
    ap.add_argument("--follower-id", default="my_awesome_follower_arm")
    ap.add_argument("--config", default=str(ROOT / "outputs/single.local.yaml"))
    ap.add_argument("--route", default=None)
    ap.add_argument("--corr", default="21,-1.3,20",
                    help="command minus target: radial mm, rotation about the base deg (+ towards +y), z mm "
                         "(11/26 aim checks: radial 17-25, rotation -0.6..-2.6)")
    ap.add_argument("--out", default=str(ROOT / "outputs/d455"))
    a = ap.parse_args(argv)
    if a.mode == "reach" and not a.route:
        ap.error("reach needs --route")
    if a.mode == "zone-image":
        if not (a.reference and a.image_out):
            ap.error("zone-image needs --reference and --image-out")
        return zone_image(a)
    if not a.serial:
        ap.error("--serial is required")
    return preview(a) if a.mode == "preview" else reach(a)


if __name__ == "__main__":
    sys.exit(main())
