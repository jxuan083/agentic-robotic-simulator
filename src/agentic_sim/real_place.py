"""Programmed follower moves shared by the finals reach tool and the park collection: IK with the gravity-sag
correction, gripper waits, and carrying a held block into a beige slot.

Used by tools/finals_reach.py (program grasp / place) and agentic_sim.real_grasp (park rounds: the operator
demonstrates park -> grasp -> stable lift, presses c, and the program places the block; the demonstration ends at the
lift, the part ACT learns). Joint readings are LeRobot degrees; positions are the right follower's mount frame (m).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EDGE_M = 0.0295
HOVER_M = 0.045      # grasp centre above the block centre while hovering (fingertips ~3 cm over the top)
GRASP_M = 0.015      # grasp centre above the block centre when lowered (11/26 aims: command top + 20..30 mm grasps)
TILT = {"above": 30.0, "hover": 15.0, "grasp": 8.0, "place": 25.0}   # max gripper tilt from vertical per phase
FAR_TILT = 30.0      # fallback tilt when the phase limit has no IK solution
SAFE_Z = 0.05        # lowest grasp-centre height along a move between start pose and hover
OPEN_PCT, CLOSED_PCT = 40.0, 8.0
HELD_PCT = 12.0      # gripper reading above this after closing: a block is between the jaws (held ~16.5 %)
PLACE_Z = 0.026      # grasp centre (= block centre) height when releasing in the beige tray (bottom ~6 mm above floor)
CARRY_Z = 0.11       # grasp-centre height while carrying a block between trays (block bottom ~6 cm over the rims)
GRIP_MIN_S, GRIP_STILL_S, GRIP_STILL_PCT, GRIP_MAX_S = 0.4, 0.3, 0.6, 2.5   # gripper wait: stopped = < 0.6 % in 0.3 s
# Beige slots, tray-local mm: one set for demonstrations and programmed placing (configs/real_slots.json).
SLOTS_MM = json.loads((ROOT / "configs/real_slots.json").read_text(encoding="utf-8"))["beige"]
DEFAULT_CORR = "21,-1.3,20"
# A tilted gripper's open fingers reach below the grasp centre on the low side (11/26: far blocks touched the table):
# grasp / place heights rise by TILT_RAISE_M * sin(tilt). 30 mm ~ half the open finger span plus the jaw edge.
TILT_RAISE_M = 0.030


def make_corr(spec=DEFAULT_CORR):
    """"radial mm, rotation deg, z mm" -> corr_at(xyz): command minus target (gravity sag), mount m."""
    radial_c, tangent_c, z_c = (float(c) for c in spec.split(","))

    def corr_at(xyz):
        x, y = float(xyz[0]), float(xyz[1])
        r = math.hypot(x, y) or 1.0
        u, v = np.array([x, y]) / r, np.array([-y, x]) / r
        xy = u * radial_c / 1000 + v * r * math.radians(tangent_c)
        return np.array([xy[0], xy[1], z_c / 1000])
    return corr_at


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


def camera_from_pose(path):
    from agentic_sim.real_d455 import Camera
    pose = json.loads(Path(path).read_text(encoding="utf-8"))
    K = pose.get("intrinsics_640x480") or pose["intrinsics"]
    return Camera(K, pose["mount_from_color"]["R"], pose["mount_from_color"]["t"])


class Placer:
    """IK moves of the follower through an agentic_sim.real_route Driver (Esc pauses like any playback)."""

    def __init__(self, driver, model, corr_at, on_move=None):
        self.driver, self.model, self.corr_at, self.on_move = driver, model, corr_at, on_move

    def pose_for(self, xyz, yaw, seed, phase, grip):
        """Joint target for the grasp centre at xyz (corrected for sag). The IK tilts only as much as the reach
        needs; for the low phases (grasp, place) the height then rises by TILT_RAISE_M * sin(tilt) so the low
        finger clears the table. self.last_tilt / self.last_raise report what was used."""
        xyz = np.asarray(xyz, float)
        self.last_tilt, self.last_raise = 0.0, 0.0
        reading, info = self._solve(xyz, yaw, seed, phase)
        if reading is not None and phase in ("grasp", "place"):
            tilt = float(info.get("tilt_deg", 0.0))
            if tilt > 3.0:
                raise_m = TILT_RAISE_M * math.sin(math.radians(tilt))
                lifted, info2 = self._solve(xyz + [0, 0, raise_m], yaw, reading, phase)
                if lifted is not None:
                    reading, info = lifted, info2
                    self.last_raise = raise_m
            self.last_tilt = float(info.get("tilt_deg", 0.0))
        return None if reading is None else reading | {"gripper": float(grip)}

    def _solve(self, xyz, yaw, seed, phase):
        for tilt in (TILT[phase], FAR_TILT):   # far blocks (r > ~0.30 m) need the gripper tilted more
            reading, info = self.model.hover(xyz + self.corr_at(xyz), yaw, seed, tilt)
            if reading is not None:
                return reading, info
        return None, {}

    def path_ok(self, a, b, min_z=SAFE_Z):
        for s in np.linspace(0, 1, 15):
            if self.model.ee({j: a[j] + (b[j] - a[j]) * s for j in a})[2] < min_z - 1e-3:
                return False
        return True

    def move(self, points, label):
        from agentic_sim.real_route import Track, follow
        follow(self.driver, Track([dict(self.driver.target), *points], [0.0] * len(points)), (label[0], label[1]))
        if self.on_move:
            self.on_move()

    def set_grip(self, pct, label):
        """Move the gripper and hold until it has stopped (closed on a block or fully open). Driver.settle only
        watches the body joints: 11/26 the arm lifted while the gripper still read ~30 % and dropped blocks."""
        from agentic_sim.real_route import Paused
        d = self.driver
        self.move([dict(d.target) | {"gripper": pct}], ("夾爪", label))
        began, seen = d.clock(), []
        while True:
            t = d.clock()
            if d.stop_requested():
                d.hold("operator stop")
                raise Paused("操作者暫停")
            _, actual, _ = d.send(d.target)
            g = float(actual["gripper"])
            seen = [(ts, v) for ts, v in seen + [(t, g)] if t - ts <= GRIP_STILL_S * 1e9]
            settled = t - began >= GRIP_MIN_S * 1e9 and t - seen[0][0] >= GRIP_STILL_S * 0.9e9 and \
                max(v for _, v in seen) - min(v for _, v in seen) < GRIP_STILL_PCT
            if settled or t - began >= GRIP_MAX_S * 1e9:
                return g
            d.tick(t)

    def carry_and_place(self, xy, tray_yaw, held_yaw, label, grip):
        """Holding a block: straight up to the carry height, across on Cartesian waypoints (a joint-space move
        between the trays dips and caught the beige rim on 11/26), down into the slot, open, rise over the slot.
        Returns an error string (nothing moved) or None."""
        xy = np.asarray(xy, float)
        top = np.array([xy[0], xy[1], PLACE_Z + HOVER_M])
        here = self.model.ee(self.driver.target)
        here = here - self.corr_at(here)
        path, seed = [], self.driver.target
        for k, s_ in enumerate(np.linspace(0, 1, 6)):
            p_xy = here[:2] + (xy - here[:2]) * s_
            p = None
            for z in (CARRY_Z, 0.095, 0.08, max(0.065, here[2])):   # far blocks: start lower, rise as the path comes in
                p = self.pose_for([p_xy[0], p_xy[1], z], tray_yaw if k == 5 else held_yaw, seed, "above", grip)
                if p is not None:
                    break
            if p is None:
                return f"搬運路徑第 {k + 1} 點 IK 無解，不移動。"
            path.append(p)
            seed = p
        over = self.pose_for(top, tray_yaw, seed, "place", grip)
        low = self.pose_for([xy[0], xy[1], PLACE_Z], tray_yaw, over or seed, "place", grip)
        if over is None or low is None:
            return "格位 IK 無解，不移動。"
        self.move(path + [over, low], ("方塊上方", label))
        self.set_grip(OPEN_PCT, "張開")
        self.move([over | {"gripper": OPEN_PCT}], ("格位", "格位上方"))
        return None
