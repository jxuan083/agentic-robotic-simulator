"""LeRobot SO-101 readings <-> SO-101 joint angles from the arm's own calibration file, FK and hover IK.

LeRobot 0.4.4 with use_degrees=True reports  deg = (present - (range_min + range_max) / 2) * 360 / 4095,  where the
calibration homing makes the middle pose of the calibration read 2047. The MuJoCo Menagerie SO-101 (TheRobotStudio
so101_new_calib) has its joint zeros at that middle pose, so

    menagerie angle [deg] = deg + ((range_min + range_max) / 2 - 2047) * 360 / 4095

with gain 1 (the encoder counts 4096 steps per output turn). The shared chain (configs/so101_chain.yaml) equals the
Menagerie chain up to configs/mujoco_so101.yaml joint_offset (q_mujoco = q_shared + offset).

Checked on the 2026-11-25 park episodes (107 depth frames, 2 episodes held out): the Menagerie meshes posed this way
match the D455 depth of the arm with a 6.8 mm median residual together with the table plane; fitting per-joint
offsets and gains freely returns the same offsets within 1 deg and gains 0.98-1.06. The provisional Workshop map
(agentic_sim.real_joints) scales LeRobot -100..100 values and does not apply to degree readings.

Frame: mount frame of the right follower, +X forward, +Y left, +Z up, origin on the table below the base (metres).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from agentic_sim.config import load, v
from agentic_sim.kinematics import So101Chain

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
RES = 4095
MIDDLE = 2047
BASE_POS = (0.0, 0.0, -0.0301)    # base link origin in the mount frame (base bottom on the table)


def calibration_offsets(calibration: dict) -> np.ndarray:
    """Degrees to add to a LeRobot reading to get the Menagerie joint angle, per arm joint."""
    return np.array([((calibration[j]["range_min"] + calibration[j]["range_max"]) / 2 - MIDDLE) * 360 / RES
                     for j in ARM], float)


def find_calibration(arm_id: str, role: str = "follower") -> Path:
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
    hits = sorted(root.glob(f"*/{arm_id}.json"))
    if not hits:
        raise FileNotFoundError(f"no LeRobot calibration {arm_id}.json under {root}")
    return hits[0]


def wrap_quarter(a: float) -> float:
    """Angle (rad) wrapped to [-pi/4, pi/4): a square block looks the same every 90 deg."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


class ArmModel:
    def __init__(self, calibration: dict, root: Path | str = "."):
        root = Path(root)
        self.offsets_deg = calibration_offsets(calibration)
        self.chain = So101Chain.from_dict(load(root / "configs" / "so101_chain.yaml"))
        self.chain.ee_offset = np.array(v(load(root / "configs" / "teleop.yaml")["ee_offset_in_gripper_link"]), float)
        self.chain.base_pos = np.array(BASE_POS, float)
        offs = (load(root / "configs" / "mujoco_so101.yaml").get("alignment") or {}).get("joint_offset") or {}
        names = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
        self.mujoco_offset = np.array([float(v(offs[n])) if n in offs else 0.0 for n in names])
        self.pan_origin = self.chain.pan_origin_mount()

    @classmethod
    def from_file(cls, path, root="."):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), root)

    # LeRobot degrees <-> shared chain radians
    def to_shared(self, reading) -> np.ndarray:
        deg = np.array([reading[j] for j in ARM], float) if isinstance(reading, dict) else np.asarray(reading, float)[:5]
        return np.radians(deg + self.offsets_deg) - self.mujoco_offset

    def to_lerobot(self, q) -> dict:
        deg = np.degrees(np.asarray(q, float) + self.mujoco_offset) - self.offsets_deg
        return {j: float(d) for j, d in zip(ARM, deg)}

    def ee(self, reading) -> np.ndarray:
        """Grasp centre (between the finger pads) in the mount frame, metres."""
        return self.chain.ee_pose_mount(self.to_shared(reading))[:3, 3].copy()

    def radial(self, xy) -> float:
        return math.atan2(xy[1] - self.pan_origin[1], xy[0] - self.pan_origin[0])

    def wrist_turn(self, xy, block_yaw_rad: float) -> float:
        """Smallest wrist turn (rad, +-45 deg) from radial fingers that squares the fingers with the block faces."""
        return wrap_quarter(block_yaw_rad - self.radial(xy))

    def hover(self, xyz, block_yaw_rad: float, seed_reading, max_tilt_deg: float = 4.0):
        """Joint reading (LeRobot degrees, arm joints) that puts the grasp centre at xyz with a vertical gripper whose
        fingers are squared with the block; None when out of reach."""
        yaw = self.radial(xyz) + self.wrist_turn(xyz, block_yaw_rad)
        seed = self.to_shared(seed_reading)
        q, ok, info = self.chain.solve_ik_tilt(np.asarray(xyz, float), yaw, seed, max_tilt_deg)
        if not ok:
            return None, info
        return self.to_lerobot(q), info
