"""LeRobot SO-101 joint readings <-> shared joint angles (configs/so101_chain.yaml, radians).

The lab follower reports LeRobot degrees (5 arm joints, calibrated range around 0) and the gripper in percent open.
The NVIDIA Sim-to-Real SO-101 Workshop maps LeRobot's -100..100 range linearly onto each USD joint's range
(lerobot_interface.SO101_USD_MAPPING); the same map fits the lab captures best: the D455 depth points of the arm
match the meshes posed this way with a 4.6 mm median residual (identity degrees: 5.8 mm and a 5 deg tilted table).
It depends on each arm's calibration file and stays provisional until a joint-by-joint check.
"""

from __future__ import annotations

import math

import numpy as np

# (min, max) degrees of each USD joint that LeRobot -100 / +100 map to (Workshop lerobot_interface.py)
WORKSHOP_RANGE_DEG = ((-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160))
GRIPPER_RANGE_DEG = (-10, 100)   # LeRobot gripper 0..100 % open


def lerobot_to_shared(readings) -> tuple:
    """LeRobot [pan, lift, elbow, wrist_flex, wrist_roll, gripper %] -> (5 arm angles rad, jaw angle rad)."""
    r = np.asarray(readings, float)
    lo = np.array([a for a, _ in WORKSHOP_RANGE_DEG], float)
    hi = np.array([b for _, b in WORKSHOP_RANGE_DEG], float)
    arm = lo + (r[:5] + 100.0) / 200.0 * (hi - lo)
    g0, g1 = GRIPPER_RANGE_DEG
    jaw = g0 + r[5] / 100.0 * (g1 - g0)
    return np.radians(arm), float(np.radians(jaw))


def shared_to_lerobot(arm_rad, jaw_rad: float) -> list:
    """Inverse of lerobot_to_shared: (5 arm angles rad, jaw angle rad) -> LeRobot [pan, lift, elbow, wrist_flex,
    wrist_roll, gripper %] (same provisional Workshop map)."""
    lo = np.array([a for a, _ in WORKSHOP_RANGE_DEG], float)
    hi = np.array([b for _, b in WORKSHOP_RANGE_DEG], float)
    arm = (np.degrees(np.asarray(arm_rad, float)) - lo) / (hi - lo) * 200.0 - 100.0
    g0, g1 = GRIPPER_RANGE_DEG
    grip = (math.degrees(float(jaw_rad)) - g0) / (g1 - g0) * 100.0
    return [float(x) for x in arm] + [float(grip)]
