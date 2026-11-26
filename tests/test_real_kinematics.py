"""Calibration-file joint map, FK / hover IK and the D455 block geometry (finals reach)."""
import math

import numpy as np

from agentic_sim.real_d455 import EDGE_M, Camera, corners
from agentic_sim.real_kinematics import ArmModel, calibration_offsets, wrap_quarter

CAL = {j: {"range_min": lo, "range_max": hi} for j, (lo, hi) in {
    "shoulder_pan": (814, 3396), "shoulder_lift": (786, 3281), "elbow_flex": (871, 3085),
    "wrist_flex": (781, 3144), "wrist_roll": (0, 4095)}.items()}


def test_offsets_are_the_range_middle_from_the_calibration_pose():
    off = calibration_offsets(CAL)
    assert np.allclose(off, [5.10, -1.19, -6.07, -7.43, 0.04], atol=0.01)


def test_reading_roundtrip_and_hover_reaches_the_target():
    arm = ArmModel(CAL)
    start = {"shoulder_pan": 1.67, "shoulder_lift": -49.49, "elbow_flex": 15.21, "wrist_flex": 93.63, "wrist_roll": 3.03}
    back = arm.to_lerobot(arm.to_shared(start))
    assert all(abs(back[j] - start[j]) < 1e-9 for j in start)
    target = np.array([0.22, 0.15, 0.06])
    reading, _ = arm.hover(target, math.radians(20), start, 15.0)
    assert reading is not None and np.linalg.norm(arm.ee(reading) - target) < 2e-3
    assert reading["shoulder_pan"] < 0          # +y (left) is negative pan on the right follower


def test_wrist_turn_is_the_smallest_quarter_turn():
    assert abs(wrap_quarter(math.radians(80)) - math.radians(-10)) < 1e-9
    arm = ArmModel(CAL)
    xy = (0.2, 0.0)
    assert abs(arm.wrist_turn(xy, arm.radial(xy) + math.radians(30))) - math.radians(30) < 1e-9


def test_block_corners_project_back_through_the_camera():
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    cam = Camera({"fx": 380, "fy": 380, "cx": 320, "cy": 240}, R, [0.35, 0.0, 0.6])
    b = {"x": 0.25, "y": 0.1, "z_top": 0.034, "yaw": math.radians(20)}
    c = corners(b)
    assert np.allclose(np.linalg.norm(c[0] - c[1]), EDGE_M)
    uv = cam.project(c)
    assert np.allclose(cam.on_plane(uv, 0.034), c, atol=1e-9)


def test_colour_names_cover_the_finals_palette():
    from agentic_sim.real_d455 import COLOR_REF, classify
    assert set(COLOR_REF) == {"red", "orange", "yellow", "light_green", "dark_green", "cyan", "blue", "purple"}
    assert classify((101, 255, 145))[0] == "blue" and classify((177, 228, 160))[0] == "red"
    assert classify((0, 10, 200))[0] == "unknown"      # grey / white is not a block colour
