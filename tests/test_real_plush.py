"""Plush toy in D455 depth (agentic_sim.real_plush): found from heights, block detections on it dropped."""

import glob
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from agentic_sim import real_plush as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def table(rng):
    xy = rng.uniform([-0.05, -0.35], [0.55, 0.40], (60000, 2))
    return np.c_[xy, rng.normal(0.0, 0.0015, len(xy))]


def plush_cloud(x, y, yaw, rng, upright=True):
    """The camera's view of the toy: tram roof (rounded edges) and, upright, the dog's head on one end. Fallen on
    its side: the tram's side (57 mm high) with the head beside it."""
    L, W, B = (0.086, 0.057, 0.048) if upright else (0.086, 0.095, 0.057)   # depth edges lose a few mm
    u, v = rng.uniform(-L / 2, L / 2, 4000), rng.uniform(-W / 2, W / 2, 4000)
    z = B - 0.01 * ((np.abs(u) / (L / 2)) ** 6 + (np.abs(v) / (W / 2)) ** 6)
    if upright:
        hu, hv = rng.uniform(-0.025, 0.025, (2, 2000))
        k = np.hypot(hu, hv) < 0.025
        hz = B + 0.025 + np.sqrt(np.maximum(0.025 ** 2 - hu ** 2 - hv ** 2, 0))
        u, v, z = np.r_[u, hu[k] + 0.012], np.r_[v, hv[k]], np.r_[z, hz[k]]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * u - s * v, y + s * u + c * v, z], 1)


def block_cloud(x, y, rng):
    u, v = rng.uniform(-0.0147, 0.0147, (2, 400))
    return np.stack([x + u, y + v, np.full_like(u, 0.0295)], 1)


def test_plush_pose_and_head():
    rng = np.random.default_rng(1)
    pts = np.r_[table(rng), plush_cloud(0.30, -0.12, math.radians(57), rng), block_cloud(0.25, 0.15, rng)]
    plushes, _, info = rp.detect(pts)
    assert len(plushes) == 1 and abs(info["table_z_mm"]) < 1
    p = plushes[0]
    assert math.hypot(p["x"] - 0.30, p["y"] + 0.12) < 0.006
    assert abs(((math.degrees(p["yaw"]) - 57) + 90) % 180 - 90) < 6
    assert 85 <= p["height_mm"] <= 105 and p["upright"]
    # the head sits on the +u end of the tram (12 mm off centre along the long axis)
    along = (p["head"][0] - p["x"]) * math.cos(math.radians(57)) + (p["head"][1] - p["y"]) * math.sin(math.radians(57))
    assert along > 0.005


def test_fallen_plush_is_still_found_and_flagged():
    rng = np.random.default_rng(2)
    plushes, _, _ = rp.detect(np.r_[table(rng), plush_cloud(0.25, 0.0, 0.3, rng, upright=False)])
    assert len(plushes) == 1 and not plushes[0]["upright"]


def test_blocks_arm_and_hand_are_not_plushes():
    rng = np.random.default_rng(3)
    arm = np.c_[rng.uniform([0.10, 0.03], [0.16, 0.09], (3000, 2)), rng.uniform(0.0, 0.18, 3000)]
    pts = np.r_[table(rng), block_cloud(0.30, 0.20, rng), block_cloud(0.33, 0.20, rng), arm]
    plushes, rejected, _ = rp.detect(pts)
    assert plushes == []
    assert any("too tall" in r["why"] for r in rejected)


def test_block_detections_on_the_plush_are_dropped():
    p = {"x": 0.30, "y": -0.12, "yaw": 0.0, "size_mm": [90.0, 60.0]}
    blocks = [{"x": 0.33, "y": -0.10}, {"x": 0.30, "y": -0.16}, {"x": 0.25, "y": 0.15}]
    assert rp.drop_blocks_on(blocks, [p]) == [blocks[2]]


@pytest.mark.skipif(not glob.glob(str(ROOT / "outputs/d455/*.npz")), reason="no saved D455 frames on this machine")
def test_no_plush_in_the_saved_lab_frames():
    from agentic_sim.real_place import camera_from_pose
    cam = camera_from_pose(ROOT / "outputs/d455/pose_261125.json")
    for f in sorted(glob.glob(str(ROOT / "outputs/d455/*.npz"))):
        assert rp.detect(rp.mount_points(np.load(f)["depth"], cam))[0] == [], f
