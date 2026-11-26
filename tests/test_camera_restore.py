"""D455 re-mount (agentic_sim.real_camera_restore): height and tilt from the table plane against a fitted pose."""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_camera_restore import TABLE_Z, advice, compare, table_plane, target_from_pose  # noqa: E402

K = {"fx": 383.7, "fy": 383.4, "cx": 320.1, "cy": 245.6}


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def pose(height, tilt_x=0.0):
    """Camera looking down from `height` over the table, image x along -Y, tilted tilt_x deg about its x axis."""
    R = np.array([[0.0, -1, 0], [-1, 0, 0], [0, 0, -1]]) @ rot_x(tilt_x)
    return {"mount_from_color": {"R": R.tolist(), "t": [0.39, 0.17, height + TABLE_Z]}}


def render(p, tray=(slice(150, 400), slice(200, 620)), tray_floor=0.006, block=(slice(260, 300), slice(300, 340))):
    """Aligned depth of the table, a tray floor 6 mm up covering most of the view and a 3 cm block."""
    R, t = np.asarray(p["mount_from_color"]["R"]), np.asarray(p["mount_from_color"]["t"])
    vv, uu = np.mgrid[0:480, 0:640]
    rays = np.stack([(uu - K["cx"]) / K["fx"], (vv - K["cy"]) / K["fy"], np.ones_like(uu, float)], -1)
    up = (rays @ R.T)[..., 2]
    surface = np.full((480, 640), TABLE_Z)
    surface[tray] += tray_floor
    surface[block] += 0.0295
    depth = (surface - t[2]) / up
    return depth + np.random.default_rng(0).normal(0, 0.0015, depth.shape)   # D455-like noise at 0.6 m


def test_the_table_not_the_tray_floor_gives_height_and_tilt():
    target = target_from_pose(pose(0.58))
    plane = table_plane(render(pose(0.58)), K)
    c = compare(plane, target, K)
    assert abs(c["height_mm"]) < 0.5 and abs(c["forward_deg"]) < 0.05 and abs(c["side_deg"]) < 0.05 and c["ok"]

    moved = compare(table_plane(render(pose(0.592, tilt_x=1.0)), K), target, K)   # 12 mm higher, tilted 1 deg
    assert abs(moved["height_mm"] - 12) < 0.7 and abs(abs(moved["forward_deg"]) - 1.0) < 0.05
    assert abs(moved["side_deg"]) < 0.05 and not moved["ok"]
    lines = advice(moved)
    assert "往下降" in lines[0] and "OK" not in lines[1] and "OK" in lines[2]
    # the top / bottom depth difference says which way: about 10 mm per degree across the frame
    assert abs(abs(moved["depth_mm"]["top"] - moved["depth_mm"]["bottom"]) - 10.5) < 1.5


def test_restore_once_on_a_saved_frame(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from d455_restore import main
    p = pose(0.58)
    p["intrinsics_640x480"] = K
    (tmp_path / "pose.json").write_text(json.dumps(p))
    np.savez(tmp_path / "frame.npz", bgr=np.zeros((480, 640, 3), np.uint8), depth=render(pose(0.575)))
    assert main(["--npz", str(tmp_path / "frame.npz"), "--target", str(tmp_path / "pose.json"), "--once",
                 "--out", str(tmp_path)]) == 1        # 5 mm low: not restored
    saved = json.loads(next(tmp_path.glob("restore-*.json")).read_text())
    assert abs(saved["compare"]["height_mm"] + 5) < 0.7
