"""Lab scene of 2026-11-20 (configs/scenes/lab_20261120.yaml): sensor cameras, LeRobot joint mapping, layout.
Pure Python (no simulator)."""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.kinematics import So101Chain, quat_wxyz_to_mat  # noqa: E402
from agentic_sim.real_joints import lerobot_to_shared  # noqa: E402
from agentic_sim.scene_layout import (block_in_tray, block_layout, global_camera_quat, sensor_cameras, trays,  # noqa: E402
                                      wrist_camera_mount)

LAB = load("configs/scenes/lab_20261120.yaml")
MINIMAL = load("configs/scenes/lab_current_minimal.yaml")


def _sources(node, path=""):
    if isinstance(node, dict):
        if "value" in node:
            yield path, node.get("source")
        for k, x in node.items():
            yield from _sources(x, f"{path}.{k}")


def test_every_lab_value_names_its_source():
    bad = [(p, s) for p, s in _sources(LAB) if s not in ("measured", "estimated", "provisional")]
    assert not bad, bad


def test_minimal_scene_has_no_sensor_cameras_and_no_mount():
    assert sensor_cameras(MINIMAL) == {} and not wrist_camera_mount(MINIMAL)


def test_global_camera_projects_like_the_d455():
    """A point on the optical axis lands on the principal point; the camera looks down with image up = +X."""
    g = sensor_cameras(LAB)["global"]
    assert (g["width"], g["height"]) == (1280, 720)
    assert abs(g["fovy_deg"] - math.degrees(2 * math.atan(360 / g["fy"]))) < 1e-9
    R, t = np.array(g["rot_cv"]), np.array(g["pos"])
    assert abs(np.linalg.det(R) - 1) < 1e-3 and np.allclose(R.T @ R, np.eye(3), atol=2e-3)

    def project(p):
        c = R.T @ (np.asarray(p, float) - t)
        return g["fx"] * c[0] / c[2] + g["cx"], g["fy"] * c[1] / c[2] + g["cy"]
    u0, v0 = project(t + 0.5 * R[:, 2])
    assert abs(u0 - g["cx"]) < 0.01 and abs(v0 - g["cy"]) < 0.01   # rot rounded to 5 decimals
    assert R[2, 2] < -0.99                                   # optical axis points down
    (_, vn), (_, vf) = project([0.10, 0.18, 0.0]), project([0.40, 0.18, 0.0])
    assert vf < vn                                           # farther ahead = higher in the image
    (ul, _), (ur, _) = project([0.3, 0.30, 0.0]), project([0.3, -0.10, 0.0])
    assert ul < ur                                           # robot's left = image left
    # USD / MuJoCo camera convention: -Z along the optical axis, +Y = image up
    m = quat_wxyz_to_mat(global_camera_quat(g))
    assert np.allclose(m[:, 2], -R[:, 2], atol=1e-4) and np.allclose(m[:, 1], -R[:, 1], atol=1e-4)


def test_trays_in_the_d455_view_and_blocks_start_inside_the_pink_tray():
    g = sensor_cameras(LAB)["global"]
    R, t = np.array(g["rot_cv"]), np.array(g["pos"])
    for name, tr in trays(LAB).items():
        c = R.T @ (np.array([*tr["center"], 0.0]) - t)
        u, vv = g["fx"] * c[0] / c[2] + g["cx"], g["fy"] * c[1] / c[2] + g["cy"]
        assert 0 < u < g["width"] and 0 < vv < g["height"], name
    lay = block_layout(LAB, 8)
    pink = trays(LAB)["storage_pink"]
    assert all(block_in_tray(p, pink, lay.edge) for p in lay.init_pos)


def test_wrist_camera_intrinsics_follow_the_fov():
    w = sensor_cameras(LAB)["wrist"]
    assert (w["width"], w["height"]) == (640, 480)
    assert abs(math.degrees(2 * math.atan(w["height"] / 2 / w["fy"])) - w["fovy_deg"]) < 1e-9
    assert abs(np.linalg.norm(w["quat_in_gripper"]) - 1) < 1e-3


def test_lerobot_mapping_workshop_ranges():
    arm, jaw = lerobot_to_shared([0, 0, 0, 0, 0, 0])
    assert np.allclose(np.degrees(arm), [0, 0, -5, 0, 0]) and abs(math.degrees(jaw) + 10) < 1e-9
    arm, jaw = lerobot_to_shared([100, -100, 100, -100, 100, 100])
    assert np.allclose(np.degrees(arm), [110, -100, 90, -95, 160]) and abs(math.degrees(jaw) - 100) < 1e-9
    # the lab rest pose maps inside the shared chain's joint limits
    chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
    arm, _ = lerobot_to_shared([-6.04, -97.27, 99.0, 71.48, 1.29, 17.0])
    assert np.all(arm >= np.array(chain.lower) - 1e-6) and np.all(arm <= np.array(chain.upper) + 1e-6)
    assert v(LAB["robot"]["base_pos"]) == v(MINIMAL["robot"]["base_pos"])
