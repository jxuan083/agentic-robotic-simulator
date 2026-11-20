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


# ---- #11: wrist lens, start pose, lab experiment -------------------------------------------------------------------
from agentic_sim.wrist_lens import WristLens, apply_wrist_lens, lens_mode  # noqa: E402


def _lab(mode=None):
    return apply_wrist_lens(load("configs/scenes/lab_20261120.yaml"), mode)


def test_wrist_lens_round_trip_and_corner_clamp():
    lens = WristLens.from_scene(LAB)
    assert (lens.width, lens.height, lens.fx, lens.dist[0]) == (640, 480, 400.0, -0.15)
    u, vv = np.meshgrid(np.linspace(20, 620, 13), np.linspace(20, 460, 9))
    x, y = lens.undistort(u, vv)
    ub, vb = lens.distort(x, y)
    assert np.max(np.abs(ub - u)) < 0.01 and np.max(np.abs(vb - vv)) < 0.01
    # barrel: an edge pixel's ray lies farther out than the pinhole with the same focal length would put it
    x0, _ = lens.undistort(np.array([0.0]), np.array([240.0]))
    assert abs(x0[0]) > 320 / 400
    # the extreme corners lie past the model's fold: they map to the fold radius, a few pixels from the corner
    xc, yc = lens.undistort(np.array([0.0]), np.array([0.0]))
    uc, vc = lens.distort(xc, yc)
    assert np.hypot(uc[0], vc[0]) < 4.0


def test_wrist_lens_modes_and_render_view():
    assert lens_mode(LAB) == "real" and lens_mode(LAB, "pinhole") == "pinhole"
    assert lens_mode(MINIMAL) == "pinhole"
    real, pin = sensor_cameras(_lab())["wrist"], sensor_cameras(_lab("pinhole"))["wrist"]
    assert pin["lens_mode"] == "pinhole" and "lens" not in pin and pin["render_fovy_deg"] == pin["fovy_deg"]
    assert real["lens_mode"] == "real" and real["render_size"] == (960, 720)
    assert real["render_fovy_deg"] > real["fovy_deg"] + 15          # wider pinhole that covers every raw ray
    lens = real["lens"]
    w, h = real["render_size"]
    iy, ix = lens.maps(w, h, real["render_fovy_deg"])
    assert iy.shape == (480, 640) and 0 <= iy.min() and iy.max() < h and 0 <= ix.min() and ix.max() < w
    # the raw centre looks at the render centre; a scaled output (GUI inset) samples the same rays
    assert abs(iy[240, 320] - (h - 1) / 2) <= 1.5 and abs(ix[240, 320] - (w - 1) / 2) <= 1.5
    sy, sx = lens.maps(w, h, real["render_fovy_deg"], 320, 240)
    assert sy.shape == (240, 320) and abs(int(sx[120, 160]) - int(ix[240, 320])) <= 2
    try:
        lens_mode(LAB, "fisheye")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lens mode accepted")


def test_lab_start_pose_moves_home_and_minimal_starts_at_home():
    from agentic_sim.teleop.ee_target import load_controller
    chain = lambda: So101Chain.from_dict(load("configs/so101_chain.yaml"))  # noqa: E731
    cfg = load("configs/teleop.yaml")
    lab = load_controller(chain(), cfg, base_pos=v(LAB["robot"]["base_pos"]), scene=LAB)
    mini = load_controller(chain(), cfg, base_pos=v(MINIMAL["robot"]["base_pos"]), scene=MINIMAL)
    assert mini.p.start_q is None and np.allclose(mini.state.q_cmd, mini.q_home) and not mini.homing
    rest = lerobot_to_shared(v(LAB["robot"]["start_pose_lerobot"]))[0]
    assert np.allclose(lab.state.q_cmd, rest, atol=1e-9) and lab.homing
    assert np.allclose(lab.q_home, mini.q_home, atol=1e-6)           # same neutral grasp (tangential fingers)
    t, steps = 0.0, 0
    while lab.homing and steps < 2000:
        prev = lab.state.q_cmd.copy()
        lab.joint_command(1 / 60)
        assert np.max(np.abs(lab.state.q_cmd - prev)) <= lab.p.homing_rate / 60 + 1e-9
        t += 1 / 60
        steps += 1
    assert not lab.homing and np.allclose(lab.state.q_cmd, lab.q_home) and 0.8 < t < 3.0
    lab.reset()
    assert lab.homing and np.allclose(lab.state.q_cmd, rest, atol=1e-9)


def test_lab_experiment_and_wrist_lens_run_option():
    from agentic_sim.experiments import RUN_OPTIONS, load_experiment
    from agentic_sim.launch import RunOptions
    e = load_experiment("lab_20261120")
    assert e.scene == "configs/scenes/lab_20261120.yaml" and set(e.backends) == {"isaac", "mujoco"}
    assert "wrist_lens" in RUN_OPTIONS and RunOptions(wrist_lens="pinhole").wrist_lens == "pinhole"


def test_undistort_tool_straightens_a_distorted_line(tmp_path):
    """Raw frame of a straight vertical line (rendered through the lens model) -> tools/undistort_wrist.py -> the
    line is straight again at its ideal column; a frame of another size is refused."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import undistort_wrist as uw
    from render_sensor_cameras import write_png
    lens = WristLens.from_scene(LAB)
    col = 520.0                                              # ideal column (off centre: clearly bent when raw)
    uu, vv = np.meshgrid(np.arange(640), np.arange(480))
    x, _ = lens.undistort(uu, vv)
    raw = np.full((480, 640, 3), 230, np.uint8)
    raw[np.abs(x * lens.fx + lens.cx - col) < 1.5] = 20     # a dark 3 px line in the ideal image, seen raw
    rows = range(40, 440, 40)
    raw_cols = [np.nonzero(raw[r, :, 0] < 128)[0].mean() for r in rows]
    assert np.ptp(raw_cols) > 6                              # bent in the raw frame
    flat = uw.undistort(raw, uw.flat_maps(lens))
    flat_cols = [np.nonzero(flat[r, :, 0] < 128)[0].mean() for r in rows]
    assert np.ptp(flat_cols) < 1.5 and abs(np.mean(flat_cols) - col) < 1.0
    src = tmp_path / "raw.png"
    write_png(str(src), raw)
    assert uw.main([str(src), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "raw_flat.png").exists() and src.exists()
    small = tmp_path / "small.png"
    write_png(str(small), raw[:240, :320])
    assert uw.main([str(small)]) == 1 and not (tmp_path / "small_flat.png").exists()
