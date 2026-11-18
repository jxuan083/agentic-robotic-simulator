"""MuJoCo backend (headless; no window). Needs the uv environment's mujoco and the menagerie SO-101
(tools/fetch_mujoco_assets.sh); skipped where those are missing (e.g. inside the Isaac container)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from agentic_sim.config import load  # noqa: E402
from agentic_sim.teleop.keys import KeyState  # noqa: E402

SCENE = load("configs/scenes/lab_current_minimal.yaml")

# ---- MuJoCo backend ---------------------------------------------------------------------------
mujoco = pytest.importorskip("mujoco")
from agentic_sim.mujoco_backend import model as mjm  # noqa: E402

needs_model = pytest.mark.skipif(
    not os.path.isfile(os.path.join(mjm.menagerie_dir(), "robotstudio_so101/so101.xml")),
    reason="menagerie SO-101 not fetched (tools/fetch_mujoco_assets.sh)")


@needs_model
def test_fk_matches_shared_chain():
    from agentic_sim.kinematics import So101Chain
    chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
    chain.ee_offset = np.array([0.0069, 0.0, -0.085])
    chain.base_pos = np.array(SCENE["robot"]["base_pos"]["value"], float)
    sc = mjm.build(SCENE, 1)
    m, d = sc.model, mujoco.MjData(sc.model)
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.uniform(chain.lower, chain.upper)
        for adr, qi, off in zip(sc.arm_qpos, q, sc.arm_offset):
            d.qpos[adr] = qi + off
        mujoco.mj_kinematics(m, d)
        g = m.body("gripper").id
        ee = d.xmat[g].reshape(3, 3) @ chain.ee_offset + d.xpos[g]
        assert np.linalg.norm(ee - chain.ee_pose_mount(q)[:3, 3]) < 1e-3
    assert "camera_mount" not in [m.body(i).name for i in range(m.nbody)]


@needs_model
def test_session_keys_move_target_and_reset():
    from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    s = MjSession(SessionConfig(blocks=3), log=lambda *_: None, keys=keys)
    home, session0 = s.ctrl.state.pos.copy(), s.ctrl.state.session
    keys.on_key("W", True)
    for _ in range(30):
        s.step(dt=1 / 60)
    keys.on_key("W", False)
    for _ in range(30):
        s.step(dt=1 / 60)
    assert s.ctrl.state.pos[0] - home[0] == pytest.approx(0.04, abs=2e-3)       # 0.08 m/s for 0.5 s
    assert np.linalg.norm(s.last_frame["ee"] - s.ctrl.state.pos) < 0.005         # arm follows the target
    keys.on_key("P", True)
    s.step(dt=1 / 60)
    keys.on_key("W", True)
    for _ in range(10):
        s.step(dt=1 / 60)
    assert s.paused and s.ctrl.state.pos[0] - home[0] == pytest.approx(0.04, abs=2e-3)  # no motion while paused
    keys.on_key("W", False)
    keys.on_key("R", True)
    s.step(dt=1 / 60)
    assert not s.paused and np.allclose(s.ctrl.state.pos, home, atol=1e-6) and s.ctrl.state.session == session0 + 1
    assert len(s.last_frame["blocks"]) == 3 and s.last_frame["n_placed"] == 0


@needs_model
def test_session_focus_loss_drops_keys():
    from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    focus = {"on": True}
    s = MjSession(SessionConfig(), log=lambda *_: None, keys=keys, focused=lambda: focus["on"])
    keys.on_key("A", True)
    s.step(dt=1 / 60)
    focus["on"] = False
    y = s.ctrl.state.pos[1]
    for _ in range(10):
        s.step(dt=1 / 60)
    focus["on"] = True
    for _ in range(10):
        s.step(dt=1 / 60)
    assert s.ctrl.state.pos[1] == y and not keys.held  # dropped, not resumed while the key stays down


@needs_model
def test_color_dr_writes_drawn_colours_and_restores():
    from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
    s = MjSession(SessionConfig(blocks=2, color_dr=True, color_dr_seed=7, color_dr_log="outputs/test_mj_dr.jsonl"),
                  log=lambda *_: None)
    rec = s.color_dr.current
    assert rec["session"] == 0
    drawn = [rec["block"]["rgb"]] + [b["rgb"] for b in rec["extra_blocks"]]
    for g, rgb in zip(s.sc.block_geoms, drawn):
        assert np.allclose(s.m.geom_rgba[g][:3], rgb)
    assert all(np.allclose(s.m.mat_rgba[i][:3], rec["arm_shell"]["rgb"]) for i in s.sc.shell_materials)
    s.request("reset")
    s.step(dt=1 / 60)
    assert s.color_dr.current["session"] == 1 and len(s.color_dr.records) == 2
    s.color_dr.configure(False)
    assert np.allclose(s.m.geom_rgba[s.sc.block_geoms[0]][:3], SCENE["block"]["color"])


@needs_model
def test_script_mouse_steps_need_the_window():
    from agentic_sim.mujoco_backend.script import MjScript
    from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
    with pytest.raises(ValueError, match="needs the MuJoCo window"):
        MjSession(SessionConfig(script="configs/acceptance/robustness.yaml"), log=lambda *_: None)
    s = MjSession(SessionConfig(script="configs/acceptance/robustness_keyboard.yaml"), log=lambda *_: None)
    assert isinstance(s.script, MjScript)
    with pytest.raises(ValueError, match="unsupported"):
        MjScript([{"focus_view": 1}], KeyState(), lambda *_: None)


@needs_model
def test_viewer_camera_moves_without_window():
    """Mouse camera path (mjv_moveCamera signature changed in MuJoCo 4); no GLFW window needed."""
    from agentic_sim.mujoco_backend.viewer import Viewer
    v = Viewer.__new__(Viewer)  # skip the window
    v.model, v.cam = mjm.build(SCENE, 1).model, mujoco.MjvCamera()
    errors = []
    v.log = errors.append
    v.cam.distance, v.cam.azimuth = 1.0, 10.0
    v.move_camera(mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0.1, 0.0)
    v.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05)
    v.move_camera(mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.05, 0.05)
    assert not errors and v.cam.azimuth != 10.0 and v.cam.distance != 1.0

