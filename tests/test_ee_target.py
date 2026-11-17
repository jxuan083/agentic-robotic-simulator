"""Behaviour tests for the shared EE target (numpy only, no simulator)."""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agentic_sim.config import load  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.teleop.ee_target import (  # noqa: E402
    REJECT_STATUSES, STATUS_IK, STATUS_JOINT_LIMIT, STATUS_JUMP, STATUS_LIMIT, STATUS_OK, load_controller)


def make():
    chain = So101Chain.from_dict(yaml.safe_load(open(os.path.join(ROOT, "configs/so101_chain.yaml"))))
    return load_controller(chain, load("configs/teleop.yaml"))


def test_home_matches_target():
    c = make()
    assert np.allclose(c.state.pos, c.p.home_ee, atol=2e-3)


def test_increment_is_time_based_not_frame_based():
    a, b = make(), make()
    for _ in range(10):  # 10 frames of 0.05 s
        a.step_velocity([1, 0, 0], 0, 0.05, "keyboard")
    for _ in range(50):  # 50 frames of 0.01 s
        b.step_velocity([1, 0, 0], 0, 0.01, "keyboard")
    assert np.allclose(a.state.pos, b.state.pos, atol=1e-6)
    assert np.isclose(a.state.pos[0] - a.p.home_ee[0], 0.5 * a.p.linear_speed, atol=2e-3)


def test_dt_clamp_after_stall():
    c = make()
    x0 = c.state.pos[0]
    c.step_velocity([1, 0, 0], 0, 5.0, "keyboard")  # a 5 s stall must not produce a 0.4 m jump
    assert c.state.pos[0] - x0 <= c.p.linear_speed * c.p.dt_max + 1e-9


def test_workspace_limit_clamps():
    c = make()
    for _ in range(200):
        c.step_velocity([0, 0, -1], 0, 0.05, "keyboard")
    assert np.isclose(c.state.pos[2], c.p.ws_lo[2])
    assert c.state.status == STATUS_LIMIT


def test_unreachable_keeps_last_valid_target_without_joint_jump():
    c = make()
    q_prev = c.state.q_goal.copy()
    last_ok = c.state.pos.copy()
    statuses = []
    for _ in range(400):  # push forward + up beyond reach
        ok = c.step_velocity([1, 0, 1], 0, 0.05, "keyboard")
        statuses.append(c.state.status)
        assert np.max(np.abs(c.state.q_goal - q_prev)) <= c.p.max_joint_step + 1e-9
        q_prev = c.state.q_goal.copy()
        if ok:
            last_ok = c.state.pos.copy()
    assert any(st in REJECT_STATUSES for st in statuses) or STATUS_LIMIT in statuses
    assert np.allclose(c.state.pos, last_ok)


def push_to_boundary(c, direction=(1, 0, 0), step=0.01):
    """Drag outward until the target stops making progress (reach boundary)."""
    for _ in range(100):
        if c.move_by(np.array(direction, float) * step, "mouse") < 0.05:
            break
    return c.state.pos.copy()


def test_drag_is_relative_and_exact():
    c = make()
    p0 = c.state.pos.copy()
    assert np.isclose(c.drag([0.02, 0.01]), 1.0)
    assert np.allclose(c.state.pos, p0 + [0.02, 0.01, 0.0], atol=1e-9)
    assert c.state.status == STATUS_OK


def test_fine_drag_and_fine_wheel():
    c = make()
    p0 = c.state.pos.copy()
    c.drag([0.02, 0.0], fine=True)
    assert np.isclose(c.state.pos[0] - p0[0], 0.02 * c.p.fine_drag_scale, atol=1e-9)
    z0 = c.state.pos[2]
    c.wheel(-1, fine=True)
    assert np.isclose(z0 - c.state.pos[2], c.p.fine_wheel_step, atol=1e-9)
    c.wheel(-1)
    assert np.isclose(z0 - c.state.pos[2], c.p.fine_wheel_step + c.p.wheel_step, atol=1e-9)


def test_boundary_stops_without_accumulating_and_recovers_immediately():
    c = make()
    edge = push_to_boundary(c)
    assert c.state.status in REJECT_STATUSES
    for _ in range(50):  # keep dragging outward: nothing accumulates
        c.drag([0.01, 0.0])
    assert c.state.pos[0] <= edge[0] + 0.003
    q_prev = c.state.q_goal.copy()
    before = c.state.pos.copy()
    assert np.isclose(c.drag([-0.01, 0.0]), 1.0)  # first step back is applied in full
    assert np.isclose(before[0] - c.state.pos[0], 0.01, atol=1e-9)
    assert c.state.status == STATUS_OK
    assert np.max(np.abs(c.state.q_goal - q_prev)) <= c.p.max_joint_step


def test_slides_along_boundary():
    c = make()
    push_to_boundary(c)
    y0 = c.state.pos[1]
    for _ in range(10):  # diagonal drag, outward + sideways: the sideways part still moves
        c.drag([0.01, 0.01])
    assert c.state.pos[1] - y0 > 0.02


def test_wheel_floor_is_workspace_limit_not_rejection():
    c = make()
    for _ in range(60):
        c.wheel(-1)
    assert np.isclose(c.state.pos[2], c.p.ws_lo[2])
    assert c.state.status == STATUS_LIMIT and c.state.status not in REJECT_STATUSES


def test_rejection_reasons_are_classified():
    c = make()
    push_to_boundary(c)
    assert c.state.status in (STATUS_IK, STATUS_JOINT_LIMIT)
    assert c.state.message
    c2 = make()
    c2.p.max_joint_step = 1e-4  # any real move becomes a jump
    c2.drag([0.01, 0.0])
    assert c2.state.status == STATUS_JUMP and "joint jump" in c2.state.message
    assert c2.state.reject_counts[STATUS_JUMP] >= 1


def test_max_drag_step_caps_one_event():
    c = make()
    p0 = c.state.pos.copy()
    c.drag([0.5, 0.0])
    assert c.state.pos[0] - p0[0] <= c.p.max_drag_step + 1e-9


def test_joint_command_is_rate_limited():
    c = make()
    c.state.q_goal = c.state.q_goal + 1.0
    q0 = c.state.q_cmd.copy()
    q1 = c.joint_command(0.02)
    assert np.max(np.abs(q1 - q0)) <= c.p.joint_rate_limit * 0.02 + 1e-9


def test_reset_restores_home_and_new_session():
    c = make()
    c.step_velocity([1, 1, 0], 0.5, 0.1, "keyboard")
    c.set_gripper(False, "keyboard")
    c.nudge_yaw(0.3, "mouse")
    s0 = c.state.session
    c.reset()
    assert c.state.session == s0 + 1
    assert c.state.yaw_goal is None and c.state.status == STATUS_OK
    assert np.allclose(c.state.q_cmd, c.q_home) and c.state.gripper == c.p.gripper_open


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)


def test_yaw_button_is_gradual_and_accepted():
    c = make()
    c.nudge_yaw(np.radians(15), "mouse")
    c.nudge_yaw(np.radians(15), "mouse")  # two clicks accumulate
    q_prev = c.state.q_goal.copy()
    for _ in range(120):
        c.step_goals(1 / 60)
        assert np.max(np.abs(c.state.q_goal - q_prev)) <= c.p.max_joint_step
        q_prev = c.state.q_goal.copy()
    assert c.state.status == STATUS_OK and c.state.rejected == 0
    assert np.isclose(c.state.yaw_offset, np.radians(30), atol=1e-6)
    assert c.state.yaw_goal is None


def test_gripper_button_is_rate_limited():
    c = make()
    c.set_gripper(False, "mouse")
    g_prev = c.state.gripper
    steps = 0
    while c.state.gripper_goal is not None and steps < 200:
        c.joint_command(1 / 60)
        assert abs(c.state.gripper - g_prev) <= c.p.gripper_speed / 60 + 1e-9
        g_prev = c.state.gripper
        steps += 1
    assert np.isclose(c.state.gripper, c.p.gripper_closed)
    assert steps >= int((c.p.gripper_open - c.p.gripper_closed) / (c.p.gripper_speed / 60))
