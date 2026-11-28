"""The ACT loop on the follower (agentic_sim.real_policy) with a scripted policy and a fake driver."""

import numpy as np

from agentic_sim.real_control import JOINTS
from agentic_sim.real_policy import run, to_inputs


class FakeDriver:
    """Commands are reached at once; the gripper stops at 20 % when closed on something (holding)."""

    def __init__(self, holding=True):
        self.t, self.holding, self.sent, self.held_reason = 0, holding, [], None
        self.target = {j: 0.0 for j in JOINTS} | {"shoulder_lift": 100.0, "gripper": 40.0}

    def clock(self):
        return self.t

    def tick(self, began):
        self.t = began + 33_333_333

    def stop_requested(self):
        return False

    def read(self):
        g = self.target["gripper"]
        return dict(self.target) | {"gripper": max(g, 20.0) if self.holding else g}

    def send(self, target):
        self.target = dict(target)
        self.sent.append(dict(target))

    def hold(self, reason):
        self.held_reason = reason


def scripted(path):
    """Policy replaying a list of (shoulder_lift, gripper) commands, then holding the last one."""
    seen = []

    def policy(state, images, condition):
        seen.append((list(state), sorted(images), condition))
        lift, grip = path[min(len(seen) - 1, len(path) - 1)]
        return [0.0, lift, 0.0, 0.0, 0.0, grip]
    policy.reset = lambda: seen.clear()
    policy.seen = seen
    return policy


def frames():
    return {"global": np.zeros((48, 64, 3), np.uint8), "wrist": np.zeros((48, 64, 3), np.uint8)}


def descend_close_lift():
    down = [(100.0 - 2.5 * k, 40.0) for k in range(21)]                    # z 0.100 -> 0.050
    close = [(50.0, 40.0 - 3 * k) for k in range(1, 11)]                     # gripper 40 -> 10
    up = [(50.0 + 2.5 * k, 10.0) for k in range(1, 21)]                      # z back to 0.100
    return down + close + up


def test_the_policy_runs_until_closed_lifted_and_still():
    d = FakeDriver()
    p = scripted(descend_close_lift())
    out = run(d, p, frames, condition=[300.0, -100.0, 0.0, 1.0], ee_z=lambda q: q["shoulder_lift"] / 1000)
    assert out["ok"] and out["why"] == "done" and out["held"]
    assert abs(out["grasp_z"] - 0.050) < 1e-9
    assert p.seen[0][1] == ["global", "wrist"] and p.seen[0][2] == [300.0, -100.0, 0.0, 1.0]
    assert out["seconds"] < 3.0


def test_an_empty_close_is_not_held():
    out = run(FakeDriver(holding=False), scripted(descend_close_lift()), frames,
              ee_z=lambda q: q["shoulder_lift"] / 1000)
    assert out["ok"] and not out["held"]


def test_a_jump_is_ramped_and_an_out_of_bounds_one_stops_and_holds():
    d = FakeDriver()
    out = run(d, scripted([(110.0, 40.0), (110.0, 40.0), (90.0, 40.0)]), frames,
              ee_z=lambda q: q["shoulder_lift"] / 1000)
    assert [s["shoulder_lift"] for s in d.sent] == [106.0, 110.0]           # 10 deg in two steps (6 deg / step)
    assert not out["ok"] and out["why"].startswith("step") and d.held_reason.startswith("policy step")
    assert out["ramped"] == 1                                               # the 20 deg jump was never sent


def test_with_open_pct_a_low_first_gripper_action_is_no_close():
    """11/28 sim: the first action asked 10 % less than the start, then the arm rose to the approach and settled:
    without open_pct that was done with nothing in the gripper."""
    rise = [(100.0 + 2.5 * k, 28.0) for k in range(21)] + [(150.0, 28.0)] * 15
    out = run(FakeDriver(holding=False), scripted(rise), frames, ee_z=lambda q: q["shoulder_lift"] / 1000,
              timeout_s=1.5, open_pct=45.0)
    assert out["why"] == "timeout"
    opened = [(150.0, 50.0)] + [(150.0 - 5 * k, 50.0) for k in range(1, 21)]
    opened += [(50.0, 50.0 - 4 * k) for k in range(1, 11)]
    up = [(50.0 + 2.5 * k, 10.0) for k in range(1, 21)]
    out = run(FakeDriver(), scripted(rise + opened + up), frames, ee_z=lambda q: q["shoulder_lift"] / 1000,
              open_pct=45.0)
    assert out["ok"] and out["held"] and abs(out["grasp_z"] - 0.050) < 1e-9


def test_a_policy_that_never_lifts_times_out():
    d = FakeDriver()
    out = run(d, scripted([(100.0, 40.0)]), frames, ee_z=lambda q: q["shoulder_lift"] / 1000, timeout_s=1.0)
    assert out["why"] == "timeout" and d.held_reason == "policy timeout"


def test_inputs_are_rgb_with_the_marker_on_the_global_camera():
    bgr = np.zeros((120, 160, 3), np.uint8)
    bgr[..., 0] = 255                                                       # blue in BGR
    out = to_inputs({"global": bgr, "wrist": bgr}, {"target": [80, 60]})
    assert out["wrist"][0, 0].tolist() == [0, 0, 255]
    assert (out["global"] != out["wrist"]).any()                            # the circle is drawn on global only
