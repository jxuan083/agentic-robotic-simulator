"""Scripted park rounds (tools/auto_collect.py): reset sampling and the recorded grasp on the fake follower."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from agentic_sim.real_kinematics import ARM, ArmModel  # noqa: E402

MID = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1047, "range_max": 3047}
       for i, j in enumerate((*ARM, "gripper"))}


def model():
    return ArmModel(MID, ROOT)


def test_reset_spots_are_inside_the_tray_the_zone_and_apart():
    from auto_collect import RESET_GAP_M, inside, sample_free
    from finals_reach import in_zone

    poly = np.array([[0.20, 0.10], [0.32, 0.10], [0.32, 0.22], [0.20, 0.22]])
    zone = {"phi_deg": [-90, 90], "r_safe_m": [0.33, 0.33], "r_min_m": 0.12}
    rng = np.random.default_rng(0)
    taken = []
    for _ in range(4):
        xy, yaw = sample_free(rng, poly, zone, taken, lambda xy, yaw: True)
        assert inside(poly, xy) and in_zone(zone, *xy) and 0 <= yaw < math.pi / 2
        assert all(math.hypot(*(xy - t)) >= RESET_GAP_M for t in taken)
        taken.append(xy)
    # a full tray: no spot left
    assert sample_free(rng, poly, zone, [np.array([x, y]) for x in np.arange(0.2, 0.33, 0.03)
                                         for y in np.arange(0.1, 0.23, 0.03)], lambda xy, yaw: True) is None


def test_scripted_grasp_runs_start_to_lift_with_a_perturbed_approach(tmp_path):
    from auto_collect import LIFT_M, Episodes
    from agentic_sim.real_place import GRASP_M, Placer, make_corr
    from agentic_sim.real_route import Driver
    from agentic_sim.real_route_fake import CHANNEL, FakeClock, FakeFollower, ScriptedTerminal

    clock = FakeClock()
    follower = FakeFollower(clock, start=CHANNEL, torque=True)
    term = ScriptedTerminal(clock)
    driver = Driver(follower, term, tmp_path / "auto", hz=60, clock=clock, wait=clock.sleep, drive_leader=False)
    driver.target = dict(CHANNEL)
    m = model()
    placer = Placer(driver, m, make_corr("0,0,0"))
    sent = []
    driver.on_send = lambda cmd, actual: sent.append(dict(cmd))
    block = {"x": 0.22, "y": 0.12, "z_top": 0.0295, "yaw": math.radians(20), "color": "red"}
    held, err = Episodes(driver, placer, np.random.default_rng(0)).grasp(block, perturb=(8.0, -6.0, 5.0))
    assert err is None
    assert held < 12.0                         # the fake follower closes on nothing: no block held
    ee = m.ee(driver.target)
    assert abs(ee[0] - 0.22) < 0.003 and abs(ee[1] - 0.12) < 0.003
    assert abs(ee[2] - (0.0295 / 2 + GRASP_M + LIFT_M)) < 0.004     # ends lifted over the block
    lowest = min(m.ee(c)[2] for c in sent)
    assert lowest < 0.0295 / 2 + GRASP_M + 0.004                    # went down to the grasp height
    assert min(c["gripper"] for c in sent) <= 8.0 + 1e-6           # and closed
