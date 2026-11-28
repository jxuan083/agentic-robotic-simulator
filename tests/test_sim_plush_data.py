"""Plush ACT episodes in MuJoCo (agentic_sim.plush_sim_data): layout sampling, condition, dataset features."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim import plush_sim_data as D  # noqa: E402
from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.real_dataset import CONDITION_NAMES  # noqa: E402

SCENE = ROOT / "configs/scenes/lab_20261128_plush.yaml"


def test_layouts_stay_in_the_documented_spread_and_clear_of_the_tram():
    scene = load(SCENE)
    x0, y0 = v(scene["plush"]["pos"])
    counts = set()
    for seed in range(200):
        lay = D.sample_layout(scene, np.random.default_rng(seed))
        x, y = lay["plush_xy"]
        assert -0.015 <= x - x0 <= 0.015 and 0.0 <= y - y0 <= 0.020
        assert 0.0 <= lay["plush_yaw_deg"] < 180.0
        assert all(abs(n) <= D.START_NOISE_DEG for n in lay["start_noise_deg"])
        for b in lay["blocks"]:
            assert D.gap_to_tram(*b["xy"], x, y, math.radians(lay["plush_yaw_deg"])) >= D.BLOCK_GAP_M
        counts.add(len(lay["blocks"]))
    assert counts == {0, 1, 2, 3}
    a, b = (D.sample_layout(scene, np.random.default_rng(7)) for _ in range(2))
    assert a == b


def test_condition_is_the_plush_footprint_with_yaw_fold_2():
    from agentic_sim.mujoco_backend.model import build

    scene = load(SCENE)
    lay = {"plush_xy": [0.21, 0.14], "plush_yaw_deg": 120.0, "blocks": [], "start_noise_deg": [0.0] * 5}
    sc_cfg, colors = D.scene_for(scene, lay)
    sc = build(sc_cfg, colors=colors)
    d = mujoco.MjData(sc.model)
    mujoco.mj_forward(sc.model, d)
    cond, info = D.plush_condition(sc, d)
    assert cond[0] == pytest.approx(210.0, abs=0.5) and cond[1] == pytest.approx(140.0, abs=0.5)
    assert info["yaw_deg"] == pytest.approx(120.0, abs=0.1)
    assert cond[2:] == pytest.approx([math.sin(math.radians(240)), math.cos(math.radians(240))], abs=1e-6)
    lay2 = dict(lay, plush_yaw_deg=lay["plush_yaw_deg"] - 180 + 360)   # same tram axis: same condition
    sc2 = build(D.scene_for(scene, lay2)[0], colors=colors)
    d2 = mujoco.MjData(sc2.model)
    mujoco.mj_forward(sc2.model, d2)
    assert D.plush_condition(sc2, d2)[0] == pytest.approx(cond, abs=1e-6)


def test_features_match_the_real_act_dataset():
    f = D.features()
    assert f["observation.environment_state"]["names"] == CONDITION_NAMES[:4]
    assert f["observation.images.global"]["shape"] == f["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert f["observation.state"]["shape"] == f["action"]["shape"] == (6,)
    assert {"task_is_policy", "is_handoff"} <= set(f)
