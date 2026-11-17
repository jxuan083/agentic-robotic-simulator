"""Experiment registry and launch-command tests (pure Python)."""
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agentic_sim.experiments import ExperimentError, list_experiments, load_experiment, parse  # noqa: E402
from agentic_sim.launch import LaunchError, RunOptions, build  # noqa: E402

GOOD = {
    "name": "x", "description": "d", "backends": ["isaac"],
    "scene": "configs/scenes/lab_current_minimal.yaml", "robots": [{"id": "arm0", "model": "so101"}],
    "teleop": "configs/teleop.yaml", "color_dr": "configs/color_randomization.yaml", "cameras": [],
    "acceptance": ["robustness", {"scenario": "color_dr", "options": {"color_dr": True, "color_dr_seed": 7}}],
}


def test_repo_experiments_are_valid():
    exps = list_experiments()
    assert [e.name for e in exps] == sorted(e.name for e in exps)
    e = load_experiment("minimal_pick_place")
    assert e.supports("isaac") and len(e.robots) == 1
    assert e.scenario("color_dr").options == {"color_dr": True, "color_dr_seed": 7}
    assert all(x.supports("isaac") for x in list_experiments(backend="isaac"))


def test_parse_collects_all_errors():
    bad = dict(GOOD, name="y", backends=["isaac", "gazebo"], scene="configs/nope.yaml", typo=1,
               robots=[{"id": "a", "model": "so101"}, {"id": "b", "model": "so101"}], cameras=[{"id": "top"}],
               acceptance=["missing_scenario", {"scenario": "robustness", "options": {"speed": 2}}])
    with pytest.raises(ExperimentError) as ei:
        parse(bad, "configs/experiments/x.yaml")
    msg = str(ei.value)
    for part in ("must match the file name", "gazebo", "nope.yaml", "unknown key 'typo'", "exactly one arm",
                 "cameras are not supported", "missing_scenario", "unknown option 'speed'"):
        assert part in msg, part


def test_robot_entry_rejects_mount_override():
    bad = dict(GOOD, robots=[{"id": "arm0", "model": "so101", "pos": [0, 0, 0]}])
    with pytest.raises(ExperimentError, match="mount pose comes from the scene"):
        parse(bad, "configs/experiments/x.yaml")


def test_unknown_experiment_lists_available(tmp_path):
    with pytest.raises(ExperimentError, match="minimal_pick_place"):
        load_experiment("does_not_exist")


def test_list_from_directory(tmp_path):
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(GOOD))
    (tmp_path / "y.yaml").write_text(yaml.safe_dump(dict(GOOD, name="y", backends=["mujoco"])))
    assert [e.name for e in list_experiments(str(tmp_path))] == ["x", "y"]
    assert [e.name for e in list_experiments(str(tmp_path), backend="mujoco")] == ["y"]


def test_build_teleop_and_acceptance():
    e = parse(GOOD, "configs/experiments/x.yaml")
    c = build("isaac", e, "teleop", RunOptions(focus=True, color_dr=True, color_dr_seed=3))
    assert c.argv[:3] == ["docker/isaac/run.sh", "/isaac-sim/python.sh", "scripts/isaac_ee_teleop.py"]
    assert c.argv[3:] == ["--scene", GOOD["scene"], "--teleop", GOOD["teleop"], "--color_dr_config", GOOD["color_dr"],
                          "--focus", "--color_dr", "--color_dr_seed", "3"]
    a = build("isaac", e, "acceptance", RunOptions(video=False), scenario="robustness")
    assert a.argv == ["tools/run_acceptance.sh", "robustness", "--no-video"]
    assert "--color_dr" not in a.env["ISAAC_ARGS"].split() and a.env["ACCEPT_OUT"] == "outputs/acceptance/x"
    f = build("isaac", e, "acceptance", RunOptions(), scenario="color_dr")  # forced options win
    assert "--color_dr --color_dr_seed 7" in f.env["ISAAC_ARGS"] and f.notes


def test_build_rejects_invalid_requests():
    e = parse(GOOD, "configs/experiments/x.yaml")
    with pytest.raises(LaunchError, match="does not support"):
        build("mujoco", e, "teleop")
    m = parse(dict(GOOD, backends=["mujoco"]), "configs/experiments/x.yaml")
    with pytest.raises(LaunchError, match="not implemented"):
        build("mujoco", m, "teleop")
    with pytest.raises(LaunchError, match="needs a scenario"):
        build("isaac", e, "acceptance")
    with pytest.raises(KeyError):
        build("isaac", e, "acceptance", scenario="focus_mode")  # not listed in this experiment
    with pytest.raises(LaunchError, match="no colour DR"):
        build("isaac", parse(dict(GOOD, color_dr=None), "configs/experiments/x.yaml"), "teleop",
              RunOptions(color_dr=True))
