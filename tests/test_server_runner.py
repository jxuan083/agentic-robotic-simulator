"""Client-side server flows (tools/sim.py) against a fake simulator server on localhost."""
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agentic_sim.experiments import parse  # noqa: E402
from agentic_sim.launch import RunOptions  # noqa: E402
from agentic_sim.server import runner  # noqa: E402
from agentic_sim.server.client import SimClient  # noqa: E402
from agentic_sim.server.core import SimServer  # noqa: E402

EXP = parse({
    "name": "x", "description": "d", "backends": ["isaac"],
    "scene": "configs/scenes/lab_current_minimal.yaml", "robots": [{"id": "arm0", "model": "so101"}],
    "teleop": "configs/teleop.yaml", "color_dr": "configs/color_randomization.yaml",
    "acceptance": ["robustness", {"scenario": "color_dr", "options": {"color_dr": True, "color_dr_seed": 7}}],
}, "configs/experiments/x.yaml")


class FakeIsaac:
    """Mimics scripts/isaac_sim_server.py: a job runs for a few status polls, then is done."""

    def __init__(self, polls_until_done=3, passed=17, warning=None):
        self.experiment, self.options, self.job = None, None, None
        self.loads = []
        self.polls_until_done, self.passed, self.warning = polls_until_done, passed, warning

    def cmd_status(self):
        if self.job and self.job["state"] == "running":
            self.job["polls"] += 1
            if self.job["polls"] >= self.polls_until_done:
                self.job.update(state="done", passed=self.passed, checks=17)
                if self.warning:
                    self.job["warning"] = self.warning
        return {"experiment": self.experiment, "options": self.options, "job": self.job}

    def cmd_load_experiment(self, name, options=None):
        self.loads.append((name, options))
        self.experiment, self.options = name, options
        return {"loaded": name, "seconds": 0.9}

    def cmd_get_state(self):
        return {"script": {"step": self.job["polls"], "steps": 5, "passed": 0, "checks": 0}} if self.job else {}

    def cmd_run_acceptance(self, scenario):
        self.job = {"scenario": scenario, "state": "running", "polls": 0, "result": f"out/{scenario}.json",
                    "forced_options": {"color_dr": True} if scenario == "color_dr" else {}}
        return dict(self.job)


@pytest.fixture
def fake():
    sims = []

    def start(**kw):
        sim = FakeIsaac(**kw)
        srv = SimServer(sim, port=0, backend="isaac", log=lambda *a: None)
        stop = threading.Event()
        t = threading.Thread(target=lambda: [srv.poll(0.01) for _ in iter(lambda: stop.is_set(), True)])
        t.start()
        sims.append((srv, stop, t))
        return sim, SimClient(*srv.address, timeout=2).__enter__()

    yield start
    for srv, stop, t in sims:
        stop.set()
        t.join(2)
        srv.close()


def test_teleop_loads_with_options(fake):
    sim, c = fake()
    said = []
    runner.teleop(c, EXP, RunOptions(focus=True, color_dr=True, color_dr_seed=3), say=said.append)
    assert sim.loads == [("x", {"focus": True, "color_dr": True, "color_dr_seed": 3, "blocks": 1})]
    assert "已載入 x" in said[0]


def test_acceptance_loads_once_and_follows_job(fake):
    sim, c = fake()
    said = []
    job = runner.acceptance(c, EXP, RunOptions(), "robustness", say=said.append, poll_s=0.01)
    assert job["state"] == "done" and job["passed"] == 17
    assert len(sim.loads) == 1  # experiment was not loaded yet
    job = runner.acceptance(c, EXP, RunOptions(), "robustness", say=said.append, poll_s=0.01)
    assert len(sim.loads) == 1  # already loaded with the same options: no rebuild
    runner.acceptance(c, EXP, RunOptions(focus=True), "robustness", say=said.append, poll_s=0.01)
    assert len(sim.loads) == 2  # options changed: reload
    assert any("17/17" in s for s in said)


def test_acceptance_reports_forced_options_and_warning(fake):
    sim, c = fake(warning="no focus")
    said = []
    job = runner.acceptance(c, EXP, RunOptions(video=False), "color_dr", say=said.append, poll_s=0.01)
    assert job["warning"] == "no focus"
    assert any("固定使用選項" in s for s in said) and any("WARNING" in s for s in said)


def test_acceptance_unknown_scenario(fake):
    _, c = fake()
    with pytest.raises(KeyError):
        runner.acceptance(c, EXP, RunOptions(), "focus_mode", poll_s=0.01)


def test_recorder_started_and_stopped(fake):
    _, c = fake()
    events = []

    class Rec:
        def __init__(self, path):
            events.append(("start", path))

        def stop(self):
            events.append(("stop",))

    runner.acceptance(c, EXP, RunOptions(video=True), "robustness", say=lambda *_: None, poll_s=0.01,
                      recorder_factory=Rec)
    assert events == [("start", os.path.join("outputs", "acceptance", "x", "robustness.mp4")), ("stop",)]


def test_same_options_ignores_seed_when_dr_off_and_defaults_blocks():
    assert runner.same_options({"focus": False, "color_dr": False, "color_dr_seed": 0, "blocks": 1},
                               {"focus": False, "color_dr": False, "color_dr_seed": None, "blocks": 1})
    assert runner.same_options({"focus": False, "color_dr": True, "color_dr_seed": 5},
                               {"focus": False, "color_dr": True, "color_dr_seed": None, "blocks": 1})
    assert not runner.same_options({"color_dr": True, "color_dr_seed": 5}, {"color_dr": True, "color_dr_seed": 6})
    assert not runner.same_options({"blocks": 4}, {"blocks": 1})  # acceptance after the panel set 4 blocks: reload


def test_acceptance_reloads_after_panel_changed_blocks(fake):
    sim, c = fake()
    runner.acceptance(c, EXP, RunOptions(), "robustness", say=lambda *_: None, poll_s=0.01)
    sim.options = dict(sim.options, blocks=4)  # the operator set 4 blocks in the panel
    runner.acceptance(c, EXP, RunOptions(), "robustness", say=lambda *_: None, poll_s=0.01)
    assert len(sim.loads) == 2 and sim.loads[-1][1]["blocks"] == 1
