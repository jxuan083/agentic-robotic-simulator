"""Finals whole-order scene (agentic_sim.task.finals): the task flow in the kinematic fake world of test_task.

Both answers end with the right placement and ledger; the scene removes the shortage colour only at a settled
standby and puts it back elsewhere after the "wait" answer; the question is asked once and worded for the chat.
Camera observations need rendering and are covered by the MuJoCo GUI acceptance (tools/finals.py batch).
"""

import copy
import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

from agentic_sim.chat.zh import Words  # noqa: E402
from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.scene_layout import trays, yaw_quat  # noqa: E402
from agentic_sim.task.finals import layout, resolve  # noqa: E402
from agentic_sim.task.runner import TaskRunner  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402
from test_task import DT, FakeWorld  # noqa: E402

LAB = load("configs/scenes/lab_20261125.yaml")


class FinalsWorld(FakeWorld):
    """The fake world on the lab scene with the finals layout, plus the scene operations the director uses."""

    def __init__(self, out_dir, roles=("red", "blue", "yellow"), seed=11, answer="substitute", move=False):
        path = "configs/tasks/finals_shortage.yaml"
        spec = load(path)
        spec["finals"].update(roles=dict(zip(("shortage", "priority", "substitute"), roles)), answer=answer,
                              perception="sim_truth")
        spec["finals"]["layout"]["seed"] = seed
        spec["finals"]["move_substitute"] = move
        self.spec_path = os.path.join(out_dir, f"finals_{answer}.yaml")
        with open(self.spec_path, "w") as f:
            yaml.safe_dump(spec, f)
        _, colors, rec = layout(copy.deepcopy(LAB), resolve(spec), "configs/teleop.yaml", log=lambda m: None)
        self.chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
        self.ctrl = load_controller(self.chain, load("configs/teleop.yaml"), base_pos=v(LAB["robot"]["base_pos"]),
                                    scene=LAB)
        self.edge = float(v(LAB["block"]["edge"]))
        self.blocks = [np.array(p, float) for p in rec["positions_m"]]
        self.yaws = list(rec["yaw_degs"])
        self.trays = trays(LAB)
        self.attached, self.t = {}, 0.0
        self.q_prev = self.ctrl.state.q_cmd.copy()
        self.removed = set()
        self.runner = TaskRunner(self.ctrl, LAB, [p.copy() for p in self.blocks], colors, DT, log=lambda m: None,
                                 scenario=self.spec_path, world_ops=self)
        self.ex = self.runner.ex

    def block_pose(self, b):
        return self.blocks[b].copy(), np.array(yaw_quat(self.yaws[b]))

    def remove_block(self, b):
        self.blocks[b] = np.array([-1.2, -0.6 + 0.06 * b, 0.4])
        self.removed.add(b)

    def place_block(self, b, xyz, yaw_deg):
        self.blocks[b] = np.array(xyz, float)
        self.yaws[b] = yaw_deg
        self.removed.discard(b)


@pytest.mark.parametrize("answer", ["substitute", "wait"])
def test_both_answers_finish_with_the_right_order(tmp_path, answer):
    w = FinalsWorld(tmp_path, answer=answer)
    w.run(lambda w: w.runner.done, max_s=300.0)
    res = w.runner.result(w.obs(0.0), "fake")
    failed = [c for c in res["checks"] if not c["ok"]]
    assert not failed, failed
    fin = res["finals"]
    assert fin["removed"] and fin["standby"]["ok"]
    if answer == "wait":
        assert fin["restocked"][0]["pose"]["from_removed_mm"] > 1000 * w.edge
        assert w.placed() == {"red": 2, "blue": 1}
    else:
        assert w.placed() == {"red": 1, "blue": 1, "yellow": 1}


def test_substitute_is_looked_for_again_after_the_answer(tmp_path):
    w = FinalsWorld(tmp_path, answer="substitute", move=True)
    w.run(lambda w: w.runner.done, max_s=300.0)
    res = w.runner.result(w.obs(0.0), "fake")
    assert not [c for c in res["checks"] if not c["ok"]]
    moved = res["finals"]["moved"][0]
    t_a = res["finals"]["answer"]["t"]
    looks = [e for e in w.ex.events if e["kind"] == "observation" and e["t"] >= t_a]
    pick = next(e for e in w.ex.events if e["kind"] == "skill_start" and e.get("skill") == "pick" and e["t"] >= t_a)
    assert looks and looks[0]["t"] <= pick["t"]
    seen = looks[0]["poses"][str(pick["block"])]["xy"]
    assert np.linalg.norm(np.subtract(seen, moved["to"]["xy"])) < 1e-3     # found where the scene moved it
    assert w.placed() == {"red": 1, "blue": 1, "yellow": 1}


@pytest.mark.parametrize(("shortage", "seen"), [("red", "red"), ("light_green", "yellow")])
def test_unclear_restock_polls_hold_position(tmp_path, shortage, seen):
    from agentic_sim.task.camera import TaskCamera

    w = FinalsWorld(tmp_path, roles=(shortage, "blue", "purple"), answer="wait")
    w.run(lambda w: w.ex.restock_wait)
    start = len(w.ex.events)
    joints = w.ctrl.state.q_cmd.copy()
    cam = TaskCamera(LAB, None, w.ex.world.trays["storage"], w.edge, w.ctrl, {})
    w.ex.camera = cam
    cam.record = Mock(side_effect=lambda key: (None, {"frame_number": cam.record.call_count}, None))

    # Red case: an unresolved outline remains even after red is put back. Green case: a yellow fit may be green.
    def fits(rec):
        if w.removed:
            return []
        b = w.runner.finals.restocked[0]["block"]
        return [dict(color=seen, xy=list(w.blocks[b][:2]), yaw=0.0, fit=1.0, source="global")]

    cam.tray_fits = fits
    cam.tray_unfitted = lambda rec: [{"xy": [0.22, 0.20]}] if shortage == "red" and rec["frame_number"] <= 4 else []
    cam.view_plan = Mock(side_effect=AssertionError("wrist motion during a restock poll"))

    def polls():
        return [e for e in w.ex.events[start:] if e["kind"] == "observation"]

    w.run(lambda w: len(polls()) >= 4)
    assert polls()[-1]["clear"] is False
    assert all(e["reason"] == "restock_wait" and e["views"] == 1 for e in polls())
    assert w.ex.restock_wait and w.ex.skill is None
    np.testing.assert_allclose(w.ctrl.state.q_cmd, joints)
    assert all(call.args == ("global",) for call in cam.record.call_args_list)
    if shortage == "red":
        w.run(lambda w: w.ex.skill is not None and w.ex.skill.name == "pick")
        assert polls()[-1]["clear"] and w.ex.skill.params["color"] == shortage
        assert w.ex.skill.params["target"]["xy"] == w.runner.finals.restocked[0]["pose"]["xy"]


def test_roles_are_read_from_the_scene_not_fixed(tmp_path):
    w = FinalsWorld(tmp_path, roles=("purple", "orange", "dark_green"), seed=21, answer="substitute")
    w.run(lambda w: w.runner.done, max_s=300.0)
    res = w.runner.result(w.obs(0.0), "fake")
    assert not [c for c in res["checks"] if not c["ok"]]
    first = next(e for e in w.ex.events if e["kind"] == "verified")
    assert first["color"] == "orange" and w.placed() == {"purple": 1, "orange": 1, "dark_green": 1}


def test_finals_question_wording():
    words = Words(load("configs/chat.yaml")["colors"])
    q = {"qkind": "shortage", "id": 1, "options": ["substitute", "wait"], "substitute_candidates": ["yellow"],
         "delivery_sequence": ["blue", "red"], "missing": {"red": 1}}
    assert words.question(q) == "一藍一紅已放好，還缺一紅。要改成黃的，還是等你補紅？（問題 1）"
    assert "看不清" in words.event({"kind": "observation_unclear", "reason": "x"})


def test_resolve_rejects_repeated_roles():
    with pytest.raises(ValueError):
        resolve({"finals": {"roles": {"shortage": "red", "priority": "red", "substitute": "blue"}}})
