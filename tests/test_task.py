"""Requirement ledger, task executor and scenarios (issue #5), without a physics engine.

The fake world tracks the commanded joints exactly (EE = FK of the command), attaches a block when the jaw closes
around it and sets it down on the floor below when the jaw opens. That is enough to check the executor's decisions
(cancel by phase, hold and ask, retrieve, shortage, revisions); the physics is covered by the GUI acceptance runs.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.scene_layout import block_in_tray, block_layout, trays  # noqa: E402
from agentic_sim.task.ledger import Ledger  # noqa: E402
from agentic_sim.task.runner import TaskRunner, default_color_names  # noqa: E402
from agentic_sim.task.skills import Obs  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402

SCENE = load("configs/scenes/lab_current_minimal.yaml")
DT = 1.0 / 60.0


# ---- ledger ------------------------------------------------------------------------------------------------
def test_ledger_total_is_a_new_total_and_add_adds():
    led = Ledger()
    led.set_order({"red": 3}, "total", 0.0)
    led.verified("red", +1, 1.0, block=0)
    assert led.need("red") == 2
    led.set_order({"red": 1, "blue": 2}, "total", 2.0)
    assert led.revision == 2 and led.need("red") == 0 and led.need("blue") == 2 and led.over("red") == 0
    led.set_order({"blue": 1}, "add", 3.0)
    assert led.target == {"red": 1, "blue": 3}
    led.set_order({"blue": 1}, "total", 4.0)        # red no longer wanted: the delivered one is over
    assert led.over("red") == 1 and led.overs() == {"red": 1}
    led.accept_extra("red", 1, 5.0)
    assert led.over("red") == 0
    with pytest.raises(ValueError):
        led.set_order({"red": -1}, "total", 6.0)
    with pytest.raises(ValueError):
        led.set_order({"red": 1}, "replace", 6.0)


def test_ledger_replay_rebuilds_the_same_state():
    led = Ledger()
    led.set_order({"red": 3}, "total", 0.0)
    led.verified("red", +1, 1.0, 0)
    led.set_order({"red": 1, "blue": 2}, "total", 2.0)
    led.verified("blue", +1, 3.0, 3)
    led.mark_uncertain("blue", 4.0, 4, "not_verified")
    other = [{"t": 2.5, "rev": 2, "kind": "phase", "skill": "pick"}]   # executor events are skipped
    again = Ledger.replay(led.events[:3] + other + led.events[3:])
    assert again.snapshot() == led.snapshot()
    bad = [dict(e) for e in led.events]
    bad[1]["ledger"] = dict(bad[1]["ledger"], delivered={"red": 2})
    with pytest.raises(ValueError):
        Ledger.replay(bad)


# ---- fake world ------------------------------------------------------------------------------------------
class FakeWorld:
    def __init__(self, colors, scenario=None):
        self.chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
        self.ctrl = load_controller(self.chain, load("configs/teleop.yaml"), base_pos=v(SCENE["robot"]["base_pos"]))
        lay = block_layout(SCENE, colors=colors)
        self.edge = lay.edge
        self.blocks = [np.array(p, float) for p in lay.init_pos]
        self.trays = trays(SCENE)
        self.attached = {}          # block -> offset from the EE
        self.t = 0.0
        self.q_prev = self.ctrl.state.q_cmd.copy()
        self.runner = TaskRunner(self.ctrl, SCENE, lay.init_pos, colors, DT, log=lambda m: None, scenario=scenario)
        self.ex = self.runner.ex

    def ee(self):
        return self.chain.ee_pose_mount(self.ctrl.state.q_cmd)[:3, 3].copy()

    def obs(self, qd):
        speed = [0.0] * len(self.blocks)
        return Obs(t=self.t, ee=self.ee(), qd_max=qd, jaw=self.ctrl.state.gripper, blocks=[b.copy() for b in self.blocks],
                   block_yaw=[0.0] * len(self.blocks), block_speed=speed)

    def floor_z(self, p):
        for t in self.trays.values():
            if block_in_tray([p[0], p[1], 0.0], t, 0.0):
                return t["floor_top"] + self.edge / 2
        return self.edge / 2

    def step(self):
        s = self.ctrl.state
        qd = float(np.max(np.abs(s.q_cmd - self.q_prev))) / DT
        self.q_prev = s.q_cmd.copy()
        self.runner.step(self.obs(qd))
        self.ctrl.joint_command(DT)
        ee = self.ee()
        g = s.gripper
        if g < 0.18:            # closed onto whatever is between the fingers
            for i, p in enumerate(self.blocks):
                if i not in self.attached and np.linalg.norm(p - ee) < 0.02:
                    self.attached[i] = p - ee
        elif g > 0.25:
            for i in list(self.attached):
                p = self.blocks[i]
                p[2] = self.floor_z(p)
                del self.attached[i]
        for i, off in self.attached.items():
            self.blocks[i] = ee + off
        self.t += DT

    def run(self, until, max_s=200.0):
        while self.t < max_s and not until(self):
            self.step()
        assert self.t < max_s, f"timed out: state {self.ex.state} skill {self.ex.skill and self.ex.skill.phase}"

    def placed(self):
        out = {}
        for i, p in enumerate(self.blocks):
            if block_in_tray(p, self.trays["placement_beige"], self.edge):
                c = self.ex.world.colors[i]
                out[c] = out.get(c, 0) + 1
        return out


COLORS = ["red", "red", "red", "blue", "blue", "yellow"]


def test_default_colour_names():
    assert default_color_names(SCENE, 3) == ["red", "orange", "yellow"]


def test_change_between_picks_keeps_old_revision_out():
    w = FakeWorld(COLORS)
    w.ex.order({"red": 3})
    w.run(lambda w: w.ex.ledger.delivered.get("red") == 1)
    w.ex.order({"red": 1, "blue": 2}, "total")
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"red": 1, "blue": 2}
    assert np.linalg.norm(w.blocks[1] - w.ex.world.init_pos[1]) < 1e-9     # red 2 never touched
    t_change = [e["t"] for e in w.ex.events if e["kind"] == "order"][1]
    assert all(r == 2 for t, r in w.ex.cmd_log if t >= t_change)


def test_cancel_while_descending_retreats_without_the_block():
    w = FakeWorld(COLORS)
    w.ex.order({"red": 3})
    w.run(lambda w: w.ex.skill is not None and w.ex.skill.name == "pick" and w.ex.skill.params["block"] == 1
          and w.ex.skill.phase == "descend")
    w.ex.order({"red": 1, "blue": 2})
    w.run(lambda w: any(e["kind"] == "skill_done" and e["status"] == "cancelled" for e in w.ex.events))
    assert w.ex.holding is None and 1 not in w.attached
    assert w.ctrl.state.pos[2] == pytest.approx(w.ex.cfg["motion"]["safe_z"], abs=1e-6)   # went straight up
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"red": 1, "blue": 2}
    assert w.ex.cancels and w.ex.cancels[0]["phase"] == "descend"


def _carry_red2_then_change(w):
    w.ex.order({"red": 3})
    w.run(lambda w: w.ex.skill is not None and w.ex.skill.name == "place" and w.ex.skill.params["block"] == 1
          and w.ex.skill.phase == "transport")
    for _ in range(20):
        w.step()
    w.ex.order({"red": 1, "blue": 2})
    w.run(lambda w: w.ex.question is not None)


def test_holding_an_unneeded_block_stops_and_asks_then_returns_it():
    w = FakeWorld(COLORS)
    _carry_red2_then_change(w)
    q = w.ex.question
    assert q["kind"] == "held_item" and q["block"] == 1 and set(q["options"]) == {"return", "place_at", "deliver"}
    pos0 = w.ctrl.state.pos.copy()
    for _ in range(120):                          # no answer: nothing moves, the block stays in the gripper
        w.step()
    assert np.allclose(w.ctrl.state.pos, pos0) and 1 in w.attached
    w.ex.answer(q["id"] + 7, "return")            # wrong id: rejected, still waiting
    w.ex.answer(q["id"], "fly away")              # not an option: rejected
    w.step()
    assert w.ex.question is not None
    assert [e["reason"] for e in w.ex.events if e["kind"] == "answer_rejected"][0] == "no such open question"
    w.ex.answer(q["id"], "return")
    w.run(lambda w: w.ex.state == "done")
    assert block_in_tray(w.blocks[1], w.trays["storage_pink"], w.edge)
    assert np.linalg.norm(w.blocks[1][:2] - w.ex.world.init_pos[1][:2]) < 0.01
    assert w.placed() == {"red": 1, "blue": 2}


def test_deliver_anyway_is_kept_and_not_retrieved():
    w = FakeWorld(COLORS)
    _carry_red2_then_change(w)
    w.ex.answer(w.ex.question["id"], "deliver")
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"red": 2, "blue": 2}
    assert not [e for e in w.ex.events if e["kind"] == "skill_start" and e.get("purpose") == "retrieve"]
    over = [e for e in w.ex.events if e["kind"] == "over_delivery"]
    assert over and over[0]["accepted"] == 1 and over[0]["over"] == 0


def test_order_change_back_while_holding_delivers_without_asking_again():
    w = FakeWorld(COLORS)
    _carry_red2_then_change(w)
    w.ex.order({"red": 2, "blue": 2})            # the held red is needed again: the question is dropped
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"red": 2, "blue": 2}
    assert [e for e in w.ex.events if e["kind"] == "question_dropped"]


def test_over_delivery_is_retrieved():
    w = FakeWorld(COLORS)
    w.ex.order({"red": 3})
    w.run(lambda w: w.ex.ledger.delivered.get("red") == 2)
    w.ex.order({"red": 1, "blue": 2})
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"red": 1, "blue": 2}
    assert w.ex.ledger.snapshot()["delivered"] == {"red": 1, "blue": 2}
    assert [e for e in w.ex.events if e["kind"] == "over_delivery" and e["over"] == 1]


def test_shortage_asks_and_never_substitutes_by_itself():
    w = FakeWorld(["red", "red", "red", "blue", "yellow"])
    w.ex.order({"red": 1, "blue": 2})
    w.run(lambda w: w.ex.question is not None)
    q = w.ex.question
    assert q["kind"] == "shortage" and q["missing"] == {"blue": 1} and q["available"] == {"red": 2, "yellow": 1}
    for _ in range(300):
        w.step()
    assert w.ex.skill is None or w.ex.skill.name != "pick"
    assert np.linalg.norm(w.blocks[4] - w.ex.world.init_pos[4]) < 1e-9
    w.ex.answer(q["id"], "substitute", color="yellow")
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"red": 1, "blue": 1, "yellow": 1}
    assert w.ex.ledger.target == {"red": 1, "blue": 1, "yellow": 1}


def test_shortage_reduce_and_wait():
    w = FakeWorld(["red", "blue"])
    w.ex.order({"blue": 2})
    w.run(lambda w: w.ex.question is not None)
    w.ex.answer(w.ex.question["id"], "wait")
    for _ in range(120):
        w.step()
    assert w.ex.question is None and w.ex.state == "waiting"
    w.ex.order({"blue": 1})                       # a new order ends the wait
    w.run(lambda w: w.ex.state == "done")
    assert w.placed() == {"blue": 1}


@pytest.mark.parametrize("name", ["change_between_picks", "change_during_descend", "hold_return", "hold_deliver",
                                  "retrieve_over_delivery", "shortage_substitute"])
def test_scenarios_pass_in_the_fake_world(name):
    path = f"configs/tasks/{name}.yaml"
    w = FakeWorld(load(path)["blocks"], scenario=path)
    w.run(lambda w: w.runner.done, max_s=250.0)
    res = w.runner.result(w.obs(0.0), "fake")
    failed = [c for c in res["checks"] if not c["ok"]]
    # hold_still / stop latency depend on real dynamics; the fake world tracks the command exactly
    assert not failed, failed


# ---- protocol / options ------------------------------------------------------------------------------------
def test_task_protocol_commands_and_block_colour_option():
    from agentic_sim.experiments import load_experiment
    from agentic_sim.launch import RunOptions, build
    from agentic_sim.server import protocol as P
    from agentic_sim.server.runner import same_options, session_options
    P.encode_request(1, "task_order", counts={"red": 1, "blue": 2}, mode="total")
    P.encode_request(2, "task_answer", id=1, choice="substitute", color="yellow")
    P.encode_request(3, "task_answer", id=1, choice="place_at", xy=[0.3, 0.05])
    for cmd in ("task_state", "task_stop"):
        P.encode_request(4, cmd)
    with pytest.raises(P.ProtocolError):
        P.encode_request(5, "task_order", mode="total")          # counts missing
    with pytest.raises(P.ProtocolError):
        P.encode_request(6, "task_answer", id="1", choice="return")
    opts = RunOptions(block_colors=["red", "blue"], blocks=2)
    assert session_options(opts)["block_colors"] == ["red", "blue"]
    assert "block_colors" not in session_options(RunOptions())
    assert not same_options({"blocks": 2, "block_colors": ["red", "red"]}, session_options(opts))
    assert same_options({"blocks": 2, "block_colors": ["red", "blue"]}, session_options(opts))
    cmd = build("mujoco", load_experiment("minimal_pick_place"), "teleop", opts)
    assert "--block_colors" in cmd.argv and "red,blue" in cmd.argv and "--blocks" not in cmd.argv


def test_unknown_block_colour_is_rejected():
    with pytest.raises(ValueError):
        block_layout(SCENE, colors=["red", "magenta"])
    lay = block_layout(SCENE, colors=["blue", "red"])
    assert len(lay.init_pos) == 2 and lay.display_rgb[0] != lay.display_rgb[1]
