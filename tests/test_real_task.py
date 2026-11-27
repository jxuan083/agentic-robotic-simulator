"""The task executor on the real-arm adapter (agentic_sim.real_task) with a simulated table (FakeArm)."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_task.arm import FakeArm  # noqa: E402
from agentic_sim.real_task.session import RealHandler, RealSession  # noqa: E402
from agentic_sim.real_task.tracker import Tracker  # noqa: E402

FINALS = [{"color": "blue", "x": 0.20, "y": 0.10}, {"color": "blue", "x": 0.26, "y": 0.18},
          {"color": "red", "x": 0.22, "y": 0.15}, {"color": "yellow", "x": 0.30, "y": 0.12}]


def run_until(session, cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        session.step()
        if cond():
            return True
        time.sleep(0.002)
    return False


def kinds(session):
    return [e["kind"] for e in session.task.ex.events]


def test_tracker_keeps_ids_and_takes_restocked_blocks():
    t = Tracker()
    t.update([{"color": "red", "tray": "pink", "x": 0.2, "y": 0.1, "z_top": 0.03}])
    t.update([{"color": "red", "tray": "pink", "x": 0.205, "y": 0.1, "z_top": 0.03}])
    assert t.colors == ["red"] and t.tray == ["storage"]
    t.update([])                                     # taken away
    assert t.tray == [None]
    t.update([{"color": "red", "tray": "pink", "x": 0.30, "y": 0.2, "z_top": 0.03}])   # restocked elsewhere
    assert t.colors == ["red"] and t.tray == ["storage"] and abs(t.pos[0][0] - 0.30) < 1e-9


def test_finals_order_on_the_real_adapter_delivers_asks_on_shortage_and_substitutes():
    arm = FakeArm(FINALS, seconds=0.01)
    s = RealSession(arm, log=lambda *a: None)
    h = RealHandler(s, fake=True)
    assert h.cmd_task_state()["storage"] == {"blue": 2, "red": 1, "yellow": 1}
    h.cmd_task_order(counts={"red": 1, "blue": 2}, mode="total")
    removed = []

    def take_last_blue(e):        # the operator takes the last blue once one red and one blue are delivered
        if e["kind"] == "verified" and not removed and sum(e["ledger"]["delivered"].values()) == 2:
            removed.append(arm.remove("blue"))
    s.task.ex.hooks.append(take_last_blue)
    assert run_until(s, lambda: s.task.ex.question is not None)
    q = h.cmd_task_state()["question"]
    assert q["kind"] == "shortage" and q["missing"] == {"blue": 1} and "yellow" in q["substitute_candidates"]
    seq = s.task.ex.delivery_seq
    assert seq == ["red", "blue"]                    # red first (the order's first colour), then a blue
    h.cmd_task_answer(id=q["id"], choice="substitute", color="yellow")
    assert run_until(s, lambda: s.task.ex.state == "done")
    st = h.cmd_task_state()
    assert st["ledger"]["delivered"] == {"red": 1, "blue": 1, "yellow": 1}
    assert st["placement"] == {"red": 1, "blue": 1, "yellow": 1} and st["storage"] == {}
    assert arm.at_park()
    ev = h.cmd_task_events(since=0)["events"]
    assert [e["skill"] for e in ev if e["kind"] == "skill_start" and e["skill"] in ("pick", "place")] == \
        ["pick", "place"] * 3
    # each delivery lands in its own slot
    xs = sorted((round(b["x"], 2), round(b["y"], 2)) for b in arm.blocks)
    assert len(set(xs)) == 3


def test_a_missed_grasp_is_retried_and_waiting_for_a_restock_picks_the_new_block():
    arm = FakeArm([{"color": "red", "x": 0.22, "y": 0.15}], seconds=0.01, misses=["red"])
    s = RealSession(arm, log=lambda *a: None)
    h = RealHandler(s, fake=True)
    h.cmd_task_order(counts={"red": 2}, mode="total")
    assert run_until(s, lambda: s.task.ex.question is not None)
    assert s.task.ex.ledger.delivered == {"red": 1}
    assert [r["status"] for r in s.task.ex.results if r["skill"] == "pick"][:2] == ["failed", "succeeded"]
    q = s.task.ex.question
    h.cmd_task_answer(id=q["id"], choice="wait")
    time.sleep(0.05)
    h.cmd_fake_scene("add", "red", [0.28, 0.20])     # the operator restocks a red
    assert run_until(s, lambda: s.task.ex.state == "done")
    assert s.task.ex.ledger.delivered == {"red": 2}


def test_finals_standby_waits_for_the_operator_before_looking_again():
    arm = FakeArm(FINALS, seconds=0.01)
    s = RealSession(arm, log=lambda *a: None, standby_after=2)
    h = RealHandler(s, fake=True)
    h.cmd_task_order(counts={"red": 1, "blue": 2}, mode="total")
    ex = s.task.ex
    assert run_until(s, lambda: ex.waiting_operator and ex.skill is None)
    assert ex.delivery_seq == ["red", "blue"] and arm.at_park()
    looks = ex.observations
    run_until(s, lambda: False, timeout=0.3)          # nothing happens while the operator works
    assert ex.observations == looks and ex.skill is None
    arm.remove("blue")                                # the shortage colour is taken away at standby
    h.cmd_task_continue()
    assert run_until(s, lambda: ex.question is not None)
    assert ex.question["kind"] == "shortage" and ex.question["missing"] == {"blue": 1}
    assert "waiting_operator" in kinds(s) and "standby_reached" in kinds(s)
