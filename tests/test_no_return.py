"""Finals rule no_return (docs/real-progress.md): the real arm only picks from storage into placement. An
over-delivery is reported and left, a held block the order no longer needs is delivered anyway or held until someone
takes it off the gripper; nothing goes back to the storage tray. Off by default: the original behaviour."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.llm import ReplayLLM  # noqa: E402
from chat_web import quick_replies  # noqa: E402
from test_chat import CFG, W, FakeServer, I, run_until  # noqa: E402
from test_task import FakeWorld  # noqa: E402

NR = {**CFG, "no_return": True}
COLORS = ["red", "red", "red", "blue", "blue", "yellow"]


def kinds(w, k):
    return [e for e in w.ex.events if e["kind"] == k]


def carrying_red_2(w):
    return w.ex.skill is not None and w.ex.skill.name == "place" and w.ex.skill.params["nth"] == 2 \
        and w.ex.skill.phase == "transport"


def test_hold_then_removed_through_the_chat():
    w = FakeWorld(COLORS)
    w.ex.no_return = True
    llm = ReplayLLM({"我要三顆紅色": I("order_total", [("red", 3)]),
                     "總共一紅兩藍": I("order_total", [("red", 1), ("blue", 2)]),
                     "先拿著我來拿": I("answer", question_id=1, choice="hold"),
                     "取下了": I("answer", question_id=2, choice="removed")})
    agent = ChatAgent(FakeServer(w), llm, NR)
    agent.handle("我要三顆紅色")
    run_until(w, carrying_red_2)
    agent.handle("總共一紅兩藍")
    run_until(w, lambda w: w.ex.question is not None)
    reports = agent.poll_reports()
    q = w.ex.question
    assert q["kind"] == "held_item" and q["options"] == ["deliver", "hold"]
    assert any("手臂不會放回粉盤" in r and "問題 1" in r for r in reports)
    assert quick_replies(q, W) == ["照樣送出", "停住等我取下"]
    assert agent.handle("先拿著我來拿") == ["好的，停住等你取下。"]
    run_until(w, lambda w: w.ex.question is not None and w.ex.question["kind"] == "removal")
    reports += agent.poll_reports()
    assert any("請從夾爪取下" in r and "問題 2" in r for r in reports)
    assert quick_replies(w.ex.question, W) == ["取下了"]
    agent.handle("取下了")
    run_until(w, lambda w: w.ex.state == "done")
    reports += agent.poll_reports()
    assert any("紅色積木已取下" in r for r in reports)
    assert not any("放回原位" in r for r in reports)
    assert not [e for e in kinds(w, "skill_start") if e.get("purpose") in ("return", "retrieve")]
    led = w.ex.ledger.snapshot()
    assert led["delivered"] == {"red": 1, "blue": 2} and led["done"] and w.placed() == {"red": 1, "blue": 2}


def test_over_delivery_is_reported_and_left():
    w = FakeWorld(COLORS)
    w.ex.no_return = True
    agent = ChatAgent(FakeServer(w), ReplayLLM([]), NR)
    w.ex.order({"red": 3})
    run_until(w, lambda w: w.ex.ledger.delivered.get("red", 0) == 2)
    w.ex.order({"red": 1, "blue": 2})
    run_until(w, lambda w: w.ex.state == "done")
    reports = agent.poll_reports()
    assert any("比需求多 1 顆，手臂不會夾回" in r for r in reports)
    assert not [e for e in kinds(w, "skill_start") if e.get("purpose") == "retrieve"]
    assert w.placed() == {"red": 2, "blue": 2} and w.ex.ledger.snapshot()["over"] == {"red": 1}


def test_a_new_order_that_needs_the_held_block_ends_the_removal_wait():
    w = FakeWorld(COLORS)
    w.ex.no_return = True
    w.ex.order({"red": 3})
    run_until(w, carrying_red_2)
    w.ex.order({"red": 1, "blue": 2})
    run_until(w, lambda w: w.ex.question is not None)
    w.ex.answer(w.ex.question["id"], "hold")
    run_until(w, lambda w: w.ex.question is not None and w.ex.question["kind"] == "removal")
    w.ex.order({"red": 2, "blue": 2})
    run_until(w, lambda w: w.ex.state == "done")
    assert kinds(w, "question_dropped") and not kinds(w, "block_removed")
    assert w.placed() == {"red": 2, "blue": 2}


def test_off_by_default_and_the_measured_schema_is_unchanged():
    w = FakeWorld(COLORS)
    assert w.ex.no_return is False
    base, on = ChatAgent(lambda *a, **k: {}, ReplayLLM([]), CFG), ChatAgent(lambda *a, **k: {}, ReplayLLM([]), NR)
    enum = lambda a: a.schema["properties"]["choice"]["anyOf"][0]["enum"]  # noqa: E731
    assert "hold" not in enum(base) and "removed" not in enum(base) and "removed＝" not in base.system
    assert enum(on)[-2:] == ["hold", "removed"] and "removed＝" in on.system
    w.ex.order({"red": 3})
    run_until(w, carrying_red_2)
    w.ex.order({"red": 1, "blue": 2})
    run_until(w, lambda w: w.ex.question is not None)
    assert w.ex.question["options"] == ["return", "place_at", "deliver"]
