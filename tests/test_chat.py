"""Chat layer with fixed model replies (no model service, no physics): intent checks, templates, and full flows
through the real task_* server commands on the kinematic fake world of test_task.py."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.intent import plan, schema  # noqa: E402
from agentic_sim.chat.llm import ReplayLLM, first_json_object  # noqa: E402
from agentic_sim.chat.zh import Words  # noqa: E402
from agentic_sim.config import load  # noqa: E402
from agentic_sim.task.server_cmds import TaskCommands  # noqa: E402
from test_task import SCENE, FakeWorld  # noqa: E402

CFG = load("configs/chat.yaml")
W = Words(CFG["colors"])


def I(intent, items=(), **kw):  # noqa: E743
    d = {"intent": intent, "items": [{"color": c, "count": n} for c, n in items], "question_id": None,
         "choice": None, "substitute_color": None, "xy": None, "reply": ""}
    d.update(kw)
    return d


ST = {"active": True, "colors_in_scene": ["blue", "red", "yellow"], "state": "running",
      "ledger": {"revision": 1, "target": {"red": 3}, "delivered": {"red": 1}, "need": {"red": 2}, "over": {}},
      "holding": None, "question": None}


# ---- intent checks ----------------------------------------------------------------------------------------
def test_plan_total_add_set():
    a = plan(I("order_total", [("red", 1), ("blue", 2)]), ST, W, 8)
    assert (a.kind, a.cmd, a.args) == ("send", "task_order", {"counts": {"red": 1, "blue": 2}, "mode": "total"})
    a = plan(I("order_add", [("blue", 1)]), ST, W, 8)
    assert a.args == {"counts": {"blue": 1}, "mode": "add"}
    st = dict(ST, ledger=dict(ST["ledger"], target={"red": 1, "blue": 2}))
    a = plan(I("order_set", [("blue", 3)]), st, W, 8)       # other colours kept
    assert a.args == {"counts": {"red": 1, "blue": 3}, "mode": "total"}
    a = plan(I("order_set", [("red", 0)]), st, W, 8)        # "no more red"
    assert a.args == {"counts": {"blue": 2}, "mode": "total"}


def test_plan_rejects_and_notes():
    assert plan(I("order_total", [("red", 9)]), ST, W, 8).kind == "say"
    assert plan(I("order_total", []), ST, W, 8).kind == "say"
    a = plan(I("order_total", [("purple", 1)]), ST, W, 8)    # not on the table: sent, with a note
    assert a.kind == "send" and "紫色" in a.text
    assert plan(I("answer", choice="return"), ST, W, 8).text == "目前沒有待回答的問題。"
    assert plan(I("stop"), ST, W, 8).kind == "say" and plan(I("bogus"), ST, W, 8).kind == "say"


def test_plan_answers_are_bound_to_the_open_question():
    q = {"id": 3, "kind": "held_item", "color": "red", "block": 1, "options": ["return", "place_at", "deliver"]}
    st = dict(ST, question=q)
    assert plan(I("answer", question_id=3, choice="return"), st, W, 8).args == {"id": 3, "choice": "return"}
    assert plan(I("answer", question_id=None, choice="return"), st, W, 8).args["id"] == 3
    assert "過期" in plan(I("answer", question_id=2, choice="return"), st, W, 8).text
    assert plan(I("answer", question_id=3, choice="substitute"), st, W, 8).kind == "say"   # not an option here
    assert plan(I("answer", question_id=3, choice="place_at"), st, W, 8).kind == "say"     # no xy: ask
    a = plan(I("answer", question_id=3, choice="place_at", xy=[0.3, 0.05]), st, W, 8)
    assert a.args["xy"] == [0.3, 0.05]
    q2 = {"id": 4, "kind": "shortage", "options": ["substitute", "reduce", "wait"], "missing": {"blue": 1}}
    st2 = dict(ST, question=q2)
    assert plan(I("answer", question_id=4, choice="substitute", substitute_color="yellow"), st2, W, 8).args == \
        {"id": 4, "choice": "substitute", "color": "yellow"}
    assert plan(I("answer", question_id=4, choice="substitute", substitute_color="purple"), st2, W, 8).kind == "say"


def test_schema_and_json_extraction():
    s = schema(list(CFG["colors"]), 8)
    assert s["properties"]["items"]["items"]["properties"]["color"]["enum"] == list(CFG["colors"])
    assert first_json_object('好的 ```json\n{"intent": "status", "a": "}"}\n```') == {"intent": "status", "a": "}"}


def test_templates():
    assert W.counts({"red": 1, "blue": 2}) == "紅色 1、藍色 2"
    e = {"kind": "over_delivery", "color": "red", "over": 1, "delivered": 2, "target": 1}
    assert "取回" in W.event(e)
    assert W.event({"kind": "phase"}) is None
    q = {"kind": "shortage", "id": 2, "delivered": {"red": 1, "blue": 1}, "missing": {"blue": 1},
         "available": {"red": 2, "yellow": 1}, "options": ["substitute", "reduce", "wait"]}
    t = W.question(q)
    assert "還缺 藍色 1" in t and "黃色 1" in t and "問題 2" in t


# ---- full flows on the fake world through the real server commands ------------------------------------------
class FakeSession:
    def __init__(self, world):
        self.task, self.w = world.runner, world
        self.block_color_names = world.ex.world.colors
        self.scene_cfg = SCENE

    def settings(self):
        return {"blocks": len(self.block_color_names)}

    def start_task(self):
        return self.task

    def stop_task(self):
        self.task = None


class FakeServer(TaskCommands):
    def __init__(self, world):
        self.session, self.job = FakeSession(world), None

    def _require_session(self):
        pass

    def _job_running(self):
        return False

    def __call__(self, cmd, **args):
        return getattr(self, f"cmd_{cmd}")(**args)


def run_until(w, cond, max_s=200.0):
    w.run(lambda w: cond(w), max_s=w.t + max_s)


def test_hold_return_dialogue_replay():
    w = FakeWorld(["red", "red", "red", "blue", "blue", "yellow"])
    srv = FakeServer(w)
    llm = ReplayLLM({"我要三顆紅色": I("order_total", [("red", 3)]),
                     "總共一紅兩藍": I("order_total", [("red", 1), ("blue", 2)]),
                     "放回去": I("answer", question_id=1, choice="return"),
                     "現在做到哪？": I("status")})
    agent = ChatAgent(srv, llm, CFG)
    reports = []
    assert agent.handle("我要三顆紅色") == ["好的，需求改為 紅色 3。"]
    run_until(w, lambda w: w.ex.skill is not None and w.ex.skill.name == "place" and w.ex.skill.params["nth"] == 2
              and w.ex.skill.phase == "transport")
    agent.handle("總共一紅兩藍")
    run_until(w, lambda w: w.ex.question is not None)
    reports += agent.poll_reports()
    assert any("需求已生效（第 2 版）" in r for r in reports)
    assert any("問題 1" in r and "照樣送出" in r for r in reports)
    assert "待回答" in agent.handle("現在做到哪？")[0]
    agent.handle("放回去")
    run_until(w, lambda w: w.ex.state == "done")
    reports += agent.poll_reports()
    assert any(r.startswith("完成") for r in reports)
    led = w.ex.ledger.snapshot()
    assert led["delivered"] == {"red": 1, "blue": 2} and led["revision"] == 2 and led["done"]
    sent = [a["sent"] for r in agent.transcript if "attempts" in r for a in r["attempts"] if "sent" in a]
    assert [s["cmd"] for s in sent] == ["task_order", "task_order", "task_answer"]
    assert sent[2]["args"] == {"id": 1, "choice": "return"}
    assert len({s["request_id"] for s in sent}) == 3


def test_shortage_dialogue_replay():
    w = FakeWorld(["red", "red", "red", "blue", "yellow"])
    srv = FakeServer(w)
    llm = ReplayLLM([I("order_total", [("red", 1), ("blue", 2)]),
                     I("answer", question_id=1, choice="substitute", substitute_color="yellow")])
    agent = ChatAgent(srv, llm, CFG)
    agent.handle("一紅兩藍")
    run_until(w, lambda w: w.ex.question is not None)
    rep = agent.poll_reports()
    assert any("還缺 藍色 1" in r and "黃色 1" in r for r in rep)
    agent.handle("換成黃色")
    run_until(w, lambda w: w.ex.state == "done")
    assert w.ex.ledger.snapshot()["delivered"] == {"red": 1, "blue": 1, "yellow": 1}
    assert w.ex.ledger.revision == 2


def test_transport_retry_applies_once_and_repeat_adds_twice():
    w = FakeWorld(["red", "blue", "blue", "blue"])
    srv = FakeServer(w)
    lost = {"n": 1}

    def flaky(cmd, **args):                 # the server applies the request, then the reply is lost once
        out = srv(cmd, **args)
        if cmd == "task_order" and lost["n"]:
            lost["n"] -= 1
            raise ConnectionError("reply lost")
        return out

    llm = ReplayLLM([I("order_add", [("blue", 1)]), I("order_add", [("blue", 1)])])
    agent = ChatAgent(flaky, llm, CFG)
    agent.handle("再加一藍")
    w.step()
    assert w.ex.ledger.target == {"blue": 1} and w.ex.ledger.revision == 1          # applied once
    sent = agent.transcript[-1]["attempts"][-1]["sent"]
    assert sent["response"].get("duplicate") is True
    agent.handle("再加一藍")                 # said again on purpose: a new request, applied again
    w.step()
    assert w.ex.ledger.target == {"blue": 2} and w.ex.ledger.revision == 2


def test_state_change_while_thinking_replans():
    w = FakeWorld(["red", "red", "blue", "blue"])
    srv = FakeServer(w)
    srv("task_order", counts={"red": 2})
    w.step()

    class Racing(ReplayLLM):
        def complete(self, messages, schema=None):
            if self.i == 0:                  # someone else changes the order while the model is thinking
                srv("task_order", counts={"blue": 1}, mode="add")
                w.step()
            return super().complete(messages, schema)

    llm = Racing([I("order_set", [("red", 1)]), I("order_set", [("red", 1)])])
    agent = ChatAgent(srv, llm, CFG)
    agent.handle("紅色改成一顆")
    w.step()
    rec = agent.transcript[-1]
    assert rec["attempts"][0].get("stale") and "sent" not in rec["attempts"][0]
    assert rec["attempts"][1]["sent"]["args"] == {"counts": {"red": 1, "blue": 1}, "mode": "total"}   # blue kept
    assert w.ex.ledger.target == {"red": 1, "blue": 1}


def test_model_service_down_sends_nothing():
    class Down:
        def complete(self, messages, schema=None):
            raise OSError("connection refused")

    w = FakeWorld(["red"])
    srv = FakeServer(w)
    agent = ChatAgent(srv, Down(), CFG)
    assert "沒有送出" in agent.handle("我要一顆紅色")[0]
    w.step()
    assert w.ex.ledger.revision == 0


def test_inactive_state_lists_scene_colours():
    w = FakeWorld(["red", "blue", "blue"])
    srv = FakeServer(w)
    srv.session.task = None
    st = srv("task_state")
    assert st["active"] is False and st["colors_in_scene"] == ["blue", "red"] and st["storage"] == {"red": 1, "blue": 2}


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_input_is_ignored(text):
    w = FakeWorld(["red"])
    assert ChatAgent(FakeServer(w), ReplayLLM([]), CFG).handle(text) == []


def test_ambiguity_guard():
    st = dict(ST, ledger=dict(ST["ledger"], target={"red": 1, "blue": 2}))
    a = plan(I("order_total", [("red", 1), ("blue", 2), ("yellow", 3)]), st, W, 8, "改成三黃")
    assert a.kind == "say" and "保留" in a.text
    assert plan(I("order_add", [("blue", 2)]), st, W, 8, "再來幾顆藍色").kind == "say"
    assert plan(I("order_set", [("blue", 3)]), st, W, 8, "藍色改成三顆").kind == "send"      # colour named first
    assert plan(I("order_total", [("red", 1), ("blue", 2)]), st, W, 8, "全部改成一顆紅色兩顆藍色").kind == "send"
    assert plan(I("order_total", [("blue", 2)]), dict(ST, ledger=dict(ST["ledger"], target={})), W, 8,
                "改成兩藍").kind == "send"                                              # nothing else in the order
    assert plan(I("order_add", [("blue", 2)]), st, W, 8, "再加兩顆藍色").kind == "send"
