"""Chat layer with fixed model replies (no model service, no physics): intent checks, templates, and full flows
through the real task_* server commands on the kinematic fake world of test_task.py."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.intent import plan, schema  # noqa: E402
from agentic_sim.chat.llm import ReplayLLM, ServiceError, first_json_object  # noqa: E402
from agentic_sim.chat.scoring import summarize, verdict  # noqa: E402
from agentic_sim.task.ledger import Ledger  # noqa: E402
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
    a = plan(I("order_set", [("blue", 3)]), st, W, 8)       # only blue changes, applied by the executor (mode set)
    assert a.args == {"counts": {"blue": 3}, "mode": "set"} and a.text == "好的，藍色改為 3 顆，其他顏色不變。"
    a = plan(I("order_set", [("red", 0)]), st, W, 8)        # "no more red"
    assert a.args == {"counts": {"red": 0}, "mode": "set"} and a.text == "好的，不要紅色，其他顏色不變。"


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
    assert W.event({"kind": "skill_start", "skill": "pick"}) is None      # progress is a separate line kind
    assert W.progress({"kind": "skill_start", "skill": "pick", "color": "red", "nth": 2, "purpose": "deliver"}) \
        == "正在夾取第二顆紅色積木…"
    assert "取回" in W.progress({"kind": "skill_start", "skill": "pick", "color": "blue", "purpose": "retrieve"})
    assert "放回原位" in W.progress({"kind": "skill_start", "skill": "place", "color": "red", "purpose": "return"})
    assert W.progress({"kind": "skill_start", "skill": "home"}) == "回到待命位置…"
    assert W.progress({"kind": "order"}) is None
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
    prog = [r for r in reports if r.startswith("[進度]")]
    assert "[進度] 正在夾取第一顆紅色積木…" in prog and "[進度] 把紅色積木放回原位…" in prog
    assert "[進度] 正在夾取第一顆藍色積木…" in prog and prog[-1] == "[進度] 回到待命位置…"
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
    assert rec["attempts"][1]["sent"]["args"] == {"counts": {"red": 1}, "mode": "set"}
    assert w.ex.ledger.target == {"red": 1, "blue": 1}


def test_model_service_down_sends_nothing():
    class Down:
        def complete(self, messages, schema=None):
            raise ServiceError("model service not reachable: connection refused")

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


# ---- PR #8 review cases ------------------------------------------------------------------------------------
HELD = {"kind": "held_item", "color": "red", "block": 1, "options": ["return", "place_at", "deliver"]}


class StubServer:
    """task_state from a dict the test changes; commands are recorded, not applied."""
    def __init__(self, st):
        self.st, self.sent = st, []

    def __call__(self, cmd, request_id=None, **args):
        if cmd == "task_state":
            return json.loads(json.dumps(self.st))
        if cmd == "task_events":
            return {"next": 0, "events": []}
        self.sent.append({"cmd": cmd, "request_id": request_id, **args})
        return {"queued": True}


def _held_state(qid):
    return dict(ST, state="waiting", holding={"block": 1, "color": "red"},
                ledger=dict(ST["ledger"], revision=2, target={"red": 1, "blue": 2}),
                question=dict(HELD, id=qid))


def test_answer_is_not_sent_to_a_question_that_changed_while_thinking():
    srv = StubServer(_held_state(1))

    class Switching(ReplayLLM):            # question 1 is replaced by question 2 (also accepts "return")
        def complete(self, messages, schema=None):
            srv.st = _held_state(2)
            return super().complete(messages, schema)

    agent = ChatAgent(srv, Switching([I("answer", question_id=1, choice="return")]), CFG)
    agent.q_shown = 1
    out = agent.handle("放回去")
    assert srv.sent == []                                  # not sent to question 2
    assert "問題 2" in out[0] and "沒有送出" in out[0]
    assert agent.q_shown == 2 and agent.transcript[-1]["attempts"][-1]["reply"] == "ask"


def test_answer_is_not_sent_to_a_question_the_user_has_not_seen():
    srv = StubServer(_held_state(2))                       # question 2 came up after question 1 was shown
    agent = ChatAgent(srv, ReplayLLM([I("answer", question_id=None, choice="return")]), CFG)
    agent.q_shown = 1
    out = agent.handle("放回去")
    assert srv.sent == [] and "問題 2" in out[0]


def test_answer_to_a_question_still_open_is_sent_once():
    srv = StubServer(_held_state(1))
    agent = ChatAgent(srv, ReplayLLM([I("answer", question_id=1, choice="return")]), CFG)
    agent.q_shown = 1
    agent.handle("放回去")
    assert [(s["cmd"], s["id"], s["choice"]) for s in srv.sent] == [("task_answer", 1, "return")]


# ---- CLI reconnect: the real Repl of tools/chat.py --------------------------------------------------------
def _chat_cli():
    import importlib.util
    if "chat_cli" not in sys.modules:
        spec = importlib.util.spec_from_file_location("chat_cli", os.path.join(ROOT, "tools", "chat.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["chat_cli"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["chat_cli"]


class EventServer(StubServer):
    """StubServer with an event log; events before a client's cursor are history it never receives."""
    def __init__(self, st, events=()):
        super().__init__(st)
        self.events = [dict(e, i=i, t=0.0, rev=2) for i, e in enumerate(events)]

    def ask(self, qid, state=None):          # the executor opens question qid (default: holding a red block)
        self.st = (state or _held_state)(qid)
        q = self.st["question"]
        self.events.append({"kind": "question", "qkind": q["kind"], "i": len(self.events), "t": 0.0, "rev": 2,
                            **{k: v for k, v in q.items() if k != "kind"}})

    def __call__(self, cmd, request_id=None, **args):
        if cmd == "task_events":
            since = args.get("since", len(self.events))
            return {"next": len(self.events), "events": self.events[since:]}
        return super().__call__(cmd, request_id=request_id, **args)


def _repl(srv, llm):
    shown_log = []
    agent, reporter = ChatAgent(srv, llm, CFG), ChatAgent(srv, ReplayLLM([]), CFG)
    return _chat_cli().Repl(agent, reporter, srv, shown_log.append), agent, shown_log


def test_cli_reconnect_answer_before_the_question_is_shown_is_not_sent():
    srv = EventServer(_held_state(1))
    srv.ask(2)                                                      # question 2 is open before the CLI connects
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=2, choice="return")] * 2))
    repl.turn("放回去", shown=agent.q_shown)                         # arrived before anything was shown
    assert srv.sent == [] and "沒有送出" in screen[-1] and "問題 2" in screen[-1]
    repl.connect()                                                  # the open question is shown on connect
    assert "目前待答的問題" in screen[-1] and "問題 2" in screen[-1]
    repl.pump()
    assert len(screen) == 2                                         # its event is history: not shown twice


def test_cli_reconnect_shows_the_open_question_then_one_answer_is_sent():
    srv = EventServer(_held_state(1))
    srv.ask(2)
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=2, choice="return")]))
    repl.connect()
    assert screen == [f"[系統] 目前待答的問題：{W.question(dict(HELD, id=2))}"]
    repl.turn("放回去", shown=agent.q_shown)
    repl.pump()
    assert [(s["cmd"], s["id"], s["choice"]) for s in srv.sent] == [("task_answer", 2, "return")]
    assert screen[-1] == agent.transcript[-1]["replies"][0]


def test_cli_question_replaced_while_thinking_old_answer_not_sent():
    srv = EventServer(_held_state(1))
    srv.ask(1)
    repl, agent, screen = _repl(srv, None)

    class Background(ReplayLLM):            # while the model thinks, question 2 (also accepts return) is reported
        def complete(self, messages, schema=None):
            if srv.st["question"]["id"] == 1:
                srv.ask(2)
                repl.pump()                 # printed and copied to the agent during the turn
            return super().complete(messages, schema)

    agent.llm = Background([I("answer", question_id=1, choice="return"), I("answer", question_id=2, choice="return")])
    repl.connect()
    shown = agent.q_shown
    assert shown == 1
    repl.turn("放回去", shown=shown)
    assert srv.sent == []
    assert "問題 2" in screen[-2] and "問題已經換了" in screen[-1] and "問題 2" in screen[-1]
    assert agent.transcript[-1]["shown"] == 1
    repl.turn("放回去", shown=agent.q_shown)                         # the user saw question 2 and answers again
    assert [(s["cmd"], s["id"]) for s in srv.sent] == [("task_answer", 2)]


# ---- PR #8 review 2: binding sampled when reading a line starts; every answer path checked --------------------
def _shortage_state(qid):
    return dict(ST, state="waiting", ledger=dict(ST["ledger"], revision=2, target={"red": 2, "blue": 1},
                                                 delivered={"red": 1, "blue": 1}, need={"red": 1}),
                question={"id": qid, "kind": "shortage", "options": ["substitute", "reduce", "wait"],
                          "delivered": {"red": 1, "blue": 1}, "missing": {"red": 1}, "available": {"blue": 1}})


def _idle_state(qid=None):
    return dict(_held_state(1), question=None)


def _lines(*steps):
    """read_line for Repl.run: each step is a text, or (callback run while the user types, text)."""
    it = iter(steps)

    def read_line(prompt):
        step = next(it, None)
        if step is None:
            raise EOFError
        if isinstance(step, tuple):
            step[0]()
            return step[1]
        return step
    return read_line


def _answers(srv):
    return [(s["id"], s["choice"], s.get("xy"), s.get("color")) for s in srv.sent if s["cmd"] == "task_answer"]


def test_case_a_question_changes_while_typing():
    srv = EventServer(_held_state(1))
    srv.ask(1)
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=2, choice="return")] * 2))
    repl.connect()                                                      # question 1 on screen

    def q2_while_typing():                  # question 2 (also accepts return) is reported before Enter
        srv.ask(2)
        repl.pump()

    repl.run(_lines((q2_while_typing, "放回去"), "放回去"))
    assert _answers(srv) == [(2, "return", None, None)]                   # only the second line, to question 2
    assert "問題 1" in screen[0] and "問題 2" in screen[1]                 # connect, report while typing
    assert "問題已經換了" in screen[2] and "手上的紅色積木" in screen[2] and "問題 2" in screen[2]
    assert screen[3].startswith("好的")
    assert agent.transcript[-2]["shown"] == 1 and agent.transcript[-1]["shown"] == 2


def test_case_b_follow_up_for_coordinates_needs_the_question_shown_first():
    srv = EventServer(_idle_state())
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=2, choice="place_at")] * 2 +
                                               [I("answer", question_id=2, choice="place_at", xy=[0.30, 0.05])]))
    repl.connect()
    assert screen == []
    srv.ask(2)                                                          # opened, report not printed yet
    repl.run(_lines("放到別處"))
    assert srv.sent == [] and len(screen) == 1
    assert "沒有送出" in screen[0] and "手上的紅色積木" in screen[0]
    assert "放回原位、放到其他位置，還是照樣送出" in screen[0] and "問題 2" in screen[0]
    repl.run(_lines("放到別處", "0.30 0.05"))
    assert "要放到哪裡" in screen[1]
    assert _answers(srv) == [(2, "place_at", [0.30, 0.05], None)]


def test_case_c_follow_up_for_a_substitute_colour_needs_the_question_shown_first():
    srv = EventServer(_idle_state())
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=2, choice="substitute")] * 2 +
                                               [I("answer", question_id=2, choice="substitute",
                                                  substitute_color="blue")]))
    repl.connect()
    srv.ask(2, _shortage_state)
    repl.run(_lines("換一個顏色"))
    assert srv.sent == [] and "還缺 紅色 1" in screen[0] and "要換成其他顏色、減少數量，還是等補料" in screen[0]
    repl.run(_lines("換一個顏色", "藍色"))
    assert "要換成哪個顏色" in screen[1]
    assert _answers(srv) == [(2, "substitute", None, "blue")]


def test_case_d_question_changes_while_typing_the_coordinates():
    srv = EventServer(_held_state(1))
    srv.ask(1)
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=1, choice="place_at"),
                                               I("answer", question_id=2, choice="place_at", xy=[0.30, 0.05]),
                                               I("answer", question_id=2, choice="place_at"),
                                               I("answer", question_id=2, choice="place_at", xy=[0.25, 0.0])]))
    repl.connect()

    def q2_while_typing():
        srv.ask(2)
        repl.pump()

    repl.run(_lines("放到別處", (q2_while_typing, "0.30 0.05")))
    assert srv.sent == []                                               # neither question 1 nor question 2
    assert "要放到哪裡" in screen[1] and "問題已經換了" in screen[-1] and "問題 2" in screen[-1]
    assert agent.history == [] and agent.transcript[-1]["history_dropped"] == 1   # old follow-up forgotten
    repl.run(_lines("放到別處", "0.25 0"))
    assert _answers(srv) == [(2, "place_at", [0.25, 0.0], None)]


# ---- PR #8 review 3: a follow-up is bound to the question that asked it ------------------------------------
def _held_yellow(qid):
    return dict(_held_state(qid), holding={"block": 4, "color": "yellow"},
                question=dict(HELD, id=qid, block=4, color="yellow"))


def _short(color):
    def st(qid):
        return dict(_shortage_state(qid), question=dict(_shortage_state(qid)["question"], missing={color: 1},
                                                         available={"blue": 1}))
    return st


class Hooked(ReplayLLM):
    """Fixed replies; hook[n]() runs while the model is answering call n (before it returns)."""
    def __init__(self, replies, hooks):
        super().__init__(replies)
        self.hooks, self.n = hooks, 0

    def complete(self, messages, schema=None):
        h = self.hooks.get(self.n)
        self.n += 1
        if h:
            h()
        return super().complete(messages, schema)


def _check_sends(srv, n):
    """read_line step that checks how many answers were sent before the user types the next line."""
    def chk():
        assert len(_answers(srv)) == n, _answers(srv)
    return chk


def _followup_switch(state1, state2, first, stale, choose, final, texts):
    srv = EventServer(state1(1))
    srv.ask(1, state1)
    repl, agent, screen = _repl(srv, None)

    def q2_while_thinking():
        srv.ask(2, state2)
        repl.pump()                                    # the full question 2 is printed before the model returns

    agent.llm = Hooked([first, stale, choose, final], {0: q2_while_thinking})
    repl.connect()
    repl.run(_lines(texts[0], (_check_sends(srv, 0), texts[1]), (_check_sends(srv, 0), texts[2]),
                    (_check_sends(srv, 0), texts[3])))
    return srv, agent, screen


def test_case_e_followup_for_coordinates_dropped_when_the_question_changes():
    srv, agent, screen = _followup_switch(
        _held_state, _held_yellow,
        I("answer", question_id=1, choice="place_at"),
        I("answer", question_id=2, choice="place_at", xy=[0.30, 0.05]),       # old coordinates, bound to question 2
        I("answer", question_id=2, choice="place_at"),
        I("answer", question_id=2, choice="place_at", xy=[0.25, 0.0]),
        ["放到別處", "0.30 0.05", "放到別處", "0.25 0.00"])
    assert _answers(srv) == [(2, "place_at", [0.25, 0.0], None)]
    i2 = next(i for i, l in enumerate(screen) if "黃色" in l and "問題 2" in l)   # question 2 reported
    first_turn = screen[i2 + 1]
    assert "要放到哪裡" not in first_turn and "黃色" in first_turn and "問題 2" in first_turn
    assert "請先選擇" in screen[i2 + 2] and "黃色" in screen[i2 + 2]               # old coordinates: 0 sent
    assert "要放到哪裡" in screen[i2 + 3] and screen[i2 + 4].startswith("好的")
    assert sum("要放到哪裡" in l for l in screen) == 1                              # only question 2's follow-up


def test_case_f_followup_for_a_colour_dropped_when_the_question_changes():
    srv, agent, screen = _followup_switch(
        _short("red"), _short("yellow"),
        I("answer", question_id=1, choice="substitute"),
        I("answer", question_id=2, choice="substitute", substitute_color="blue"),
        I("answer", question_id=2, choice="substitute"),
        I("answer", question_id=2, choice="substitute", substitute_color="blue"),
        ["換一個顏色", "藍色", "換一個顏色", "藍色"])
    assert _answers(srv) == [(2, "substitute", None, "blue")]
    i2 = next(i for i, l in enumerate(screen) if "還缺 黃色 1" in l and l.startswith("[系統]"))
    assert "要換成哪個顏色" not in screen[i2 + 1] and "還缺 黃色 1" in screen[i2 + 1]
    assert "請先選擇" in screen[i2 + 2]
    assert "要換成哪個顏色" in screen[i2 + 3] and screen[i2 + 4].startswith("好的")
    assert sum("要換成哪個顏色" in l for l in screen) == 1


def test_followup_shown_then_question_changes_before_the_next_line():
    srv = EventServer(_held_state(1))
    srv.ask(1)
    repl, agent, screen = _repl(srv, ReplayLLM([I("answer", question_id=1, choice="place_at"),
                                               I("answer", question_id=2, choice="place_at", xy=[0.30, 0.05]),
                                               I("answer", question_id=2, choice="place_at"),
                                               I("answer", question_id=2, choice="place_at", xy=[0.25, 0.0])]))
    repl.connect()

    def q2_before_reading():                 # question 1's follow-up is on screen, then question 2 is reported
        srv.ask(2, _held_yellow)
        repl.pump()

    repl.run(_lines("放到別處"))
    assert "要放到哪裡" in screen[-1]
    q2_before_reading()
    repl.run(_lines("0.30 0.05"))                                   # shown is already 2 when this line starts
    assert srv.sent == [] and "問題已經換了" in screen[-1] and "黃色" in screen[-1]
    repl.run(_lines("放到別處", "0.25 0.00"))
    assert _answers(srv) == [(2, "place_at", [0.25, 0.0], None)]


def test_set_keeps_an_order_that_arrives_first():
    """Order red 1 + blue 1. The chat reads it, then another source adds one red before "藍色改成兩顆" is applied:
    both changes are kept (red 2, blue 2)."""
    w = FakeWorld(["red", "red", "blue", "blue"])
    srv = FakeServer(w)
    srv("task_order", counts={"red": 1, "blue": 1})
    w.step()
    other = {"done": False}

    def racing(cmd, **args):
        if cmd == "task_order" and not other["done"]:      # after the chat's last state read, before its command
            other["done"] = True
            srv("task_order", counts={"red": 1}, mode="add", request_id="panel-1")
        return srv(cmd, **args)

    agent = ChatAgent(racing, ReplayLLM([I("order_set", [("blue", 2)])]), CFG)
    agent.handle("藍色改成兩顆")
    w.step()
    assert w.ex.ledger.target == {"red": 2, "blue": 2} and w.ex.ledger.revision == 3
    assert [e["mode"] for e in w.ex.ledger.events if e["kind"] == "order"] == ["total", "add", "set"]
    assert Ledger.replay(w.ex.ledger.events).target == {"red": 2, "blue": 2}


def test_add_is_capped_per_colour_and_retry_applies_once():
    w = FakeWorld(["red"] * 8)
    srv = FakeServer(w)
    srv("task_order", counts={"red": 7})
    w.step()
    lost = {"n": 1}

    def flaky(cmd, **args):                 # applied, reply lost once: the retry carries the same request_id
        out = srv(cmd, **args)
        if cmd == "task_order" and lost["n"]:
            lost["n"] -= 1
            raise ConnectionError("reply lost")
        return out

    agent = ChatAgent(flaky, ReplayLLM([I("order_add", [("red", 1)]), I("order_add", [("red", 1)])]), CFG)
    agent.handle("再加一顆紅色")
    w.step()
    assert w.ex.ledger.target == {"red": 8} and w.ex.ledger.revision == 2
    assert agent.transcript[-1]["attempts"][-1]["sent"]["response"]["duplicate"] is True
    out = agent.handle("再加一顆紅色")       # a new request: 9 red is over the cap, refused by the chat
    w.step()
    assert "最多 8 顆" in out[0] and agent.transcript[-1]["attempts"][-1]["reply"] == "reject"
    assert w.ex.ledger.target == {"red": 8} and w.ex.ledger.revision == 2
    with pytest.raises(ValueError, match="at most 8"):     # and by the server, if sent anyway
        srv("task_order", counts={"red": 1}, mode="add", request_id="other-1")
    w.step()
    assert w.ex.ledger.target == {"red": 8} and w.ex.ledger.revision == 2


def test_cap_counts_orders_still_queued():
    w = FakeWorld(["red"] * 8)
    srv = FakeServer(w)
    srv("task_order", counts={"red": 7})
    srv("task_order", counts={"red": 1}, mode="add")         # queued, not applied yet
    with pytest.raises(ValueError):
        srv("task_order", counts={"red": 1}, mode="add")
    w.step()
    assert w.ex.ledger.target == {"red": 8} and w.ex.ledger.revision == 2


def test_refused_first_order_leaves_the_arm_to_the_operator():
    w = FakeWorld(["red"])
    srv = FakeServer(w)
    srv.session.task = None
    srv.session.start_task = lambda: setattr(srv.session, "task", w.runner) or w.runner
    with pytest.raises(ValueError):
        srv("task_order", counts={"red": 9})
    assert srv.session.task is None


def test_ledger_set_mode():
    assert Ledger.compose({"red": 1, "blue": 2}, {"blue": 0, "yellow": 1}, "set") == {"red": 1, "yellow": 1}
    assert Ledger.compose({"red": 1}, {"blue": 2}, "total") == {"blue": 2}
    assert Ledger.compose({"red": 1}, {"red": 2}, "add") == {"red": 3}


def _rec(**a):
    return {"attempts": [a], "replies": [a.pop("text", "")]}


def test_scoring_separates_askback_reject_and_failures():
    clar = {"expect": "clarify", "about": "count"}
    assert verdict(clar, _rec(reply="ask", text="要幾顆？請說數字")) == "correct_askback"
    assert verdict(clar, _rec(reply="ask", text="要哪個顏色？")) == "wrong_askback"
    assert verdict({"expect": "clarify", "about": "color"},
                   _rec(reply="reject", text="不支援水藍色")) == "wrong_reply(reject)"
    assert verdict({"expect": "reject"}, _rec(reply="reject")) == "correct_reject"
    assert verdict({"expect": "reject"}, _rec(reply="ask")) == "wrong_reply(ask)"
    send = {"expect": "send", "command": {"cmd": "task_order", "mode": "add", "counts": {"blue": 1}}}
    assert verdict(send, _rec(sent={"cmd": "task_order", "args": {"mode": "add", "counts": {"blue": 1}}})) \
        == "correct_command"
    assert verdict(send, _rec(sent={"cmd": "task_order", "args": {"mode": "add", "counts": {"blue": 2}}})) \
        == "wrong_command"
    assert verdict(send, _rec(reply="ask")) == "missed"
    assert verdict({"expect": "reject"}, _rec(sent={"cmd": "task_order", "args": {}})) == "misexecuted"
    # the model service down: nothing sent, but a failure, not a correct "no command"
    assert verdict({"expect": "reject"}, _rec(failure="service")) == "service_failure"
    assert verdict(send, _rec(failure="format")) == "format_failure"
    s = summarize(["correct_command", "wrong_command", "misexecuted", "service_failure", "wrong_reply(ask)"])
    assert (s["correct"], s["mis_sent"], s["failures"], s["wrong_reply"]) == (1, 2, 1, 1)


def test_model_service_down_scores_as_service_failure():
    class Down:
        def complete(self, messages, schema=None):
            raise ServiceError("model service not reachable")

    agent = ChatAgent(StubServer(ST), Down(), CFG)
    agent.handle("手臂往左移五公分")
    assert verdict({"expect": "reject"}, agent.transcript[-1]) == "service_failure"


# ---- one block named by position ("左上角那顆紅色") --------------------------------------------------------
def test_plan_pick_answers_from_the_block_list():
    reds4 = load("configs/chat/utterances.yaml")["presets"]["reds4"]
    P = lambda ref, c="red": plan(I("order_pick", [(c, 1)], pick_ref=ref), reds4, W, 8)   # noqa: E731
    a = P("leftmost")
    assert a.kind == "send" and a.args == {"counts": {"red": 1}, "mode": "add", "pick": {"color": "red", "ref": "leftmost"}}
    assert a.text == "好的，夾最左邊那顆紅色。"
    a = P("bottom_right")                                      # block 3: out of reach, others are not
    assert a.kind == "say" and a.reply == "ask" and "夾不到" in a.text and "另外有 3 顆紅色" in a.text
    assert P("top_left", "purple").reply == "reject"           # not in the scene
    assert P("top_left", "yellow").text == "備料盤裡已經沒有黃色積木了。"
    assert P(None).reply == "ask"                              # position words outside the list
    tie = dict(reds4, pan_xy=[0.0, 0.0], storage_blocks=[{"block": 0, "color": "red", "xy": [0.2, 0.1], "reachable": True},
                                      {"block": 1, "color": "red", "xy": [0.2, -0.1], "reachable": True}])
    assert plan(I("order_pick", [("red", 1)], pick_ref="farthest"), tie, W, 8).reply == "ask"
    # no task yet: no block list, sent; the executor checks when it picks
    assert plan(I("order_pick", [("red", 1)], pick_ref="top_left"), EMPTY_ST, W, 8).kind == "send"


EMPTY_ST = {"active": False, "colors_in_scene": ["blue", "red", "yellow"],
            "ledger": {"revision": 0, "target": {}, "delivered": {}, "need": {}, "over": {}}}


def test_pick_dialogue_replay():
    w = FakeWorld(["red", "red", "red", "blue"])
    srv = FakeServer(w)
    llm = ReplayLLM([I("order_pick", [("red", 1)], pick_ref="top_left")])
    agent = ChatAgent(srv, llm, CFG)
    assert agent.handle("我要左上角那顆紅色") == ["好的，夾左上角那顆紅色。"]
    run_until(w, lambda w: w.ex.state == "done")
    rep = agent.poll_reports()
    assert "[進度] 找到左上角那顆紅色積木，準備夾取…" in rep
    assert list(w.ex.delivered_blocks) == [1]                  # the red farthest from the base (+x)
    sent = [a["sent"] for r in agent.transcript if "attempts" in r for a in r["attempts"] if "sent" in a]
    assert sent[0]["args"]["pick"] == {"color": "red", "ref": "top_left"}


def test_pick_rejected_report():
    e = {"kind": "pick_rejected", "color": "red", "ref": "bottom_right", "reachable_left": 2}
    assert W.event(e) == "右下角那顆紅色積木在手臂範圍外，夾不到，已從需求拿掉這一顆。另外還有 2 顆夾得到。"
