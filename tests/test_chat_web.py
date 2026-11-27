"""Chat web page (tools/chat_web.py): the same agent through the page's message log and HTTP endpoints, on the fake
world of the task tests (no simulator, no model service)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from agentic_sim.chat.llm import ReplayLLM  # noqa: E402
from chat_web import Stt, WebChat, handler, quick_replies  # noqa: E402
from test_chat import CFG, W, FakeServer, I, run_until  # noqa: E402
from test_task import FakeWorld  # noqa: E402


def wait_idle(app, timeout=10.0):
    end = time.monotonic() + timeout
    while app.busy and time.monotonic() < end:
        time.sleep(0.01)
    assert not app.busy


def texts(app, kind=None):
    out = []
    for m in app.log:
        if m["kind"] == "lines":
            out += [l["text"] for l in m["lines"] if kind is None or l["tag"] == kind]
    return out


def make(colors, replies):
    w = FakeWorld(colors)
    srv = FakeServer(w)
    app = WebChat(lambda name: srv, ReplayLLM(replies), CFG)
    app.connect()
    return w, app


def test_quick_replies_follow_the_question_options():
    q = {"options": ["substitute", "wait"], "substitute_candidates": ["yellow"]}
    assert quick_replies(q, W) == ["等補料", "改成黃色"]
    assert quick_replies({"options": ["return", "place_at", "deliver"]}, W) == ["放回去", "照樣送出"]
    assert quick_replies(None, W) == []


def test_hold_return_through_the_page_log():
    w, app = make(["red", "red", "red", "blue", "blue", "yellow"],
                  {"我要三顆紅色": I("order_total", [("red", 3)]),
                   "總共一紅兩藍": I("order_total", [("red", 1), ("blue", 2)]),
                   "放回去": I("answer", question_id=1, choice="return")})
    assert app.send("我要三顆紅色", None, None)
    wait_idle(app)
    run_until(w, lambda w: w.ex.skill is not None and w.ex.skill.name == "place" and w.ex.skill.params["nth"] == 2
              and w.ex.skill.phase == "transport")
    app.pump()
    assert "[進度] 正在夾取第二顆紅色積木…" in texts(app, "progress")
    app.send("總共一紅兩藍", None, None)
    wait_idle(app)
    run_until(w, lambda w: w.ex.question is not None)
    app.pump()
    card = [m for m in app.log if m["kind"] == "question"][-1]
    assert card["qid"] == 1 and "照樣送出" in card["text"] and card["quick"] == ["放回去", "照樣送出"]
    # the page displayed question 1 when the line was started: bound to it, sent once
    assert app.send("放回去", 1, 1)
    wait_idle(app)
    run_until(w, lambda w: w.ex.state == "done")
    app.pump()
    assert any(t.startswith("[系統] 完成") for t in texts(app, "system"))
    assert [m for m in app.log if m["kind"] == "question"][-1]["text"] is None      # card cleared
    assert [m["text"] for m in app.log if m["kind"] == "user"] == ["我要三顆紅色", "總共一紅兩藍", "放回去"]
    assert w.ex.ledger.snapshot()["delivered"] == {"red": 1, "blue": 2}
    assert [m["seq"] for m in app.log] == list(range(len(app.log)))


def test_answer_to_a_question_the_page_never_showed_is_not_sent():
    w, app = make(["red", "red", "red", "blue", "blue", "yellow"],
                  [I("order_total", [("red", 3)]), I("order_total", [("red", 1), ("blue", 2)]),
                   I("answer", question_id=1, choice="return")])
    app.send("三紅", None, None)
    wait_idle(app)
    run_until(w, lambda w: w.ex.skill is not None and w.ex.skill.name == "place" and w.ex.skill.params["nth"] == 2
              and w.ex.skill.phase == "transport")
    app.send("一紅兩藍", None, None)
    wait_idle(app)
    run_until(w, lambda w: w.ex.question is not None)
    # no pump: the page has not drawn question 1, so the line carries shown=None
    app.send("放回去", None, None)
    wait_idle(app)
    sent = [a["sent"]["cmd"] for r in app.agent.transcript if "attempts" in r for a in r["attempts"] if "sent" in a]
    assert sent == ["task_order", "task_order"]
    assert any("問題 1" in t for t in texts(app))     # the full open question is shown instead


def test_one_turn_at_a_time():
    w, app = make(["red"], [I("status")])
    gate = threading.Event()
    real = app.agent.handle
    app.agent.handle = lambda text, shown=None: (gate.wait(5), real(text, shown=shown))[1]
    assert app.send("現在做到哪？", None, None)
    assert not app.send("再一次", None, None)
    gate.set()
    wait_idle(app)
    assert app.send("   ", None, None) and [m["text"] for m in app.log if m["kind"] == "user"] == ["現在做到哪？"]


def test_http_page_send_and_events():
    w, app = make(["red", "blue"], [I("order_total", [("red", 1)])])
    app.put("ready")
    ref = {"app": app, "boot": []}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler(ref, Stt(None, "cpu")))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "SO-101 任務助理" in page and "/api/events" in page
        assert json.loads(urllib.request.urlopen(base + "/api/info").read()) == {"stt": False, "stt_error": None}
        req = urllib.request.Request(base + "/api/send", data=json.dumps({"text": "一顆紅色"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True}
        wait_idle(app)
        stt = urllib.request.Request(base + "/api/stt", data=b"x", method="POST")
        try:
            urllib.request.urlopen(stt)
            raise AssertionError("stt without a model must answer 501")
        except urllib.error.HTTPError as e:
            assert e.code == 501
        r = urllib.request.urlopen(base + "/api/events", timeout=5)
        got = []
        while len(got) < len(app.log):
            line = r.readline().decode()
            if line.startswith("data: "):
                got.append(json.loads(line[6:]))
        r.close()
        assert [m["kind"] for m in got][:2] == ["ready", "user"]
        assert any(m["kind"] == "lines" and m["lines"][0]["text"] == "好的，需求改為 紅色 1。" for m in got)
    finally:
        srv.shutdown()


def test_block_map_follows_the_task_state():
    w, app = make(["red", "red", "blue"], [])
    w.step()
    app.pump()
    m = [x for x in app.log if x["kind"] == "scene"][-1]["scene"]
    assert sorted(b["block"] for b in m["blocks"]) == [0, 1, 2] and all(b["reachable"] for b in m["blocks"])
    assert m["names"] == {"red": "紅色", "blue": "藍色"} and m["frame"]["up"] == [1.0, 0.0]
    n = len(app.log)
    app.pump()
    assert len(app.log) == n          # unchanged: not sent again
