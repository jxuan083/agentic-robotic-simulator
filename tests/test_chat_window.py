"""PR #12 review 1: an answer typed in the chat window is bound to the question the window *displayed* when the line
was started (keys, input-method commits and pastes alike), not to one that only arrived in the background.

The pure tests drive agentic_sim.chat.display_binding with the real ChatAgent and an event server; the Tk tests run
the real tools/chat_window.py window (needs an X display; skipped otherwise) and decide by hand when background
messages are drawn, so "received" and "displayed" are ordered by the test, not by sleeps."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.display_binding import DisplayBinding  # noqa: E402
from agentic_sim.chat.llm import ReplayLLM  # noqa: E402
from test_chat import CFG, EventServer, I, _held_state, _idle_state, _shortage_state  # noqa: E402


def _answers(srv):
    return [(s["id"], s["choice"], s.get("color")) for s in srv.sent if s["cmd"] == "task_answer"]


# ---- pure: binding + real agent ---------------------------------------------------------------------------------
def test_line_started_while_a_is_displayed_does_not_answer_b():
    srv = EventServer(_shortage_state(1))
    agent = ChatAgent(srv, ReplayLLM([I("answer", question_id=2, choice="substitute", substitute_color="blue"), I("answer", question_id=2, choice="reduce")]), CFG)
    b = DisplayBinding()
    b.displayed(1)                                   # A on screen
    srv.ask(2, _shortage_state)                      # B received in the background, not drawn yet
    b.text_changed("換")                             # typing starts: bound to A
    b.text_changed("換成藍色")
    b.displayed(2)                                   # B drawn before Enter
    out = agent.handle("換成藍色", shown=b.take())
    assert _answers(srv) == [] and "沒有送出" in out[0]          # B stays open, nothing sent
    # the user saw B, clears and types again: one answer to B
    b.text_changed("")
    b.text_changed("減少數量")
    agent.handle("減少數量", shown=b.take())
    assert _answers(srv) == [(2, "reduce", None)]       # after a discarded line: the disposal first (PR #8 rule)


def test_line_started_with_no_question_does_not_answer_the_next_one():
    srv = EventServer(_idle_state())
    agent = ChatAgent(srv, ReplayLLM([I("answer", question_id=2, choice="substitute", substitute_color="blue"), I("answer", question_id=2, choice="reduce")]), CFG)
    b = DisplayBinding()
    b.text_changed("換成藍色")                       # nothing on screen yet
    srv.ask(2, _shortage_state)
    b.displayed(2)
    agent.handle("換成藍色", shown=b.take())
    assert _answers(srv) == []
    b.text_changed("")
    b.text_changed("減")
    agent.handle("減少數量", shown=b.take())
    assert _answers(srv) == [(2, "reduce", None)]


def test_binding_paste_equals_typing_and_clear_rebinds():
    b = DisplayBinding()
    b.displayed(3)
    b.text_changed("放回去")                        # a paste puts the whole line at once
    b.displayed(4)
    assert b.bound == 3 and b.take() == 3 and b.bound == 4
    b.text_changed("放")
    b.text_changed("")                              # cleared: released
    b.displayed(5)
    b.text_changed("放回去")
    assert b.take() == 5


# ---- Tk: the real chat window -----------------------------------------------------------------------------------
tk = pytest.importorskip("tkinter")


@pytest.fixture
def window(monkeypatch):
    if not os.environ.get("DISPLAY"):
        pytest.skip("no X display")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no X display")
    import chat_window as cw

    class Sync:                                     # run the agent turn on the calling thread
        def __init__(self, target, args=(), daemon=None):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)
    monkeypatch.setattr(cw.threading, "Thread", Sync)
    a = type("A", (), {"type": None, "type_gap": 1.0})()
    w = cw.ChatWindow(root, a, CFG, start=False)
    yield w, root
    root.destroy()


def _setup(w, srv, replies):
    w.agent = ChatAgent(srv, ReplayLLM(replies), CFG)
    w.reporter = ChatAgent(srv, ReplayLLM([]), CFG)
    w.connect()
    _draw(w)
    w.show("ready", None)


def _draw(w):
    """The UI draws everything received so far."""
    while not w.q.empty():
        w.show(*w.q.get_nowait())
    w.root.update()


def _type(w, text):
    w.entry.focus_force()
    w.root.update()
    for ch in text:
        if ch.isascii():
            w.entry.event_generate("<KeyPress>", keysym=ch)
        else:                                        # input-method commit: the character is inserted
            w.entry.insert("end", ch)
        w.root.update()


def _paste(w, text):
    w.root.clipboard_clear()
    w.root.clipboard_append(text)
    w.entry.event_generate("<<Paste>>")
    w.root.update()


@pytest.mark.parametrize("enter", ["typed", "pasted"])
def test_window_answer_bound_to_the_displayed_question(window, enter):
    w, root = window
    srv = EventServer(_shortage_state(1))
    srv.ask(1, _shortage_state)
    rev, delivered = srv.st["ledger"]["revision"], dict(srv.st["ledger"]["delivered"])
    _setup(w, srv, [I("answer", question_id=2, choice="substitute", substitute_color="blue"), I("answer", question_id=2, choice="reduce")])
    assert w.binding.on_screen == 1                  # A shown on connect
    srv.ask(2, _shortage_state)
    w.pump()                                         # B received in the background; not drawn
    assert w.agent.q_shown == 1 and w.binding.on_screen == 1
    (_type if enter == "typed" else _paste)(w, "換成藍色")
    _draw(w)                                         # now B is drawn, before Enter
    assert w.binding.on_screen == 2
    w.send()
    assert _answers(srv) == []                       # not answered: B stays open
    assert srv.st["question"]["id"] == 2 and srv.st["ledger"]["revision"] == rev
    assert srv.st["ledger"]["delivered"] == delivered
    _draw(w)
    (_type if enter == "typed" else _paste)(w, "reduce")        # started while B is displayed
    w.send()
    assert _answers(srv) == [(2, "reduce", None)]


def test_window_line_started_before_any_question(window):
    w, root = window
    srv = EventServer(_idle_state())
    _setup(w, srv, [I("answer", question_id=2, choice="substitute", substitute_color="blue"), I("answer", question_id=2, choice="reduce")])
    _type(w, "yellow")
    srv.ask(2, _shortage_state)
    w.pump()
    _draw(w)
    w.send()
    assert _answers(srv) == []
    _draw(w)
    _type(w, "reduce")
    w.text.set("")                                   # cleared and typed again while B is displayed
    _type(w, "reduce")
    w.send()
    assert _answers(srv) == [(2, "reduce", None)]
