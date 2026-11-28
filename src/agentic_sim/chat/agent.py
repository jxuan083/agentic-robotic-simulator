"""Chat agent: one user turn -> (model) intent -> checked task command or reply; executor events -> reports.

  agent = ChatAgent(call, llm, cfg)       # call(cmd, **args) talks to a simulator server (task_* commands)
  for line in agent.handle("總共一紅兩藍"): print(line)
  for line in agent.poll_reports():        print(line)     # order applied, questions, shortage, over-delivery,
                                                             # done, anomalies (fixed Traditional Chinese templates)
  agent.transcript                         # every turn: input, state revision, model output, action, command sent

Before a command is sent the task state is read again: if the order revision or the open question changed while the
model was thinking, the turn is planned again on the new state (once). An answer is bound to the question the user
saw (`shown`: the CLI samples q_shown when it starts reading the line, so a question printed while the user types
does not move it). Every answer path (a full answer, or a follow-up that only asks for coordinates / a colour) is
checked: if no question had been shown (None, e.g. a CLI that just connected) or the open question is another one
(it changed before the line was read or while the model was thinking), nothing is sent, the full open question is
shown, and the conversation history is dropped so a follow-up for the old question cannot complete the new one.
q_shown only moves when a full question text is returned (or printed as a report). The executor checks the id again
when it applies the answer.

Follow-ups (asking for the coordinates / colour of a chosen disposal) are bound to the question that asked them
(`followup`). Before a follow-up is shown the task state is read again; if the question changed, the follow-up is
dropped and the full new question is shown instead. After any dropped answer the user must choose a disposal for the
new question again (`need_choice`): an answer that only completes parameters (place_at with coordinates, substitute
with a colour) is not sent until a disposal was chosen for that question in this chat. Every command carries a request_id unique to the
turn, so a transport retry is applied once; saying the same thing again is a new turn with a new id.

Each turn's record says what happened: sent, or a reply (ask / reject / info / error), or a failure
(failure: service = model service down, format = unusable model output, transport = command could not be delivered).
"""

from __future__ import annotations

import time
import uuid

from agentic_sim.chat.llm import ServiceError
from agentic_sim.chat.intent import plan, schema, state_for_prompt, system_prompt
from agentic_sim.chat.zh import Words


class ChatAgent:
    def __init__(self, call, llm, cfg: dict, log=None, clock=time.time):
        self.call, self.llm, self.cfg = call, llm, cfg
        self.words = Words(cfg["colors"])
        self.palette = list(cfg["colors"])
        self.max_count = int(cfg.get("max_count", 8))
        pick = bool(cfg.get("pick_by_position", False))     # order_pick: off = the measured schema and prompt
        no_return = bool(cfg.get("no_return", False))      # hold / removed answers: off = the measured schema
        self.schema = schema(self.palette, self.max_count, pick, no_return)
        self.system = system_prompt(cfg["colors"], self.max_count, pick, no_return)
        self.history: list = []
        self.transcript: list = []
        self.events_seen = 0
        self.session = uuid.uuid4().hex[:8]
        self.turn = 0
        self.q_shown = None        # id of the last question shown to the user (None: none shown yet)
        self.followup = None       # {"qid", "need"}: a follow-up for this question is on screen
        self._last_progress = None
        self.progress_reports = bool(cfg.get("progress_reports", True))   # skill_start -> [進度] lines
        self.need_choice = None    # question id whose disposal must be chosen again before parameters are accepted
        self.log = log or (lambda m: None)
        self.clock = clock

    # ---- task state ------------------------------------------------------------------------------
    def state(self) -> dict:
        return self.call("task_state")

    @staticmethod
    def _key(st: dict):
        q = st.get("question")
        return ((st.get("ledger") or {}).get("revision", 0), q["id"] if q else None)

    def warmup(self) -> float:
        """One request before the first turn: the server compiles the intent schema's grammar on first use (tens of
        seconds on the tested setup), which must not eat the first real turn's timeout."""
        t0 = time.monotonic()
        old = getattr(self.llm, "timeout_s", None)
        if old is not None:
            self.llm.timeout_s = 300.0
        try:
            self.llm.complete([{"role": "system", "content": self.system},
                               {"role": "user", "content": "目前狀態：{}\n使用者：現在做到哪？"}], self.schema)
        finally:
            if old is not None:
                self.llm.timeout_s = old
        return time.monotonic() - t0

    def _ask_model(self, text: str, st: dict):
        msgs = [{"role": "system", "content": self.system},
                *self.history[-2 * int(self.cfg.get("history_turns", 6)):],
                {"role": "user", "content": f"目前狀態：{state_for_prompt(st, self.words)}\n使用者：{text}"}]
        return self.llm.complete(msgs, self.schema)

    def _send(self, cmd: str, args: dict, request_id: str) -> dict:
        tries = int(self.cfg.get("send_retries", 2)) + 1
        for i in range(tries):
            try:
                return self.call(cmd, request_id=request_id, **args)
            except (ConnectionError, OSError, TimeoutError) as e:   # transport: retry with the same request_id
                self.log(f"send {cmd} failed ({e}); retry {i + 1}/{tries - 1}")
                if i == tries - 1:
                    raise
                time.sleep(0.2)

    # ---- one user turn ---------------------------------------------------------------------------
    def _new_question(self, st: dict, shown) -> list:
        q = st.get("question")
        self.followup = None
        if q:
            self.q_shown = self.need_choice = q["id"]
            head = "這句沒有送出，請看目前的問題再回答。" if shown is None else "問題已經換了，剛才那句沒有送出。"
            return [f"{head}{self.words.question(q)}"]
        return ["剛才的問題已經不需要回答了，這句沒有送出。"]

    def show_open_question(self) -> list:
        """On connect: show the question the task is waiting on (its event came before this client's cursor)."""
        q = self.state().get("question")
        if not q:
            return []
        self.q_shown = q["id"]
        return [f"目前待答的問題：{self.words.question(q)}"]

    def handle(self, text: str, shown=...) -> list:
        """One user turn. `shown`: the question id on screen when the text was received (default: q_shown now)."""
        text = text.strip()
        if not text:
            return []
        if shown is ...:
            shown = self.q_shown
        self.turn += 1
        rec = {"turn": self.turn, "t": round(self.clock(), 3), "user": text, "shown": shown, "attempts": []}
        st = self.state()
        out = []
        drop_context = False
        for attempt in range(2):
            try:
                rep = self._ask_model(text, st)
            except Exception as e:  # noqa: BLE001  (model service down / bad reply: tell the user, send nothing)
                kind = "service" if isinstance(e, (ServiceError, OSError, TimeoutError)) else "format"
                rec["attempts"].append({"state": self._key(st), "error": f"{type(e).__name__}: {e}", "failure": kind})
                out = [f"模型服務沒有回應，這句沒有送出（{type(e).__name__}）。" if kind == "service"
                       else "模型的回覆無法使用，這句沒有送出，請再說一次。"]
                break
            try:
                act = plan(rep.data, st, self.words, self.max_count, text)
            except (KeyError, TypeError, ValueError) as e:   # schema-valid but unusable output
                rec["attempts"].append({"state": self._key(st), "model_output": rep.data, "failure": "format",
                                        "error": f"{type(e).__name__}: {e}"})
                out = ["模型的回覆無法使用，這句沒有送出，請再說一次。"]
                break
            a = {"state": self._key(st), "model": rep.model, "latency_s": round(rep.latency_s, 3),
                 "model_output": rep.data, "action": act.kind, "intent": act.intent, "reply": act.reply}
            rec["attempts"].append(a)
            q = st.get("question")
            if act.intent == "answer" and q and shown != q["id"]:     # any answer path, follow-ups included
                a["action"], a["reply"], a["unseen_question"] = "say", "ask", {"shown": shown, "open": q["id"]}
                out = self._new_question(st, shown)
                drop_context = True
                break
            if act.intent == "answer" and q and act.cmd == "task_answer" and ("xy" in act.args or "color" in act.args):
                if self.followup and self.followup["qid"] != q["id"]:        # completes an old question's follow-up
                    a["action"], a["reply"], a["stale_followup"] = "say", "ask", dict(self.followup, open=q["id"])
                    out = self._new_question(st, -1)
                    drop_context = True
                    break
                if self.need_choice == q["id"]:                               # parameters before a new choice
                    a["action"], a["reply"], a["choice_needed"] = "say", "ask", q["id"]
                    out = [f"這句沒有送出。請先選擇這一題要怎麼處理：{self.words.question(q)}"]
                    break
            if act.kind != "send":
                if act.followup and q:
                    st2 = self.state()                     # the question may have changed while the model thought
                    q2 = st2.get("question")
                    if not q2 or q2["id"] != q["id"]:
                        a["action"], a["stale"] = "say", {"was": self._key(st), "now": self._key(st2)}
                        a["followup_dropped"] = act.followup
                        out = self._new_question(st2, -1)
                        drop_context = True
                        break
                    self.followup, self.need_choice = {"qid": q["id"], "need": act.followup}, None
                elif act.intent == "answer" and q and act.reply == "ask":
                    self.followup = None
                out = [act.text]
                break
            st2 = self.state()
            if self._key(st2) != self._key(st):     # the order or the question changed meanwhile
                a["stale"] = {"was": self._key(st), "now": self._key(st2)}
                if act.cmd == "task_answer" and self._key(st2)[1] != self._key(st)[1]:
                    a["action"], a["reply"] = "say", "ask"   # the user answered another question: ask again
                    out = self._new_question(st2, shown)
                    drop_context = True
                    break
                st = st2
                if attempt == 0:
                    continue
                a["action"], a["reply"] = "say", "ask"
                out = ["狀態剛剛變了，這句沒有送出，請再說一次。"]
                break
            rid = f"{self.session}-{self.turn}"
            try:
                res = self._send(act.cmd, act.args, rid)
            except (ConnectionError, OSError, TimeoutError) as e:
                a["action"], a["failure"], a["error"] = "say", "transport", f"{type(e).__name__}: {e}"
                a["unsent"] = {"cmd": act.cmd, "args": act.args, "request_id": rid}
                out = [f"命令送不出去（{type(e).__name__}），狀態未知，請用「現在做到哪？」確認。"]
                break
            except Exception as e:  # noqa: BLE001  (the server refused the command: nothing changed)
                a["action"], a["reply"], a["refused"] = "say", "reject", f"{type(e).__name__}: {e}"
                out = [f"系統沒有接受這個命令，帳本沒有改變（{e}）。"]
                break
            a["sent"] = {"cmd": act.cmd, "args": act.args, "request_id": rid, "response": res}
            if act.cmd == "task_answer":
                self.followup = self.need_choice = None
            out = [act.text]
            break
        rec["replies"] = out
        self.transcript.append(rec)
        if drop_context:      # the answer was for a question that is no longer the one on screen: forget it
            rec["history_dropped"] = len(self.history) // 2
            self.history = []
        else:
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(out)}]
        return out

    # ---- reports ------------------------------------------------------------------------------------
    def poll_reports(self) -> list:
        r = self.call("task_events", since=self.events_seen)
        self.events_seen = r.get("next", self.events_seen)
        out = []
        for e in r.get("events", []):
            line = self.words.event(e)
            if line is None and self.progress_reports:
                p = self.words.progress(e)
                line = f"[進度] {p}" if p and p != self._last_progress else None   # repeated looks: said once
                self._last_progress = p or self._last_progress
            elif line is not None:
                self._last_progress = None
            if e["kind"] == "question":
                self.q_shown = e["id"]
            self.transcript.append({"event": e["kind"], "i": e["i"], "t_sim": e["t"], "rev": e["rev"],
                                    "report": line, **({"ledger": e["ledger"]} if "ledger" in e else {})})
            if line:
                out.append(line)
        return out
