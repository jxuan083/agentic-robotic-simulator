"""Chat agent: one user turn -> (model) intent -> checked task command or reply; executor events -> reports.

  agent = ChatAgent(call, llm, cfg)       # call(cmd, **args) talks to a simulator server (task_* commands)
  for line in agent.handle("總共一紅兩藍"): print(line)
  for line in agent.poll_reports():        print(line)     # order applied, questions, shortage, over-delivery,
                                                             # done, anomalies (fixed Traditional Chinese templates)
  agent.transcript                         # every turn: input, state revision, model output, action, command sent

Before a command is sent the task state is read again: if the order revision or the open question changed while the
model was thinking, the turn is planned again on the new state (once). Every command carries a request_id unique to
the turn, so a transport retry is applied once; saying the same thing again is a new turn with a new id.
"""

from __future__ import annotations

import time
import uuid

from agentic_sim.chat.intent import plan, schema, state_for_prompt, system_prompt
from agentic_sim.chat.zh import Words


class ChatAgent:
    def __init__(self, call, llm, cfg: dict, log=None, clock=time.time):
        self.call, self.llm, self.cfg = call, llm, cfg
        self.words = Words(cfg["colors"])
        self.palette = list(cfg["colors"])
        self.max_count = int(cfg.get("max_count", 8))
        self.schema = schema(self.palette, self.max_count)
        self.system = system_prompt(cfg["colors"], self.max_count)
        self.history: list = []
        self.transcript: list = []
        self.events_seen = 0
        self.session = uuid.uuid4().hex[:8]
        self.turn = 0
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
    def handle(self, text: str) -> list:
        text = text.strip()
        if not text:
            return []
        self.turn += 1
        rec = {"turn": self.turn, "t": round(self.clock(), 3), "user": text, "attempts": []}
        st = self.state()
        out = []
        for attempt in range(2):
            try:
                rep = self._ask_model(text, st)
            except Exception as e:  # noqa: BLE001  (model service down / bad reply: tell the user, send nothing)
                rec["attempts"].append({"state": self._key(st), "error": f"{type(e).__name__}: {e}"})
                out = [f"模型服務沒有回應，這句沒有送出（{type(e).__name__}）。"]
                break
            act = plan(rep.data, st, self.words, self.max_count, text)
            a = {"state": self._key(st), "model": rep.model, "latency_s": round(rep.latency_s, 3),
                 "model_output": rep.data, "action": act.kind, "intent": act.intent}
            rec["attempts"].append(a)
            if act.kind != "send":
                out = [act.text]
                break
            st2 = self.state()
            if self._key(st2) != self._key(st):     # the order or the question changed meanwhile: plan again
                a["stale"] = {"was": self._key(st), "now": self._key(st2)}
                st = st2
                if attempt == 0:
                    continue
                out = ["狀態剛剛變了，這句沒有送出，請再說一次。"]
                break
            rid = f"{self.session}-{self.turn}"
            res = self._send(act.cmd, act.args, rid)
            a["sent"] = {"cmd": act.cmd, "args": act.args, "request_id": rid, "response": res}
            out = [act.text]
            break
        rec["replies"] = out
        self.transcript.append(rec)
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(out)}]
        return out

    # ---- reports ------------------------------------------------------------------------------------
    def poll_reports(self) -> list:
        r = self.call("task_events", since=self.events_seen)
        self.events_seen = r.get("next", self.events_seen)
        out = []
        for e in r.get("events", []):
            line = self.words.event(e)
            self.transcript.append({"event": e["kind"], "i": e["i"], "t_sim": e["t"], "rev": e["rev"],
                                    "report": line, **({"ledger": e["ledger"]} if "ledger" in e else {})})
            if line:
                out.append(line)
        return out
