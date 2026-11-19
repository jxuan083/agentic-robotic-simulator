"""Scoring one chat turn against an utterance-table case (tools/chat_eval.py, tests).

  verdict(case, rec) -> "correct_command" | "correct_askback" | ...      rec = ChatAgent.transcript[-1]

A case says what must happen (`expect`):
  send      a command equal to `command` (cmd + args) is sent
  clarify   nothing is sent and the system asks back; `about` (count / keep / color / place, or a list of them) is
            what the question must be about, checked on the reply text
  reject    nothing is sent and the system says it cannot do it
  status    progress answered from the ledger; stop: pointed to the stop controls
  no_send   any reply without a command (no open question, colour not on the table, ...)

Verdicts (nothing else counts as correct):
  correct_command / correct_askback / correct_reject / correct_status / correct_stop / correct_no_command
  wrong_command      a command was sent, but not the expected one          } mis-sent: the task would change wrongly
  misexecuted        a command was sent where none should be               }
  missed             nothing was sent where a command was expected (the model asked back or refused)
  wrong_askback      asked back, but not about what the case needs clarified
  wrong_reply(kind)  nothing sent, but the wrong kind of reply (e.g. reject where an ask-back was expected)
  service_failure    the model service was down / timed out / HTTP error: nothing sent, still a failure
  format_failure     the model's output could not be used
  transport_failure  the command could not be delivered to the simulator
"""

from __future__ import annotations

import re

TOPICS = {
    "count": re.compile(r"幾顆|幾個|數量|數字|多少"),
    "keep": re.compile(r"保留"),
    "color": re.compile(r"顏色|哪一?種|哪個|淺綠|深綠|墨綠|藍色|青色"),
    "place": re.compile(r"哪裡|位置|座標"),
}
MIS_SENT = ("wrong_command", "misexecuted")
FAILURES = ("service_failure", "format_failure", "transport_failure")


def _eq(a, b):
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b))
    return a == b


def same_command(want: dict, sent: dict) -> bool:
    got = {"cmd": sent["cmd"], **sent["args"]}
    return all(_eq(got.get(k), v) for k, v in want.items()) and set(got) <= set(want)


def verdict(case: dict, rec: dict) -> str:
    a = rec["attempts"][-1] if rec.get("attempts") else {}
    sent = a.get("sent")
    exp = case["expect"]
    if sent:
        if exp == "send":
            return "correct_command" if same_command(case["command"], sent) else "wrong_command"
        return "misexecuted"
    failure = a.get("failure")
    if failure:
        return f"{failure}_failure"
    if exp == "send":
        return "missed"
    kind = a.get("reply", "")
    if exp == "clarify":
        if kind != "ask":
            return f"wrong_reply({kind})"
        about = case.get("about")
        if about:
            text = " ".join(rec.get("replies") or [])
            topics = [about] if isinstance(about, str) else list(about)
            if not any(TOPICS[t].search(text) for t in topics):
                return "wrong_askback"
        return "correct_askback"
    if exp == "reject":
        return "correct_reject" if kind == "reject" else f"wrong_reply({kind})"
    if exp in ("status", "stop"):
        return f"correct_{exp}" if a.get("intent") == exp else f"wrong_reply({a.get('intent')})"
    return "correct_no_command"            # no_send: any reply without a command


def summarize(verdicts: list) -> dict:
    """Counts per verdict, plus the headline numbers: correct, mis-sent, missed, failures."""
    by: dict = {}
    for v in verdicts:
        k = v.split("(")[0]
        by[k] = by.get(k, 0) + 1
    return {"total": len(verdicts), "correct": sum(n for k, n in by.items() if k.startswith("correct")),
            "mis_sent": sum(by.get(k, 0) for k in MIS_SENT), "missed": by.get("missed", 0),
            "wrong_askback": by.get("wrong_askback", 0), "wrong_reply": by.get("wrong_reply", 0),
            "failures": sum(by.get(k, 0) for k in FAILURES), "by_verdict": by}
