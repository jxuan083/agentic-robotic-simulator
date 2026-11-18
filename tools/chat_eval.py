"""Utterance table for the chat layer, run with the real model (no simulator needed: states are presets).

  uv run tools/chat_eval.py [--table configs/chat/utterances.yaml] [--out outputs/chat/eval] [--repeat 1]

For every case the model's intent goes through the same checks as in the chat (agentic_sim.chat.intent.plan) against
the preset state; the command that would be sent is compared with the expected one. Writes result.json (per case:
model output, action, command, verdict, latency) and prints the counts per group: correct command, correct ask-back,
correct no-command, missed (should have sent), mis-executed (sent when it should not / sent another command).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.llm import OpenAICompatLLM  # noqa: E402
from agentic_sim.config import load  # noqa: E402

ASK_INTENTS = ("clarify", "clarify(guard)", "answer", "order_total", "order_add", "order_set")


def verdict(case: dict, rec: dict) -> str:
    a = rec["attempts"][-1] if rec.get("attempts") else {}
    sent = a.get("sent")
    exp = case["expect"]
    if exp == "send":
        if not sent:
            return "missed"
        want = dict(case["command"])
        got = {"cmd": sent["cmd"], **sent["args"]}
        ok = all(_eq(got.get(k), v) for k, v in want.items()) and set(got) - {"cmd"} <= set(want) - {"cmd"} | {"mode"}
        return "correct_command" if ok else "misexecuted"
    if sent:
        return "misexecuted"
    intent = a.get("intent", "")
    if exp == "clarify":
        return "correct_askback" if intent in ASK_INTENTS else f"wrong_reply({intent})"
    if exp in ("no_send", "reject"):        # judged by the outcome: nothing was sent (model refusal or program check)
        return "correct_no_command"
    return "correct_no_command" if intent == exp else f"wrong_reply({intent})"


def _eq(a, b):
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b))
    return a == b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="configs/chat/utterances.yaml")
    ap.add_argument("--config", default="configs/chat.yaml")
    ap.add_argument("--out", default="outputs/chat/eval")
    ap.add_argument("--repeat", type=int, default=1, help="run the table this many times (model variability)")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    table = load(a.table)
    llm = OpenAICompatLLM.from_cfg(cfg["llm"])
    models = llm.models()
    warm = ChatAgent(lambda *a, **k: {}, llm, cfg).warmup()
    print(f"warm-up {warm:.1f} s", flush=True)
    results = []
    for rep in range(a.repeat):
        for n, case in enumerate(table["cases"]):
            st = json.loads(json.dumps(table["presets"][case["preset"]]))
            sent: list = []

            def call(cmd, request_id=None, _st=st, _sent=sent, **args):
                if cmd == "task_state":
                    return _st
                _sent.append({"cmd": cmd, **args})
                return {"queued": True}

            agent = ChatAgent(call, llm, cfg)
            t0 = time.monotonic()
            agent.handle(case["text"])
            rec = agent.transcript[-1]
            v = verdict(case, rec)
            a_ = rec["attempts"][-1] if rec.get("attempts") else {}
            results.append({"repeat": rep, "case": n, "group": case["group"], "preset": case["preset"],
                            "text": case["text"], "expect": case["expect"], "expected_command": case.get("command"),
                            "verdict": v, "model_output": a_.get("model_output"), "error": a_.get("error"),
                            "sent": a_.get("sent", {}).get("cmd") and {"cmd": a_["sent"]["cmd"], **a_["sent"]["args"]},
                            "reply": rec.get("replies"), "latency_s": round(time.monotonic() - t0, 3)})
            mark = "OK " if v.startswith("correct") else "NG "
            print(f"{mark}[{case['group']:<9}] {case['text']}  -> {v}" + ("" if v.startswith("correct") else
                  f"  {json.dumps(results[-1]['sent'] or results[-1]['reply'], ensure_ascii=False)}"), flush=True)
    groups: dict = {}
    for r in results:
        g = groups.setdefault(r["group"], {})
        key = r["verdict"].split("(")[0]
        g[key] = g.get(key, 0) + 1
    totals: dict = {}
    for g in groups.values():
        for k, v in g.items():
            totals[k] = totals.get(k, 0) + v
    lat = sorted(r["latency_s"] for r in results)
    summary = {"model": cfg["llm"]["model"], "served": models, "cases": len(table["cases"]), "repeat": a.repeat,
               "totals": totals, "by_group": groups,
               "latency_s": {"p50": lat[len(lat) // 2], "max": lat[-1]} if lat else None}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
