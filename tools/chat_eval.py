"""Utterance table for the chat layer, run with the real model (no simulator needed: states are presets).

  uv run tools/chat_eval.py [--table configs/chat/utterances.yaml] [--out outputs/chat/eval] [--repeat 1]
                            [--base-url http://127.0.0.1:8000/v1 --model NAME]   # another model service

For every case the model's intent goes through the same checks as in the chat (agentic_sim.chat.intent.plan) against
the preset state; the command that would be sent is compared with the expected one. Verdicts are in
agentic_sim.chat.scoring: correct (command / ask-back on the right topic / reject / status / stop / no command),
mis-sent (wrong command, or a command where none was expected), missed, wrong ask-back, wrong kind of reply, and
service / format / transport failures (a failure counts even when nothing was sent). Writes result.json (per case:
model output, action, command, reply, verdict, latency) and prints the counts per group.
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
from agentic_sim.chat.scoring import summarize, verdict  # noqa: E402
from agentic_sim.config import load  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="configs/chat/utterances.yaml")
    ap.add_argument("--config", default="configs/chat.yaml")
    ap.add_argument("--out", default="outputs/chat/eval")
    ap.add_argument("--repeat", type=int, default=1, help="run the table this many times (model variability)")
    ap.add_argument("--base-url", help="model service URL (default: configs/chat.yaml llm.base_url)")
    ap.add_argument("--model", help="served model name (default: configs/chat.yaml llm.model)")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.base_url:
        cfg["llm"]["base_url"] = a.base_url
    if a.model:
        cfg["llm"]["model"] = a.model
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
            if st.get("question"):            # a table case assumes the user is looking at the open question
                agent.q_shown = st["question"]["id"]
            t0 = time.monotonic()
            agent.handle(case["text"])
            rec = agent.transcript[-1]
            v = verdict(case, rec)
            a_ = rec["attempts"][-1] if rec.get("attempts") else {}
            results.append({"repeat": rep, "case": n, "group": case["group"], "preset": case["preset"],
                            "text": case["text"], "expect": case["expect"], "expected_command": case.get("command"),
                            "verdict": v, "model_output": a_.get("model_output"), "error": a_.get("error"),
                            "sent": a_.get("sent", {}).get("cmd") and {"cmd": a_["sent"]["cmd"], **a_["sent"]["args"]},
                            "reply_kind": a_.get("reply"), "failure": a_.get("failure"),
                            "reply": rec.get("replies"), "latency_s": round(time.monotonic() - t0, 3)})
            mark = "OK " if v.startswith("correct") else "NG "
            print(f"{mark}[{case['group']:<9}] {case['text']}  -> {v}" + ("" if v.startswith("correct") else
                  f"  {json.dumps(results[-1]['sent'] or results[-1]['reply'], ensure_ascii=False)}"), flush=True)
    groups: dict = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r["verdict"])
    lat = sorted(r["latency_s"] for r in results if not r["verdict"].endswith("_failure"))
    pct = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]  # noqa: E731
    summary = {"model": cfg["llm"]["model"], "base_url": cfg["llm"]["base_url"], "served": models,
               "table": a.table, "cases": len(table["cases"]), "repeat": a.repeat, "warmup_s": round(warm, 1),
               "totals": summarize([r["verdict"] for r in results]),
               "per_repeat": [summarize([r["verdict"] for r in results if r["repeat"] == i]) for i in range(a.repeat)],
               "by_group": {g: summarize(v)["by_verdict"] for g, v in groups.items()},
               "latency_s": {"p50": pct(0.5), "p95": pct(0.95), "max": lat[-1], "n": len(lat)} if lat else None}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
