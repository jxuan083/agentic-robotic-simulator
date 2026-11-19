"""Traditional Chinese chat with the task executor of a running simulator server.

  docker/vllm/serve.sh up                                      # model service (Qwen3.5-9B, OpenAI-compatible API)
  tools/mujoco_server.sh up                                    # or docker/isaac/server.sh up
  uv run tools/chat.py --backend mujoco --load --block-colors red,red,red,blue,blue,yellow
  > 我要三顆紅色
  > 總共一紅兩藍就好
  > 放回去
  > 現在做到哪？

Scripted, reproducible dialogue (waits on the task state between lines; saves the transcript):
  uv run tools/chat.py --backend mujoco --script configs/chat/dialogues/hold_return.yaml \
      --out outputs/chat/hold_return_mujoco

The chat only sends task commands (task_order / task_answer); to stop the arm use the simulator's Pause (P).
Reports (order applied, questions, shortage, over-delivery, done, anomalies) are printed as they happen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.llm import LLMError, OpenAICompatLLM  # noqa: E402
from agentic_sim.config import load  # noqa: E402
from agentic_sim.server.client import RemoteError, SimClient  # noqa: E402


class Link:
    """SimClient that reconnects once per call after a dropped connection (the caller retries with its id)."""

    def __init__(self, backend: str, name: str):
        self.backend, self.name, self.c = backend, name, None

    def __call__(self, cmd, **args):
        if self.c is None:
            self.c = SimClient.for_backend(self.backend, name=self.name, timeout=10.0)
            self.c.connect()
        try:
            return self.c.call(cmd, **args)
        except (ConnectionError, OSError):
            try:
                self.c.close()
            finally:
                self.c = None
            raise


def matches(cond: dict, st: dict, events: list) -> bool:
    if "skill" in cond:
        sk = st.get("skill") or {}
        want = cond["skill"]
        if sk.get("name") != want or any(sk.get(k) != cond[k] for k in ("color", "nth", "phase") if k in cond):
            return False
    if "question" in cond:
        q = st.get("question")
        if not q or q.get("kind") != cond["question"]:
            return False
    if "state" in cond and st.get("state") != cond["state"]:
        return False
    if "event" in cond and not any(e["kind"] == cond["event"] for e in events):
        return False
    return True


def versions(llm, call) -> dict:
    out = {"model": llm.model}
    try:
        import urllib.request
        root = llm.base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(root + "/version", timeout=3) as r:
            out["model_server"] = "vllm " + json.loads(r.read()).get("version", "?")
    except Exception:  # noqa: BLE001
        try:   # llama.cpp server
            with urllib.request.urlopen(root + "/props", timeout=3) as r:
                p = json.loads(r.read())
            out["model_server"] = f"llama.cpp {p.get('build_info', '?')}"
            out["model_file"] = os.path.basename(p.get("model_path", ""))
        except Exception:  # noqa: BLE001
            out["model_server"] = "unknown"
    st = call("status")
    out["engine"] = st.get("backend")
    out["experiment"] = st.get("experiment")
    out["options"] = st.get("options")
    return out


class Repl:
    """Interactive session: reports are polled on a second connection and printed; the question id on screen when a
    line arrives is the one its answer is bound to.

      repl.connect()            # report cursor, then show a question that is already open
      repl.pump()               # background thread: print new reports; the agent learns which question is on screen
      repl.run(read_line)       # input loop: the question on screen is sampled when reading a line starts
    """

    def __init__(self, agent: ChatAgent, reporter: ChatAgent, poll, say):
        self.agent, self.reporter, self.poll, self.say = agent, reporter, poll, say
        self.copied = None

    def connect(self):
        self.reporter.events_seen = self.poll("task_events").get("next", 0)   # later questions arrive as events
        for line in self.agent.show_open_question():                         # its event was before the cursor
            self.say(f"[系統] {line}")
        self.reporter.q_shown = self.copied = self.agent.q_shown

    def pump(self):
        for line in self.reporter.poll_reports():
            self.say(f"[系統] {line}")
        if self.reporter.q_shown != self.copied:          # printed above: the user has now seen it
            self.agent.q_shown = self.copied = self.reporter.q_shown

    def turn(self, text: str, shown):
        for line in self.agent.handle(text, shown=shown):
            self.say(line)

    def run(self, read_line):
        """read_line(prompt) -> str (raises EOFError / KeyboardInterrupt to end). The question id is sampled before
        reading starts: a question printed while the user types does not become the one the line answers."""
        while True:
            shown = self.agent.q_shown
            try:
                text = read_line("> ")
            except (EOFError, KeyboardInterrupt):
                return
            if text.strip() in ("exit", "quit", "離開"):
                return
            try:
                self.turn(text, shown)
            except RemoteError as e:
                self.say(f"錯誤：{e}")


def run_script(agent: ChatAgent, call, spec: dict, out_dir: str, say) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    seen_events: list = []
    t0 = time.monotonic()
    timeout = float(spec.get("timeout_s", 300))
    log = []

    def pump():
        nonlocal agent
        for line in agent.poll_reports():
            say(f"[系統] {line}")
            log.append({"t": round(time.monotonic() - t0, 3), "who": "system", "text": line})
        seen_events[:] = [r for r in agent.transcript if "event" in r]

    for turn in spec["turns"]:
        cond = turn.get("wait")
        while cond:
            pump()
            st = call("task_state")
            if matches(cond, st, [{"kind": r["event"]} for r in seen_events]):
                break
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"waiting for {cond}")
            time.sleep(0.05)
        if turn.get("after_s"):
            end = time.monotonic() + float(turn["after_s"])
            while time.monotonic() < end:
                pump()
                time.sleep(0.05)
        if turn.get("reconnect"):     # a new CLI connects: report cursor now, no question shown yet, fresh history
            new = ChatAgent(call, agent.llm, agent.cfg, log=agent.log)
            new.transcript = agent.transcript
            new.events_seen = call("task_events").get("next", 0)
            agent = new
            say("（聊天重新連線）")
            log.append({"t": round(time.monotonic() - t0, 3), "who": "system", "text": "reconnect"})
            agent.transcript.append({"reconnect": True, "events_from": agent.events_seen})
            for line in agent.show_open_question():
                say(f"[系統] {line}")
                log.append({"t": round(time.monotonic() - t0, 3), "who": "system", "text": line})
        if "say" in turn:
            say(f"> {turn['say']}")
            log.append({"t": round(time.monotonic() - t0, 3), "who": "user", "text": turn["say"]})
            for line in agent.handle(turn["say"]):
                say(line)
                log.append({"t": round(time.monotonic() - t0, 3), "who": "assistant", "text": line})
    for _ in range(10):   # last reports
        pump()
        time.sleep(0.1)
    final = call("task_state")
    checks = []
    exp = spec.get("expect", {})
    led = final.get("ledger") or {}
    if "delivered" in exp:
        checks.append({"name": "final_delivered", "ok": led.get("delivered") == exp["delivered"],
                       "expected": exp["delivered"], "observed": led.get("delivered")})
    if "target" in exp:
        checks.append({"name": "final_target", "ok": led.get("target") == exp["target"], "expected": exp["target"],
                       "observed": led.get("target")})
    if "revision" in exp:
        checks.append({"name": "final_revision", "ok": led.get("revision") == exp["revision"],
                       "expected": exp["revision"], "observed": led.get("revision")})
    if exp.get("done"):
        checks.append({"name": "done", "ok": final.get("state") == "done" and bool(led.get("done")),
                       "state": final.get("state")})
    sent = [a["sent"] for r in agent.transcript if "attempts" in r for a in r["attempts"] if "sent" in a]
    if "commands" in exp:
        got = [{"cmd": s["cmd"], **s["args"]} for s in sent]
        ok = len(got) == len(exp["commands"]) and all(all(g.get(k) == v for k, v in e.items())
                                                       for g, e in zip(got, exp["commands"]))
        checks.append({"name": "commands", "ok": ok, "expected": exp["commands"], "observed": got})
    checks.append({"name": "no_duplicate_effects", "ok": not any(s["response"].get("duplicate") for s in sent)})
    return {"dialogue": spec["name"], "passed": sum(c["ok"] for c in checks), "checks_total": len(checks),
            "checks": checks, "final_state": final, "log": log, "wall_s": round(time.monotonic() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["isaac", "mujoco"], required=True)
    ap.add_argument("--config", default="configs/chat.yaml")
    ap.add_argument("--load", action="store_true", help="(re)load minimal_pick_place first (fresh scene, no task)")
    ap.add_argument("--block-colors", default=None, help="with --load: colour name per block, comma-separated")
    ap.add_argument("--script", default=None, help="dialogue YAML (configs/chat/dialogues/*.yaml)")
    ap.add_argument("--out", default=None, help="script mode: output directory (transcript, result)")
    ap.add_argument("--base-url", default=None, help="model service URL (default: configs/chat.yaml llm.base_url)")
    ap.add_argument("--model", default=None, help="served model name (default: configs/chat.yaml llm.model)")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.base_url:
        cfg["llm"]["base_url"] = a.base_url
    if a.model:
        cfg["llm"]["model"] = a.model
    llm = OpenAICompatLLM.from_cfg(cfg["llm"])
    try:
        llm.models()
    except LLMError as e:
        print(f"錯誤：{e}\n先啟動模型服務：docker/vllm/serve.sh up", file=sys.stderr)
        return 2
    call = Link(a.backend, "chat")
    lock = threading.Lock()

    def say(line):
        with lock:
            print(line, flush=True)

    spec = load(a.script) if a.script else None
    colors = (spec or {}).get("block_colors") or (a.block_colors.split(",") if a.block_colors else None)
    try:
        if a.load or spec:
            opts = {"block_colors": colors, "blocks": len(colors)} if colors else {}
            call("load_experiment", name="minimal_pick_place", options=opts)
            say(f"已載入場景：{', '.join(colors) if colors else '預設積木'}")
        agent = ChatAgent(call, llm, cfg, log=lambda m: say(f"[chat] {m}"))
        say("準備模型中…")
        say(f"模型就緒（{agent.warmup():.1f} 秒）")
        info = versions(llm, call)
        if spec:
            out_dir = a.out or os.path.join("outputs", "chat", f"{spec['name']}_{a.backend}")
            res = run_script(agent, call, spec, out_dir, say)
            res["versions"] = info
            with open(os.path.join(out_dir, "transcript.jsonl"), "w") as f:
                for r in agent.transcript:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            say(f"\n對話 {spec['name']}：{res['passed']}/{res['checks_total']} 項通過；結果在 {out_dir}")
            for c in res["checks"]:
                if not c["ok"]:
                    say(f"  未通過：{c}")
            return 0 if res["passed"] == res["checks_total"] else 1
        say(f"模型 {info['model']}（{info['model_server']}），模擬器 {info['engine']}。輸入 exit 離開。")
        stop = threading.Event()
        poll = Link(a.backend, "chat-reports")
        reporter = ChatAgent(poll, llm, cfg)
        repl = Repl(agent, reporter, poll, say)
        repl.connect()

        def loop():
            while not stop.is_set():
                try:
                    repl.pump()
                except (RemoteError, ConnectionError, OSError):
                    pass
                stop.wait(float(cfg.get("poll_s", 0.25)))

        threading.Thread(target=loop, daemon=True).start()
        repl.run(input)
        stop.set()
        return 0
    except (RemoteError, ConnectionError, OSError, TimeoutError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
