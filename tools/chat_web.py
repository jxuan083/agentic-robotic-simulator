#!/usr/bin/env python3
"""Chat web page: the Traditional Chinese chat (tools/chat.py, tools/chat_window.py) in a browser, with progress lines,
quick replies to the open question, pause / resume and voice input. Same agent and rules as the Tk window.

  uv run tools/chat_web.py --backend mujoco [--load --experiment lab_20261120 --block-colors red,red,blue] [--port 8765]
  open http://127.0.0.1:8765

Standard library only (http.server + server-sent events); the page is tools/chat_web/index.html, no CDN (the venue
may have no network). Voice: the page records while the mic button is held and posts the audio to /api/stt, which
uses faster-whisper when it is installed (`--stt-model small`); without it the page falls back to the browser's
speech recognition when the browser has one. Recognised text goes into the input box and is sent only by the user.

Answer binding works as in the Tk window (agentic_sim.chat.display_binding), but on the page: the page remembers the
question id it displayed when the input box went from empty to non-empty and sends it with the line (`shown`), plus
the id displayed now (`on_screen`, what the user has seen). Pause / resume are the simulator panel's buttons
(`pause` / `resume`); the chat itself still never drives or stops the arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.llm import LLMError, OpenAICompatLLM  # noqa: E402
from agentic_sim.chat.zh import report_line  # noqa: E402
from agentic_sim.config import load  # noqa: E402
from agentic_sim.server.client import RemoteError  # noqa: E402

PAGE = os.path.join(ROOT, "tools", "chat_web", "index.html")
LINK_ERRORS = (RemoteError, ConnectionError, OSError, TimeoutError)

# quick replies for the open question: option -> what the user would say (sent through the model like typed text)
QUICK = {"return": "放回去", "deliver": "照樣送出", "wait": "等補料", "reduce": "不用補了"}


def tag(line: str) -> str:
    if line.startswith("[進度]"):
        return "progress"
    if line.startswith("[系統]"):
        return "system"
    return "error" if line.startswith("錯誤") else "agent"


def quick_replies(q: dict | None, words) -> list:
    if not q:
        return []
    out = [QUICK[o] for o in q.get("options", []) if o in QUICK]
    if "substitute" in q.get("options", []):
        out += [f"改成{words.color(c)}" for c in q.get("substitute_candidates") or []]
    return out


class WebChat:
    """Everything except HTTP: the agents, the message log the pages read, one turn at a time.

    call(name) -> a callable call(cmd, **args) to the simulator server for that client name. Messages are appended to
    `log` (each with `seq`); pages read them from any seq on, so a reloaded page gets the whole conversation again."""

    def __init__(self, call, llm, cfg: dict):
        self.cfg, self.llm = cfg, llm
        self.agent = ChatAgent(call("chat-web"), llm, cfg)
        self.reporter = ChatAgent(call("chat-web-reports"), llm, cfg)
        self.control = call("chat-web-control")
        self.log: list = []
        self.cv = threading.Condition()
        self.busy = False
        self.question = None          # (text, id, quick replies) shown in the question card
        self.conn = True

    # ---- message log ------------------------------------------------------------------------------
    def put(self, kind: str, **data):
        with self.cv:
            self.log.append({"seq": len(self.log), "kind": kind, **data})
            self.cv.notify_all()

    def since(self, seq: int, wait: float = 15.0) -> list:
        with self.cv:
            if seq >= len(self.log):
                self.cv.wait(wait)
            return self.log[seq:]

    def lines(self, lines, qid=...):
        if lines:
            self.put("lines", lines=[{"text": x, "tag": tag(x)} for x in lines],
                     **({} if qid is ... else {"qid": qid}))

    # ---- simulator side ---------------------------------------------------------------------------
    def connect(self):
        """Report cursor first, then the question the task already waits on (its event is before the cursor)."""
        self.reporter.events_seen = self.reporter.call("task_events").get("next", 0)
        shown = [f"[系統] {x}" for x in self.agent.show_open_question()]
        self.reporter.q_shown = self.agent.q_shown
        self.lines(shown, self.agent.q_shown if shown else ...)

    def pump(self):
        before = self.reporter.q_shown
        lines = [report_line(x) for x in self.reporter.poll_reports()]
        after = self.reporter.q_shown
        self.lines(lines, after if after != before else ...)
        qn = self.reporter.state().get("question")
        card = (self.reporter.words.question(qn), qn["id"], quick_replies(qn, self.reporter.words)) if qn else None
        if card != self.question:
            self.question = card
            self.put("question", text=card[0] if card else None, qid=card[1] if card else None,
                     quick=card[2] if card else [])

    def run_polling(self, stop: threading.Event):
        while not stop.is_set():
            try:
                self.pump()
                if not self.conn:
                    self.conn = True
                    self.put("conn", ok=True)
            except LINK_ERRORS:
                if self.conn:
                    self.conn = False
                    self.put("conn", ok=False)
            stop.wait(float(self.cfg.get("poll_s", 0.25)))

    # ---- one user turn ---------------------------------------------------------------------------------
    def send(self, text: str, shown, on_screen) -> bool:
        """Start one turn; False while the previous one is still running. `on_screen`: the question the page shows
        now (the user has seen it); `shown`: the one displayed when this line was started (the answer's binding)."""
        text = (text or "").strip()
        if not text:
            return True
        with self.cv:
            if self.busy:
                return False
            self.busy = True
        if on_screen is not None:
            self.agent.q_shown = on_screen
        self.put("user", text=text)
        threading.Thread(target=self._turn, args=(text, shown), daemon=True).start()
        return True

    def _turn(self, text, shown):
        try:
            self.lines(self.agent.handle(text, shown=shown), self.agent.q_shown)
        except LINK_ERRORS as e:
            self.lines([f"錯誤：{e}"])
        finally:
            with self.cv:
                self.busy = False
            self.put("idle")

    def simulator(self, cmd: str) -> dict:
        """pause / resume: the simulator panel's buttons."""
        try:
            r = self.control(cmd)
        except LINK_ERRORS as e:
            return {"ok": False, "error": str(e)}
        if isinstance(r, dict) and r.get("error"):
            return {"ok": False, "error": r["error"]}
        self.put("paused", paused=cmd == "pause")
        return {"ok": True}


class Stt:
    """faster-whisper on the audio the page recorded (optional; the page falls back to the browser's recogniser)."""

    def __init__(self, model: str | None, device: str):
        self.model_name, self.device, self.model, self.error = model, device, None, None
        self.lock = threading.Lock()
        if model:
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                self.error = "faster-whisper 沒有安裝（uv pip install faster-whisper）"
                self.model_name = None

    @property
    def available(self) -> bool:
        return self.model_name is not None

    def transcribe(self, audio: bytes, suffix: str) -> str:
        with self.lock:
            if self.model is None:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_name, device=self.device,
                                          compute_type="int8" if self.device == "cpu" else "float16")
            with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                f.write(audio)
                f.flush()
                segs, _ = self.model.transcribe(f.name, language="zh", vad_filter=True,
                                                initial_prompt="以下是繁體中文：我要兩顆紅色、一顆藍色，藍色先放。")
                return "".join(s.text for s in segs).strip()


def handler(app_ref: dict, stt: Stt):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):   # quiet
            pass

        def _json(self, code: int, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                with open(PAGE, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/info":
                self._json(200, {"stt": stt.available, "stt_error": stt.error})
            elif path == "/api/events":
                self._events()
            else:
                self._json(404, {"error": "not found"})

        def _events(self):
            """Server-sent events: every message from Last-Event-ID (or 0) on; reloading the page replays all."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seq = int(self.headers.get("Last-Event-ID", -1)) + 1
            try:
                while True:
                    app = app_ref["app"]
                    msgs = app.since(seq) if app else (time.sleep(0.5) or [])
                    msgs = msgs or ([] if app else app_ref["boot"][seq:])
                    for m in msgs:
                        self.wfile.write(f"id: {m['seq']}\ndata: {json.dumps(m, ensure_ascii=False)}\n\n".encode())
                        seq = m["seq"] + 1
                    if not msgs:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            app = app_ref["app"]
            if path == "/api/stt":
                if not stt.available:
                    return self._json(501, {"error": stt.error or "沒有啟用語音辨識（--stt-model）"})
                ctype = self.headers.get("Content-Type", "")
                suffix = ".webm" if "webm" in ctype else ".ogg" if "ogg" in ctype else ".mp4" if "mp4" in ctype else ".wav"
                try:
                    return self._json(200, {"text": stt.transcribe(self._body(), suffix)})
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": f"語音辨識失敗：{e}"})
            if app is None:
                return self._json(503, {"error": "還在啟動"})
            try:
                req = json.loads(self._body() or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            if path == "/api/send":
                ok = app.send(req.get("text", ""), req.get("shown"), req.get("on_screen"))
                return self._json(200 if ok else 409, {"ok": ok})
            if path in ("/api/pause", "/api/resume"):
                return self._json(200, app.simulator(path.rsplit("/", 1)[1]))
            self._json(404, {"error": "not found"})

    return H


class Boot:
    """Messages before the agents exist (start-up status, fatal errors), with the same seq numbering as WebChat."""

    def __init__(self, app_ref: dict):
        self.ref = app_ref

    def put(self, kind: str, **data):
        b = self.ref["boot"]
        b.append({"seq": len(b), "kind": kind, **data})


def start(a, cfg: dict, app_ref: dict, stop: threading.Event):
    from chat import Link, versions
    boot = Boot(app_ref)
    boot.put("status", text="連線中…")
    try:
        llm = OpenAICompatLLM.from_cfg(cfg["llm"])
        llm.models()
    except LLMError as e:
        boot.put("fatal", text=f"模型服務沒有回應：{e}\n先啟動模型服務（tools/lab.py 會自動啟動）")
        return
    try:
        if a.load:
            colors = a.block_colors.split(",") if a.block_colors else None
            opts = {"block_colors": colors, "blocks": len(colors)} if colors else {}
            Link(a.backend, "chat-web-load")("load_experiment", name=a.experiment, options=opts)
        app = WebChat(lambda name: Link(a.backend, name), llm, cfg)
        app.log = list(app_ref["boot"])          # keep the start-up messages and their seq numbers
        if a.load:
            app.lines([f"[系統] 已載入 {a.experiment}：{a.block_colors or '預設積木'}"])
        app.put("status", text="準備模型中…")
        dt = app.agent.warmup()
        info = versions(llm, app.agent.call)
        app_ref["app"] = app
        app.connect()
        app.put("status", text=f"模擬器 {info['engine']}｜模型 {info['model']}（{info['model_server']}，暖機 {dt:.0f} 秒）")
        app.put("ready")
    except LINK_ERRORS as e:
        boot.put("fatal", text=f"連不上模擬器（{a.backend}）：{e}")
        return
    app.run_polling(stop)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["isaac", "mujoco"], required=True)
    ap.add_argument("--config", default="configs/chat.yaml")
    ap.add_argument("--load", action="store_true", help="(re)load the experiment first (fresh scene, no task)")
    ap.add_argument("--experiment", default="lab_20261120")
    ap.add_argument("--block-colors", default=None, help="with --load: colour name per block, comma-separated")
    ap.add_argument("--base-url", default=None, help="model service URL (default: configs/chat.yaml llm.base_url)")
    ap.add_argument("--model", default=None, help="served model name (default: configs/chat.yaml llm.model)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the page from another device on the LAN")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stt-model", default=None, help="faster-whisper model for voice input (e.g. small, medium)")
    ap.add_argument("--stt-device", default="cpu", help="cpu (default: keeps the GPU for the simulator) or cuda")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load(a.config)
    if a.base_url:
        cfg["llm"]["base_url"] = a.base_url
    if a.model:
        cfg["llm"]["model"] = a.model
    app_ref = {"app": None, "boot": []}
    stop = threading.Event()
    threading.Thread(target=start, args=(a, cfg, app_ref, stop), daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), handler(app_ref, Stt(a.stt_model, a.stt_device)))
    srv.daemon_threads = True
    print(f"聊天頁面：http://{a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
