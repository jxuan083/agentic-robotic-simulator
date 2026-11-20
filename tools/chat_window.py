#!/usr/bin/env python3
"""Chat window: the Traditional Chinese chat (tools/chat.py) in its own small desktop window, separate from the
simulator. Same agent and rules as the CLI (answers bound to the question on screen, follow-ups bound to their
question, the open question shown again on reconnect); only the input / output is a Tk window.

  uv run tools/chat_window.py --backend mujoco [--load --experiment lab_20261120 --block-colors red,red,blue]

Window: status line (simulator, model, connection), the question the task is waiting on, the conversation, an input
box (Enter sends). Closing the window ends only the chat; the simulator keeps running. tools/lab.py starts it.
The question an answer is bound to is the one on screen when typing of that line started (as the CLI samples it
before reading a line), so a question that appears while typing does not receive the half-typed answer.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from agentic_sim.chat.agent import ChatAgent  # noqa: E402
from agentic_sim.chat.llm import LLMError, OpenAICompatLLM  # noqa: E402
from agentic_sim.config import load  # noqa: E402
from agentic_sim.server.client import RemoteError  # noqa: E402
from chat import Link, Repl, versions  # noqa: E402

TITLE = "SO-101 聊天"
COLORS = {"bg": "#1e2127", "panel": "#2a2e36", "text": "#e6e6e6", "muted": "#9aa3b2", "user": "#8ec7ff",
          "system": "#f2c879", "agent": "#e6e6e6", "error": "#ff8a80", "question_bg": "#3b3420",
          "question_fg": "#ffd77a", "ok": "#7fd18b"}


class ChatWindow:
    def __init__(self, root: tk.Tk, a, cfg: dict):
        self.root, self.a, self.cfg = root, a, cfg
        self.q: queue.Queue = queue.Queue()      # (kind, payload) from the worker / poll threads to the UI
        self.stop = threading.Event()
        self.busy = False
        self.shown_at_typing = ...             # question id on screen when typing of the current line started
        self.agent = self.repl = self.reporter = None
        self._build()
        self.typing = list(getattr(a, "type", None) or [])   # acceptance: lines typed into the box like a user
        self.question_open = False
        threading.Thread(target=self._start, daemon=True).start()
        self.root.after(80, self._drain)

    # ---- layout -----------------------------------------------------------------------------------
    def _build(self):
        r = self.root
        r.title(TITLE)
        r.geometry("520x640")
        r.configure(bg=COLORS["bg"])
        base = tkfont.nametofont("TkDefaultFont")
        fam = next((f for f in ("Noto Sans CJK TC", "Noto Sans TC", "Noto Sans CJK JP", "AR PL UMing TW",
                                "WenQuanYi Micro Hei", "Droid Sans Fallback") if f in tkfont.families()), base.actual("family"))
        self.f = (fam, 12)
        self.fb = (fam, 12, "bold")
        self.status = tk.Label(r, text="連線中…", anchor="w", bg=COLORS["panel"], fg=COLORS["muted"], font=(fam, 10),
                               padx=8, pady=4)
        self.status.pack(fill="x")
        self.question = tk.Label(r, text="（沒有待回答的問題）", anchor="w", justify="left", wraplength=496,
                                 bg=COLORS["question_bg"], fg=COLORS["question_fg"], font=self.fb, padx=8, pady=6)
        self.question.pack(fill="x")
        self.log = scrolledtext.ScrolledText(r, wrap="word", state="disabled", bg=COLORS["bg"], fg=COLORS["text"],
                                             font=self.f, relief="flat", padx=8, pady=6, insertbackground="white")
        self.log.pack(fill="both", expand=True)
        for tag in ("user", "system", "agent", "error", "ok", "muted"):
            self.log.tag_configure(tag, foreground=COLORS[tag if tag != "muted" else "muted"])
        bar = tk.Frame(r, bg=COLORS["panel"], padx=6, pady=6)
        bar.pack(fill="x")
        self.entry = tk.Entry(bar, font=self.f, bg="#11141a", fg=COLORS["text"], insertbackground="white",
                              relief="flat")
        self.entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.entry.bind("<Return>", lambda e: self.send())
        self.entry.bind("<KP_Enter>", lambda e: self.send())
        self.entry.bind("<Key>", self._on_key, add="+")
        self.btn = tk.Button(bar, text="送出", command=self.send, font=self.f, relief="flat", padx=12)
        self.btn.pack(side="left", padx=(6, 0))
        self.entry.configure(state="disabled")
        self.btn.configure(state="disabled")
        r.protocol("WM_DELETE_WINDOW", self.close)

    def write(self, text: str, tag: str = "agent"):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.configure(state="disabled")
        self.log.see("end")

    # ---- threads ----------------------------------------------------------------------------------
    def _say(self, line: str):
        tag = "system" if line.startswith("[系統]") else "error" if line.startswith("錯誤") else "agent"
        self.q.put(("line", (line, tag)))

    def _start(self):
        a, cfg = self.a, self.cfg
        try:
            llm = OpenAICompatLLM.from_cfg(cfg["llm"])
            llm.models()
        except LLMError as e:
            self.q.put(("fatal", f"模型服務沒有回應：{e}\n先啟動模型服務（tools/lab.py 會自動啟動）"))
            return
        try:
            call = Link(a.backend, "chat-window")
            if a.load:
                colors = a.block_colors.split(",") if a.block_colors else None
                opts = {"block_colors": colors, "blocks": len(colors)} if colors else {}
                call("load_experiment", name=a.experiment, options=opts)
                self._say(f"[系統] 已載入 {a.experiment}：{', '.join(colors) if colors else '預設積木'}")
            self.agent = ChatAgent(call, llm, cfg)
            self.q.put(("status", "準備模型中…"))
            dt = self.agent.warmup()
            info = versions(llm, call)
            poll = Link(a.backend, "chat-window-reports")
            self.reporter = ChatAgent(poll, llm, cfg)
            self.repl = Repl(self.agent, self.reporter, poll, self._say)
            self.repl.connect()
            self.q.put(("status", f"模擬器 {info['engine']}｜模型 {info['model']}（{info['model_server']}，暖機 {dt:.0f} 秒）"))
            self.q.put(("ready", None))
        except (RemoteError, ConnectionError, OSError, TimeoutError) as e:
            self.q.put(("fatal", f"連不上模擬器（{a.backend}）：{e}"))
            return
        while not self.stop.is_set():
            try:
                self.repl.pump()
                st = self.reporter.state()
                qn = st.get("question")
                self.q.put(("question", self.reporter.words.question(qn) if qn else None))
                self.q.put(("conn", True))
            except (RemoteError, ConnectionError, OSError, TimeoutError):
                self.q.put(("conn", False))
            self.stop.wait(float(self.cfg.get("poll_s", 0.25)))

    def _turn(self, text: str, shown):
        try:
            self.repl.turn(text, shown)
        except (RemoteError, ConnectionError, OSError, TimeoutError) as e:
            self._say(f"錯誤：{e}")
        finally:
            self.q.put(("idle", None))

    # ---- UI events --------------------------------------------------------------------------------
    def _on_key(self, event):
        # the first character of a new line fixes the question this line answers (CLI: sampled before reading)
        if self.agent is not None and not self.entry.get() and event.char and event.char.isprintable():
            self.shown_at_typing = self.agent.q_shown

    def send(self):
        if self.busy or self.repl is None:
            return
        text = self.entry.get().strip()
        if not text:
            return
        shown = self.shown_at_typing if self.shown_at_typing is not ... else self.agent.q_shown
        self.shown_at_typing = ...
        self.entry.delete(0, "end")
        self.write(f"你：{text}", "user")
        self.busy = True
        self.btn.configure(state="disabled", text="處理中…")
        threading.Thread(target=self._turn, args=(text, shown), daemon=True).start()

    def _drain(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    self.write(*val)
                elif kind == "status":
                    self.status.configure(text=val, fg=COLORS["muted"])
                elif kind == "ready":
                    self.entry.configure(state="normal")
                    self.btn.configure(state="normal")
                    self.entry.focus_set()
                    self.write("可以開始下單，例如「我要兩顆紅色、一顆藍色」。", "muted")
                    if self.typing:
                        self.root.after(1500, self._type_next)
                elif kind == "question":
                    self.question_open = bool(val)
                    self.question.configure(text=f"待回答：{val}" if val else "（沒有待回答的問題）")
                elif kind == "conn":
                    if not val:
                        self.status.configure(text="模擬器連線中斷，重試中…", fg=COLORS["error"])
                elif kind == "idle":
                    self.busy = False
                    self.btn.configure(state="normal", text="送出")
                elif kind == "fatal":
                    self.status.configure(text="無法啟動", fg=COLORS["error"])
                    self.write(val, "error")
        except queue.Empty:
            pass
        if not self.stop.is_set():
            self.root.after(80, self._drain)

    def _type_next(self):
        """Acceptance input: type the next line character by character through the same key path, then send it.
        Each line waits until the previous turn is answered (and `--type-gap` seconds); a line starting with "?" also
        waits until the task shows a question (answers to shortage / held-item questions)."""
        if not self.typing or self.stop.is_set():
            return
        if self.busy:
            self.root.after(300, self._type_next)
            return
        if self.typing[0].startswith("?") and not self.question_open:
            self.root.after(300, self._type_next)   # "?answer": typed once the task asks a question
            return
        line = self.typing.pop(0).lstrip("?")

        def put(i=0):
            if i < len(line):
                ev = type("E", (), {"char": line[i]})()
                self._on_key(ev)
                self.entry.insert("end", line[i])
                self.root.after(60, put, i + 1)
            else:
                self.send()
                if self.typing:
                    self.root.after(int(1000 * self.a.type_gap), self._type_next)
        put()

    def close(self):
        self.stop.set()
        self.root.destroy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["isaac", "mujoco"], required=True)
    ap.add_argument("--config", default="configs/chat.yaml")
    ap.add_argument("--load", action="store_true", help="(re)load the experiment first (fresh scene, no task)")
    ap.add_argument("--experiment", default="lab_20261120")
    ap.add_argument("--block-colors", default=None, help="with --load: colour name per block, comma-separated")
    ap.add_argument("--base-url", default=None, help="model service URL (default: configs/chat.yaml llm.base_url)")
    ap.add_argument("--model", default=None, help="served model name (default: configs/chat.yaml llm.model)")
    ap.add_argument("--type", action="append", default=None, metavar="LINE",
                    help="acceptance: type this line into the window (repeatable, in order; '?text' waits for a question)")
    ap.add_argument("--type-gap", type=float, default=3.0, help="seconds between typed lines after each answer")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.base_url:
        cfg["llm"]["base_url"] = a.base_url
    if a.model:
        cfg["llm"]["model"] = a.model
    root = tk.Tk()
    ChatWindow(root, a, cfg)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
