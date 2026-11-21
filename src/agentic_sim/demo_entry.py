"""Launcher side of demonstration capture and readback (tools/sim.py, "錄示範" / "檢查上一筆"). Pure Python.

Recording: runs the capture command (agentic_sim.launch.build_record, the same command shown for copying) with
its output in a log file, and shows the recorder's state from <out_dir>/recorder_events.jsonl (written by
agentic_sim.capture.recorder): waiting for the cameras, recording with the frame count, saved (result and path).
Only events appended after the launch, from the process that wrote the first of them, belong to this run, so an
older episode in the same directory is never taken for the one just recorded.

Readback: runs the readback command (tools/read_episode.py, the same one shown for copying) as an offline tool;
no simulator or model service is started.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time

from agentic_sim.launch import Command

EVENTS_FILE = "recorder_events.jsonl"   # agentic_sim.capture.recorder.EVENTS_FILE (that module needs NumPy)


def readback_command(episode: str, figures: bool = False) -> Command:
    if figures:
        return Command(["uv", "run", "--with", "matplotlib", "tools/read_episode.py", episode, "--figures"])
    return Command(["uv", "run", "tools/read_episode.py", episode])


class EventReader:
    """New lines of the recorder's event file since the launch, from one recorder process."""

    def __init__(self, out_dir: str):
        self.path = os.path.join(out_dir, EVENTS_FILE)
        self.offset = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        self.pid = None
        self._partial = ""

    def read(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            f.seek(self.offset)
            data = f.read()
            self.offset = f.tell()
        data = self._partial + data
        lines = data.split("\n")
        self._partial = lines.pop()          # an incomplete last line waits for the next read
        out = []
        for line in lines:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self.pid is None:
                self.pid = ev.get("pid")
            if ev.get("pid") == self.pid:
                out.append(ev)
        return out


class Status:
    """The recorder state as one terminal line, from the events."""

    def __init__(self, target_color: str):
        self.state = "starting"
        self.target = target_color
        self.episode = None
        self.frames = 0
        self.saved = []                      # "saved" events of this run, in order

    def apply(self, ev: dict) -> str | None:
        """Update from one event; returns a line to print once (state changes), else None."""
        kind = ev.get("event")
        if kind == "episode":
            self.state, self.episode, self.frames = "waiting_cameras", ev.get("episode"), 0
            return f"新的一筆：{self.episode}（目標 {ev.get('target_color', self.target)}），等待兩台相機的第一張影像…"
        if kind in ("recording", "progress"):
            first = self.state != "recording"
            self.state, self.episode, self.frames = "recording", ev.get("episode", self.episode), ev.get("frames", 0)
            return "兩台相機就緒，開始錄製" if first else None
        if kind == "saved":
            self.state, self.frames = "saved", ev.get("frames", self.frames)
            self.saved.append(ev)
            reason = {"placed": "成功：放進米色盤", "reset": "未完成：按了 R", "closed": "未完成：視窗關閉"}.get(
                ev.get("reason"), ev.get("reason"))
            nxt = "；按 R 錄下一筆，或關閉 Isaac 視窗結束" if ev.get("reason") != "closed" else ""
            return (f"已存檔（{reason}）：{ev.get('frames')} 幀，{ev.get('duration_s')} s → {ev.get('episode')}{nxt}")
        if kind == "closed":
            self.state = "closed"
        return None

    def line(self) -> str:
        if self.state == "recording":
            return f"錄製中　目標 {self.target}　已錄 {self.frames} 幀　{self.episode}"
        if self.state == "waiting_cameras":
            return "等待相機…"
        if self.state == "saved":
            return "已存檔，等待 R 或關閉視窗"
        if self.state == "starting":
            return "啟動 Isaac 中（第一次載入約 1 分鐘）…"
        return ""


def run_recording(cmd: Command, root: str, out_dir: str, target_color: str, env: dict, log_dir: str,
                  out=sys.stdout, poll_s: float = 0.5) -> dict:
    """Run the capture command until the Isaac window closes; show the recorder state. Returns
    {"exit_code", "saved": [saved events], "log"}."""
    os.makedirs(os.path.join(root, out_dir), exist_ok=True)
    os.makedirs(os.path.join(root, log_dir), exist_ok=True)
    log_path = os.path.join(log_dir, datetime.datetime.now().strftime("record_%Y%m%d_%H%M%S.log"))
    reader = EventReader(os.path.join(root, out_dir))
    status = Status(target_color)
    tty = out.isatty()
    shown = ""

    def show(text_once=None):
        nonlocal shown
        if text_once:
            if tty and shown:
                out.write("\r\033[K")
            out.write(text_once + "\n")
            shown = ""
        cur = status.line()
        if tty:
            if cur != shown:
                out.write("\r\033[K" + cur)
                shown = cur
        out.flush()

    def drain():
        for ev in reader.read():
            show(status.apply(ev))

    print(f"Isaac 的輸出寫在 {log_path}", file=out)
    interrupted = False
    with open(os.path.join(root, log_path), "w") as log:
        # stdin is not the terminal: docker runs without a TTY and Ctrl+C stays with this menu (forwarded below)
        p = subprocess.Popen(cmd.argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT)
        show()
        try:
            while p.poll() is None:
                drain()
                time.sleep(poll_s)
        except KeyboardInterrupt:
            interrupted = True
            show("中斷：結束 Isaac（未存檔的那一筆會停在 status: recording；正常結束請關閉 Isaac 視窗）")
            try:
                p.wait(timeout=60)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                p.kill()
                p.wait()
        drain()
    if tty and shown:
        out.write("\r\033[K")
    code = p.returncode
    if code not in (0, None) and not interrupted:
        print(f"Isaac 以結束碼 {code} 結束；log 最後幾行：", file=out)
        with open(os.path.join(root, log_path), errors="replace") as f:
            for line in f.readlines()[-15:]:
                print("  " + line.rstrip(), file=out)
    return {"exit_code": code, "saved": status.saved, "log": log_path, "interrupted": interrupted}


def recent_episodes(root: str, out_dir: str, n: int = 10) -> list:
    """Existing episode directories (with a manifest) under out_dir, newest first."""
    base = os.path.join(root, out_dir)
    if not os.path.isdir(base):
        return []
    eps = [d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, "manifest.json"))]
    eps.sort(key=lambda d: os.path.getmtime(os.path.join(base, d, "manifest.json")), reverse=True)
    return [os.path.join(out_dir, d) for d in eps[:n]]
