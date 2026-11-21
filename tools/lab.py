#!/usr/bin/env python3
"""One command for the lab environment: model service, simulator server (with the sensor camera views) and the chat
window, started in order; Ctrl+C stops the services this command started.

  uv run tools/lab.py                                   # MuJoCo, Q4 model, lab_20261120, default blocks
  uv run tools/lab.py --backend isaac --model bf16 --block-colors red,red,blue,blue,yellow
  uv run tools/lab.py --wrist-lens pinhole --no-chat

Steps (each waits until the previous one answers; a failure stops here, names the component and exits 1):
  1. model service   Q4_K_M (llama.cpp, docker/llamacpp/serve.sh, port 8001) or BF16 (vLLM, docker/vllm/serve.sh,
                     port 8000)
  2. simulator       persistent server (tools/mujoco_server.sh / docker/isaac/server.sh); the experiment is
                     (re)loaded with the options given here
  3. chat window     tools/chat_window.py
Ownership: each control script's `up` creates the service only when none exists (atomic: the container name, or a
lock for the simulator) and writes the identity of what it created (container ID / process ID) to ARS_CREATED_FILE.
Only that identity is stopped on exit (`down <id>`), also when it never became ready; a service that already existed
(even one still warming up, or started by another lab.py in between) is reused and left running. Names, ports and the
served model come from the scripts themselves (`serve.sh info`), so LLM_CONTAINER_NAME / LLM_PORT apply everywhere.
Closing the chat window keeps everything running: press Enter here to open it again (it shows the open question
first and continues the same task), or run tools/chat_window.py yourself. Ctrl+C (or the simulator window closing)
ends the session.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.server.client import SimClient, probe  # noqa: E402

MODELS = {   # model service: control script, ready timeout (s); container name, URL and model name: `<script> info`
    "q4": (["docker/llamacpp/serve.sh"], 300),
    "bf16": (["docker/vllm/serve.sh"], 900),
}
SERVERS = {  # simulator server: control script, experiment env var, ready timeout (s)
    "mujoco": (["tools/mujoco_server.sh"], "MUJOCO_SERVER_EXPERIMENT", 90),
    "isaac": (["docker/isaac/server.sh"], "ISAAC_SERVER_EXPERIMENT", 300),
}


def say(msg: str):
    print(f"[lab] {msg}", flush=True)


def model_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def script_info(script: list, env: dict) -> dict:
    """`<script> info`: name=, url=, model= lines (the script's own settings, environment included)."""
    r = subprocess.run(script + ["info"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


def wait(cond, timeout: float, what: str, every: float = 2.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(every)
    raise TimeoutError(f"{what} did not become ready within {timeout:.0f} s")


class Lab:
    def __init__(self, a):
        self.a = a
        self.owned = []            # (description, stop command) of services this command created, in start order
        self.env = dict(os.environ)
        self.env.setdefault("DISPLAY", ":1")
        self.env.setdefault("XAUTHORITY", f"/run/user/{os.getuid()}/gdm/Xauthority")

    def run(self, cmd, timeout=None, env=None):
        return subprocess.run(cmd, cwd=ROOT, env=env or self.env, timeout=timeout)

    def _up(self, what: str, script: list, timeout: float, ready) -> None:
        """`script up` (bounded by timeout), then ready(). The script writes the identity of an instance it created to
        ARS_CREATED_FILE; only that one is recorded as ours (and stopped by `script down <id>`), also when `up` failed
        timed out or was interrupted (Ctrl+C while it warms up: the ID is read and recorded first, then the interrupt
        goes on to the caller, whose stop() removes it). Nothing written: the service already existed and is reused."""
        fd, created = tempfile.mkstemp(prefix="ars-created-")
        os.close(fd)
        os.unlink(created)                           # docker --cidfile wants a path that does not exist yet
        t0, err = time.monotonic(), None
        try:
            r = self.run(script + ["up"], timeout=timeout, env=dict(self.env, ARS_CREATED_FILE=created))
            if r.returncode != 0:
                err = RuntimeError(f"{what}: {' '.join(script)} up failed (exit {r.returncode})")
        except subprocess.TimeoutExpired:
            err = TimeoutError(f"{what} did not become ready within {timeout:.0f} s")
        except KeyboardInterrupt as e:               # Ctrl+C during `up`: record what was created, then re-raise
            err = e
        ident = ""
        for _ in range(10 if err else 1):            # a killed `up` may leave docker writing the ID for a moment
            if os.path.exists(created):
                with open(created) as f:
                    ident = f.read().strip()
                if ident:
                    break
            if err:
                time.sleep(0.2)
        if os.path.exists(created):
            os.unlink(created)
        if ident:
            self.owned.append((what, script + ["down", ident]))
            say(f"{what}由這裡建立（{ident[:12]}），結束時關閉")
        elif err:
            say(f"{what}沒有建立任何實例，不必清理")
        else:
            say(f"{what}已存在，沿用（不會由這裡關閉）")
        if err:
            raise err
        wait(ready, max(timeout - (time.monotonic() - t0), min(timeout, 5.0)), what)

    # ---- steps -----------------------------------------------------------------------------------------
    def model(self):
        script, timeout = MODELS[self.a.model]
        timeout = self.a.start_timeout or timeout
        what = f"模型服務 {self.a.model}"
        info = script_info(script, self.env)
        url, name = info["url"], info["model"]
        if model_up(url):
            say(f"{what}已在回應（{url}），沿用（不會由這裡關閉）")
        else:
            say(f"啟動{what}（容器 {info['name']}；第一次可能要幾分鐘）")
            self._up(what, script, timeout, lambda: model_up(url))
        say(f"{what}就緒")
        return url, name

    def simulator(self):
        script, env_key, timeout = SERVERS[self.a.backend]
        timeout = self.a.start_timeout or timeout
        what = f"{self.a.backend} 模擬伺服器"
        ready = lambda: probe(self.a.backend, timeout=1.0) is not None  # noqa: E731
        if ready():
            say(f"{what}已在回應，沿用（不會由這裡關閉）")
        else:
            say(f"啟動{what}（{self.a.experiment}）")
            self.env[env_key] = self.a.experiment
            self._up(what, script, timeout, ready)
        opts = {}
        if self.a.block_colors:
            colors = self.a.block_colors.split(",")
            opts.update(block_colors=colors, blocks=len(colors))
        elif self.a.blocks:
            opts["blocks"] = self.a.blocks
        if self.a.wrist_lens:
            opts["wrist_lens"] = self.a.wrist_lens
        c = SimClient.for_backend(self.a.backend, name="lab", timeout=120.0)
        c.connect()
        try:
            c.call("load_experiment", name=self.a.experiment, options=opts)
        finally:
            c.close()
        say(f"已載入 {self.a.experiment}{'（' + ', '.join(f'{k}={v}' for k, v in opts.items()) + '）' if opts else ''}")

    def chat(self, url, name, typing=()):
        cmd = ["uv", "run", "--quiet", "tools/chat_window.py", "--backend", self.a.backend, "--base-url", url,
               "--model", name]
        for line in typing:
            cmd += ["--type", line]
        return subprocess.Popen(cmd, cwd=ROOT, env=self.env)

    def stop(self):
        for what, cmd in reversed(self.owned):
            say(f"關閉{what}")
            try:
                self.run(cmd, timeout=120)
            except Exception as e:  # noqa: BLE001
                say(f"關閉{what}失敗：{e}")
        self.owned = []


def enter_pressed() -> bool:
    """Enter on stdin (non-blocking); False when stdin is closed or not readable."""
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return False
    if not r:
        return False
    line = sys.stdin.readline()
    return line != ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(SERVERS), default="mujoco")
    ap.add_argument("--model", choices=list(MODELS), default="q4")
    ap.add_argument("--experiment", default="lab_20261120")
    ap.add_argument("--blocks", type=int, default=None, help="number of blocks (default: the scene's count)")
    ap.add_argument("--block-colors", default=None, help="colour name per block, comma-separated (sets the count)")
    ap.add_argument("--wrist-lens", choices=["real", "pinhole"], default=None,
                    help="wrist camera image (default: the scene's lens_mode, real)")
    ap.add_argument("--no-chat", action="store_true", help="simulator only (no model service, no chat window)")
    ap.add_argument("--start-timeout", type=float, default=None,
                    help="seconds a service may take to become ready (default: per service)")
    ap.add_argument("--type", action="append", default=None, metavar="LINE",
                    help="acceptance: lines typed into the first chat window (tools/chat_window.py --type)")
    ap.add_argument("--type-reopen", action="append", default=None, metavar="LINE",
                    help="acceptance: lines typed into a chat window reopened with Enter")
    a = ap.parse_args(argv)
    lab = Lab(a)
    stopping = {"flag": False}

    def on_signal(signum, frame):  # noqa: ARG001
        stopping["flag"] = True
    signal.signal(signal.SIGTERM, on_signal)
    # Ctrl+C ends the session even when started from a script in the background (where SIGINT is inherited ignored)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    rc = 0
    try:
        url = name = None
        if not a.no_chat:
            url, name = lab.model()
        lab.simulator()
        chat = None
        if not a.no_chat:
            say("開啟聊天視窗")
            chat = lab.chat(url, name, a.type or ())
        say("環境就緒。Ctrl+C 結束整套環境。")
        while not stopping["flag"]:
            if chat is not None and chat.poll() is not None:
                chat = None
                say("聊天視窗已關閉，模擬器與模型繼續執行。按 Enter 重開聊天視窗，Ctrl+C 結束整套環境。")
            if chat is None and not a.no_chat and enter_pressed():
                say("重開聊天視窗")
                chat = lab.chat(url, name, a.type_reopen or ())
            if probe(a.backend, timeout=1.0) is None:
                say("模擬伺服器沒有回應（視窗可能已關閉），結束")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        say("中斷，結束整套環境")
    except Exception as e:  # noqa: BLE001
        say(f"錯誤：{e}")
        rc = 1
    finally:
        try:
            if "chat" in locals() and chat is not None and chat.poll() is None:
                chat.terminate()
        except Exception:  # noqa: BLE001
            pass
        lab.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
