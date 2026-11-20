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
A service that already exists (even if it is still warming up) is reused and never stopped here; one this command
creates is recorded as its own before it is started, so it is stopped on exit even when it failed to become ready.
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
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.server.client import SimClient, probe  # noqa: E402

MODELS = {   # model service: control script, container, base URL, served model name, ready timeout (s)
    "q4": (["docker/llamacpp/serve.sh"], "ars-llm-q4", "http://127.0.0.1:8001/v1", "Qwen3.5-9B-Q4_K_M", 300),
    "bf16": (["docker/vllm/serve.sh"], "ars-llm", "http://127.0.0.1:8000/v1", "Qwen/Qwen3.5-9B", 900),
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


def container_exists(name: str) -> bool:
    r = subprocess.run(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
                       capture_output=True, text=True)
    return name in r.stdout.split()


def sim_exists(backend: str) -> bool:
    """A server process / container exists (it may still be starting)."""
    if backend == "isaac":
        return container_exists("ars-isaac-server")
    r = subprocess.run(["pgrep", "-f", "scripts/mujoco_sim_server.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


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

    def run(self, cmd, timeout=None):
        return subprocess.run(cmd, cwd=ROOT, env=self.env, timeout=timeout)

    def _start(self, what: str, script: list, timeout: float, ready):
        """Create a service: recorded as ours first, then `script up` (bounded by timeout), then ready()."""
        self.owned.append((what, script + ["down"]))
        t0 = time.monotonic()
        try:
            r = self.run(script + ["up"], timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"{what} did not become ready within {timeout:.0f} s") from None
        if r.returncode != 0:
            raise RuntimeError(f"{what}: {' '.join(script)} up failed (exit {r.returncode})")
        wait(ready, max(timeout - (time.monotonic() - t0), 5.0), what)

    # ---- steps -----------------------------------------------------------------------------------------
    def model(self):
        script, name_c, url, name, timeout = MODELS[self.a.model]
        timeout = self.a.start_timeout or timeout
        what = f"模型服務 {self.a.model}"
        if container_exists(name_c) or model_up(url):
            say(f"{what}已存在，沿用（不會由這裡關閉）")
            wait(lambda: model_up(url), timeout, what)
        else:
            say(f"啟動{what}（第一次可能要幾分鐘）")
            self._start(what, script, timeout, lambda: model_up(url))
        say(f"{what}就緒")
        return url, name

    def simulator(self):
        script, env_key, timeout = SERVERS[self.a.backend]
        timeout = self.a.start_timeout or timeout
        what = f"{self.a.backend} 模擬伺服器"
        ready = lambda: probe(self.a.backend, timeout=1.0) is not None  # noqa: E731
        if sim_exists(self.a.backend) or ready():
            say(f"{what}已存在，沿用（不會由這裡關閉），重新載入 {self.a.experiment}")
            wait(ready, timeout, what)
        else:
            say(f"啟動{what}（{self.a.experiment}）")
            self.env[env_key] = self.a.experiment
            self._start(what, script, timeout, ready)
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
