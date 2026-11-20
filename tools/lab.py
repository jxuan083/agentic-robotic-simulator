#!/usr/bin/env python3
"""One command for the lab environment: simulator server (with the sensor camera views), model service and the chat
window, started in order and stopped together.

  uv run tools/lab.py                                   # MuJoCo, Q4 model, lab_20261120, default blocks
  uv run tools/lab.py --backend isaac --model bf16 --block-colors red,red,blue,blue,yellow
  uv run tools/lab.py --wrist-lens pinhole --no-chat

Steps (each waits until the previous one answers; a failure stops here and says why):
  1. model service   Q4_K_M (llama.cpp, docker/llamacpp/serve.sh, port 8001) or BF16 (vLLM, docker/vllm/serve.sh,
                     port 8000); reused when already running
  2. simulator       persistent server (tools/mujoco_server.sh / docker/isaac/server.sh) with the experiment loaded;
                     reused when already running (the experiment is reloaded with the options given here)
  3. chat window     tools/chat_window.py (separate window; closing it ends the session)
The session ends when the chat window closes or on Ctrl+C (with --no-chat: Ctrl+C or the simulator window closing).
Only what this command started is stopped; services that were already running keep running.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.server.client import SimClient, probe  # noqa: E402

MODELS = {   # model service: start / stop script, base URL, served model name, start timeout (s)
    "q4": (["docker/llamacpp/serve.sh"], "http://127.0.0.1:8001/v1", "Qwen3.5-9B-Q4_K_M", 240),
    "bf16": (["docker/vllm/serve.sh"], "http://127.0.0.1:8000/v1", "Qwen/Qwen3.5-9B", 600),
}
SERVERS = {  # simulator server control script, experiment env var, start timeout (s)
    "mujoco": (["tools/mujoco_server.sh"], "MUJOCO_SERVER_EXPERIMENT", 60),
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


def wait(cond, timeout: float, what: str, every: float = 2.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(every)
    raise TimeoutError(f"{what} did not answer within {timeout:.0f} s")


class Lab:
    def __init__(self, a):
        self.a = a
        self.started = []          # (description, stop command) in start order
        self.env = dict(os.environ)
        self.env.setdefault("DISPLAY", ":1")
        self.env.setdefault("XAUTHORITY", f"/run/user/{os.getuid()}/gdm/Xauthority")

    def run(self, cmd, **kw):
        return subprocess.run(cmd, cwd=ROOT, env=self.env, **kw)

    # ---- steps -----------------------------------------------------------------------------------------
    def model(self):
        script, url, name, timeout = MODELS[self.a.model]
        if model_up(url):
            say(f"模型服務已在執行（{self.a.model}，{url}），沿用")
            return url, name
        say(f"啟動模型服務 {self.a.model}（第一次可能要幾分鐘）")
        r = self.run(script + ["up"])
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(script)} up failed (exit {r.returncode})")
        self.started.append((f"模型服務 {self.a.model}", script + ["down"]))
        wait(lambda: model_up(url), timeout, f"model service {self.a.model}")
        say("模型服務就緒")
        return url, name

    def simulator(self):
        script, env_key, timeout = SERVERS[self.a.backend]
        if probe(self.a.backend, timeout=1.0):
            say(f"{self.a.backend} 模擬伺服器已在執行，沿用並重新載入 {self.a.experiment}")
        else:
            say(f"啟動 {self.a.backend} 模擬伺服器（{self.a.experiment}）")
            self.env[env_key] = self.a.experiment
            r = self.run(script + ["up"])
            if r.returncode != 0:
                raise RuntimeError(f"{' '.join(script)} up failed (exit {r.returncode}); see `{script[0]} logs`")
            self.started.append((f"{self.a.backend} 模擬伺服器", script + ["down"]))
            wait(lambda: probe(self.a.backend, timeout=1.0) is not None, timeout, f"{self.a.backend} server")
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

    def chat(self, url, name):
        cmd = ["uv", "run", "--quiet", "tools/chat_window.py", "--backend", self.a.backend, "--base-url", url,
               "--model", name]
        for line in self.a.type or []:
            cmd += ["--type", line]
        say("開啟聊天視窗（關掉聊天視窗就結束這次的環境）")
        return subprocess.Popen(cmd, cwd=ROOT, env=self.env)

    def stop(self):
        for what, cmd in reversed(self.started):
            say(f"關閉{what}")
            try:
                self.run(cmd, timeout=120)
            except Exception as e:  # noqa: BLE001
                say(f"關閉{what}失敗：{e}")
        self.started = []


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
    ap.add_argument("--type", action="append", default=None, metavar="LINE",
                    help="acceptance: lines typed into the chat window (tools/chat_window.py --type)")
    a = ap.parse_args(argv)
    lab = Lab(a)
    stopping = {"flag": False}

    def on_signal(signum, frame):  # noqa: ARG001
        stopping["flag"] = True
    signal.signal(signal.SIGTERM, on_signal)
    try:
        url = name = None
        if not a.no_chat:
            url, name = lab.model()
        lab.simulator()
        chat = None if a.no_chat else lab.chat(url, name)
        say("環境就緒。Ctrl+C 結束。")
        while not stopping["flag"]:
            if chat is not None and chat.poll() is not None:
                say("聊天視窗已關閉")
                break
            if probe(a.backend, timeout=1.0) is None:
                say("模擬伺服器沒有回應（視窗可能已關閉）")
                break
            time.sleep(1.0)
        if chat is not None and chat.poll() is None:
            chat.terminate()
        return 0
    except KeyboardInterrupt:
        say("中斷")
        return 0
    except Exception as e:  # noqa: BLE001
        say(f"錯誤：{e}")
        return 1
    finally:
        lab.stop()


if __name__ == "__main__":
    sys.exit(main())
