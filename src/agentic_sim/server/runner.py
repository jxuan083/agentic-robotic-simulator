"""Client-side flows on a running simulator server (used by tools/sim.py; no engine imports).

  teleop:      load the experiment with the chosen options; the operator then works in the simulator window
  acceptance:  make sure the experiment is loaded, start the scenario, follow its progress until it is done
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict

from agentic_sim.config import REPO_ROOT
from agentic_sim.experiments import Experiment
from agentic_sim.launch import RunOptions
from agentic_sim.server.client import SimClient

SESSION_OPTIONS = ("live_perception", "perception_camera", "perception_hz", "perception_json", "focus", "color_dr", "color_dr_seed", "blocks", "block_colors", "random_blocks", "random_seed", "random_region", "random_config", "random_episode")


def session_options(opts: RunOptions) -> dict:
    d = asdict(opts)
    return {k: d[k] for k in SESSION_OPTIONS if not (k == "block_colors" and d[k] is None)
            and not (k in ("live_perception", "perception_camera", "perception_hz", "perception_json") and not d["live_perception"])
            and not (k.startswith("random_") and not d["random_blocks"])}


def same_options(loaded: dict, wanted: dict) -> bool:
    """Loaded session options (server status) match the wanted ones. The seed only counts with colour DR on, and a
    missing seed means the config default (any seed is accepted)."""
    loaded = loaded or {}
    if bool(loaded.get("live_perception")) != bool(wanted.get("live_perception")):
        return False
    if wanted.get("live_perception"):
        if any(loaded.get(k) != wanted.get(k) for k in ("perception_camera", "perception_hz", "perception_json")):
            return False
    if bool(loaded.get("random_blocks")) != bool(wanted.get("random_blocks")):
        return False
    if wanted.get("random_blocks"):
        for k in ("random_seed", "random_region", "random_config", "random_episode"):
            if wanted.get(k) is not None and loaded.get(k) != wanted[k]:
                return False
    for k in ("focus", "color_dr"):
        if bool(loaded.get(k)) != bool(wanted.get(k)):
            return False
    if not wanted.get("random_blocks") and int(loaded.get("blocks") or 1) != int(wanted.get("blocks") or 1):
        return False
    if (loaded.get("block_colors") or None) != (wanted.get("block_colors") or None):
        return False
    if wanted.get("color_dr") and wanted.get("color_dr_seed") is not None:
        return loaded.get("color_dr_seed") == wanted["color_dr_seed"]
    return True


def teleop(client: SimClient, exp: Experiment, opts: RunOptions, say=print) -> dict:
    r = client.call("load_experiment", name=exp.name, options=session_options(opts))
    say(f"已載入 {exp.name}（{r.get('seconds')} 秒），請到模擬器視窗操作。關閉這個 terminal 不會關閉模擬器。")
    return r


def window_title(backend: str) -> str | None:
    """Title of the backend's simulator window (for window-only recording); None = the recorder's default (Isaac)."""
    if backend == "mujoco":
        from agentic_sim.config import load
        return load("configs/mujoco_so101.yaml")["viewer"]["title"]
    return None


class WindowRecorder:
    """tools/record_isaac_window.sh in its own process group; stop() sends SIGINT so ffmpeg finalizes the file."""

    def __init__(self, out_path: str, max_s: int = 900, window: str | None = None):
        self.out = out_path
        env = dict(os.environ, **({"ISAAC_WINDOW_NAME": window} if window else {}))
        self.proc = subprocess.Popen(["tools/record_isaac_window.sh", "video", out_path, str(max_s)], cwd=REPO_ROOT,
                                     start_new_session=True, env=env)

    def stop(self, timeout: float = 15.0):
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:  # ffmpeg (a grandchild) may outlive the script: wait for the group
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        os.killpg(self.proc.pid, signal.SIGKILL)


def acceptance(client: SimClient, exp: Experiment, opts: RunOptions, scenario: str, say=print,
               poll_s: float = 0.5, timeout_s: float = 900.0, recorder_factory=None, backend: str | None = None) -> dict:
    """Run one scenario on the server; returns the finished job (state, passed, checks, warning, ...)."""
    exp.scenario(scenario, backend)  # KeyError early if the experiment does not list it for this backend
    st = client.call("status")
    loaded = st.get("options") or {}
    if st.get("experiment") != exp.name or not same_options(loaded, session_options(opts)):
        say(f"載入 {exp.name} ...")
        client.call("load_experiment", name=exp.name, options=session_options(opts))
    rec = None
    if opts.video and recorder_factory is not None:
        sub = [backend] if backend and backend != "isaac" else []
        rec = recorder_factory(os.path.join("outputs", "acceptance", *sub, exp.name, f"{scenario}.mp4"))
    try:
        job = client.call("run_acceptance", scenario=scenario)
        if job.get("forced_options"):
            say(f"注意：情境 {scenario} 固定使用選項 {job['forced_options']}")
        say(f"驗收 {scenario} 開始，執行中請勿操作桌面（模擬器視窗需保持焦點）")
        t0 = time.monotonic()
        last = None
        while True:
            time.sleep(poll_s)
            st = client.call("status")
            job = st.get("job") or {}
            if job.get("state") != "running":
                break
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"acceptance {scenario} still running after {timeout_s:.0f} s")
            try:
                prog = client.call("get_state").get("script") or {}
            except Exception:  # noqa: BLE001  (session being rebuilt)
                prog = {}
            cur = (prog.get("step"), prog.get("checks"))
            if prog and cur != last:
                say(f"  步驟 {prog.get('step', 0) + 1}/{prog.get('steps')}，檢查 {prog.get('passed')}/{prog.get('checks')} 通過")
                last = cur
    finally:
        if rec is not None:
            rec.stop()
    if job.get("state") == "done":
        say(f"驗收 {scenario}：{job.get('passed')}/{job.get('checks')} 通過；結果 {job.get('result')}")
    else:
        say(f"驗收 {scenario} 失敗：{job.get('error') or job}")
    if job.get("warning"):
        say(f"WARNING: {job['warning']}")
    return job
