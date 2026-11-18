"""Persistent MuJoCo server: same JSON-lines protocol and commands as the Isaac server, on its own port.

Started by tools/mujoco_server.sh (`up`), stopped with `down` or the `shutdown` command; closing the window also
stops it (as with Isaac). Clients: tools/sim.py, scripts, agents (docs/sim-server.md). Runs on the host with the uv
environment, no container or NVIDIA GPU needed.

  uv run scripts/mujoco_sim_server.py [--experiment minimal_pick_place] [--port 47802] [--headless]
"""

import argparse
import os
import signal
import sys
import time
import traceback
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.config import load  # noqa: E402
from agentic_sim.experiments import RUN_OPTIONS, list_experiments, load_experiment  # noqa: E402
from agentic_sim.launch import RunOptions  # noqa: E402
from agentic_sim.mujoco_backend.model import ROBOT_CONFIG  # noqa: E402
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig, session_config_with  # noqa: E402
from agentic_sim.server.core import SimServer  # noqa: E402
from agentic_sim.teleop.keys import KeyState  # noqa: E402

BACKEND = "mujoco"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--experiment", default=None, help="experiment loaded at start-up (default: none)")
parser.add_argument("--port", type=int, default=None, help="default: configs/sim_server.yaml")
parser.add_argument("--headless", action="store_true", help="no window (scripted runs and agents only)")
args = parser.parse_args()


def log(msg):
    print(f"[mujoco_server] {msg}", flush=True)


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


class MujocoServer:
    """Command handlers (cmd_<name>), called by SimServer.poll() from the main loop between frames."""

    def __init__(self, viewer, keys: KeyState):
        self.viewer = viewer
        self.keys = keys
        self.session: MjSession | None = None
        self.experiment = None
        self.options = RunOptions()
        self.job = None
        self.rebuilds = 0
        self.rss0 = rss_mb()
        self.started = time.time()
        self._resume_options = RunOptions()

    def _open(self, exp, opts: RunOptions, script=None, out_dir=None, carry_script=None):
        cfg = SessionConfig(scene=exp.scene, teleop=exp.teleop,
                            color_dr_config=exp.color_dr or "configs/color_randomization.yaml",
                            color_dr=opts.color_dr, color_dr_seed=opts.color_dr_seed, blocks=opts.blocks)
        if script:
            cfg.script = f"configs/acceptance/{script}.yaml"
            cfg.log = os.path.join(out_dir, f"{script}.csv")
            cfg.result = os.path.join(out_dir, f"{script}.json")
            if opts.color_dr:
                cfg.color_dr_log = os.path.join(out_dir, f"{script}_color_dr.jsonl")
        t0 = time.monotonic()
        if self.session is not None:
            self.session.finish()
            self.rebuilds += 1
        self.keys.clear()
        self.keys.pop_actions()
        self.session = MjSession(cfg, log, keys=self.keys, focused=self.viewer.focused if self.viewer else None,
                                 script=carry_script, viewer=self.viewer)
        if self.viewer:
            self.viewer.attach(self.session)
        self.experiment, self.options = exp, opts
        dt = time.monotonic() - t0
        log(f"loaded {exp.name} in {dt:.2f}s (options {opts.__dict__}{', scenario ' + script if script else ''}); "
            f"rss {rss_mb():.0f} MB")
        return dt

    def rebuild_from_panel(self):
        """The panel asked for another block count: rebuild the session with the panel's settings."""
        want = self.session.rebuild_request
        opts = replace(self.current_options(), **want)
        carry = self.session.handover_script() if self._job_running() else None
        if carry is not None:  # an acceptance script changed the block count: it continues in the new scene
            self._open(self.experiment, opts, script=self.job["scenario"], out_dir=self.job["out_dir"],
                       carry_script=carry)
        else:
            self._open(self.experiment, opts)

    def current_options(self) -> RunOptions:
        if self.session is None:
            return self.options
        return replace(self.options, **self.session.settings())

    def _require_session(self):
        if self.session is None:
            raise RuntimeError("no experiment loaded (send load_experiment first)")

    def _job_running(self) -> bool:
        return self.job is not None and self.job["state"] == "running"

    # ---- protocol commands ------------------------------------------------------------
    def cmd_status(self):
        cur = self.current_options()
        st = {"backend": BACKEND, "experiment": self.experiment.name if self.experiment else None,
              "options": {k: getattr(cur, k) for k in RUN_OPTIONS} if self.experiment else None,
              "rebuilds": self.rebuilds, "rss_mb": round(rss_mb()), "rss_growth_mb": round(rss_mb() - self.rss0),
              "uptime_s": round(time.time() - self.started), "job": self.job, "window": self.viewer is not None}
        if self.session is not None:
            st.update({"session": self.session.ctrl.state.session, "paused": self.session.paused,
                       "frames": self.session.frames})
        return st

    def cmd_list_experiments(self):
        return {"experiments": [{"name": e.name, "description": e.description,
                                 "acceptance": [a.scenario for a in e.acceptance_for(BACKEND)]}
                                for e in list_experiments(backend=BACKEND)]}

    def cmd_load_experiment(self, name, options=None):
        if self._job_running():
            raise RuntimeError(f"acceptance run {self.job['scenario']} is in progress")
        exp = load_experiment(name)
        if not exp.supports(BACKEND):
            raise ValueError(f"experiment {name!r} does not support {BACKEND}")
        unknown = set(options or {}) - set(RUN_OPTIONS)
        if unknown:
            raise ValueError(f"unknown options {sorted(unknown)}")
        opts = RunOptions(**{k: v for k, v in (options or {}).items() if v is not None})
        dt = self._open(exp, opts)
        return {"loaded": exp.name, "seconds": round(dt, 3)}

    def cmd_reset(self):
        self._require_session()
        self.session.request("reset")
        return {}

    def cmd_pause(self):
        self._require_session()
        self.session.request("pause")
        return {}

    def cmd_resume(self):
        self._require_session()
        self.session.request("resume")
        return {}

    def cmd_get_state(self):
        self._require_session()
        st = self.session.state()
        st["experiment"] = self.experiment.name
        return st

    def cmd_run_acceptance(self, scenario):
        if self._job_running():
            raise RuntimeError(f"acceptance run {self.job['scenario']} is in progress")
        if self.experiment is None:
            raise RuntimeError("no experiment loaded (send load_experiment first)")
        exp = self.experiment
        entry = exp.scenario(scenario, BACKEND)
        opts = self.current_options().merged(entry.options)
        opts.video = False
        out_dir = os.path.join("outputs", "acceptance", BACKEND, exp.name)
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("csv", "json"):
            p = os.path.join(out_dir, f"{scenario}.{ext}")
            if os.path.exists(p):
                os.remove(p)
        self._resume_options = self.current_options()
        self._open(exp, opts, script=scenario, out_dir=out_dir)
        focused = self.focus_window()
        self.job = {"scenario": scenario, "state": "running", "started": time.time(), "out_dir": out_dir,
                    "result": os.path.join(out_dir, f"{scenario}.json"), "focus_requested": focused,
                    "forced_options": entry.options}
        return dict(self.job)

    def focus_window(self) -> bool:
        if self.viewer is None:
            return True
        try:
            import glfw
            glfw.focus_window(self.viewer.window)
            return True
        except Exception as e:  # noqa: BLE001
            log(f"could not focus the window ({e})")
            return False

    def after_frame(self):
        if not self._job_running() or not self.session.script_done:
            return
        s = self.session
        out = s.finish()
        focus_frames = 0
        if s.cfg.log and os.path.exists(s.cfg.log):
            import csv
            with open(s.cfg.log) as f:
                focus_frames = sum(1 for row in csv.DictReader(f) if row.get("os_focus") == "0")
        self.job.update({"state": "done", "finished": time.time(), "passed": out.get("passed"),
                         "checks": out.get("checks"), "os_focus_lost_frames": focus_frames})
        if focus_frames:
            self.job["warning"] = (f"the MuJoCo window lacked OS focus for {focus_frames} frames; input was dropped "
                                   "by design, rerun with the window focused")
        log(f"acceptance {self.job['scenario']}: {out.get('passed')}/{out.get('checks')} checks passed"
            + (f" (WARNING: no OS focus for {focus_frames} frames)" if focus_frames else ""))
        self._open(self.experiment, self._resume_options)


def main():
    host, port = (lambda s: (s["host"], int(s["port"])))(load("configs/sim_server.yaml")["servers"][BACKEND])
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    viewer = None
    if not args.headless:
        from agentic_sim.mujoco_backend.viewer import Viewer
        vc = load(ROBOT_CONFIG)["viewer"]
        viewer = Viewer(keys, vc["title"], vc["size"], log)
    handler = MujocoServer(viewer, keys)
    server = SimServer(handler, host, args.port or port, backend=BACKEND, log=log)
    signal.signal(signal.SIGTERM, lambda *_: setattr(server, "stop_requested", True))
    log(f"listening on {server.address[0]}:{server.address[1]} (localhost only)")
    if args.experiment:
        handler.cmd_load_experiment(args.experiment, {})
    log("ready")
    while not server.stop_requested:
        now = time.monotonic()
        if viewer is not None and viewer.should_close():
            log("window closed")
            break
        try:
            server.poll(0.0)
        except Exception:  # a broken request must never take the server down
            log(f"poll error:\n{traceback.format_exc()}")
        s = handler.session
        if s is not None:
            try:
                s.step()
                handler.after_frame()
                if handler.session is not None and handler.session.rebuild_request is not None:
                    handler.rebuild_from_panel()
            except Exception:
                log(f"session error:\n{traceback.format_exc()}")
                if handler.job and handler.job["state"] == "running":
                    handler.job.update({"state": "failed", "error": traceback.format_exc(limit=3)})
                handler.session = None
        if viewer is not None:
            if handler.session is not None:
                job = handler.job if handler._job_running() else None
                viewer.render(handler.session, extra=f"acceptance {job['scenario']} running" if job else
                              f"{handler.experiment.name} (port {server.address[1]})")
            else:
                viewer.render_idle()
        spare = (handler.session.control_dt if handler.session else 0.05) - (time.monotonic() - now)
        if spare > 0:
            try:
                server.poll(spare)
            except Exception:
                log(f"poll error:\n{traceback.format_exc()}")
    log("shutting down")
    if handler.session is not None:
        handler.session.finish()
    server.close()
    if viewer is not None:
        viewer.close()


main()
