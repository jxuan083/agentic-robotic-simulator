"""Persistent Isaac Sim server: Kit and the assets load once; clients load experiments, reset, run acceptance.

Started by docker/isaac/compose.yaml (`docker compose -f docker/isaac/compose.yaml up -d`); stopped with
`docker compose ... down` or the `shutdown` command. Clients (tools/sim.py, scripts, agents) talk the JSON-lines
protocol of agentic_sim.server on 127.0.0.1 (docs/sim-server.md). The Isaac window stays on the desktop; manual
teleop happens in that window as before.

  /isaac-sim/python.sh scripts/isaac_sim_server.py [--experiment minimal_pick_place] [--port 47801]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--experiment", default=None, help="experiment loaded at start-up (default: none)")
parser.add_argument("--focus", action="store_true", help="start-up experiment in the focus layout")
parser.add_argument("--port", type=int, default=None, help="default: configs/sim_server.yaml")
parser.add_argument("--rss_warn_mb", type=float, default=4000.0,
                    help="status warns once the process grew this much since start-up (each scene rebuild "
                         "leaves about 70 MB inside Kit; restart the server to reclaim it)")
parser.add_argument("--profile", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", kit_args="--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false")
args = parser.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402
import traceback  # noqa: E402

from agentic_sim.config import load  # noqa: E402
from agentic_sim.experiments import RUN_OPTIONS, list_experiments, load_experiment  # noqa: E402
from agentic_sim.isaac.focus_mode import FocusMode  # noqa: E402
from agentic_sim.isaac.session import CONTROL_DT, SessionConfig, TeleopSession  # noqa: E402
from dataclasses import replace  # noqa: E402
from agentic_sim.launch import RunOptions  # noqa: E402
from agentic_sim.server.core import SimServer  # noqa: E402

BACKEND = "isaac"


def log(msg):
    print(f"[isaac_server] {msg}", flush=True)


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def chown_for_host(*paths):
    """The container runs as root; hand result files back to the host user (HOST_UID / HOST_GID)."""
    uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
    if not uid or not gid:
        return
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        for root, dirs, files in os.walk(p) if os.path.isdir(p) else [(os.path.dirname(p), [], [os.path.basename(p)])]:
            for n in [root] + [os.path.join(root, x) for x in dirs + files]:
                try:
                    os.chown(n, int(uid), int(gid))
                except OSError:
                    pass


def focus_app_window() -> bool:
    """Ask the window manager to raise and focus the Isaac window (injected acceptance input needs OS focus)."""
    try:
        import carb.windowing
        import omni.appwindow
        win = omni.appwindow.get_default_app_window().get_window()
        carb.windowing.acquire_windowing_interface().focus_window(win)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"could not focus the app window ({e})")
        return False


class IsaacServer:
    """Command handlers (cmd_<name>), called by SimServer.poll() on the Kit main thread between frames."""

    def __init__(self):
        self.session: TeleopSession | None = None
        self.experiment = None
        self.options = RunOptions()
        self.layout: FocusMode | None = None  # focus layout state shared by consecutive sessions
        self.job = None                        # running / last acceptance run
        self.rebuilds = 0
        self.rss0 = rss_mb()
        self.started = time.time()

    # ---- session management ---------------------------------------------------------
    def _close_session(self, keep_layout: bool):
        if self.session is not None:
            self.session.close(keep_layout=keep_layout)
            self.session = None
            self.rebuilds += 1

    def _open(self, exp, opts: RunOptions, script=None, out_dir=None, carry_script=None):
        if self.layout is not None and self.layout.active and not opts.focus:
            self.layout.exit()
        cfg = SessionConfig(scene=exp.scene, teleop=exp.teleop,
                            color_dr_config=exp.color_dr or "configs/color_randomization.yaml",
                            focus=opts.focus, color_dr=opts.color_dr, color_dr_seed=opts.color_dr_seed,
                            blocks=opts.blocks, profile=args.profile, device=args.device)
        if script:
            cfg.script = f"configs/acceptance/{script}.yaml"
            cfg.log = os.path.join(out_dir, f"{script}.csv")
            cfg.result = os.path.join(out_dir, f"{script}.json")
            if opts.color_dr:
                cfg.color_dr_log = os.path.join(out_dir, f"{script}_color_dr.jsonl")
        t0 = time.monotonic()
        self.session = TeleopSession(cfg, log, focus_layout=self.layout, carry_script=carry_script)
        self.layout = self.session.focus_mode
        self.experiment, self.options = exp, opts
        log(f"loaded {exp.name} in {time.monotonic() - t0:.1f}s (options {opts.__dict__}"
            f"{', scenario ' + script if script else ''}); rss {rss_mb():.0f} MB")
        return time.monotonic() - t0

    def current_options(self) -> RunOptions:
        """Loaded options, updated with the scene settings changed in the panel (colour DR, seed, blocks)."""
        if self.session is None or self.session.closed:
            return self.options
        return replace(self.options, **self.session.settings())

    def rebuild_from_panel(self):
        """The panel asked for another block count: rebuild the session with the panel's settings."""
        want = self.session.rebuild_request
        opts = replace(self.current_options(), **want)
        carry = self.session.handover_script() if self._job_running() else None
        self._close_session(keep_layout=opts.focus)
        if carry is not None:  # an acceptance script changed the block count: it continues in the new scene
            self._open(self.experiment, opts, script=self.job["scenario"], out_dir=self.job["out_dir"],
                       carry_script=carry)
        else:
            self._open(self.experiment, opts)

    def _require_session(self):
        if self.session is None:
            raise RuntimeError("no experiment loaded (send load_experiment first)")

    def _job_running(self) -> bool:
        return self.job is not None and self.job["state"] == "running"

    # ---- protocol commands ------------------------------------------------------------
    def cmd_status(self):
        grow = rss_mb() - self.rss0
        cur = self.current_options()
        opts = {k: getattr(cur, k) for k in RUN_OPTIONS}
        st = {"backend": BACKEND, "experiment": self.experiment.name if self.experiment else None,
              "options": opts if self.experiment else None,
              "rebuilds": self.rebuilds, "rss_mb": round(rss_mb()), "rss_growth_mb": round(grow),
              "uptime_s": round(time.time() - self.started), "job": self.job}
        if self.session is not None:
            st.update({"session": self.session.ctrl.state.session, "paused": self.session.paused,
                       "frames": self.session.frames})
        if grow > args.rss_warn_mb:
            st["warning"] = (f"memory grew {grow:.0f} MB since start-up ({self.rebuilds} scene rebuilds); "
                             "restart the server to reclaim it")
        return st

    def cmd_list_experiments(self):
        return {"experiments": [{"name": e.name, "description": e.description,
                                 "acceptance": [a.scenario for a in e.acceptance]}
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
        self._close_session(keep_layout=opts.focus)
        dt = self._open(exp, opts)
        return {"loaded": exp.name, "seconds": round(dt, 2)}

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
        entry = exp.scenario(scenario)
        opts = self.current_options().merged(entry.options)
        opts.video = False
        out_dir = os.path.join("outputs", "acceptance", exp.name)
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("csv", "json"):
            p = os.path.join(out_dir, f"{scenario}.{ext}")
            if os.path.exists(p):
                os.remove(p)
        self._resume_options = self.current_options()
        self._close_session(keep_layout=opts.focus)
        self._open(exp, opts, script=scenario, out_dir=out_dir)
        focused = focus_app_window()
        self.job = {"scenario": scenario, "state": "running", "started": time.time(), "out_dir": out_dir,
                    "result": os.path.join(out_dir, f"{scenario}.json"), "focus_requested": focused,
                    "forced_options": entry.options}
        return dict(self.job)

    # ---- main-loop hooks ------------------------------------------------------------------
    def after_frame(self):
        """Finish an acceptance run once its script is done, then go back to a fresh teleop session."""
        if not self._job_running() or not self.session.script_done:
            return
        s = self.session
        out = s.finish()
        csv_path = s.cfg.log
        focus_frames = 0
        if csv_path and os.path.exists(csv_path):
            import csv
            with open(csv_path) as f:
                r = csv.DictReader(f)
                focus_frames = sum(1 for row in r if row.get("os_focus") == "0")
        self.job.update({"state": "done", "finished": time.time(), "passed": out.get("passed"),
                         "checks": out.get("checks"), "os_focus_lost_frames": focus_frames})
        if focus_frames:
            self.job["warning"] = (f"the Isaac window lacked OS focus for {focus_frames} frames; input was dropped "
                                   "by design, rerun with the window focused")
        log(f"acceptance {self.job['scenario']}: {out.get('passed')}/{out.get('checks')} checks passed"
            + (f" (WARNING: no OS focus for {focus_frames} frames)" if focus_frames else ""))
        chown_for_host(self.job["out_dir"])
        # back to a plain teleop session of the same experiment with the options it had before the run
        self._close_session(keep_layout=self._resume_options.focus)
        self._open(self.experiment, self._resume_options)


def main():
    host, port = (lambda s: (s["host"], int(s["port"])))(load("configs/sim_server.yaml")["servers"][BACKEND])
    handler = IsaacServer()
    server = SimServer(handler, host, args.port or port, backend=BACKEND, log=log)
    import signal
    # docker compose down / stop sends SIGTERM: leave the loop and write what is pending
    signal.signal(signal.SIGTERM, lambda *_: setattr(server, "stop_requested", True))
    log(f"listening on {server.address[0]}:{server.address[1]} (localhost only)")
    if args.experiment:
        handler.cmd_load_experiment(args.experiment, {"focus": args.focus})
    handler.rss0 = rss_mb()  # growth is reported from here: the first scene is the baseline
    log("ready")
    while app.is_running() and not server.stop_requested:
        now = time.monotonic()
        try:
            server.poll(0.0)
        except Exception:  # a broken request must never take the server down
            log(f"poll error:\n{traceback.format_exc()}")
        if handler.session is not None:
            try:
                handler.session.step()
                handler.after_frame()
                if handler.session is not None and handler.session.rebuild_request is not None:
                    handler.rebuild_from_panel()
            except Exception:  # keep the server alive; the client sees the error in status
                log(f"session error:\n{traceback.format_exc()}")
                if handler.job and handler.job["state"] == "running":
                    handler.job.update({"state": "failed", "error": traceback.format_exc(limit=3)})
                try:
                    handler._close_session(keep_layout=False)
                except Exception:  # noqa: BLE001
                    log(f"could not close the failed session:\n{traceback.format_exc()}")
                    handler.session = None
            spare = CONTROL_DT - (time.monotonic() - now)
        else:
            app.update()  # idle: keep the window responsive
            spare = 0.05 - (time.monotonic() - now)
        if spare > 0:
            try:
                server.poll(spare)  # wait for requests instead of sleeping
            except Exception:
                log(f"poll error:\n{traceback.format_exc()}")
    log("shutting down")
    if handler.session is not None:
        handler.session.finish()
    server.close()
    chown_for_host("outputs")


main()
sys.stdout.flush()
os._exit(0)  # SimulationApp.close() can hang in Isaac Sim 5.1
