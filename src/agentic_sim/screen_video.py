"""Screen video of the simulator window from the right column ("Record screen" / "Stop"), Isaac and MuJoCo alike.

The recording itself is tools/record_isaac_window.sh (one X window by id, never the desktop; a resize such as the
operator workspace going fullscreen continues as a new segment, joined at the end). It runs on the host: the Isaac
container has no ffmpeg. The simulator and the host side talk through two small files in a control directory under
the repo (the container mounts the repo):

  request.json   written by the simulator: {"seq": n, "op": "start" | "stop", "window_id": X id, "engine": ...}
  status.json    written by the agent:     {"pid", "heartbeat" (wall time), "handled_seq", "state", "path", ...}

state: idle | recording | saving | saved | failed. "saved" is written only after the tool exited cleanly and the MP4
exists and is not empty; "failed" carries the reason. Every start makes a new file (outputs/screen_video/<time>_<engine>
.mp4). Closing the simulator ends the recording like a stop: the window is gone, the tool joins what it has.

Who runs the agent:
  - tools/sim.py starts one on the host for every simulator it launches (Isaac and MuJoCo, sim and record modes);
  - the MuJoCo teleop script, already on the host, starts one in its own process when none is running;
  - by hand, next to an Isaac command run on its own: uv run tools/screen_video.py agent <control dir>.
Without an agent the right column says so (no file is ever claimed). Pure Python, no NumPy.

Platforms: the recording needs X11 window ids and the shell tool, i.e. Linux with an X display. Elsewhere (Windows,
macOS, Wayland without DISPLAY) platform_support() says so: the button is disabled, the right column reads "not
available on this platform" and no helper is started. ARS_SCREEN_VIDEO=off forces that state (tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join("tools", "record_isaac_window.sh")
OUT_DIR = os.path.join("outputs", "screen_video")
CONTROL_ROOT = os.path.join(OUT_DIR, "control")
ALIVE_S = 3.0              # no heartbeat for this long: the agent is gone
START_GRACE_S = 1.5        # the tool exiting within this time after a start is a failed start
MAX_S = 3600


def platform_support() -> tuple:
    """(supported, reason): window recording needs Linux with an X display (X11 window ids, ffmpeg x11grab)."""
    if os.environ.get("ARS_SCREEN_VIDEO", "").lower() in ("off", "0", "no"):
        return False, "turned off (ARS_SCREEN_VIDEO=off)"
    if not sys.platform.startswith("linux"):
        return False, f"not available on this platform ({platform.system() or sys.platform})"
    if not os.environ.get("DISPLAY"):
        return False, "not available without an X display (DISPLAY unset)"
    return True, None


def control_dir(run_id: str) -> str:
    """Repo-relative control directory of one simulator run."""
    return (Path(CONTROL_ROOT) / run_id).as_posix()


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _write_json(path: str, data: dict):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---- simulator side ------------------------------------------------------------------------------------------------
class ScreenVideoClient:
    """Right-column control of one simulator window. window_id: a callable returning the X window id (asked at
    every start: the id survives fullscreen, but the window may be rebuilt)."""

    def __init__(self, ctl_dir: str | None, window_id, engine: str, log=print):
        self.ctl_dir = ctl_dir
        self.window_id = window_id
        self.engine = engine
        self.log = log
        self.seq = 0
        self.op = None
        self.local_error = None
        self._status = None
        self._read_at = 0.0
        self.supported, self.unsupported_reason = platform_support()
        if not self.supported:
            log(f"screen video: {self.unsupported_reason}; the right-column button is disabled")
            return
        if ctl_dir:
            os.makedirs(_abs(ctl_dir), exist_ok=True)
            req = _read_json(os.path.join(_abs(ctl_dir), "request.json")) or {}
            self.seq = int(req.get("seq", 0))   # a restarted session keeps counting

    # status ------------------------------------------------------------------------
    def status(self, now: float | None = None) -> dict | None:
        now = time.monotonic() if now is None else now
        if self.supported and self.ctl_dir and now - self._read_at >= 0.25:
            self._read_at = now
            self._status = _read_json(os.path.join(_abs(self.ctl_dir), "status.json"))
        return self._status

    def agent_alive(self) -> bool:
        st = self.status()
        return bool(st) and time.time() - float(st.get("heartbeat", 0)) < ALIVE_S

    def recording(self) -> bool:
        st = self.status() or {}
        pending = self.op is not None and int(st.get("handled_seq", 0)) < self.seq
        if pending:
            return self.op == "start"
        return st.get("state") in ("recording", "saving") and self.agent_alive()

    # actions -----------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.supported

    def toggle(self):
        if not self.supported:
            return
        if self.recording():
            self.stop()
        else:
            self.start()

    def start(self):
        self.local_error = None
        if not self.supported:
            return
        if not self.ctl_dir:
            self.local_error = "no control directory (run from tools/sim.py)"
            return
        if not self.agent_alive():
            self.local_error = ("screen video helper not running: start the simulator from tools/sim.py or run "
                                f"`uv run tools/screen_video.py agent {self.ctl_dir}`")
            self.log(f"screen video: {self.local_error}")
            return
        try:
            wid = int(self.window_id())
        except Exception as e:  # noqa: BLE001 - shown in the right column
            self.local_error = f"cannot find this window's X id: {e}"
            self.log(f"screen video: {self.local_error}")
            return
        self._request("start", window_id=wid)

    def stop(self):
        self.local_error = None
        if self.supported and self.ctl_dir:
            self._request("stop")

    def _request(self, op, **kw):
        self.seq += 1
        self.op = op
        _write_json(os.path.join(_abs(self.ctl_dir), "request.json"),
                    {"seq": self.seq, "op": op, "engine": self.engine, "t": time.time(), **kw})
        self._read_at = 0.0
        self.log(f"screen video: {op} requested (seq {self.seq})")

    # text --------------------------------------------------------------------------
    def button_text(self) -> str:
        if not self.supported:
            return "Screen video unavailable"
        return "Stop screen video" if self.recording() else "Record screen"

    def text(self) -> str:
        """One line for the right column ("Screen video: ...")."""
        if not self.supported:
            return f"Screen video: {self.unsupported_reason}"
        if self.local_error:
            return f"Screen video: not started - {self.local_error}"
        st = self.status()
        if not st:
            return "Screen video: off" if self.ctl_dir else "Screen video: unavailable (no control directory)"
        if not self.agent_alive():
            return "Screen video: helper not running"
        pending = self.op is not None and int(st.get("handled_seq", 0)) < self.seq
        if pending:
            return f"Screen video: {'starting' if self.op == 'start' else 'stopping'}..."
        state = st.get("state")
        if state == "recording":
            el = max(0.0, time.time() - float(st.get("started", time.time())))
            return f"Screen video: REC {int(el // 60):02d}:{int(el % 60):02d} -> {st.get('path')}"
        if state == "saving":
            return "Screen video: saving..."
        if state == "saved":
            return f"Screen video: saved {st.get('path')}"
        if state == "failed":
            return f"Screen video: failed - {st.get('error')}"
        return "Screen video: off"


def isaac_window_id() -> int:
    """X window id of the Isaac app window (carb windowing gives the native window as a capsule)."""
    import ctypes

    import carb.windowing
    import omni.appwindow
    win = omni.appwindow.get_default_app_window().get_window()
    cap = carb.windowing.acquire_windowing_interface().get_native_window(win)
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))
    if not ptr:
        raise RuntimeError("no native window")
    return int(ptr)


# ---- host side -----------------------------------------------------------------------------------------------------
class ScreenVideoAgent:
    """Watches one control directory and runs the window recording tool. poll() does one round; run() loops."""

    def __init__(self, ctl_dir: str, tool: str | None = None, out_dir: str = OUT_DIR, log=print):
        self.ctl_dir = ctl_dir
        self.tool = tool or os.environ.get("ARS_SCREEN_VIDEO_TOOL") or TOOL
        self.out_dir = out_dir
        self.log = log
        os.makedirs(_abs(ctl_dir), exist_ok=True)
        req = _read_json(os.path.join(_abs(ctl_dir), "request.json")) or {}
        self.handled = int(req.get("seq", 0))     # requests from before this agent are not replayed
        self.proc = None
        self.state = "idle"
        self.path = None
        self.error = None
        self.started = None
        self.saved = []
        self._err = None
        self.stopped_by_request = False

    def _status(self):
        _write_json(os.path.join(_abs(self.ctl_dir), "status.json"),
                    {"pid": os.getpid(), "heartbeat": time.time(), "handled_seq": self.handled, "state": self.state,
                     "path": self.path, "error": self.error, "started": self.started, "saved": self.saved})

    def _new_path(self, engine: str) -> str:
        base = os.path.join(self.out_dir, time.strftime("%Y%m%d_%H%M%S") + f"_{engine or 'sim'}")
        path, i = base + ".mp4", 1
        while os.path.exists(_abs(path)):
            i += 1
            path = f"{base}_{i}.mp4"
        return path

    def _start(self, req: dict):
        if self.proc is not None:
            return                                  # already recording: a second start is ignored
        self.path = self._new_path(req.get("engine"))
        self.error = None
        os.makedirs(os.path.dirname(_abs(self.path)), exist_ok=True)
        env = dict(os.environ, ISAAC_WINDOW_ID=str(int(req["window_id"])))
        self._err = open(_abs(self.path) + ".log", "w")
        try:
            self.proc = subprocess.Popen([_abs(self.tool), "video", _abs(self.path), str(MAX_S)], cwd=ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=self._err, stderr=self._err,
                                         start_new_session=True)
        except OSError as e:
            self._err.close()
            self.proc, self.state, self.error = None, "failed", f"cannot start {self.tool}: {e.strerror or e}"
            self.log(f"screen video: {self.error}")
            return
        self.state, self.started, self.stopped_by_request = "recording", time.time(), False
        self.log(f"screen video: recording window {req['window_id']} -> {self.path}")

    def _stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.stopped_by_request = True
            self.state = "saving"
            try:
                self.proc.send_signal(signal.SIGINT)   # the tool ends its segment and still writes the file
            except ProcessLookupError:
                pass

    def _reap(self):
        if self.proc is None or self.proc.poll() is None:
            return
        rc = self.proc.returncode
        self.proc = None
        self._err.close()
        log_path = _abs(self.path) + ".log"
        tail = ""
        try:
            lines = [ln.strip() for ln in open(log_path) if ln.strip()]
            tail = lines[-1] if lines else ""
        except OSError:
            pass
        ok = rc == 0 and os.path.exists(_abs(self.path)) and os.path.getsize(_abs(self.path)) > 0
        if ok:
            self.state, self.error = "saved", None
            self.saved.append(self.path)
            os.remove(log_path)
            self.log(f"screen video: saved {self.path}")
        else:
            quick = self.started is not None and time.time() - self.started < START_GRACE_S
            self.state = "failed"
            self.error = (("could not start: " if quick else "") + (tail or f"tool exited with {rc}"))[:200]
            self.log(f"screen video: failed ({self.error}); tool output in {os.path.relpath(log_path, ROOT)}")

    def poll(self):
        req = _read_json(os.path.join(_abs(self.ctl_dir), "request.json"))
        if req and int(req.get("seq", 0)) > self.handled:
            self.handled = int(req["seq"])
            if req.get("op") == "start":
                self._start(req)
            elif req.get("op") == "stop":
                self._stop()
        self._reap()
        self._status()                              # also the heartbeat (poll runs every 0.2 s)

    def finish(self, timeout_s: float = 120.0):
        """The simulator ended: let a running recording finish (its window is gone, the tool joins the segments),
        stop it if it does not, and write the final status."""
        t0 = time.time()
        while self.proc is not None and time.time() - t0 < timeout_s:
            if time.time() - t0 > 10:
                self._stop()
            self._reap()
            if self.proc is not None:
                time.sleep(0.2)
            self._status()
        if self.proc is not None:
            self.proc.kill()
            self._reap()
        self._status()
        return list(self.saved)

    def run(self, stop_event: threading.Event | None = None, parent_pid: int | None = None, period_s: float = 0.2):
        while not (stop_event is not None and stop_event.is_set()):
            self.poll()
            if parent_pid is not None and not _alive(parent_pid):
                break
            time.sleep(period_s)
        return self.finish()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class AgentThread:
    """An agent in this process (the MuJoCo script on the host, tests)."""

    def __init__(self, ctl_dir: str, log=print, tool: str | None = None):
        self.agent = ScreenVideoAgent(ctl_dir, tool=tool, log=log)
        self._stop = threading.Event()
        self._saved = []
        self.thread = threading.Thread(target=self._run, name="screen-video-agent", daemon=True)
        self.thread.start()

    def _run(self):
        self._saved = self.agent.run(self._stop)

    def close(self) -> list:
        self._stop.set()
        self.thread.join(timeout=130)
        return self._saved


def agent_running(ctl_dir: str) -> bool:
    st = _read_json(os.path.join(_abs(ctl_dir), "status.json"))
    return bool(st) and time.time() - float(st.get("heartbeat", 0)) < ALIVE_S and _alive(int(st.get("pid", -1)))


class HostAgent:
    """tools/sim.py: an agent process on the host for the simulator it launches (context manager)."""

    def __init__(self, ctl_dir: str, python=None):
        self.ctl_dir = ctl_dir
        self.python = python
        self.proc = None

    def __enter__(self):
        import sys
        os.makedirs(_abs(self.ctl_dir), exist_ok=True)
        self.proc = subprocess.Popen([self.python or sys.executable, os.path.join(ROOT, "tools", "screen_video.py"),
                                      "agent", self.ctl_dir, "--parent-pid", str(os.getpid())], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        return self

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)    # the agent lets a running recording finish, then exits
            try:
                self.proc.wait(timeout=130)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False

    def saved(self) -> list:
        st = _read_json(os.path.join(_abs(self.ctl_dir), "status.json")) or {}
        return list(st.get("saved") or [])
