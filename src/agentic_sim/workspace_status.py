"""Operator workspace status shared by the Isaac and MuJoCo windows (pure Python): clocks, perception snapshot ages,
the status column text, and which image a camera cell shows.

Clocks (kept apart, never mixed):
  - UI FPS: rendered frames per host second (time.monotonic), over the last `window_s`.
  - RTF (real-time factor): delta simulated time / delta host monotonic time over the same window. Paused frames
    and Reset start a new segment: the window is emptied, so no RTF spans a pause or a Reset ("—" until two frames
    of the new segment exist).
  - Perception result age: host monotonic now minus the capture time of the image the result was computed from
    (a late result keeps its original capture time). No result: "no result yet" (尚無結果).
  - Perception latency: completion minus capture time of the latest result (how long the estimate took).
Physical waits in the scripted operator use simulated time (agentic_sim.teleop.script_core: settle, sim_timeout);
UI waits and service timeouts use host monotonic time.
"""

from __future__ import annotations

import json
import os
from collections import deque

STALE_S = 2.0          # a perception result older than this is shown as a dated snapshot, never as the live image
NO_RESULT = "no result yet"   # window text is English: the Kit and MuJoCo fonts have no CJK glyphs


class Clocks:
    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self._frames = deque()        # host times of rendered frames
        self._seg = deque()           # (host time, simulated time) of the current running segment
        self.paused = False

    def reset(self):
        """Reset pressed (or a new session): the next RTF starts from scratch."""
        self._seg.clear()

    def frame(self, now: float, t_sim: float, paused: bool):
        self._frames.append(now)
        while self._frames and now - self._frames[0] > self.window_s:
            self._frames.popleft()
        if paused:
            self._seg.clear()         # nothing measured across a pause
        else:
            if self._seg and t_sim < self._seg[-1][1]:
                self._seg.clear()     # simulated time went back: a Reset this frame
            self._seg.append((now, t_sim))
            while len(self._seg) > 2 and now - self._seg[0][0] > self.window_s:
                self._seg.popleft()
        self.paused = paused

    @property
    def ui_fps(self) -> float | None:
        if len(self._frames) < 2:
            return None
        span = self._frames[-1] - self._frames[0]
        return (len(self._frames) - 1) / span if span > 0 else None

    @property
    def rtf(self) -> float | None:
        if self.paused or len(self._seg) < 2:
            return None
        (h0, s0), (h1, s1) = self._seg[0], self._seg[-1]
        return (s1 - s0) / (h1 - h0) if h1 > h0 else None


def fmt(x, spec=".1f", none="—") -> str:
    return none if x is None else format(x, spec)


def perception_item(state: dict | None, now: float) -> dict:
    """Age / latency / staleness of one camera's latest result (state: the engine's per-camera perception state)."""
    latest = (state or {}).get("latest")
    if not latest:
        return {"age_s": None, "latency_s": None, "stale": True, "frame": None, "error": (state or {}).get("error")}
    cap = latest.get("captured_at_monotonic_s")
    done = latest.get("completed_at_monotonic_s")
    age = None if cap is None else now - cap
    return {"age_s": age, "latency_s": None if (cap is None or done is None) else done - cap,
            "stale": age is None or age > STALE_S, "frame": latest.get("frame_number"),
            "error": (state or {}).get("error")}


def cell_view(annotate: bool, item: dict | None) -> tuple:
    """What a camera cell shows: ("raw", text) or ("annotated", text).
    The annotated image is the snapshot the result came from (boxes are drawn on that image, never on a newer
    frame); a stale or missing result shows the raw camera with the reason."""
    if not annotate:
        return "raw", "raw"
    if item is None or item.get("age_s") is None:
        return "raw", f"raw (annotation: {NO_RESULT})"
    if item["stale"]:
        return "raw", f"raw (annotation too old: {item['age_s']:.1f} s ago)"
    return "annotated", f"annotated snapshot, {item['age_s']:.1f} s ago"


def status_lines(info: dict) -> list:
    """The status column. info: target_color, task, gripper_cmd, jaw, gripper_open, recording (dict or None),
    paused, clocks (Clocks), perception {camera: perception_item} or None, perception_stage."""
    c = info["clocks"]
    lines = [f"Target colour: {info.get('target_color') or '-'}", f"Task: {info.get('task') or 'manual teleop'}"]
    jaw, cmd = info.get("jaw"), info.get("gripper_cmd")
    if jaw is not None:
        state = "open" if info.get("gripper_open") else "closed"
        lines.append(f"Gripper: {state} (cmd {fmt(cmd, '.2f')}, measured {fmt(jaw, '.2f')} rad)")
    rec = info.get("recording")
    if rec is None:
        lines.append("Recording: off")
    else:
        text = {"waiting_cameras": "waiting for cameras", "recording": "RECORDING",
                "saved": "saved - press R for the next one"}.get(rec.get("state"), rec.get("state"))
        lines.append(f"Recording: {text}" + (f"  {rec['frames']} frames" if rec.get("state") == "recording" else ""))
        if rec.get("episode"):
            lines.append(f"  {os.path.basename(str(rec['episode']).rstrip('/'))}")
    lines.append("PAUSED (simulated time stopped)" if info.get("paused") else "")
    lines.append(f"UI FPS {fmt(c.ui_fps)}   RTF {fmt(c.rtf, '.2f')}")
    perc = info.get("perception")
    if perc is None:
        lines.append("Perception: off")
    else:
        stage = info.get("perception_stage", "estimate")
        lines.append("Perception: " + ("capture only (no estimate)" if stage == "capture" else "capture + estimate"))
        for cam, it in perc.items():
            if it.get("age_s") is None:
                lines.append(f"  {cam}: {NO_RESULT}" + (f" (error: {it['error']})" if it.get("error") else ""))
            else:
                lines.append(f"  {cam}: {it['age_s']:.1f} s old, took {fmt(it['latency_s'], '.2f')} s"
                             + ("  STALE" if it["stale"] else ""))
    return [x for x in lines if x != ""]


class MetricsLog:
    """One JSON line per host second (benchmarks): UI FPS, RTF, perception age / latency per camera."""

    def __init__(self, path: str | None, period_s: float = 1.0):
        self.path, self.period_s = path, period_s
        self._next = None
        self._f = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._f = open(path, "a")

    def sample(self, now: float, t_sim: float, clocks: Clocks, perception: dict | None, extra: dict | None = None):
        if self._f is None:
            return
        if self._next is None:
            self._next = now + self.period_s
            return
        if now < self._next:
            return
        self._next = now + self.period_s
        row = {"t_host": round(now, 4), "t_sim": round(t_sim, 4), "ui_fps": clocks.ui_fps, "rtf": clocks.rtf,
               "paused": clocks.paused}
        if perception is not None:
            row["perception"] = {k: {"age_s": v["age_s"], "latency_s": v["latency_s"], "frame": v["frame"]}
                                 for k, v in perception.items()}
        if extra:
            row.update(extra)
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
