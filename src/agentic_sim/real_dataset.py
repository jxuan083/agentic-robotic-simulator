"""Real SO-101 episodes -> training samples (standard library only).

  plan = plan_episode("outputs/261124-right/episodes/D001-01", segment="handoff")
  plan["included"], plan["reason"], plan["segments"][0]["samples"][0]

One sample per control row i of the chosen segment, in LeRobot's order "observe, then act":
  action              row i command (sent at command_ns)
  observation.state   row i-1 readback (read right after the previous command); segments start at row 1 or later
  observation images  per camera, the frame nearest the observation readback time that was captured before
                      row i's command was sent; farther than `max_offset_ms` from the readback excludes the episode
State and action stay in the recorded units (five body joints in calibrated degrees, gripper 0-100 % open), the
LeRobot SO-101 convention; the provisional joint mapping (real_joints) is never applied.

Segments (`segment`): for park-start pick-and-place episodes (review.json training_ranges) "task" from the first
motion to the end of the placement (grasp start .. place end; grasp only when there is no placement), "grasp"
[start, lift], "place" [lift, end]. Older start-pose episodes: "handoff" from review.json's handoff frame to the
end of its training_range, "review" the whole training_range. "full" the whole episode. `handoff_jitter` k adds copies starting
k frames earlier ... k frames later than the segment start (the handoff, or the first motion at park) (the approach is kept in the raw episode), so one recording
covers several starting states. The language instruction comes from the task card (VLA policies; ACT ignores it).

Target condition (`condition`, constant over the episode, written as observation.environment_state):
  "target"        [target x, y (mm, tray-local, D455 opening frame), sin 4*yaw, cos 4*yaw]   (a cube repeats every 90 deg)
  "target+place"  the above + [place x, y]: the D455 landing seen back at park, else the nominal slot
Markers (`markers`): a circle at the target's pixel and a square at the landing's pixel, drawn on the global camera
image of every frame (D455 is fixed, so training and runtime draw the same marks; see draw_markers).
"""

from __future__ import annotations

from bisect import bisect_right
import json
import math
from pathlib import Path

from agentic_sim.real_control import JOINTS
from agentic_sim.real_episode import Episode

SEGMENTS = ("task", "grasp", "place", "handoff", "review", "full")
CONDITIONS = ("none", "target", "target+place")
CONDITION_NAMES = ["target_x_mm", "target_y_mm", "target_sin4yaw", "target_cos4yaw", "place_x_mm", "place_y_mm"]
MARKER_CAMERA = "global"
MARKER = {"color": (0, 255, 0), "outline": (0, 0, 0), "radius": 16, "width": 3}   # green: no block or tray colour
COLORS = {"red": ("red", "紅色"), "blue": ("blue", "藍色"), "yellow": ("yellow", "黃色"),
          "green": ("green", "綠色"), "light_green": ("light green", "淺綠色"),
          "dark_green": ("dark green", "深綠色"), "orange": ("orange", "橘色"), "cyan": ("cyan", "青色"),
          "purple": ("purple", "紫色")}
TRAYS = {"pink": ("pink", "粉色"), "beige": ("beige", "米色")}


def instruction(card: dict | None) -> dict:
    """{'en': 'Put the red block in the beige tray.', 'zh': '把紅色方塊放進米色盤'} from a task card."""
    if not card or not card.get("color"):
        return {"en": "", "zh": ""}
    c_en, c_zh = COLORS.get(card["color"], (card["color"], card["color"]))
    if not card.get("destination"):
        return {"en": f"Pick up the {c_en} block.", "zh": f"夾起{c_zh}方塊"}
    t_en, t_zh = TRAYS.get(card["destination"], (card["destination"], card["destination"]))
    return {"en": f"Put the {c_en} block in the {t_en} tray.", "zh": f"把{c_zh}方塊放進{t_zh}盤"}


def _review(root: Path) -> dict | None:
    p = root / "review.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def segment_range(review: dict | None, n_rows: int, mode: str) -> tuple:
    """(start, end) inclusive control rows, or raise ValueError with the reason."""
    if mode == "full":     # row 0 has no earlier readback / frame to observe: the first sample is row 1
        return 1, n_rows - 1
    if review is None:
        raise ValueError("no review.json (training range not confirmed)")
    if mode in ("task", "grasp", "place"):
        ranges = review.get("training_ranges") or {}
        grasp, place = ranges.get("grasp"), ranges.get("place")
        if not grasp:
            raise ValueError("review.json has no training_ranges.grasp (close / lift not marked)")
        if mode == "place" and not place:
            raise ValueError("review.json has no training_ranges.place (no demonstrated placement)")
        start, end = {"task": (grasp[0], (place or grasp)[1]), "grasp": grasp, "place": place}[mode]
        start, end = max(1, int(start)), int(end)
        if not start <= end < n_rows:
            raise ValueError(f"segment {start}..{end} outside the {n_rows} control rows")
        return start, end
    rng = review.get("training_range")
    if not rng or len(rng) != 2:
        raise ValueError("review.json has no training_range")
    start, end = max(1, int(rng[0])), int(rng[1])   # row 0 has no earlier readback to observe
    if mode == "handoff":
        h = (review.get("frames") or {}).get("handoff")
        if not h:
            raise ValueError("review.json has no handoff frame")
        start = int(h["frame"])
    if not 1 <= start <= end < n_rows:
        raise ValueError(f"segment {start}..{end} outside the {n_rows} control rows")
    return start, end


def condition(review: dict | None, kind: str) -> list | None:
    """The episode's condition vector (see the module doc), or raise ValueError naming what is missing."""
    if kind == "none":
        return None
    target = ((review or {}).get("target") or {}).get("d455")
    if not target:
        raise ValueError("review.json has no target.d455 (D455 did not find the target block)")
    yaw = math.radians(4 * float(target.get("yaw_deg") or 0.0))
    out = [float(target["x_mm"]), float(target["y_mm"]), math.sin(yaw), math.cos(yaw)]
    if kind == "target":
        return out
    place = (review or {}).get("place_target") or {}
    landed = place.get("landed_d455")
    if landed:
        return out + [float(landed["x_mm"]), float(landed["y_mm"])]
    if place.get("xy_mm"):
        return out + [float(v) for v in place["xy_mm"]]
    raise ValueError("review.json has no place_target (needed by condition target+place)")


def markers(review: dict | None) -> dict:
    """Pixel positions on the global camera: {"target": [u, v] | None, "place": [u, v] | None}."""
    target = ((review or {}).get("target") or {}).get("d455") or {}
    landed = ((review or {}).get("place_target") or {}).get("landed_d455") or {}
    return {"target": target.get("pixel"), "place": landed.get("pixel")}


def draw_markers(image, marks: dict):
    """Circle at the target, square at the landing, on a PIL image (modified and returned). Runtime must call
    the same function on the live global frame with the chosen block's and slot's pixels."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(image)
    r, w = MARKER["radius"], MARKER["width"]
    for name, shape in (("target", d.ellipse), ("place", d.rectangle)):
        if marks.get(name):
            u, v = marks[name]
            shape([u - r - 1, v - r - 1, u + r + 1, v + r + 1], outline=MARKER["outline"], width=w + 2)
            shape([u - r, v - r, u + r, v + r], outline=MARKER["color"], width=w)
    return image


def _pair(frames: list, stamps: list, at_ns: int, before_ns: int):
    """Index of the frame nearest at_ns among those captured before before_ns (None if none)."""
    k = bisect_right(stamps, before_ns)
    if k == 0:
        return None
    cands = range(max(0, k - 2), k)
    return min(cands, key=lambda j: abs(stamps[j] - at_ns))


MAX_STILL_S = 0.3        # still stretches inside a training segment keep this much (s)
STILL_DEG_S = 3.0        # commanded body joints slower than this (fastest joint, deg/s) ...
STILL_GRIP_PCT_S = 5.0   # ... and the gripper command slower than this (%/s): standing still
CLOSE_DROP_PCT = 6.0     # a gripper close: the command drops this far below its running max (and an open: rises)


def trim_still(samples, max_still_s: float, dt: float):
    """Drop the samples of a still stretch beyond its first max_still_s. Returns (samples, dropped count)."""
    if not max_still_s or len(samples) < 3:
        return samples, 0
    keep_n = max(1, int(round(max_still_s / dt)))
    out, run, dropped = [samples[0]], 0, 0
    for a, b in zip(samples, samples[1:]):
        body = max(abs(x - y) for x, y in zip(a["action"][:5], b["action"][:5])) / dt
        grip = abs(a["action"][5] - b["action"][5]) / dt
        run = run + 1 if body < STILL_DEG_S and grip < STILL_GRIP_PCT_S else 0
        if run > keep_n:
            dropped += 1
            continue
        out.append(b)
    return out, dropped


def count_closes(gripper) -> int:
    """Gripper closes in a command series: a drop of CLOSE_DROP_PCT below the running max, re-armed by a rise of the
    same size above the running min (close, reopen, close = 2)."""
    closes, mode, hi, lo = 0, "open", None, None
    for x in gripper:
        if mode == "open":
            hi = x if hi is None else max(hi, x)
            if x < hi - CLOSE_DROP_PCT:
                closes, mode, lo = closes + 1, "closed", x
        else:
            lo = min(lo, x)
            if x > lo + CLOSE_DROP_PCT:
                mode, hi = "open", x
    return closes


def plan_episode(root, *, arm: str = "right", segment: str = "task", max_offset_ms: float = 25.0,
                 allow_fake: bool = False, max_gap_periods: float = 1.5, handoff_jitter: int = 0,
                 modes=("demonstration",), condition_kind: str = "none", include_failed: bool = False,
                 late_ms: float = 70.0, late_run: int = 2, max_still_s: float = MAX_STILL_S,
                 regrasp: str = "keep") -> dict:
    """What one episode contributes: included / reason, the instruction, and the samples of each segment copy.

    max_still_s: a stretch where the commanded arm and gripper stand still keeps only its first max_still_s (0: keep
    all). 11/27: the scripted grasps waited ~0.7 s for the gripper; pauses in the operator's grasps reached 1 s; an
    ACT trained on long still stretches tends to stall. regrasp: "keep" / "skip" episodes whose segment closes the
    gripper, reopens and closes again (11/27: 4 of 42 demonstrations, a missed grasp corrected)."""
    root = Path(root)
    out = {"episode": str(root), "included": False, "reason": "", "segments": [], "stats": {}}
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS}")
    if condition_kind not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    try:
        ep = Episode(root)
    except (OSError, ValueError, KeyError) as e:
        out["reason"] = f"unreadable episode: {e}"
        return out
    m = ep.manifest
    out.update(episode_id=m.get("episode_id"), source=m.get("source"), mode=m.get("mode"))
    if m.get("source") == "fake" and not allow_fake:
        out["reason"] = "source is fake (use allow_fake for pipeline tests only)"
        return out
    review = _review(root)
    # Park-start pick-and-place episodes (tools/real_route.py grasp) are demonstrations without a mode field.
    mode = m.get("mode") or ("demonstration" if (review or {}).get("start_pose") == "park" else "manual")
    if mode not in modes:
        out["reason"] = f"recording mode {mode!r} not selected (modes {list(modes)})"
        return out
    if arm not in m.get("arms", {}):
        out["reason"] = f"arm {arm!r} not in the episode ({sorted(m.get('arms', {}))})"
        return out
    report = ep.validate()
    if not report["ok"]:
        out["reason"] = f"readback failed: {report['issues'][:3]}"
        return out
    result = (m.get("operator_result") or {})
    if result.get("cancelled"):
        out["reason"] = "cancelled by the operator"
        return out
    if result.get("success") is False and not include_failed:
        out["reason"] = "operator marked the episode failed"
        return out
    try:
        start, end = segment_range(review, len(ep.rows), segment)
        vector = condition(review, condition_kind)
    except ValueError as e:
        out["reason"] = str(e)
        return out
    period = 1e9 / float(m["control_hz"])
    rows = ep.rows
    cams = {c: f for c, f in ep.frames.items()}
    stamps = {c: [f["capture_ns"] for f in fr] for c, fr in cams.items()}
    max_off = max_offset_ms * 1e6
    starts = [start]
    if segment != "full" and handoff_jitter > 0:
        starts = [s for s in range(start - handoff_jitter, start + handoff_jitter + 1) if 1 <= s < end]
    offsets_all = {c: [] for c in cams}
    late_max = late_ms * 1e6
    late_rows = {c: 0 for c in cams}
    for s in starts:
        samples = []
        run = {c: 0 for c in cams}
        for i in range(s, end + 1):
            if i > s and rows[i]["timestamp_ns"] - rows[i - 1]["timestamp_ns"] > max_gap_periods * period:
                out["reason"] = f"control gap before row {i} exceeds {max_gap_periods} periods"
                out["segments"] = []
                return out
            prev = rows[i - 1]
            obs = prev["arms"][arm]
            act = rows[i]["arms"][arm]
            images = {}
            for c, fr in cams.items():
                j = _pair(fr, stamps[c], obs["readback_ns"], act["command_ns"])
                off = None if j is None else fr[j]["capture_ns"] - obs["readback_ns"]
                # A single dropped camera frame (accepted by the collection readback) leaves the nearest frame up to
                # one period away: keep it for at most `late_run` consecutive rows within `late_ms`.
                late = off is not None and abs(off) > max_off
                run[c] = run[c] + 1 if late else 0
                if j is None or (late and (abs(off) > late_max or run[c] > late_run)):
                    out["reason"] = (f"{c}: no frame within {max_offset_ms} ms of the observation at row {i}"
                                     + ("" if off is None else f" (nearest {off / 1e6:.1f} ms")
                                     + ("" if off is None else f", {run[c]} rows in a row)" if late else ")"))
                    out["segments"] = []
                    return out
                if late and s == starts[0]:
                    late_rows[c] += 1
                offsets_all[c].append(off / 1e6)
                images[c] = {"frame": fr[j]["frame"], "path": str(root / c / fr[j]["rgb"]), "offset_ms": off / 1e6}
            samples.append({"row": i, "state": [float(obs["readback"][j]) for j in JOINTS],
                            "action": [float(act["command"][j]) for j in JOINTS], "images": images,
                            "task_is_policy": int(rows[i]["task_is_policy"]),
                            "command_source": rows[i]["command_source"],
                            "is_handoff": int(i == start), "timestamp_ns": int(rows[i]["timestamp_ns"])})
        samples, cut = trim_still(samples, max_still_s, period / 1e9)
        out["segments"].append({"start": s, "end": end, "start_offset": s - start, "samples": samples,
                                "trimmed_rows": cut})
    regrasps = count_closes([r["arms"][arm]["command"]["gripper"] for r in rows[start:end + 1]]) > 1
    if regrasps and regrasp == "skip":
        out["reason"] = "regrasp in the segment (--regrasp skip)"
        out["segments"] = []
        return out
    out["regrasp"] = regrasps
    out.update(included=True, reason="ok", instruction=instruction(m.get("task_card")),
               condition=vector, markers=markers(review),
               round=(review or {}).get("round"), scene=(review or {}).get("scene"),
               handoff=start, review_frames={k: v["frame"] for k, v in ((review or {}).get("frames") or {}).items()},
               stats={"rows": end - start + 1, "copies": len(starts),
                      "trimmed_rows": out["segments"][0]["trimmed_rows"] if out["segments"] else 0,
                      "camera_offset_ms": {c: {"min": round(min(v), 3), "max": round(max(v), 3)}
                                           for c, v in offsets_all.items() if v},
                      "late_camera_rows": late_rows,
                      "warnings": report.get("warnings", [])})
    return out


def find_episodes(paths) -> list:
    """Episode directories under the given session / episode paths (a directory holding manifest.json)."""
    out = []
    for p in map(Path, paths):
        if (p / "manifest.json").exists():
            out.append(p)
        else:
            out += sorted(q.parent for q in p.glob("**/manifest.json"))
    return out
