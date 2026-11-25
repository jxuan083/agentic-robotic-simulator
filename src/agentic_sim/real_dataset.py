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

Segments (`segment`): "handoff" from review.json's handoff frame to the end of its training_range (the ACT
default), "review" the whole training_range, "full" the whole episode. `handoff_jitter` k adds copies starting
k frames earlier ... k frames later than the handoff (the approach is kept in the raw episode), so one recording
covers several starting states. The language instruction comes from the task card (VLA policies; ACT ignores it).
"""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

from agentic_sim.real_control import JOINTS
from agentic_sim.real_episode import Episode

SEGMENTS = ("handoff", "review", "full")
COLORS = {"red": ("red", "紅色"), "blue": ("blue", "藍色"), "yellow": ("yellow", "黃色"),
          "green": ("green", "綠色"), "light_green": ("light green", "淺綠色"),
          "dark_green": ("dark green", "深綠色"), "orange": ("orange", "橘色"), "cyan": ("cyan", "青色"),
          "purple": ("purple", "紫色")}
TRAYS = {"pink": ("pink", "粉色"), "beige": ("beige", "米色")}


def instruction(card: dict | None) -> dict:
    """{'en': 'Put the red block in the beige tray.', 'zh': '把紅色方塊放進米色盤'} from a task card."""
    if not card or "color" not in card or "destination" not in card:
        return {"en": "", "zh": ""}
    c_en, c_zh = COLORS.get(card["color"], (card["color"], card["color"]))
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
    rng = review.get("training_range")
    if not rng or len(rng) != 2:
        raise ValueError("review.json has no training_range")
    start, end = int(rng[0]), int(rng[1])
    if mode == "handoff":
        h = (review.get("frames") or {}).get("handoff")
        if not h:
            raise ValueError("review.json has no handoff frame")
        start = int(h["frame"])
    if not 1 <= start <= end < n_rows:
        raise ValueError(f"segment {start}..{end} outside the {n_rows} control rows")
    return start, end


def _pair(frames: list, stamps: list, at_ns: int, before_ns: int):
    """Index of the frame nearest at_ns among those captured before before_ns (None if none)."""
    k = bisect_right(stamps, before_ns)
    if k == 0:
        return None
    cands = range(max(0, k - 2), k)
    return min(cands, key=lambda j: abs(stamps[j] - at_ns))


def plan_episode(root, *, arm: str = "right", segment: str = "handoff", max_offset_ms: float = 25.0,
                 allow_fake: bool = False, max_gap_periods: float = 1.5, handoff_jitter: int = 0,
                 modes=("demonstration",)) -> dict:
    """What one episode contributes: included / reason, the instruction, and the samples of each segment copy."""
    root = Path(root)
    out = {"episode": str(root), "included": False, "reason": "", "segments": [], "stats": {}}
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS}")
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
    mode = m.get("mode") or "manual"
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
    review = _review(root)
    try:
        start, end = segment_range(review, len(ep.rows), segment)
    except ValueError as e:
        out["reason"] = str(e)
        return out
    period = 1e9 / float(m["control_hz"])
    rows = ep.rows
    cams = {c: f for c, f in ep.frames.items()}
    stamps = {c: [f["capture_ns"] for f in fr] for c, fr in cams.items()}
    max_off = max_offset_ms * 1e6
    starts = [start]
    if segment == "handoff" and handoff_jitter > 0:
        starts = [s for s in range(start - handoff_jitter, start + handoff_jitter + 1) if 1 <= s < end]
    offsets_all = {c: [] for c in cams}
    for s in starts:
        samples = []
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
                if j is None or abs(off) > max_off:
                    out["reason"] = (f"{c}: no frame within {max_offset_ms} ms of the observation at row {i}"
                                     + ("" if off is None else f" (nearest {off / 1e6:.1f} ms)"))
                    out["segments"] = []
                    return out
                offsets_all[c].append(off / 1e6)
                images[c] = {"frame": fr[j]["frame"], "path": str(root / c / fr[j]["rgb"]), "offset_ms": off / 1e6}
            samples.append({"row": i, "state": [float(obs["readback"][j]) for j in JOINTS],
                            "action": [float(act["command"][j]) for j in JOINTS], "images": images,
                            "task_is_policy": int(rows[i]["task_is_policy"]),
                            "command_source": rows[i]["command_source"],
                            "is_handoff": int(i == start), "timestamp_ns": int(rows[i]["timestamp_ns"])})
        out["segments"].append({"start": s, "end": end, "start_offset": s - start, "samples": samples})
    out.update(included=True, reason="ok", instruction=instruction(m.get("task_card")),
               handoff=start, review_frames={k: v["frame"] for k, v in ((review or {}).get("frames") or {}).items()},
               stats={"rows": end - start + 1, "copies": len(starts),
                      "camera_offset_ms": {c: {"min": round(min(v), 3), "max": round(max(v), 3)}
                                           for c, v in offsets_all.items() if v},
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
