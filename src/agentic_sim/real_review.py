"""Offline keyframe candidates and camera evidence for real collection cards."""

from bisect import bisect_left
from itertools import pairwise
from pathlib import Path

from agentic_sim.real_control import JOINTS
from agentic_sim.real_episode import Episode, write_json
from agentic_sim.real_viewer import write_viewer


def _nearest(frames, stamp):
    stamps = [frame["capture_ns"] for frame in frames]
    i = bisect_left(stamps, stamp)
    return min(frames[max(0, i - 1):i + 1], key=lambda frame: abs(frame["capture_ns"] - stamp))


def _candidates(episode, arm, warnings):
    rows = episode.rows
    if len(rows) < 6:
        return {}, ["too few task frames to locate a grasp"]
    joints = [row["arms"][arm]["readback"] for row in rows]
    grip = [sample["gripper"] for sample in joints]
    close = next((i for i in range(1, len(grip) - 2)
                  if grip[i - 1] - grip[i] > .2 and grip[i - 1] - grip[i + 2] >= 3
                  and all(a >= b - .1 for a, b in pairwise(grip[i - 1:i + 3]))), None)
    if close is None:
        return {}, ["no sustained measured gripper closure"]

    def quiet(i):
        return all(abs(joints[i][j] - joints[i - 1][j]) <= .25 for j in JOINTS[:-1])

    stamps = [row["timestamp_ns"] for row in rows]
    begin = bisect_left(stamps, stamps[close] - 2_000_000_000)
    dwells = []
    for i in range(max(2, begin), close):
        if quiet(i) and quiet(i - 1) and grip[i] >= grip[close - 1] - 2:
            if dwells and dwells[-1][-1] == i - 1:
                dwells[-1].append(i)
            else:
                dwells.append([i])
    # The motion cue only proposes a window for the lift. Image evidence verifies it.
    lift = next((i for i in range(close + 2, len(joints) - 1)
                 if grip[i] < grip[close - 1] - 2
                 and (abs(joints[i]["shoulder_lift"] - joints[close]["shoulder_lift"])
                      + abs(joints[i]["elbow_flex"] - joints[close]["elbow_flex"])) >= 1.5), None)
    hints = {"close": (close, "sustained decrease in measured gripper opening")}
    reasons = []
    if dwells:
        hints["coarse"] = (dwells[0][0], "first open-gripper low-speed interval")
        hints["handoff"] = (dwells[-1][-1], "last open-gripper low-speed interval before closing")
        if len(dwells) > 1:
            reasons.append("multiple pre-close dwells; check where the final correction begins")
    else:
        reasons.append("no unambiguous open-gripper dwell before closing")
    if lift is None:
        reasons.append("no lift candidate with a closed gripper")
    else:
        hints["lift"] = (lift, "arm motion with gripper closed; check consecutive images for a stable lift")
    if any(hints.get(key, (0,))[0] >= close for key in ("coarse", "handoff")):
        reasons.append("dwell ordering is uncertain")
    candidates = {}
    for name, (index, basis) in hints.items():
        stamp = stamps[index]
        images = {}
        for camera in ("global", "wrist"):
            frame = _nearest(episode.frames[camera], stamp)
            offset = frame["capture_ns"] - stamp
            images[camera] = {"frame": frame["frame"], "path": f"{camera}/{frame['rgb']}",
                              "capture_ns": frame["capture_ns"], "offset_ns": offset}
            if abs(offset) > 1e9 / episode.manifest["cameras"][camera]["fps"]:
                reasons.append(f"{name}: {camera} image is too far from the control sample")
        candidates[name] = {"frame": index, "timestamp_ns": stamp, "basis": basis, "images": images,
                            "command_ns": rows[index]["arms"][arm]["command_ns"]}
    if "handoff" in candidates and "lift" in candidates:
        left, right = candidates["handoff"]["frame"], candidates["lift"]["frame"]
        if any(gap["before_frame"] <= right and gap["after_frame"] >= left for gap in warnings):
            reasons.append("control timing gap overlaps the grasp segment")
    if dwells:
        candidates["handoff"]["alternatives"] = [dwell[-1] for dwell in dwells]
    return candidates, reasons


def review_episode(root, *, source, arm, card=None, rgb_depth=None):
    root = Path(root)
    episode = Episode(root)
    report = episode.validate()
    m = episode.manifest
    result = m.get("operator_result", {})
    allowed = {"manual"} if card else {"manual", "policy"}
    issues = []
    if m["source"] != source or list(m["arms"]) != [arm] or not {"global", "wrist"} <= episode.frames.keys():
        issues.append("unexpected episode source, arm or camera streams")
    if {row["command_source"] for row in episode.rows} - allowed:
        issues.append("unexpected command source")
    report["issues"].extend({"stream": "collection", "frame": None, "reason": x} for x in issues)
    report["ok"] = report["ok"] and not issues
    candidates, reasons = _candidates(episode, arm, report["warnings"]) if report["ok"] else (
        {}, ["reader validation failed"]
    )
    if card and (result.get("color") != card["color"] or result.get("placed") is not True):
        reasons.append("target color or destination placement has not been confirmed")
    if card and not card.get("layout_confirmed"):
        reasons.append("actual tray layout requires review")
    global_info = m["cameras"].get("global", {})
    depth = global_info.get("depth") or {}
    depth_frames = sum("depth" in frame and (root / "global" / frame["depth"]).is_file()
                       for frame in episode.frames.get("global", []))
    report.update(candidates=candidates, review_reasons=reasons, operator_result=result,
                  opening_global=(f"global/{episode.frames['global'][0]['rgb']}" if episode.frames.get("global") else None),
                  depth_summary={"frames": depth_frames, "units_m": depth.get("units_m"),
                                 "alignment": depth.get("alignment"),
                                 "color_intrinsics": global_info.get("color_intrinsics"),
                                 "depth_intrinsics": depth.get("intrinsics"),
                                 "depth_to_color": depth.get("depth_to_color"),
                                 "right_base_extrinsics": (rgb_depth or {}).get("right_base_extrinsics"),
                                 "workspace": m.get("motion_setup", {}).get("workspace"),
                                 "reference_points": (rgb_depth or {}).get("reference_points")})
    if report["ok"]:
        write_viewer(episode)
    write_json(root / "readback.json", report)
    return report


def confirmed_frames(episode, frames):
    """Bind reviewed frame numbers to exact timestamps and the nearest camera files."""
    order = ("coarse", "handoff", "close", "lift")
    selected = [frames[key] for key in order]
    if any(type(index) is not int or not 0 <= index < len(episode.rows) for index in selected):
        raise ValueError("review frame is outside this episode")
    if not selected[0] <= selected[1] < selected[2] < selected[3]:
        raise ValueError("expected coarse ≤ handoff < close < lift")
    arm, = episode.manifest["arms"]
    result = {}
    for key, index in zip(order, selected):
        row = episode.rows[index]
        stamp = row["timestamp_ns"]
        result[key] = {"frame": index, "timestamp_ns": stamp, "command_ns": row["arms"][arm]["command_ns"],
                       "images": {camera: _nearest(episode.frames[camera], stamp)
                                  for camera in ("global", "wrist")}}
    return result


def native_depth(episode, index):
    """Read the native D455 depth image closest to a reviewed control sample."""
    import cv2
    import numpy as np

    camera = episode.manifest["cameras"]["global"]
    units = (camera.get("depth") or {}).get("units_m")
    if not units or not 0 < units < 1:
        raise ValueError("global depth scale is unavailable")
    controls = episode.rows
    if not 0 <= index < len(controls):
        raise ValueError("depth sample control frame is outside this episode")
    stamp = controls[index]["timestamp_ns"]
    frame = _nearest(episode.frames["global"], stamp)
    if "depth" not in frame or abs(frame["capture_ns"] - stamp) > 1e9 / camera["fps"]:
        raise ValueError("no sufficiently close native depth frame")
    data = (episode.root / "global" / frame["depth"]).read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16 or image.shape != (camera["height"], camera["width"]):
        raise ValueError("native depth must be a 16-bit single-channel PNG at the recorded dimensions")
    return image, units, frame


def save_depth_comparison(episode, index, top, bottom):
    """Keep the two visible locations, depth units and supporting sample counts."""
    import numpy as np

    image, units, frame = native_depth(episode, index)
    samples = {}
    for name, (x, y) in (("block_top", top), ("tray_bottom", bottom)):
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            raise ValueError("depth sample is outside the native image")
        window = image[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        valid = window[window != 0]
        if valid.size < 3:
            raise ValueError(f"{name}: fewer than three valid depth pixels")
        samples[name] = {"native_pixel": [x, y], "valid_pixels": int(valid.size),
                         "depth_m": float(np.median(valid)) * units}
    result = {"control_frame": index, "global_frame": frame["frame"], "capture_ns": frame["capture_ns"],
              "depth_path": f"global/{frame['depth']}", "units_m": units, "samples": samples,
              "range_difference_m": samples["tray_bottom"]["depth_m"] - samples["block_top"]["depth_m"]}
    write_json(episode.root / "depth_comparison.json", result)
    return result
