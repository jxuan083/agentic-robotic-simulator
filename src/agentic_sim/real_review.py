"""Offline keyframe candidates and camera evidence for real collection cards."""

from bisect import bisect_left
from pathlib import Path

from agentic_sim.real_control import JOINTS
from agentic_sim.real_episode import Episode, write_json
from agentic_sim.real_viewer import write_viewer


def _nearest(frames, stamp):
    stamps = [frame["capture_ns"] for frame in frames]
    i = bisect_left(stamps, stamp)
    return min(frames[max(0, i - 1):i + 1], key=lambda frame: abs(frame["capture_ns"] - stamp))


CLOSE_DROP = 4.0       # % below the opening reached before the grasp
HOLD_FRAMES = 5        # the drop lasts at least this many frames
RELEASE_FRAMES = 15    # a release stays open this many frames (0.5 s at 30 Hz)
STILL_DEG = 0.25       # per-frame body change counted as still
LIFT_DEG = 3.0         # body motion after the closure that counts as lifting
STABLE_FRAMES = 10     # still frames that make a stable lift
RELEASE_RISE = 4.0     # % above the held opening: the block is released (placement demonstrations)


def keyframes(joints):
    """start / close / lift / release / end from follower readbacks (slow closures and any lifting joint)."""
    grip = [sample["gripper"] for sample in joints]
    n = len(joints)
    step = [0.0] + [max(abs(joints[i][j] - joints[i - 1][j]) for j in JOINTS[:-1]) for i in range(1, n)]
    moving = [i for i in range(n) if step[i] > STILL_DEG or (i and abs(grip[i] - grip[i - 1]) > 0.5)]
    if not moving:
        return {}
    marks = {"start": max(0, moving[0] - 5), "end": min(n - 1, moving[-1] + 5)}
    peak = grip[0]
    close = None
    for i in range(0, n - 2):
        peak = max(peak, grip[i])
        if grip[i] <= peak - CLOSE_DROP and all(g <= peak - CLOSE_DROP for g in grip[i:i + HOLD_FRAMES]):
            close = i
            while close > 0 and grip[close - 1] > grip[close] + 0.1:
                close -= 1
            break
    if close is None:
        return marks
    marks["close"] = close
    held_at = next((i for i in range(close, n - 3) if abs(grip[i + 3] - grip[i]) < 0.5), close)
    hold = grip[held_at]
    base = joints[held_at]
    lifting = next((i for i in range(held_at, n)
                    if max(abs(joints[i][j] - base[j]) for j in JOINTS[:-1]) >= LIFT_DEG), None)
    if lifting is not None:
        lift = None
        for i in range(lifting, n - STABLE_FRAMES):
            if all(step[k] <= STILL_DEG for k in range(i, i + STABLE_FRAMES)):
                lift = i
                break
        marks["lift"] = lift if lift is not None else min(n - 1, lifting + 30)
        release = next((i for i in range(marks["lift"], n - RELEASE_FRAMES)
                        if all(g >= hold + RELEASE_RISE for g in grip[i:i + RELEASE_FRAMES])), None)
        if release is not None:
            while release > marks["lift"] and grip[release - 1] < grip[release] - 0.1:
                release -= 1
            marks["release"] = release
    return marks


def _candidates(episode, arm, warnings):
    rows = episode.rows
    if len(rows) < 6:
        return {}, ["too few task frames to locate a grasp"]
    joints = [row["arms"][arm]["readback"] for row in rows]
    marks = keyframes(joints)
    if "close" not in marks:
        return {}, ["no sustained measured gripper closure"]
    close, lift = marks["close"], marks.get("lift")
    stamps = [row["timestamp_ns"] for row in rows]
    # The policy learns the whole approach: training starts at the settled standard start (first task frame).
    hints = {"start": (marks["start"], "first arm motion (minus 5 frames)"),
             "close": (close, "sustained decrease in measured gripper opening"),
             "end": (marks["end"], "last arm motion (plus 5 frames)")}
    if "release" in marks:
        hints["release"] = (marks["release"], "gripper reopened after the lift (placement)")
    reasons = []
    if lift is None:
        reasons.append("no lift candidate with a closed gripper")
    else:
        hints["lift"] = (lift, "first still moment after lifting with the gripper closed")
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
    if "lift" in candidates:
        right = candidates["lift"]["frame"]
        if any(gap["before_frame"] <= right for gap in warnings):
            reasons.append("control timing gap inside the training segment (start to lift)")
    return candidates, reasons


def locate_opening(episode, card=None, quads=None):
    """D455 block positions (tray-local mm) in the opening global frame; the card's block if a card is given."""
    try:
        import cv2

        from agentic_sim.real_locate import locate, match_target
    except ImportError as error:  # camera-less test environments
        return {"error": f"D455 localisation unavailable: {error}"}

    frames = episode.frames.get("global") or []
    if not frames:
        return None
    first = frames[0]
    image = cv2.imread(str(episode.root / "global" / first["rgb"]))
    if image is None:
        return {"error": "opening global frame unreadable"}
    result = locate(image, quads)
    result["global_frame"] = first["frame"]
    if card:
        result["target"] = match_target(result, card)
    return result


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
    if card and not card.get("layout_confirmed") and not (located or {}).get("layout_from_d455"):
        reasons.append("actual tray layout requires review")
    global_info = m["cameras"].get("global", {})
    depth = global_info.get("depth") or {}
    depth_frames = sum("depth" in frame and (root / "global" / frame["depth"]).is_file()
                       for frame in episode.frames.get("global", []))
    quads = (rgb_depth or {}).get("tray_quads")
    located = locate_opening(episode, card, quads) if report["ok"] and source == "real" else None  # fake is synthetic
    if card and located is not None and "error" not in located:
        target = located.get("target")
        if not target:
            reasons.append("D455 did not find the target block in the opening frame")
        elif target["tray_clipped"]:
            reasons.append("target tray touches the D455 image border; its D455 position is unreliable")
        elif target["same_colour_in_tray"] > 1:
            reasons.append("more than one block of the target colour in the source tray; confirm which one")
        else:
            located["layout_from_d455"] = [target["x_mm"], target["y_mm"], target["yaw_deg"]]
    report.update(candidates=candidates, review_reasons=reasons, operator_result=result, d455_blocks=located,
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
    order = ("start", "close", "lift")
    selected = [frames.get(key, 0 if key == "start" else None) for key in order]
    if any(type(index) is not int or not 0 <= index < len(episode.rows) for index in selected):
        raise ValueError("review frame is outside this episode")
    if not selected[0] < selected[1] < selected[2]:
        raise ValueError("expected start < close < lift")
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
