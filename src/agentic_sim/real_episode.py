"""Portable capture directory with independent camera and control clocks.

Uses only the standard library. Device adapters encode native-size RGB and
optional native depth before writing to the episode.
"""

import json
from pathlib import Path
import re
from struct import unpack
from time import monotonic_ns, time_ns
from zlib import crc32

from agentic_sim.real_control import JOINTS, UNITS, pose


SCHEMA = "trc.so101.real_teleop.v1"
SOURCES = {"real", "converted", "fake"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def stream_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name):
        raise ValueError(f"invalid stream name: {name!r}")
    return name


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metadata(manifest):
    arms = manifest["arms"]
    cameras = manifest["cameras"]
    if (manifest["source"] not in SOURCES or not isinstance(arms, dict) or not isinstance(cameras, dict)
            or not 1 <= len(arms) <= 2 or len(cameras) < 2):
        raise ValueError("episode needs a valid source, one or two arms, and at least two cameras")
    for name, calib in arms.items():
        stream_name(name)
        for role in ("leader", "follower"):
            _calibration(calib[role])
    for name, camera in cameras.items():
        stream_name(name)
        _calibration(camera["calibration"])
        for field in ("width", "height", "fps"):
            if type(camera[field]) != int or camera[field] <= 0:
                raise ValueError(f"{name}: invalid camera {field}")


def _calibration(calib):
    if not isinstance(calib, dict) or not all(isinstance(calib.get(key), str) and calib[key].strip()
                                               for key in ("id", "version")):
        raise ValueError("calibration id and version must be supplied")


class EpisodeWriter:
    """Per-camera writers can run independently. Join them before closing."""

    def __init__(self, root, *, source, arms, cameras, provenance, hz, layout=None):
        self.root = Path(root)
        self.origin_ns = monotonic_ns()
        self.manifest = {
            "schema": SCHEMA, "episode_id": self.root.name, "status": "recording",
            "source": source, "provenance": provenance, "layout_id": layout,
            "started_unix_ns": time_ns(), "timestamp_basis": "monotonic_ns since episode start",
            "control_hz": hz, "joint_names": list(JOINTS), "joint_units": UNITS,
            "arms": arms, "cameras": cameras,
        }
        _metadata(self.manifest)
        self.root.mkdir(parents=True, exist_ok=False)
        for camera in cameras:
            (self.root / camera).mkdir()
        self.control = (self.root / "control.jsonl").open("w", encoding="utf-8", buffering=1)
        self.indexes = {name: (self.root / name / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
                        for name in cameras}
        self.count = 0
        self.camera_counts = dict.fromkeys(cameras, 0)
        write_json(self.root / "manifest.json", self.manifest)

    def clock(self):
        return monotonic_ns() - self.origin_ns

    def add_control(self, row):
        _line(self.control, dict(row, frame=self.count))
        self.count += 1

    def add_image(self, camera, png, stamp, *, depth=None, device_timing=None):
        index = self.camera_counts[camera]
        name = f"{index:06d}.png"
        folder = self.root / camera
        (folder / name).write_bytes(png)
        row = {"frame": index, "capture_ns": stamp, "rgb": name}
        if depth is not None:
            depth_name = f"{index:06d}_depth.png"
            (folder / depth_name).write_bytes(depth)
            row["depth"] = depth_name
        if device_timing is not None:
            row["device_timing"] = device_timing
        _line(self.indexes[camera], row)
        self.camera_counts[camera] += 1

    def close(self, *, error=None):
        self.control.close()
        for handle in self.indexes.values():
            handle.close()
        self.manifest.update(
            status="complete" if error is None else "interrupted",
            control_frames=self.count, camera_frames=self.camera_counts,
        )
        if error is not None:
            self.manifest["error"] = str(error)
        write_json(self.root / "manifest.json", self.manifest)


def _check_png(folder, name, expected_size):
    if not isinstance(name, str) or not re.fullmatch(r"\d{6,}(?:_depth)?\.png", name):
        raise ValueError("invalid image name")
    with (folder / name).open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError("invalid PNG signature")
        image_data = False
        first = True
        while header := handle.read(8):
            if len(header) != 8:
                raise ValueError("incomplete PNG chunk")
            count, tag = unpack(">I4s", header)
            if count > 64 * 1024 * 1024:
                raise ValueError("oversized PNG chunk")
            data = handle.read(count)
            tail = handle.read(4)
            if len(data) != count or len(tail) != 4 or unpack(">I", tail)[0] != crc32(data, crc32(tag)):
                raise ValueError("invalid PNG chunk CRC or length")
            if first:
                if tag != b"IHDR" or count != 13 or unpack(">II", data[:8]) != expected_size:
                    raise ValueError(f"PNG dimensions differ from {expected_size}")
                first = False
            if tag == b"IDAT":
                image_data = True
            if tag == b"IEND" and image_data and count == 0:
                return
        raise ValueError("incomplete PNG")


def _timestamp(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _observed_hz(count, first, last):
    if count > 1 and _timestamp(first) and _timestamp(last, minimum=first + 1):
        return round((count - 1) * 1e9 / (last - first), 3)
    return None


class Episode:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported episode schema")
        _metadata(self.manifest)
        self.rows = _read_jsonl(self.root / "control.jsonl")
        self.frames = {}
        self.index_errors = {}
        for camera in self.manifest["cameras"]:
            try:
                self.frames[camera] = _read_jsonl(self.root / camera / "frames.jsonl")
            except (OSError, ValueError) as error:
                self.frames[camera] = []
                self.index_errors[camera] = str(error)

    def validate(self):
        """Read every referenced image so transfer damage and frame gaps identify the stream."""
        m = self.manifest
        issues = []
        takeovers = []

        def issue(stream, frame, reason):
            issues.append({"stream": stream, "frame": frame, "reason": reason})

        if m.get("status") != "complete":
            issue("control", None, f"manifest status is {m.get('status')}")
        if m.get("joint_names") != list(JOINTS) or m.get("joint_units") != UNITS:
            issue("control", None, "joint order or units differ from the schema")
        if not self.rows or m.get("control_frames") != len(self.rows):
            issue("control", None, "empty control log or count mismatch")
        previous = -1
        previous_source = None
        for i, row in enumerate(self.rows):
            try:
                stamp = row["timestamp_ns"]
                if row["frame"] != i or not _timestamp(stamp, minimum=previous + 1):
                    raise ValueError("control frame gap or non-increasing timestamp")
                source = row["command_source"]
                if source not in ("manual", "policy") or row["task_is_policy"] != int(source == "policy"):
                    raise ValueError("inconsistent command source")
                if set(row["arms"]) != set(m["arms"]):
                    raise ValueError("control row arm names differ from manifest")
                for arm in row["arms"].values():
                    pose(arm["command"])
                    pose(arm["readback"])
                    if not (_timestamp(arm["command_ns"], minimum=previous + 1)
                            and _timestamp(arm["readback_ns"], minimum=arm["command_ns"])
                            and arm["readback_ns"] <= stamp):
                        raise ValueError("invalid command/readback timestamp")
                event = row.get("event")
                if event is not None:
                    if (not isinstance(event, dict) or event.get("type") != "takeover"
                            or previous_source not in (None, "policy") or source != "manual"
                            or not _timestamp(event.get("switch_ns"), minimum=previous)
                            or event["switch_ns"] > min(arm["command_ns"] for arm in row["arms"].values())
                            or not _timestamp(event.get("discarded_policy_actions"))
                            or set(event["aligned_readback"]) != set(m["arms"])):
                        raise ValueError("invalid takeover event")
                    for alignment in event["aligned_readback"].values():
                        pose(alignment)
                    takeovers.append({**event, "frame": i, "timestamp_ns": stamp})
                if previous_source == "policy" and source == "manual" and event is None:
                    raise ValueError("policy-to-manual transition has no takeover event")
                previous, previous_source = stamp, source
            except (KeyError, TypeError, ValueError) as error:
                issue("control", i, str(error))

        cameras = {}
        for name, frames in self.frames.items():
            camera = m["cameras"][name]
            if name in self.index_errors:
                issue(name, None, f"camera index unreadable: {self.index_errors[name]}")
            if not frames or m.get("camera_frames", {}).get(name) != len(frames):
                issue(name, None, "empty camera index or count mismatch")
            timestamps = {}
            previous = -1
            previous_sequence = None
            for index, row in enumerate(frames):
                try:
                    frame = row["frame"]
                    stamp = row["capture_ns"]
                    if frame != index:
                        issue(name, index, f"camera frame gap: expected {index}, found {frame}")
                    if not _timestamp(stamp, minimum=previous + 1):
                        raise ValueError("camera timestamp is missing or non-increasing")
                    sequence = (row.get("device_timing") or {}).get("color", {}).get("number")
                    if previous_sequence is not None and sequence is not None:
                        if sequence != previous_sequence + 1:
                            issue(name, index, f"device color sequence: {previous_sequence} -> {sequence}")
                    elif previous >= 0 and stamp - previous > 1.5e9 / camera["fps"]:
                        issue(name, index, "camera timestamp gap exceeds 1.5 nominal frame periods")
                    previous, previous_sequence = stamp, sequence
                    timestamps[index] = stamp
                    folder = self.root / name
                    size = camera["width"], camera["height"]
                    _check_png(folder, row["rgb"], size)
                    if camera.get("depth") or "depth" in row:
                        _check_png(folder, row["depth"], size)
                except (KeyError, OSError, TypeError, ValueError) as error:
                    issue(name, index, str(error))
            if timestamps and self.rows:
                first = self.rows[0].get("timestamp_ns")
                last = self.rows[-1].get("timestamp_ns")
                allowance = 2e9 / camera["fps"]
                if _timestamp(first) and _timestamp(last) and (
                    min(timestamps.values()) - first > allowance or last - max(timestamps.values()) > allowance
                ):
                    issue(name, None, "camera stream does not cover the control interval")
            cameras[name] = {
                "frames": len(frames), "first_ns": timestamps.get(0),
                "last_ns": timestamps.get(len(frames) - 1),
                "observed_fps": _observed_hz(len(frames), timestamps.get(0), timestamps.get(len(frames) - 1)),
            }
        first = self.rows[0].get("timestamp_ns") if self.rows else None
        last = self.rows[-1].get("timestamp_ns") if self.rows else None
        return {"ok": not issues, "episode_id": m["episode_id"], "source": m["source"],
                "arms": list(m["arms"]), "control_frames": len(self.rows),
                "first_ns": first, "last_ns": last, "observed_control_hz": _observed_hz(len(self.rows), first, last),
                "takeovers": takeovers, "cameras": cameras, "issues": issues}
