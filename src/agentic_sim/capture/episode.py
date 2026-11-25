"""Demonstration episode on disk (issue #13): writer, reader, completeness check and the three grasp moments.

Same layout as the lab's raw teleop captures (trc.so101.lab_capture.v1), with schema trc.so101.sim_capture.v1:

  <episode>/manifest.json         scene, target colour, source "sim", operator, cameras, units, result, counts
  <episode>/control.jsonl         one row per recorded control step: command and state of that step (see ROW_FIELDS)
  <episode>/<camera>/frames.jsonl one row per image: {frame, rgb, step, t_sim, capture_ns}
  <episode>/<camera>/000000.png   RGB, one image per recorded control step

Alignment: control row k and image k of every camera share `frame` = k and the simulation `step` (control step
counter, CONTROL_DT apart), so the two image streams, the state and the command are matched by frame number; `t_sim`
is step * CONTROL_DT and `capture_ns` the wall clock when the image was read back. Pure Python (numpy); images are
written by a pool of threads (PIL when available, else ffmpeg).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import numpy as np

SCHEMA = "trc.so101.sim_capture.v1"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# what one control.jsonl row carries (nested keys use dots); the reader reports any that are missing
ROW_FIELDS = (
    "frame", "step", "t_sim", "wall_ns", "source", "paused", "focused", "ik_status",
    "command.ee_pos", "command.ee_yaw_offset", "command.gripper", "command.joint_pos",
    "state.joint_pos", "state.joint_vel", "state.ee_pos",
    "target", "readback", "blocks", "images",
)

UNITS = {
    "frame": "mount frame = Isaac world: origin on the table top below the arm base, +X forward, +Y left, +Z up",
    "command": {
        "kind": "EE target (operator input after the constrained IK) and the joint position targets sent to the "
                "joint drives in the same step",
        "ee_pos": "m, mount frame (grasp point between the fingers)",
        "ee_yaw_offset": "rad, gripper yaw relative to the controller's grasp heading",
        "gripper": "rad, jaw joint position target",
        "joint_pos": "rad, joint position targets [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, "
                     "gripper]",
    },
    "state": {
        "joint_pos": "rad, measured joint positions (same order as command.joint_pos)",
        "joint_vel": "rad/s, measured joint velocities",
        "ee_pos": "m, mount frame, grasp point from forward kinematics of the measured arm joints",
    },
    "target": "LeRobot units of command.joint_pos: degrees for the arm (calibrated -100..100 range), gripper % open "
              "(provisional Workshop map, agentic_sim.real_joints); same key as the lab captures",
    "readback": "LeRobot units of state.joint_pos (as above)",
    "blocks": "simulation truth per block: colour name, pos m (mount frame), quat wxyz",
}


def _write_png(path: str, img: np.ndarray):
    try:
        from PIL import Image
    except ImportError:
        from agentic_sim.image_io import write_png
        write_png(path, img)
        return
    Image.fromarray(img).save(path, compress_level=1)


class EpisodeWriter:
    def __init__(self, root: str, manifest: dict, cameras: dict, image_writer=None, threads: int = 4,
                 max_queue: int = 64):
        """cameras: key -> {"width", "height", ...} (stored in the manifest)."""
        self.root = root
        os.makedirs(root, exist_ok=False)
        self.manifest = dict(manifest, schema=SCHEMA, status="recording", joint_names=JOINT_NAMES, units=UNITS,
                             cameras={k: dict(c, dir=k, frames="frames.jsonl") for k, c in cameras.items()})
        self.cams = list(cameras)
        for k in self.cams:
            os.makedirs(os.path.join(root, k))
        self._ctl = open(os.path.join(root, "control.jsonl"), "w")
        self._frames = {k: open(os.path.join(root, k, "frames.jsonl"), "w") for k in self.cams}
        self.n = 0
        self.missing_images = {k: 0 for k in self.cams}
        self.t0_ns = time.time_ns()
        self._write = image_writer or _write_png
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.queue_wait_s = 0.0
        self.write_errors = []
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(threads)]
        for t in self._threads:
            t.start()
        self._dump_manifest()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            path, img = item
            try:
                self._write(path, img)
            except Exception as e:  # noqa: BLE001
                self.write_errors.append(f"{path}: {e}")

    def add(self, row: dict, images: dict) -> int:
        """One recorded control step. images: camera key -> HxWx3 uint8 (None: not available this step)."""
        k = self.n
        row = dict(row, frame=k, images={})
        for cam in self.cams:
            img = images.get(cam)
            if img is None:
                self.missing_images[cam] += 1
                continue
            name = f"{k:06d}.png"
            row["images"][cam] = name
            t = time.monotonic()
            self._q.put((os.path.join(self.root, cam, name), np.ascontiguousarray(img[..., :3])))
            self.queue_wait_s += time.monotonic() - t
            self._frames[cam].write(json.dumps({"frame": k, "rgb": name, "step": row["step"], "t_sim": row["t_sim"],
                                                "capture_ns": row.get("capture_ns", {}).get(cam)}) + "\n")
        self._ctl.write(json.dumps(row) + "\n")
        self.n += 1
        return k

    def close(self, status: str, result: dict, extra: dict | None = None) -> dict:
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join()
        self._ctl.close()
        for f in self._frames.values():
            f.close()
        self.manifest.update(status=status, result=result, control_frames=self.n,
                             missing_images=self.missing_images, image_write_errors=self.write_errors[:20],
                             writer_queue_wait_s=round(self.queue_wait_s, 3), **(extra or {}))
        self._dump_manifest()
        return self.manifest

    def _dump_manifest(self):
        tmp = os.path.join(self.root, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.root, "manifest.json"))


# ---- reading back ---------------------------------------------------------------------------------------------
def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _get(row, dotted):
    for part in dotted.split("."):
        if not isinstance(row, dict) or part not in row:
            return None
        row = row[part]
    return row


class Episode:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.rows = _jsonl(os.path.join(root, "control.jsonl"))
        self.frames = {k: _jsonl(os.path.join(root, c["dir"], c["frames"]))
                       for k, c in self.manifest["cameras"].items()}

    @property
    def cameras(self):
        return list(self.manifest["cameras"])

    def image_path(self, cam: str, frame: int) -> str | None:
        name = self.rows[frame]["images"].get(cam)
        return None if name is None else os.path.join(self.root, self.manifest["cameras"][cam]["dir"], name)

    def array(self, dotted: str) -> np.ndarray:
        return np.array([_get(r, dotted) for r in self.rows], float)


def check(ep: Episode) -> dict:
    """Completeness: fields of every row, continuous frames and steps, and for every camera and frame the control row
    and the camera index name the same image of the same step and that file exists. errors: camera, frame, reason."""
    rep = {"frames": len(ep.rows), "missing_fields": {}, "frame_gaps": [], "step_gaps": [], "images": {}}
    for r in ep.rows:
        for f in ROW_FIELDS:
            if _get(r, f) is None:
                rep["missing_fields"][f] = rep["missing_fields"].get(f, 0) + 1
    frames = [r.get("frame") for r in ep.rows]
    rep["frame_gaps"] = [i for i, f in enumerate(frames) if f != i][:20]
    steps = np.array([r.get("step", -1) for r in ep.rows])
    if len(steps) > 1:
        d = np.diff(steps)
        stride = int(np.median(d))
        rep["step_stride"] = stride
        rep["step_gaps"] = [int(steps[i]) for i in np.nonzero(d != stride)[0]][:20]
    rep["errors"] = []
    for cam in ep.cameras:
        cdir = ep.manifest["cameras"][cam].get("dir", cam)
        listed = {}
        duplicate = []
        for fr in ep.frames[cam]:
            if fr.get("frame") in listed:
                duplicate.append(fr.get("frame"))
            listed[fr.get("frame")] = fr
        bad = {k: [] for k in ("missing_in_row", "missing_in_frames_jsonl", "name_mismatch", "step_mismatch",
                               "missing_files")}

        def err(kind, i, why):
            bad[kind].append(i)
            rep["errors"].append({"camera": cam, "frame": i, "reason": why})

        # per frame: the control row and the camera index must name the same image of the same step, on disk
        for i, row in enumerate(ep.rows):
            name = (row.get("images") or {}).get(cam)
            fr = listed.get(i)
            if name is None:
                err("missing_in_row", i, "control row has no image for this camera")
            if fr is None:
                err("missing_in_frames_jsonl", i, "camera index has no entry for this frame")
                continue
            if name is not None and fr.get("rgb") != name:
                err("name_mismatch", i, f"control row names {name}, camera index names {fr.get('rgb')}")
            if fr.get("step") != row.get("step"):
                err("step_mismatch", i, f"camera index step {fr.get('step')}, control row step {row.get('step')}")
            for f in {n for n in (name, fr.get("rgb")) if n}:
                if not os.path.exists(os.path.join(ep.root, cdir, f)):
                    err("missing_files", i, f"image file {cdir}/{f} does not exist")
        extra = sorted(k for k in listed if not (isinstance(k, int) and 0 <= k < len(ep.rows)))
        for k in extra:
            rep["errors"].append({"camera": cam, "frame": k, "reason": "camera index entry without a control row"})
        for k in duplicate:
            rep["errors"].append({"camera": cam, "frame": k, "reason": "camera index lists this frame twice"})
        rep["images"][cam] = {"listed": len(listed), "counts": {k: len(v) for k, v in bad.items()},
                              **{k: v[:20] for k, v in bad.items()},
                              "extra_in_frames_jsonl": extra[:20], "duplicate_in_frames_jsonl": duplicate[:20]}
    n_err = len(rep["errors"])
    rep["errors"] = rep["errors"][:50]
    rep["error_count"] = n_err
    rep["ok"] = not rep["missing_fields"] and not rep["frame_gaps"] and not rep["step_gaps"] and n_err == 0
    return rep


def grasp_moments(ep: Episode, lift_m: float = 0.01) -> dict:
    """The three moments of a pick and place, from the recorded data alone:
    close   first frame the jaw command starts closing on the target block (before it is lifted)
    lift    first frame the target block is lift_m above its initial height (it leaves the storage tray)
    release first frame after the lift the jaw command starts opening, with the block over the placement tray
    plus, for each, what the state did (jaw measured, block height) so their order can be checked."""
    m = ep.manifest
    target = m["result"].get("target_block") if m.get("result") else None
    if target is None:
        target = next(i for i, b in enumerate(ep.rows[0]["blocks"]) if b["color"] == m["target_color"])
    g_cmd = ep.array("command.gripper")
    jaw = np.array([r["state"]["joint_pos"][5] for r in ep.rows])
    z = np.array([r["blocks"][target]["pos"][2] for r in ep.rows])
    mid = 0.5 * (m["gripper"]["open"] + m["gripper"]["closed"])
    closing = np.r_[False, np.diff(g_cmd) < -1e-6]
    opening = np.r_[False, np.diff(g_cmd) > 1e-6]
    lifted = np.nonzero(z > z[0] + lift_m)[0]
    out = {"target_block": target}
    if len(lifted) == 0:
        return out
    lift = int(lifted[0])
    # the close that grasped: the last start of a closing run before the lift
    starts = [i for i in np.nonzero(closing)[0] if i < lift and (i == 0 or not closing[i - 1])]
    close = int(starts[-1]) if starts else None
    rel = [i for i in np.nonzero(opening)[0] if i > lift and (i == 0 or not opening[i - 1])]
    release = int(rel[0]) if rel else None
    for name, f in (("close", close), ("lift", lift), ("release", release)):
        if f is None:
            continue
        r = ep.rows[f]
        out[name] = {"frame": f, "step": r["step"], "t_sim": r["t_sim"], "gripper_cmd": float(g_cmd[f]),
                     "jaw": float(jaw[f]), "block_z": float(z[f]), "block_pos": r["blocks"][target]["pos"],
                     "ee_pos": r["state"]["ee_pos"], "images": r["images"]}
    if close is not None:   # the jaw stops on the block: first frame after close where it no longer moves
        dj = np.abs(np.diff(jaw[close:lift + 1]))
        still = np.nonzero(dj < 1e-3)[0]
        out["close"]["jaw_settled_frame"] = int(close + still[0] + 1) if len(still) else None
        out["close"]["gripper_below_mid_frame"] = next((int(i) for i in range(close, lift + 1) if g_cmd[i] < mid),
                                                       None)
    return out
