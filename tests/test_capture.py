"""Issue #13: demonstration capture format, readback check and the three grasp moments (pure Python, no simulator).

A synthetic pick and place (the block follows the closed gripper from the pink tray to the beige tray) goes through
the real DemoRecorder / EpisodeWriter into a temp directory and is read back with tools/read_episode.py."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agentic_sim.capture.episode import ROW_FIELDS, SCHEMA, Episode, check, grasp_moments  # noqa: E402
from agentic_sim.capture.recorder import DemoRecorder  # noqa: E402
from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.real_joints import lerobot_to_shared, shared_to_lerobot  # noqa: E402
from agentic_sim.scene_layout import block_layout, trays  # noqa: E402

LAB = load(os.path.join(ROOT, "configs/scenes/lab_20261120.yaml"))
DT = 1.0 / 60.0
GRIP = {"open": 0.9, "closed": -0.1}
CAMS = {"global": {"width": 32, "height": 18}, "wrist": {"width": 16, "height": 12}}


def _npy_writer(path, img):           # fast stand-in for PNG in tests (same file name)
    with open(path, "wb") as f:
        np.save(f, img)


def _recorder(tmp_path, **kw):
    lay = block_layout(LAB, colors=["red", "yellow"])
    rec = DemoRecorder(str(tmp_path), scene_path="configs/scenes/lab_20261120.yaml", experiment=None,
                       target_color="yellow", block_colors=["red", "yellow"], trays=trays(LAB),
                       edge=v(LAB["block"]["edge"]), gripper=GRIP, cameras=CAMS, control_dt=DT, record_hz=20.0,
                       operator="test", log=lambda *a: None,
                       writer_cls=lambda root, man, cams: __import__("agentic_sim.capture.episode", fromlist=["x"])
                       .EpisodeWriter(root, man, cams, image_writer=_npy_writer), **kw)
    return rec, lay


def _pick_place(rec, lay, missing_at=(), drop_row=None):
    """Script of a pick and place in control steps; returns the frame-step of each phase start."""
    tr = trays(LAB)
    red, y0 = np.array(lay.init_pos[0]), np.array(lay.init_pos[1])
    dst = np.array([*tr["placement_beige"]["center"], y0[2] - tr["storage_pink"]["floor_top"]
                    + tr["placement_beige"]["floor_top"]])
    q = np.zeros(6)
    grip, block, ee = GRIP["open"], y0.copy(), y0 + [0, 0, 0.08]
    phases = {}
    step = 100
    plan = [("approach", 40), ("close", 30), ("lift", 40), ("carry", 60), ("lower", 40), ("open", 30), ("wait", 90)]
    for name, n in plan:
        phases[name] = step
        for i in range(n):
            a = (i + 1) / n
            if name == "approach":
                ee = y0 + [0, 0, 0.08 * (1 - a)]
            elif name == "close":
                grip = GRIP["open"] + a * (0.2 - GRIP["open"])       # stops on the block
            elif name == "lift":
                ee = y0 + [0, 0, 0.06 * a]
            elif name == "carry":
                ee = y0 + [0, 0, 0.06] + a * (dst - y0)
            elif name == "lower":
                ee = dst + [0, 0, 0.06 * (1 - a)]
            elif name == "open":
                grip = 0.2 + a * (GRIP["open"] - 0.2)
            if name in ("lift", "carry", "lower"):
                block = ee.copy()
            jaw = grip if name != "close" else max(grip, 0.25)          # measured jaw lags / stops on the block
            q[5] = jaw
            cmd = np.r_[q[:5], grip]
            imgs = {"global": (np.full((18, 32, 3), step % 255, np.uint8), step),
                    "wrist": (np.full((12, 16, 3), step % 255, np.uint8), step)}
            if step in missing_at:
                imgs["wrist"] = None
            rec.frame(step, step * DT, paused=False, focused=True, source="keyboard", ik_status="ok", cmd_ee=ee,
                      cmd_yaw=0.0, cmd_q=cmd, q=q.copy(), qd=np.zeros(6), ee=ee, blocks=[(red, (1, 0, 0, 0)),
                                                                                       (block.copy(), (1, 0, 0, 0))],
                      grab=lambda imgs=imgs: imgs)
            step += 1
    return phases


def test_joint_map_round_trip():
    r = [-6.04, -97.27, 99.0, 71.48, 1.29, 17.0]
    assert np.allclose(shared_to_lerobot(*lerobot_to_shared(r)), r)


def test_recorded_pick_place_reads_back_complete_with_ordered_moments(tmp_path):
    rec, lay = _recorder(tmp_path)
    ph = _pick_place(rec, lay)
    assert not rec.recording and len(rec.episodes) == 1                    # ended by the success hold
    root = rec.episodes[0]["path"]
    ep = Episode(root)
    m = ep.manifest
    assert m["schema"] == SCHEMA and m["source"] == "sim" and m["status"] == "complete"
    assert m["result"]["success"] and m["result"]["target_block"] == 1 and m["target_color"] == "yellow"
    assert m["record_stride"] == 3 and m["control_frames"] == len(ep.rows)
    rep = check(ep)
    assert rep["ok"], rep
    assert all(all(f.split(".")[0] in r for f in ROW_FIELDS) for r in ep.rows)
    assert ep.rows[0]["images"] == {"global": "000000.png", "wrist": "000000.png"}
    assert ep.frames["wrist"][5]["step"] == ep.rows[5]["step"]               # images and rows share the step
    mo = grasp_moments(ep)
    assert mo["close"]["frame"] < mo["lift"]["frame"] < mo["release"]["frame"]
    assert ph["close"] <= mo["close"]["step"] < ph["close"] + 3
    assert ph["lift"] <= mo["lift"]["step"] < ph["carry"]
    assert ph["open"] <= mo["release"]["step"] < ph["open"] + 3
    import read_episode
    assert read_episode.main([root]) == 0
    out = json.load(open(os.path.join(root, "readback", "readback.json")))
    assert out["ok"] and all(out["order"].values()) and out["frames"] == len(ep.rows)


def test_missing_image_and_reset_are_reported(tmp_path):
    rec, lay = _recorder(tmp_path)
    _pick_place(rec, lay, missing_at={130})        # step 130 is a recorded step (start 100, stride 3)
    ep = Episode(rec.episodes[0]["path"])
    rep = check(ep)
    assert not rep["ok"] and rep["missing_fields"] == {} and rep["images"]["wrist"]["missing_in_frames_jsonl"] == [10]
    assert ep.manifest["missing_images"] == {"global": 0, "wrist": 1}
    rec.reset()                                    # next episode; closing it unfinished is not a success
    rec.close()
    assert len(rec.episodes) == 2 and rec.episodes[1]["status"] == "empty"
    assert rec.episodes[1]["result"] == {"success": False, "reason": "closed", "target_block": 1, "lifted": False}


def test_episode_waits_for_the_first_images(tmp_path):
    rec, lay = _recorder(tmp_path)
    blocks = [(np.array(p), (1, 0, 0, 0)) for p in lay.init_pos]
    for step in range(5):
        rec.frame(step, step * DT, paused=False, focused=True, source="none", ik_status="ok", cmd_ee=np.zeros(3),
                  cmd_yaw=0.0, cmd_q=np.r_[np.zeros(5), GRIP["open"]], q=np.zeros(6), qd=np.zeros(6),
                  ee=np.zeros(3), blocks=blocks,
                  grab=lambda s=step: {"global": None if s < 3 else (np.zeros((18, 32, 3), np.uint8), 0),
                                       "wrist": (np.zeros((12, 16, 3), np.uint8), 0)})
    rec.close()
    ep = Episode(rec.episodes[0]["path"])
    assert [r["step"] for r in ep.rows] == [3] and check(ep)["ok"]


def test_unknown_target_colour_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no green block"):
        DemoRecorder(str(tmp_path), scene_path="x", experiment=None, target_color="green", block_colors=["red"],
                     trays=trays(LAB), edge=0.03, gripper=GRIP, cameras=CAMS, control_dt=DT, record_hz=20,
                     operator="test")


def _edit_jsonl(path, frame, fn):
    rows = [json.loads(line) for line in open(path)]
    fn(next(r for r in rows if r["frame"] == frame))
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def _wrist_bad_name(root):
    _edit_jsonl(os.path.join(root, "control.jsonl"), 5, lambda r: r["images"].update(wrist="does_not_exist.png"))


def _global_neighbour(root):            # an image that exists, but of the next frame
    _edit_jsonl(os.path.join(root, "control.jsonl"), 5, lambda r: r["images"].update(**{"global": "000006.png"}))


def _index_step(root):
    _edit_jsonl(os.path.join(root, "wrist", "frames.jsonl"), 5, lambda r: r.update(step=r["step"] + 3))


def _png_gone(root):
    os.remove(os.path.join(root, "global", "000005.png"))


@pytest.mark.parametrize("corrupt,camera,kind", [
    (_wrist_bad_name, "wrist", "name_mismatch"),
    (_global_neighbour, "global", "name_mismatch"),
    (_index_step, "wrist", "step_mismatch"),
    (_png_gone, "global", "missing_files"),
])
def test_row_and_camera_index_are_cross_checked_per_frame(tmp_path, corrupt, camera, kind):
    """Each corruption of a complete episode (one at a time) fails the readback and names camera, frame, reason."""
    rec, lay = _recorder(tmp_path)
    _pick_place(rec, lay)
    root = rec.episodes[0]["path"]
    import read_episode
    assert check(Episode(root))["ok"] and read_episode.main([root]) == 0
    corrupt(root)
    rep = check(Episode(root))
    assert not rep["ok"] and rep["images"][camera][kind] == [5]
    assert any(e["camera"] == camera and e["frame"] == 5 for e in rep["errors"])
    assert read_episode.main([root]) == 1
    assert json.load(open(os.path.join(root, "readback", "readback.json")))["ok"] is False
