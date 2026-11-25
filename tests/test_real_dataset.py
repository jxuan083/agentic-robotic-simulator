"""Real episodes -> training samples (agentic_sim.real_dataset) on the fake collection demo; no LeRobot needed."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_control import JOINTS  # noqa: E402
from agentic_sim.real_dataset import (  # noqa: E402
    condition, draw_markers, find_episodes, instruction, markers, plan_episode, segment_range,
)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    root = tmp_path_factory.mktemp("session")
    subprocess.run([sys.executable, "-S", str(ROOT / "tools/real_collection.py"), "--fake", "--demo",
                    "--session", str(root)], text=True, capture_output=True, check=True)
    return root


def _episode(session, eid="D001-01"):
    return session / "episodes" / eid


def test_handoff_segment_pairs_the_previous_readback_with_this_command(session):
    ep = _episode(session)
    review = json.loads((ep / "review.json").read_text())
    rows = [json.loads(line) for line in open(ep / "control.jsonl")]
    p = plan_episode(ep, allow_fake=True, segment="review")
    assert p["included"], p["reason"]
    seg = p["segments"][0]
    handoff, end = max(1, review["training_range"][0]), review["training_range"][1]
    assert (seg["start"], seg["end"]) == (handoff, end) and len(seg["samples"]) == end - handoff + 1
    s = seg["samples"][0]
    assert s["state"] == [rows[handoff - 1]["arms"]["right"]["readback"][j] for j in JOINTS]
    assert s["action"] == [rows[handoff]["arms"]["right"]["command"][j] for j in JOINTS]
    assert s["is_handoff"] == 1 and all(x["is_handoff"] == 0 for x in seg["samples"][1:])
    for cam, im in s["images"].items():
        assert abs(im["offset_ms"]) <= 25 and Path(im["path"]).is_file()
    assert p["instruction"] == {"en": "Put the red block in the beige tray.", "zh": "把紅色方塊放進米色盤"}


def test_segment_modes_and_handoff_jitter(session):
    ep = _episode(session)
    review = json.loads((ep / "review.json").read_text())
    lo, hi = review["training_range"]
    lo = max(1, lo)
    assert len(plan_episode(ep, allow_fake=True, segment="review")["segments"][0]["samples"]) == hi - lo + 1
    assert "handoff" in plan_episode(ep, allow_fake=True, segment="handoff")["reason"]   # start-pose era only
    n = len(open(ep / "control.jsonl").readlines())
    assert len(plan_episode(ep, allow_fake=True, segment="full")["segments"][0]["samples"]) == n - 1
    j = plan_episode(ep, allow_fake=True, segment="review", handoff_jitter=1)
    assert [s["start_offset"] for s in j["segments"]] == ([-1, 0, 1] if lo > 1 else [0, 1])  # never before row 1
    assert all(s["end"] == hi for s in j["segments"])


def test_exclusions_name_the_reason(session):
    ep = _episode(session)
    assert "fake" in plan_episode(ep)["reason"]
    assert "camera" in plan_episode(ep, allow_fake=True, max_offset_ms=0.01, segment="review", late_run=0)["reason"] or \
        "no frame" in plan_episode(ep, allow_fake=True, max_offset_ms=0.01, segment="review", late_run=0)["reason"]
    assert "in a row" in plan_episode(ep, allow_fake=True, max_offset_ms=0.01, segment="review", late_run=1)["reason"]
    dagger = [p for p in find_episodes([session]) if p.parent.name == "dagger"]
    assert dagger and "mode" in plan_episode(dagger[0], allow_fake=True)["reason"]
    assert plan_episode(dagger[0], allow_fake=True, modes=("dagger",), segment="full")["included"]
    no_review = plan_episode(dagger[0], allow_fake=True, modes=("dagger",), segment="handoff")
    assert not no_review["included"] and "review" in no_review["reason"]


def test_instruction_from_task_card():
    assert instruction({"color": "yellow", "destination": "pink"}) == {
        "en": "Put the yellow block in the pink tray.", "zh": "把黃色方塊放進粉色盤"}
    assert instruction(None) == {"en": "", "zh": ""}
    assert instruction({"color": "red", "destination": None})["en"] == "Pick up the red block."


PARK_REVIEW = {
    "start_pose": "park", "outcome": {"color": "red", "destination": "beige", "success": True},
    "training_ranges": {"grasp": [3, 10], "place": [10, 15]},
    "target": {"d455": {"color": "red", "tray": "pink", "x_mm": 62.5, "y_mm": 150.7, "yaw_deg": 22.5,
                        "pixel": [305.7, 331.1]}},
    "place_target": {"tray": "beige", "slot": 2, "xy_mm": [70, 95],
                     "landed_d455": {"x_mm": 74.0, "y_mm": 99.0, "pixel": [200.0, 120.0]}},
    "round": {"n": 1, "step": 2}, "scene": {"candidates": 3},
}


def test_park_segments_from_training_ranges():
    assert segment_range(PARK_REVIEW, 40, "task") == (3, 15)
    assert segment_range(PARK_REVIEW, 40, "grasp") == (3, 10)
    assert segment_range(PARK_REVIEW, 40, "place") == (10, 15)
    grasp_only = PARK_REVIEW | {"training_ranges": {"grasp": [0, 10]}}
    assert segment_range(grasp_only, 40, "task") == (1, 10)          # row 0 has nothing to observe
    with pytest.raises(ValueError, match="place"):
        segment_range(grasp_only, 40, "place")


def test_condition_vector_and_fallbacks():
    v = condition(PARK_REVIEW, "target+place")
    assert v[:2] == [62.5, 150.7] and v[4:] == [74.0, 99.0]
    assert abs(v[2] - 1.0) < 1e-9 and abs(v[3]) < 1e-9                # 4 * 22.5 deg = 90 deg
    assert condition(PARK_REVIEW | {"target": {"d455": PARK_REVIEW["target"]["d455"] | {"yaw_deg": 90}}},
                     "target")[3] == pytest.approx(1.0)               # 90 deg looks the same as 0 deg
    slot_only = PARK_REVIEW | {"place_target": {"xy_mm": [70, 95]}}
    assert condition(slot_only, "target+place")[4:] == [70.0, 95.0]
    assert condition(PARK_REVIEW, "none") is None
    with pytest.raises(ValueError, match="target.d455"):
        condition({"target": {}}, "target")


def test_markers_are_drawn_where_the_pixels_are():
    Image = pytest.importorskip("PIL.Image")

    marks = markers(PARK_REVIEW)
    assert marks == {"target": [305.7, 331.1], "place": [200.0, 120.0]}
    img = draw_markers(Image.new("RGB", (640, 480)), marks)
    assert img.getpixel((305 + 16, 331)) == (0, 255, 0)              # on the circle
    assert img.getpixel((200 - 16, 120)) == (0, 255, 0)              # on the square's edge
    assert img.getpixel((305, 331)) == (0, 0, 0)                      # the block itself stays visible


def _park_episode(session, tmp_path, review):
    import shutil

    ep = tmp_path / "P0001"
    shutil.copytree(_episode(session), ep)
    manifest = json.loads((ep / "manifest.json").read_text())
    manifest.pop("mode", None)
    (ep / "manifest.json").write_text(json.dumps(manifest))
    (ep / "review.json").write_text(json.dumps(review))
    return ep


def test_park_episode_plan_carries_condition_and_excludes_failures(session, tmp_path):
    ep = _park_episode(session, tmp_path, PARK_REVIEW)
    p = plan_episode(ep, allow_fake=True, condition_kind="target+place")
    assert p["included"], p["reason"]
    seg = p["segments"][0]
    assert (seg["start"], seg["end"]) == (3, 15) and len(seg["samples"]) == 13
    assert len(p["condition"]) == 6 and p["markers"]["target"] == [305.7, 331.1] and p["round"]["step"] == 2
    failed = _park_episode(session, tmp_path / "f", PARK_REVIEW)
    manifest = json.loads((failed / "manifest.json").read_text())
    manifest["operator_result"] = {"success": False}
    (failed / "manifest.json").write_text(json.dumps(manifest))
    assert "failed" in plan_episode(failed, allow_fake=True)["reason"]
    assert plan_episode(failed, allow_fake=True, include_failed=True)["included"]
    missing = _park_episode(session, tmp_path / "m", PARK_REVIEW | {"target": {"d455": None}})
    assert "target.d455" in plan_episode(missing, allow_fake=True, condition_kind="target")["reason"]
