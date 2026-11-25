"""Real episodes -> training samples (agentic_sim.real_dataset) on the fake collection demo; no LeRobot needed."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_control import JOINTS  # noqa: E402
from agentic_sim.real_dataset import find_episodes, instruction, plan_episode  # noqa: E402


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
    p = plan_episode(ep, allow_fake=True)
    assert p["included"], p["reason"]
    seg = p["segments"][0]
    handoff, end = review["frames"]["handoff"]["frame"], review["training_range"][1]
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
    assert len(plan_episode(ep, allow_fake=True, segment="review")["segments"][0]["samples"]) == hi - lo + 1
    n = len(open(ep / "control.jsonl").readlines())
    assert len(plan_episode(ep, allow_fake=True, segment="full")["segments"][0]["samples"]) == n - 1
    j = plan_episode(ep, allow_fake=True, handoff_jitter=1)
    assert [s["start_offset"] for s in j["segments"]] == [-1, 0, 1]
    assert all(s["end"] == hi for s in j["segments"])


def test_exclusions_name_the_reason(session):
    ep = _episode(session)
    assert "fake" in plan_episode(ep)["reason"]
    assert "camera" in plan_episode(ep, allow_fake=True, max_offset_ms=0.01)["reason"] or \
        "no frame" in plan_episode(ep, allow_fake=True, max_offset_ms=0.01)["reason"]
    dagger = [p for p in find_episodes([session]) if p.parent.name == "dagger"]
    assert dagger and "mode" in plan_episode(dagger[0], allow_fake=True)["reason"]
    assert plan_episode(dagger[0], allow_fake=True, modes=("dagger",), segment="full")["included"]
    no_review = plan_episode(dagger[0], allow_fake=True, modes=("dagger",))
    assert not no_review["included"] and "review" in no_review["reason"]


def test_instruction_from_task_card():
    assert instruction({"color": "yellow", "destination": "pink"}) == {
        "en": "Put the yellow block in the pink tray.", "zh": "把黃色方塊放進粉色盤"}
    assert instruction(None) == {"en": "", "zh": ""}
