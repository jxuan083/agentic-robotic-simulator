"""Converter clean-up of the 11/27 collection: still stretches trimmed, corrected misses (re-grasps) flagged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_sim.real_dataset import count_closes, trim_still  # noqa: E402


def sample(i, q, g):
    return {"row": i, "action": [q, 0.0, 0.0, 0.0, 0.0, g]}


def test_a_still_stretch_keeps_its_first_third_of_a_second():
    dt = 1 / 30
    s = [sample(i, i * 1.0, 30.0) for i in range(10)]            # moving 30 deg/s
    s += [sample(10 + i, 9.0, 30.0) for i in range(30)]          # 1 s still
    s += [sample(40 + i, 9.0 + i, 30.0) for i in range(5)]       # moving again from row 41
    out, cut = trim_still(s, 0.3, dt)
    # 31 still steps (rows 10-40 repeat row 9's command); the first 9 (0.3 s) stay
    assert cut == 22 and len(out) == len(s) - cut
    assert [x["row"] for x in out][9:] == [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 41, 42, 43, 44]
    assert trim_still(s, 0, dt) == (s, 0)


def test_closing_twice_is_a_regrasp():
    once = [26] * 10 + [20, 15, 14] + [14] * 10
    again = once + [20, 26, 26] + [18, 14, 14]
    assert count_closes(once) == 1 and count_closes(again) == 2
    assert count_closes([26, 25.5, 25, 26] * 5) == 0             # jitter is not a close
