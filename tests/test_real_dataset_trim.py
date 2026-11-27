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


def test_same_colour_targets_get_a_side_in_the_instruction():
    from agentic_sim.real_dataset import instruction, same_colour_side
    review = {"target": {"d455": {"color": "blue", "tray": "pink", "x_mm": 31.8, "y_mm": 108.0, "same_colour_in_tray": 2}},
              "d455_blocks": {"blocks": [{"color": "blue", "tray": "pink", "x_mm": 31.8, "y_mm": 108.0},
                                         {"color": "blue", "tray": "pink", "x_mm": 101.5, "y_mm": 50.9},
                                         {"color": "red", "tray": "pink", "x_mm": 60.0, "y_mm": 150.0}]}}
    side = same_colour_side(review)
    assert side == "near"                     # 70 mm nearer the arm outweighs 57 mm to the left (tray x ~ mount x)
    card = {"color": "blue", "destination": "beige"}
    assert instruction(card, side) == {"en": "Put the nearer blue block in the beige tray.",
                                       "zh": "把離手臂較近的藍色方塊放進米色盤"}
    review["d455_blocks"]["blocks"][1].update(x_mm=40.0, y_mm=40.0)
    assert same_colour_side(review) == "left"
    review["target"]["d455"]["same_colour_in_tray"] = 1
    assert same_colour_side(review) is None and instruction(card)["en"] == "Put the blue block in the beige tray."


def test_plush_condition_and_instruction():
    """11/28: the plush toy's condition folds its yaw by 2 (mount mm), its instruction names the toy."""
    import math

    from agentic_sim.real_dataset import condition, instruction
    review = {"target": {"d455": {"object": "plush", "x_mm": 300.0, "y_mm": -100.0, "yaw_deg": 120.0, "yaw_fold": 2}}}
    c = condition(review, "target")
    assert c[:2] == [300.0, -100.0]
    assert math.isclose(c[2], math.sin(math.radians(240))) and math.isclose(c[3], math.cos(math.radians(240)))
    assert instruction({"color": "plush", "destination": "beige"}) == {
        "en": "Put the Shiba plush in the beige tray.", "zh": "把柴犬玩偶放進米色盤"}
    assert instruction({"color": "red", "destination": "beige"})["en"] == "Put the red block in the beige tray."
