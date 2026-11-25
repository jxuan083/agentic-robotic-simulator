"""D455 block localisation: tray-local axes and scale on a synthetic top view."""

from pathlib import Path
import sys

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_sim.real_locate import locate, match_target  # noqa: E402

PINK, BLUE, RED, TABLE = (190, 170, 245), (200, 140, 20), (30, 30, 200), (200, 200, 196)


def scene(blue_center=(335, 367), red_center=(300, 440)):
    image = np.full((480, 640, 3), TABLE, np.uint8)
    cv2.rectangle(image, (260, 270), (410, 465), PINK, -1)  # 150 px wide (y), 195 px tall (x)
    for color, (cx, cy) in ((BLUE, blue_center), (RED, red_center)):
        cv2.rectangle(image, (cx - 10, cy - 10), (cx + 10, cy + 10), color, -1)
    return image


def test_tray_local_axes_and_scale():
    result = locate(scene())
    assert result["warnings"] == ["beige tray not found"]
    blocks = {b["color"]: b for b in result["blocks"]}
    # +x up the image, +y to the left, origin at the bottom-right inner corner (outer 305 x 230 mm, 5 mm wall).
    assert blocks["blue"]["tray"] == "pink"
    assert blocks["blue"]["x_mm"] == pytest.approx((465 - 367) / 195 * 305 - 5, abs=4)
    assert blocks["blue"]["y_mm"] == pytest.approx((410 - 335) / 150 * 230 - 5, abs=4)
    assert blocks["red"]["x_mm"] < blocks["blue"]["x_mm"] and blocks["red"]["y_mm"] > blocks["blue"]["y_mm"]
    assert blocks["blue"]["yaw_deg"] == pytest.approx(0, abs=3) or blocks["blue"]["yaw_deg"] == pytest.approx(90, abs=3)


def test_match_target_picks_card_colour_and_tray():
    result = locate(scene())
    card = {"color": "blue", "source_tray": "pink", "pose": [140, 100, 0]}
    target = match_target(result, card)
    assert target["color"] == "blue" and target["card_offset_mm"] < 15
    assert match_target(result, card | {"color": "yellow"}) is None


def test_clipped_tray_is_flagged():
    image = np.full((480, 640, 3), TABLE, np.uint8)
    cv2.rectangle(image, (500, 260), (639, 470), PINK, -1)
    result = locate(image)
    assert result["trays"]["pink"]["clipped"] and result["warnings"]


def test_fixed_quads_ignore_a_hand_over_the_tray():
    from agentic_sim.real_locate import reference_quads

    clean = scene()
    quads = reference_quads(clean, require=("pink",))
    busy = clean.copy()
    cv2.rectangle(busy, (250, 280), (330, 330), (120, 150, 190), -1)  # skin-coloured hand over the far-left corner
    moved = locate(busy)
    fixed = locate(busy, quads)
    blue_clean = next(b for b in locate(clean)["blocks"] if b["color"] == "blue")
    blue_fixed = next(b for b in fixed["blocks"] if b["color"] == "blue")
    assert abs(blue_fixed["x_mm"] - blue_clean["x_mm"]) < 1 and abs(blue_fixed["y_mm"] - blue_clean["y_mm"]) < 1
    assert fixed["tray_source"] == "fixed reference quads" and moved["tray_source"] == "detected in this frame"
