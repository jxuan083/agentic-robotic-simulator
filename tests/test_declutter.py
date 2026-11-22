import numpy as np

from agentic_sim.task.declutter import free_spot, in_tray_xy, item_key, mount_conflicts, plan, push_path

E = 0.0295
CFG = {"close_gap_m": 0.015, "hull_margin_m": 0.0, "support_gap_m": 0.005, "max_failures_per_item": 2,
       "finger_room_m": 0.018, "edge_m": E, "global_lookalikes": {"dark_green": ["light_green"]}}
TRAY = {"center": [0.25, 0.20], "yaw": 0.0, "inner_half": [0.10, 0.12], "floor_top": 0.004}


def _item(color, xy, status="upright", layers=1.0, size=E):
    h = size / 2
    hull = [[xy[0] - h, xy[1] - h], [xy[0] + h, xy[1] - h], [xy[0] + h, xy[1] + h], [xy[0] - h, xy[1] + h]]
    return {"segment": 0, "color": color, "status": status, "xy": list(xy), "top_z_m": 0.004 + layers * E,
            "layers": layers, "tilt_deg": 20.0 if status == "tilted" else 0.5, "yaw_rad": 0.0, "hull": hull}


def test_upright_target_is_looked_at_first_then_localized_when_clear():
    items = [_item("red", (0.20, 0.20)), _item("blue", (0.30, 0.20))]
    d = plan(items, "red", CFG)
    assert d["action"] == "localize" and d["target"] == 0 and "colour" in d["reason"]
    d = plan(items, "red", CFG, confirmed={item_key(items[0])})
    assert d["action"] == "localize" and d["reason"] == "target confirmed and clear"


def test_crowded_confirmed_target_is_relocated_or_its_neighbour_moved():
    roomy = [_item("red", (0.20, 0.20)), _item("blue", (0.20 + E + 0.008, 0.20))]   # room on the other face pair
    d = plan(roomy, "red", CFG, confirmed={item_key(roomy[0])})
    assert d["action"] == "move" and d["item"] == 0 and "relocate" in d["reason"]
    boxed = roomy + [_item("cyan", (0.20, 0.20 + E + 0.008)), _item("cyan", (0.20 - E - 0.008, 0.20)),
                     _item("cyan", (0.20, 0.20 - E - 0.008))]
    d = plan(boxed, "red", CFG, confirmed={item_key(boxed[0])})
    assert d["action"] == "move" and d["item"] != 0 and d["target"] == 0


def test_unconfirmed_crowded_target_gets_cleared_then_looked_at_again():
    items = [_item("red", (0.20, 0.20)), _item("blue", (0.20 + E + 0.005, 0.20))]
    d = plan(items, "red", CFG, unconfirmed_crowded={item_key(items[0])})
    assert d["action"] == "move" and d["item"] == 1
    d = plan(items[:1], "red", CFG, unconfirmed_crowded={item_key(items[0])})
    assert d["action"] == "localize" and d["reason"] == "cleared: look again"


def test_stacked_and_tilted_targets():
    stacked = [_item("red", (0.20, 0.20), "stacked", 2.0)]
    assert plan(stacked, "red", CFG)["item"] == 0
    tilted = [_item("red", (0.20, 0.20), "tilted"), _item("blue", (0.20 + E + 0.002, 0.20))]
    d = plan(tilted, "red", CFG)
    assert d["action"] == "move" and d["item"] == 1 and "leans" in d["reason"]
    under = [_item("red", (0.20, 0.20), "partial", size=E * 0.6), _item("blue", (0.20 + 0.012, 0.20), "stacked", 2.0)]
    assert plan(under, "red", CFG)["item"] == 1


def test_hidden_target_moves_suspects_then_reports_not_found():
    items = [_item("blue", (0.30, 0.20)), _item("cyan", (0.20, 0.25), "stacked", 2.0),
             _item("orange", (0.22, 0.15), "tilted")]
    d = plan(items, "red", CFG)
    assert d["action"] == "move" and d["item"] == 1 and d["target"] is None      # stacked before tilted
    failed = {item_key(items[1]): 2, item_key(items[2]): 2}
    assert plan(items, "red", CFG, failed=failed)["action"] == "not_found"


def test_lookalike_after_exact_colour_and_excluded_skipped():
    items = [_item("light_green", (0.18, 0.20)), _item("dark_green", (0.30, 0.25))]
    assert plan(items, "dark_green", CFG)["target"] == 1
    assert plan(items, "dark_green", CFG, excluded={item_key(items[1])})["target"] == 0


def test_free_spot_keeps_clear_and_reachable_and_push_stays_in_tray():
    items = [_item("red", (0.20, 0.20)), _item("blue", (0.24, 0.20))]
    reach = lambda p, yaw: np.hypot(p[0], p[1]) < 0.33
    spot = free_spot(items, TRAY, E, reach, (0.0, 0.0), avoid_xy=[(0.20, 0.20)])
    assert spot is not None and np.hypot(*spot) < 0.33 - 0.015
    sq = np.array(spot)
    for it in items:
        assert np.min(np.abs(sq - np.asarray(it["xy"]))) > E or np.max(np.abs(sq - np.asarray(it["xy"]))) > E + 0.025
    assert free_spot(items, TRAY, E, lambda p, yaw: False) is None
    edge_items = [_item("red", (0.25 + 0.10 - E, 0.20)), _item("blue", (0.25 + 0.10 - 2.2 * E, 0.20))]
    start, end = push_path(edge_items, 0, TRAY, E, target=1)
    assert abs(end[0] - 0.25) <= 0.10 - E / 2 + 1e-9                            # turned away from the tray wall


def test_free_spot_inset_keeps_away_from_the_walls():
    reach = lambda p, yaw: True
    spot = free_spot([], TRAY, E, reach, (0.0, 0.0), inset_m=0.02)
    assert abs(spot[0] - 0.25) <= 0.10 - E / 2 - 0.02 + 1e-9 and abs(spot[1] - 0.20) <= 0.12 - E / 2 - 0.02 + 1e-9


def test_mount_conflicts_low_points_over_other_blocks_only():
    items = [_item("red", (0.20, 0.20)), _item("blue", (0.25, 0.20)), _item("cyan", (0.20, 0.30))]
    low_over_blue = [[0.25, 0.20, 0.02], [0.20, 0.20, 0.02]]        # also over the target: ignored
    assert mount_conflicts(items, 0, low_over_blue) == [1]
    assert mount_conflicts(items, 0, [[0.25, 0.20, 0.004 + E + 0.01]]) == []   # above the block top
    assert mount_conflicts(items, 0, [[0.20, 0.25, 0.01]]) == []                 # between blocks


def test_in_tray_xy():
    assert in_tray_xy((0.25, 0.20), TRAY) and in_tray_xy((0.355, 0.20), TRAY)
    assert not in_tray_xy((0.40, 0.20), TRAY)
