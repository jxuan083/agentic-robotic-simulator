import pytest
import json
from pathlib import Path

from agentic_sim.perception.exploration import (TargetColorSearch, WristEstimateAccumulator,
    choose_existing_color, plan_wrist_sweep, score_target_confirmation)
from agentic_sim.config import load


def _row(detection_id, center, color="red", fit=0.9):
    return {"detection_id": detection_id, "color": color,
            "geometry": {"status": "estimated", "center_xyz_m": list(center), "fit_iou": fit}}


def test_accumulator_spatially_associates_and_reports_current_and_historical():
    acc = WristEstimateAccumulator(gate_m=0.02)
    acc.add_view("view-a", 1.0, [_row(0, [0, 0, 0], "red")])
    acc.add_view("view-b", 2.0, [_row(1, [0.005, 0, 0], "blue", 0.95),
                                  _row(2, [0.1, 0, 0])])
    acc.add_view("view-c", 3.0, [_row(3, [0.006, 0, 0], "yellow")])
    records = acc.records()
    assert len(records) == 2
    assert records[0]["color"] == "blue"  # highest fit, color is never an association key
    assert records[0]["exploration"]["source_views"] == ["view-a", "view-b", "view-c"]
    assert records[0]["exploration"]["currently_visible_in_latest_view"]
    assert not records[1]["exploration"]["currently_visible_in_latest_view"]
    assert records[1]["exploration"]["last_seen_view_id"] == "view-b"
    assert acc.summary()["historical_only_n"] == 1


def test_accumulator_rejects_duplicate_view_id_and_reset_clears_map():
    acc = WristEstimateAccumulator()
    acc.add_view("same", 1.0, [_row(0, [0, 0, 0])])
    with pytest.raises(ValueError, match="already been added"):
        acc.add_view("same", 2.0, [_row(1, [0, 0, 0])])
    acc.reset()
    assert acc.records() == []
    assert acc.summary()["latest_view_id"] is None


def test_sweep_is_bounded_and_derived_from_tray_geometry():
    scene = load("configs/scenes/lab_20261120.yaml")
    sweep = plan_wrist_sweep(scene)
    assert len(sweep.points_xyz_m) == 6
    assert all(p[2] == sweep.observation_z_m for p in sweep.points_xyz_m)
    assert sweep.transit_z_m > sweep.observation_z_m
    assert sweep.max_segment_m <= 0.01


def _saved_target_records():
    path = Path(__file__).parent / "fixtures" / "target_search_saved_records.json"
    payload = json.loads(path.read_text())
    return payload["target_color"], payload["records"]


def test_target_search_confirms_from_saved_live_record_contract():
    target, records = _saved_target_records()
    search = TargetColorSearch(target)
    first, second = records
    assert search.observe(first)["status"] == "searching"
    # A new frame at the same camera origin is not a distinct physical view.
    same_pose = json.loads(json.dumps(first))
    same_pose["frame_number"] += 100
    same_pose["captured_at_monotonic_s"] += 10
    assert search.observe(same_pose)["status"] == "searching"
    result = search.observe(second)
    assert result["status"] == "confirmed"
    confirmed = result["confirmed"]
    assert confirmed["target_color"] == target
    assert len(confirmed["vertices_xyz_m"]) == 8
    criteria = confirmed["confirmation_criteria"]
    assert criteria["minimum_camera_origin_separation_mm"] >= 10
    assert criteria["center_agreement_mm"] <= 5
    assert criteria["corner_agreement_rmse_mm_cube_symmetry"] <= 5


def test_target_search_requires_corner_agreement_and_can_exhaust_reset():
    target, records = _saved_target_records()
    search = TargetColorSearch(target)
    search.observe(records[0])
    altered = json.loads(json.dumps(records[1]))
    for vertex in altered["detections"][0]["geometry"]["vertices"]:
        vertex["xyz_m"][2] += 0.02
    assert search.observe(altered)["status"] == "searching"
    assert search.exhaust("all bounded views visited")["status"] == "exhausted"
    search.reset()
    assert search.result()["status"] == "searching"
    assert search.result()["target_color"] is None
    assert search.result()["observation_n"] == 0


def test_target_color_choice_is_repeatable_and_present_only():
    colors = ["blue", "purple", "blue"]
    chosen = choose_existing_color(colors, seed=42, episode=2)
    assert chosen == choose_existing_color(colors, seed=42, episode=2)
    assert chosen in {"blue", "purple"}


def test_target_confirmation_score_requires_correct_color_and_center_and_corners():
    target, records = _saved_target_records()
    row = records[1]["detections"][0]
    geometry = row["geometry"]
    corners = [v["xyz_m"] for v in geometry["vertices"]]
    gt = [{"center_xyz_m": geometry["center_xyz_m"], "corners_xyz_m": corners,
           "yaw_rad": geometry["yaw_rad_mod_pi_over_2"]}]
    confirmed = {"center_xyz_m": geometry["center_xyz_m"], "vertices_xyz_m": corners,
                 "yaw_rad_mod_pi_over_2": geometry["yaw_rad_mod_pi_over_2"]}
    score = score_target_confirmation(confirmed, target, gt, [target])
    assert score["correct_target_confirmation"]
    off_by_six_mm = json.loads(json.dumps(confirmed))
    off_by_six_mm["center_xyz_m"][0] += 0.006
    for xyz in off_by_six_mm["vertices_xyz_m"]:
        xyz[0] += 0.006
    bad_pose = score_target_confirmation(off_by_six_mm, target, gt, [target])
    assert not bad_pose["pose_within_5mm"]
    assert not bad_pose["correct_target_confirmation"]
    bad_color = score_target_confirmation(confirmed, target, gt, ["red"])
    assert bad_color["pose_within_5mm"] and not bad_color["color_correct"]
    assert not bad_color["correct_target_confirmation"]
