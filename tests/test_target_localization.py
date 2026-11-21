import numpy as np

from agentic_sim.config import load, v
from agentic_sim.kinematics import So101Chain
from agentic_sim.perception.classical import ClassicalBlockPerception, ClassicalConfig
from agentic_sim.perception.localize import global_candidates, plan_wrist_view, refine_with_wrist, target_pose
from agentic_sim.perception.vertices import CameraGeometry
from agentic_sim.teleop.ee_target import load_controller
from agentic_sim.wrist_lens import apply_wrist_lens

CFG = {"global_lookalikes": {"dark_green": ["light_green"]}, "max_candidates": 3, "association_gate_m": 0.03}
VERTS = [{"xyz_m": [0, 0, 0]}] * 8


def _det(i, color, center, status="estimated", fit=0.9, bbox=None, reason=None):
    g = {"status": status, "center_xyz_m": list(center), "yaw_rad_mod_pi_over_2": 0.1, "vertices": VERTS,
         "fit_iou": fit} if status == "estimated" else {"status": status, "reason": reason, "vertices": []}
    return {"detection_id": i, "color": color, "bbox_xyxy": bbox, "geometry": g}


def test_global_candidates_exact_colour_first_then_lookalikes_by_fit():
    record = {"detections": [_det(0, "light_green", [0.2, 0.2, 0.02], fit=0.99),
                             _det(1, "dark_green", [0.3, 0.2, 0.02], fit=0.90),
                             _det(2, "dark_green", [0.3, 0.3, 0.02], fit=0.95),
                             _det(3, "red", [0.1, 0.1, 0.02]),
                             _det(4, "dark_green", None, status="rejected", reason="image_boundary_clipped")]}
    cands = global_candidates(record, "dark_green", CFG)
    assert [c["global_detection_id"] for c in cands] == [2, 1, 0]
    assert [c["rank"] for c in cands] == [0, 0, 1]
    assert global_candidates(record, "blue", CFG) == []


def test_refine_confirms_rejects_wrong_colour_and_reports_not_seen():
    cand = {"center_xyz_m": [0.25, 0.20, 0.019], "global_color": "red"}
    ok = refine_with_wrist({"detections": [_det(0, "red", [0.251, 0.201, 0.019]),
                                           _det(1, "red", [0.40, 0.20, 0.019])]}, cand, "red", CFG)
    assert ok["status"] == "confirmed" and ok["wrist_detection_id"] == 0 and ok["offset_from_global_mm"] < 2
    wrong = refine_with_wrist({"detections": [_det(0, "orange", [0.252, 0.2, 0.019])]}, cand, "red", CFG)
    assert wrong["status"] == "wrong_color" and wrong["wrist_color"] == "orange"
    far = refine_with_wrist({"detections": [_det(0, "red", [0.30, 0.2, 0.019])]}, cand, "red", CFG)
    assert far["status"] == "not_seen"


def _wrist_record(dets, center):
    """Straight-down camera 0.12 m above `center`; the centre projects to the image middle."""
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [center[0], center[1], center[2] + 0.12]
    return {"camera": "wrist", "image_size": [640, 480], "intrinsics": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0], "mount_from_cv": pose.tolist(), "detections": dets}


def test_refine_colour_only_when_covering_silhouette_is_rejected():
    c = [0.25, 0.20, 0.019]
    cand = {"center_xyz_m": c, "global_color": "yellow", "vertices_xyz_m": [[0, 0, 0]] * 8,
            "yaw_rad_mod_pi_over_2": 0.3, "fit_iou": 0.97}
    covered = _wrist_record([_det(0, "yellow", None, status="rejected", reason="silhouette_fit_below_threshold",
                                  bbox=[250, 180, 400, 330])], c)
    verdict = refine_with_wrist(covered, cand, "yellow", CFG)
    assert verdict["status"] == "color_only"
    pose = target_pose(verdict, cand, covered, {"frame_number": 3})
    assert pose["source"]["pose"] == "global" and pose["center_xyz_m"] == c and pose["yaw_rad_mod_pi_over_2"] == 0.3
    # a rejected silhouette of another colour may be merged blocks: not proof of a wrong colour
    other = _wrist_record([_det(0, "purple", None, status="rejected", reason="silhouette_fit_below_threshold",
                                bbox=[250, 180, 400, 330])], c)
    assert refine_with_wrist(other, cand, "yellow", CFG)["status"] == "not_seen"
    elsewhere = _wrist_record([_det(0, "yellow", None, status="rejected", reason="image_boundary_clipped",
                                    bbox=[0, 0, 60, 60])], c)
    assert refine_with_wrist(elsewhere, cand, "yellow", CFG)["status"] == "not_seen"


def test_plan_wrist_view_centres_tray_points_in_the_wrist_image():
    scene = apply_wrist_lens(load("configs/scenes/lab_20261120.yaml"), None)
    chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
    ctrl = load_controller(chain, load("configs/teleop.yaml"), v(scene["robot"]["base_pos"]), scene)
    for xy in [(0.17, 0.18), (0.38, 0.13), (0.25, 0.30)]:
        target = np.array([xy[0], xy[1], 0.0186])
        plan = plan_wrist_view(chain, ctrl.p, scene, target, 0.10, ctrl.q_home)
        assert plan is not None and plan["center_offset"] < 0.2
        cam = CameraGeometry.from_scene(scene, "wrist", joints_rad=np.array(plan["q"]))
        u, v_ = cam.project(target[None, :])[0]
        assert abs(u - cam.width / 2) < 0.2 * cam.width / 2 and abs(v_ - cam.height / 2) < 0.2 * cam.height / 2
        assert 0.08 < plan["camera_height_m"] < 0.14


def test_split_by_hue_separates_touching_blocks_of_different_colours():
    rgb = np.full((240, 400, 3), 235, np.uint8)   # each square under max_area_fraction
    rgb[30:90, 40:100] = (220, 30, 30)     # red square
    rgb[30:90, 100:160] = (40, 60, 220)    # blue square touching it
    whole = ClassicalBlockPerception(ClassicalConfig()).predict(rgb)
    assert len(whole.detections) == 1
    split = ClassicalBlockPerception(ClassicalConfig(split_by_hue=True)).predict(rgb)
    assert sorted(d.color for d in split.detections) == ["blue", "red"]


def test_split_green_by_value_separates_light_and_dark_green():
    rgb = np.full((240, 400, 3), 235, np.uint8)   # each square under max_area_fraction
    rgb[30:90, 40:100] = (150, 230, 60)    # light green (value 230)
    rgb[30:90, 100:160] = (20, 110, 30)    # dark green (value 110), same hue band
    assert len(ClassicalBlockPerception(ClassicalConfig(split_by_hue=True)).predict(rgb).detections) == 1
    both = ClassicalBlockPerception(ClassicalConfig(split_by_hue=True, split_green_by_value=True)).predict(rgb)
    assert sorted(d.color for d in both.detections) == ["dark_green", "light_green"]
