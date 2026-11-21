import math

import numpy as np

from tools.evaluate_perception import _cross_camera, _ground_truth_pose, _summary, summarize
from agentic_sim.perception.vertices import SIGNS
from agentic_sim.kinematics import quat_wxyz_to_mat


def _row(det_id, xyz, color="red"):
    return {"detection_id": det_id, "color": color,
            "geometry": {"status": "estimated", "center_xyz_m": list(xyz),
                         "yaw_rad_mod_pi_over_2": 0.0}}


def test_corner_ground_truth_uses_actual_centered_mesh_pose():
    edge = 0.0385
    center = np.array([0.2, -0.1, 0.021])
    yaw = math.radians(31)
    quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    pose = _ground_truth_pose(center, quat, edge)
    corners = np.asarray(pose["corners_xyz_m"])
    rot = quat_wxyz_to_mat(np.asarray(quat))
    expected = SIGNS.copy()
    expected[:, :2] *= edge / 2
    expected[:, 2] = (expected[:, 2] - .5) * edge
    assert np.allclose(corners, expected @ rot.T + center)
    assert np.allclose(corners[:4, 2], center[2] - edge / 2)
    assert np.allclose(corners[4:, 2], center[2] + edge / 2)


def test_cross_camera_matches_only_close_predicted_xyz_and_reports_color_separately():
    global_rows = [_row(0, (0, 0, 0), "red"), _row(1, (.08, 0, 0), "blue")]
    wrist_rows = [_row(4, (.001, 0, 0), "blue"), _row(5, (.081, 0, 0), "blue"),
                  _row(6, (.2, 0, 0), "red")]
    result = _cross_camera(global_rows, wrist_rows)
    assert result["matched_n"] == 2
    assert result["global_unpaired_n"] == 0
    assert result["wrist_unpaired_n"] == 1
    assert result["color_disagreement_n"] == 1
    assert max(pair["center_delta_mm"] for pair in result["pairs"]) < 5


def test_summary_handles_empty_and_episode_denominators():
    assert _summary([])["n"] == 0
    episode = {"settled_ground_truth": [{}],
               "cameras": {key: {"metrics": {"ground_truth_n": 1, "matched_n": 0, "missed_n": 1,
                            "false_positive_n": 0, "rejected_n": 0, "color_error_n_matched": 0},
                                 "record": {"evaluation": {"per_ground_truth": [{"status": "missed"}]}}}
                          for key in ("global", "wrist")},
               "cross_camera": {"matched_n": 0, "global_unpaired_n": 0, "wrist_unpaired_n": 0,
                                "color_disagreement_n": 0, "pairs": []}}
    result = summarize([episode], seed=7)
    assert result["episodes_n"] == 1
    assert result["cameras"]["global"]["ground_truth_n"] == 1
    assert result["cameras"]["global"]["center_le_5mm_over_all_gt"] == 0
