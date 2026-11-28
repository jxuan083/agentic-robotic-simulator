"""Check printed scale, marker coordinates and the native-depth transform."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from agentic_sim.camera_calibration import (  # noqa: E402
    charuco_board, compare_workspace, depth_holdout, detect_board, measurements, workspace_geometry,
)


@pytest.fixture
def print_sizes():
    return measurements("camera-test", [160, 224, 24, 42, 41.8, 42.2, 100, 100], 1)


def test_printed_scale_and_nonlegacy_board(print_sizes):
    assert print_sizes["table_marker_size_m"]["41"] == pytest.approx(.0418)
    assert print_sizes["board"]["square_m"] == pytest.approx(.032)
    board = charuco_board(print_sizes)
    image = board.generateImage((640, 896), marginSize=16)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    view = detect_board(color, board, cv2.aruco.CharucoDetector(board))
    assert sorted(view.ids.tolist()) == list(range(24))
    with pytest.raises(ValueError, match="check lines"):
        measurements("bad-print", [160, 224, 24, 42, 42, 42, 96, 100], 1)
    with pytest.raises(ValueError, match="unequal"):
        measurements("stretched-print", [160, 221, 24, 42, 42, 42, 100, 100], 1)


def test_relative_tray_offsets_and_native_depth_to_base(print_sizes):
    pytest.importorskip("pyrealsense2")
    intr = {"width": 640, "height": 480, "fx": 420., "fy": 420., "cx": 320., "cy": 240.,
            "distortion": "distortion.none", "coefficients": [0.] * 5}
    intrinsics = {"color_intrinsics": intr, "depth": {"units_m": .001, "intrinsics": intr,
                  "depth_to_color": {"rotation_column_major": [0., 1., 0., -1., 0., 0., 0., 0., 1.],
                                     "translation_m": [.01, 0., 0.]}}}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def scene(shift):
        raw = np.full((480, 640), 246, np.uint8)
        origin = np.float32([[0, 0], [167, 0], [167, 167], [0, 167]])
        for marker, (x, y) in ((40, (180, 160)), (41, (300 + shift, 245)), (42, (445, 345))):
            code = cv2.aruco.generateImageMarker(dictionary, marker, 168)
            target = np.float32([[x, y], [x + 32, y + 1], [x + 31, y + 33], [x - 1, y + 32]])
            matrix = cv2.getPerspectiveTransform(origin, target)
            raw = np.minimum(raw, cv2.warpPerspective(code, matrix, (640, 480), borderValue=255))
        return raw

    fixed = workspace_geometry(scene(0), intrinsics, print_sizes)
    moved = workspace_geometry(scene(12), intrinsics, print_sizes)
    delta = compare_workspace(fixed, moved)
    assert fixed["base_from_rgb_m"] is None
    assert delta["anchor_corner_shift_rms_px"] == 0
    assert delta["trays"]["41"]["offset_in_marker40_axes_mm"][0] > 10
    assert delta["trays"]["42"]["offset_in_marker40_axes_mm"] == pytest.approx([0, 0, 0])

    raw_depth = np.full((480, 640), 550, np.uint16)
    points = [{"tray": "pink", "region": "near", "depth_pixel": [362, 240], "base_xyz_mm": [10, 55, 550]}]
    with pytest.raises(ValueError, match="base-from-marker40"):
        depth_holdout(raw_depth, intrinsics, fixed, points)
    pose = fixed | {"base_from_rgb_m": np.eye(4).tolist()}
    sample, = depth_holdout(raw_depth, intrinsics, pose, points)
    assert sample["xy_error_mm"] < .001
    assert abs(sample["z_error_mm"]) < .001
