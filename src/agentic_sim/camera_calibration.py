"""Printed ChArUco and workspace-marker calibration from camera images."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

from agentic_sim.real_episode import write_json


MARKERS = (40, 41, 42)
SQUARES = (5, 7)


def measurements(layout, values, precision_mm):
    width, height, marker, id40, id41, id42, ruler1, ruler2 = values
    if not all(isfinite(value) and value > 0 for value in (*values, precision_mm)):
        raise ValueError("all printed measurements and ruler precision must be positive")
    if not layout.strip():
        raise ValueError("layout ID is required")
    if any(abs(length - 100) > precision_mm for length in (ruler1, ruler2)):
        raise ValueError("one of the 100 mm check lines is outside ruler precision; check print scaling")
    sx, sy = width / SQUARES[0], height / SQUARES[1]
    if abs(sx - sy) > precision_mm * (1 / SQUARES[0] + 1 / SQUARES[1]):
        raise ValueError("board has unequal horizontal and vertical scale; check print settings")
    square = (sx + sy) / 2
    if marker >= square:
        raise ValueError("ChArUco marker must fit within one square")
    return {
        "layout_id": layout, "dictionary": "DICT_4X4_50", "legacy_pattern": False,
        "board": {"squares": list(SQUARES), "square_m": square / 1000, "marker_m": marker / 1000,
                  "width_mm": width, "height_mm": height},
        "table_marker_size_m": {str(mark): size / 1000 for mark, size in zip(MARKERS, (id40, id41, id42))},
        "reference_lines_mm": [ruler1, ruler2], "ruler_precision_mm": precision_mm,
    }


def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def charuco_board(measured):
    spec = measured["board"]
    board = cv2.aruco.CharucoBoard(SQUARES, spec["square_m"], spec["marker_m"], dictionary())
    board.setLegacyPattern(False)
    return board


def read_png(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image {path}")
    return image


def save_png(path, image):
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError(f"cannot encode {path}")
    Path(path).write_bytes(data.tobytes())


def find_markers(image, detector):
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    result = {}
    for marker, quad in zip(ids.ravel(), corners):
        marker = int(marker)
        if marker in MARKERS:
            if marker in result:
                raise ValueError(f"duplicate workspace marker {marker}")
            result[marker] = quad.reshape(4, 2)
    return result


def coverage_regions(pixels, size):
    cols = np.minimum(2, np.maximum(0, (pixels[:, 0] * 3 / size[0]).astype(int)))
    rows = np.minimum(2, np.maximum(0, (pixels[:, 1] * 3 / size[1]).astype(int)))
    return {(int(row), int(col)) for row, col in zip(rows, cols)}


@dataclass
class View:
    path: str
    ids: np.ndarray
    points: np.ndarray
    pixels: np.ndarray
    sharpness: float
    board_span: tuple

    def signature(self, size):
        scale = np.array(size, float)
        homography, _ = cv2.findHomography(self.points[:, :2], self.pixels, 0)
        if homography is None:
            raise ValueError("could not compute board coverage")
        w, h = self.board_span
        quad = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), homography)[0] / scale
        return quad.flatten()


def detect_board(image, board, detector, path=""):
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) < 6 or board.checkCharucoCornersCollinear(ids):
        return None
    pixels = corners.reshape(-1, 2).astype(np.float32)
    points = board.getChessboardCorners()[ids.ravel()].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hull = cv2.convexHull(pixels.round().astype(np.int32))
    mask = cv2.fillConvexPoly(np.zeros_like(gray), hull, 1).astype(bool)
    gradient = cv2.Laplacian(gray, cv2.CV_32F)[mask]
    span = tuple(count * board.getSquareLength() for count in SQUARES)
    return View(path, ids.ravel(), points, pixels, float(gradient.var()) if gradient.size else 0.0, span)


def _fit(views, size, model, seed=None):
    points = [view.points for view in views]
    pixels = [view.pixels for view in views]
    if model == "opencv_brown":
        rms, k, d, _, _ = cv2.calibrateCamera(points, pixels, size, None, None)
    else:
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        if seed is not None:
            flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        rms, k, d, _, _ = cv2.fisheye.calibrate(
            [p.reshape(1, -1, 3).astype(np.float64) for p in points],
            [p.reshape(1, -1, 2).astype(np.float64) for p in pixels],
            size, seed.copy() if seed is not None else np.eye(3), np.zeros((4, 1)), flags=flags,
        )
    if not (isfinite(rms) and np.isfinite(k).all() and np.isfinite(d).all() and k[0, 0] > 0 and k[1, 1] > 0):
        raise ValueError(f"{model}: nonphysical intrinsics or nonfinite fit")
    return rms, k, d


def _rms(elements):
    return float(np.sqrt(np.mean(np.square(elements)))) if len(elements) else None


def _score(views, size, model, k, d):
    interior, edges, reports = [], [], []
    for view in views:
        points, pixels = view.points, view.pixels
        if model == "opencv_brown":
            ok, rv, tv = cv2.solvePnP(points, pixels, k, d)
            if not ok:
                raise ValueError(f"cannot estimate pose for {view.path}")
            projected, _ = cv2.projectPoints(points, rv, tv, k, d)
        else:
            undistorted = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), k, d)
            ok, rv, tv = cv2.solvePnP(points, undistorted, np.eye(3), None)
            if not ok:
                raise ValueError(f"cannot estimate fisheye pose for {view.path}")
            projected, _ = cv2.fisheye.projectPoints(points.reshape(1, -1, 3), rv, tv, k, d)
        error = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
        mask = np.max(np.abs(pixels * 2 / size - 1), axis=1) >= .67
        if not np.isfinite(error).all():
            raise ValueError(f"invalid reprojection for {view.path}")
        interior.extend(error.tolist())
        edges.extend(error[mask].tolist())
        reports.append({"image": view.path, "rms_px": _rms(error), "edge_rms_px": _rms(error[mask]),
                        "corners": len(error), "edge_corners": int(mask.sum())})
    return {"rms_px": _rms(interior), "edge_rms_px": _rms(edges), "edge_points": len(edges), "images": reports}


def fit_wrist(capture, measured, holdout_names=()):
    board = charuco_board(measured)
    detector = cv2.aruco.CharucoDetector(board)
    capture = Path(capture)
    views, errors = [], []
    size = None
    for path in sorted((capture / "frames").glob("*.png")):
        image = read_png(path)
        image_size = image.shape[1], image.shape[0]
        if size is not None and size != image_size:
            raise ValueError("wrist frames must use one native resolution")
        size = image_size
        view = detect_board(image, board, detector, path.name)
        if view is not None:
            views.append(view)
        else:
            errors.append(path.name)
    if len(views) < 10:
        raise ValueError(f"need at least ten usable views to fit and hold out frames; found {len(views)}")
    if holdout_names:
        wanted = set(holdout_names)
        holdout = [view for view in views if view.path in wanted]
        if wanted != {view.path for view in holdout}:
            raise ValueError("holdout contains an unknown or unusable frame")
    else:
        ordered = sorted(views, key=lambda view: tuple(view.signature(size)[:2]))
        count = min(len(views) - 8, max(2, len(views) // 4))
        holdout = [ordered[(i + 1) * len(views) // (count + 1)] for i in range(count)]
    held = {view.path for view in holdout}
    train = [view for view in views if view.path not in held]
    if len(holdout) < 2 or len(train) < 8:
        raise ValueError("need at least eight training and two held-out views")
    coverage = []
    for split in (train, holdout):
        covered = set().union(*(coverage_regions(view.pixels, size) for view in split))
        coverage.append({"seen": sorted(covered), "missing": sorted({(y, x) for y in range(3)
                                                                     for x in range(3)} - covered)})
    results = {}
    seed = None
    parameters = {}
    for model, coefficients in (("opencv_brown", ["k1", "k2", "p1", "p2", "k3"]),
                                ("opencv_fisheye", ["k1", "k2", "k3", "k4"])):
        try:
            rms, k, d = _fit(train, size, model, seed)
            scores = _score(holdout, np.asarray(size), model, k, d)
        except (cv2.error, ValueError) as error:
            results[model] = {"error": str(error)}
            continue
        if model == "opencv_brown":
            seed = k
        parameters[model] = (k, d)
        results[model] = {"camera_matrix_px": k.tolist(), "coefficients": d.ravel().tolist(),
                          "coefficient_order": coefficients, "training_rms_px": rms, "holdout": scores}
    report = {"camera": "wrist", "layout_id": measured["layout_id"], "image_size_px": list(size),
              "training_images": [view.path for view in train], "held_out_images": sorted(held),
              "unusable_images": errors, "training_grid": coverage[0], "holdout_grid": coverage[1],
              "grid_origin": "top-left; zero-based [row, column] in a 3x3 image grid", "models": results}
    if parameters:
        best = min(parameters, key=lambda model: results[model]["holdout"]["rms_px"])
        view = max(holdout, key=lambda item: len(coverage_regions(item.pixels, size)))
        raw = read_png(capture / "frames" / view.path)
        k, d = parameters[best]
        corrected = (cv2.undistort(raw, k, d) if best == "opencv_brown" else
                     cv2.fisheye.undistortImage(raw, k, d, Knew=k))
        save_png(capture / "undistorted_example.png", np.concatenate([raw, corrected], axis=1))
        report["example"] = {"model": best, "image": view.path, "output": "undistorted_example.png"}
    write_json(capture / "intrinsics.json", report)
    if len(parameters) != len(results):
        raise RuntimeError(f"one or more fits failed; see {capture / 'intrinsics.json'} and add tilted views")
    return report


def _rs_intrinsics(metadata):
    import pyrealsense2 as rs

    result = rs.intrinsics()
    result.width, result.height = metadata["width"], metadata["height"]
    result.fx, result.fy = metadata["fx"], metadata["fy"]
    result.ppx, result.ppy = metadata["cx"], metadata["cy"]
    model = metadata["distortion"].split(".")[-1]
    if model not in {"none", "brown_conrady", "inverse_brown_conrady", "kannala_brandt4", "ftheta"}:
        raise ValueError(f"unsupported RealSense deprojection model: {model}")
    result.model = getattr(rs.distortion, model)
    result.coeffs = metadata["coefficients"]
    return result


def marker_poses(markers, meta, sizes):
    """Return RGB-from-tag poses. Tag x is right, y is up, z faces out."""
    import pyrealsense2 as rs

    intr = _rs_intrinsics(meta)
    focal = np.array([intr.fx, intr.fy])
    result = {}
    for marker, pixels in markers.items():
        half = sizes[str(marker)] / 2
        points = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
        rays = np.array([rs.rs2_deproject_pixel_to_point(intr, pixel.tolist(), 1.0)[:2] for pixel in pixels])
        count, rotations, translations, _ = cv2.solvePnPGeneric(
            points, rays, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        candidates = []
        for rv, tv in zip(rotations, translations):
            rotation, _ = cv2.Rodrigues(rv)
            points_rgb = points @ rotation.T + tv.ravel()
            if np.any(points_rgb[:, 2] <= 0):
                continue
            residual = (points_rgb[:, :2] / points_rgb[:, 2:] - rays) * focal
            rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
            if np.isfinite(rms):
                candidates.append((rms, rotation, tv.ravel()))
        if not count or not candidates:
            raise ValueError(f"marker {marker}: no finite positive-depth pose")
        candidates.sort(key=lambda item: item[0])
        score, rotation, translation = candidates[0]
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, translation
        result[str(marker)] = {"rgb_from_marker_m": matrix.tolist(), "corners_px": pixels.tolist(),
                               "ray_equivalent_rms_px": score,
                               "alternate_rms_px": candidates[1][0] if len(candidates) > 1 else None}
    return result


def _transform(data, key):
    matrix = np.asarray(data[key], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3)):
        raise ValueError(f"{key} needs a rigid homogeneous 4x4 transform")
    return matrix


def workspace_geometry(rgb, meta, measured, *, links=None):
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary(), detector_params)
    corners = find_markers(rgb, detector)
    if set(corners) != set(MARKERS):
        raise ValueError(f"global frame missing markers {sorted(set(MARKERS) - corners.keys())}")
    poses = marker_poses(corners, meta["color_intrinsics"], measured["table_marker_size_m"])
    anchor = np.asarray(poses["40"]["rgb_from_marker_m"])
    relative = {marker: (np.linalg.inv(anchor) @ np.asarray(poses[marker]["rgb_from_marker_m"])).tolist()
                for marker in ("41", "42")}
    result = {"layout_id": measured["layout_id"], "markers": poses, "marker40_from_marker_m": relative,
              "frame_convention": "destination_from_source, right-handed, translation m",
              "base_from_rgb_m": None, "marker40_from_tray_m": {}}
    if links and "base_from_marker40_m" in links:
        result["base_from_rgb_m"] = (_transform(links, "base_from_marker40_m") @ np.linalg.inv(anchor)).tolist()
    for tray, mark in (("pink", "41"), ("beige", "42")):
        key = f"marker{mark}_from_tray_m"
        if links and key in links:
            result["marker40_from_tray_m"][tray] = (np.asarray(relative[mark]) @ _transform(links, key)).tolist()
    return result


def compare_workspace(reference, current):
    if reference["layout_id"] != current["layout_id"]:
        raise ValueError("compare captures made against the same layout ID")
    if reference.get("measurements") != current.get("measurements"):
        raise ValueError("compare captures measured using the same printed targets")
    if reference.get("camera", {}).get("device") != current.get("camera", {}).get("device"):
        raise ValueError("compare captures from the same camera serial")
    if reference.get("camera", {}).get("color_intrinsics") != current.get("camera", {}).get("color_intrinsics"):
        raise ValueError("compare captures with the same native RGB mode and intrinsics")
    result = {}
    for marker in ("41", "42"):
        old = np.asarray(reference["marker40_from_marker_m"][marker])
        new = np.asarray(current["marker40_from_marker_m"][marker])
        rotation = old[:3, :3].T @ new[:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        shift = (new[:3, 3] - old[:3, 3]) * 1000
        result[marker] = {"offset_in_marker40_axes_mm": shift.tolist(), "rotation_delta_degrees": float(angle)}
    old = np.asarray(reference["markers"]["40"]["corners_px"])
    new = np.asarray(current["markers"]["40"]["corners_px"])
    return {"anchor_corner_shift_rms_px": float(np.sqrt(np.mean(np.sum((new - old)**2, axis=1)))),
            "trays": result}


def depth_holdout(raw, camera_meta, geometry, points):
    """Check independent native-depth picks against surveyed right-base coordinates."""
    import pyrealsense2 as rs

    if geometry["base_from_rgb_m"] is None:
        raise ValueError("independent depth validation needs a measured base-from-marker40 transform")
    depth_meta = camera_meta["depth"]
    intr = _rs_intrinsics(depth_meta["intrinsics"])
    if raw.shape != (intr.height, intr.width) or raw.dtype != np.uint16:
        raise ValueError("expected native uint16 depth at the archived resolution")
    extrinsics = depth_meta["depth_to_color"]
    rotate = np.asarray(extrinsics["rotation_column_major"]).reshape(3, 3, order="F")
    translate = np.asarray(extrinsics["translation_m"])
    transform = np.asarray(geometry["base_from_rgb_m"])
    results = []
    used = set()
    for point in points:
        key = point["tray"], point["region"]
        if key in used or key[0] not in {"pink", "beige"} or key[1] not in {"near", "far"}:
            raise ValueError("expected separate pink/beige near/far holdouts")
        used.add(key)
        x, y = point["depth_pixel"]
        expected = np.asarray(point["base_xyz_mm"], dtype=float)
        if expected.shape != (3,) or not np.isfinite(expected).all():
            raise ValueError("holdout needs three independent base coordinates in mm")
        if type(x) is not int or type(y) is not int or not (0 <= x < intr.width and 0 <= y < intr.height):
            raise ValueError("holdout pixel must be a native-depth integer pixel")
        window = raw[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        valid = window[window > 0]
        outcome = dict(point, valid_depth_pixels=int(valid.size))
        if valid.size >= 3:
            distance = float(np.median(valid)) * depth_meta["units_m"]
            optical = rs.rs2_deproject_pixel_to_point(intr, [x, y], distance)
            base_mm = (transform @ np.r_[rotate @ optical + translate, 1])[:3] * 1000
            delta = base_mm - expected
            outcome.update(base_estimate_mm=base_mm.tolist(), xy_error_mm=float(np.linalg.norm(delta[:2])),
                           z_error_mm=float(delta[2]), recorded_depth_m=distance)
        else:
            outcome["error"] = "insufficient valid native depth near marked pixel"
        results.append(outcome)
    return results
