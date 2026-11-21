"""Bounded wrist-camera view planning and prediction-only multi-view accumulation.

This module consumes configured tray geometry and image-derived estimates only. It
never receives simulator block poses, spawn locations, or scene object handles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from agentic_sim.config import v

BLOCK_PALETTE = ("red", "orange", "yellow", "light_green", "dark_green", "blue", "cyan", "purple")


def choose_existing_color(colors, seed: int | None, episode: int) -> str:
    """Choose one palette color from the actual color-name list, reproducibly.

    The caller passes only the scene's color list, never object positions or IDs.
    A separate SeedSequence stream keeps this choice independent of layout draws.
    """
    present = sorted({str(c) for c in colors})
    if not present:
        raise ValueError("cannot select a target from an empty color list")
    invalid = sorted(set(present) - set(BLOCK_PALETTE))
    if invalid:
        raise ValueError(f"unsupported target color(s): {invalid}")
    root_seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, int(episode), 0x54415247]))
    return present[int(rng.integers(len(present)))]


class TargetColorSearch:
    """Image-only confirmation state for one requested palette color.

    Confirmations require the same classified color, an accepted geometric fit,
    predicted centers within 5 mm, and camera origins at least 10 mm apart.
    """

    def __init__(self, target_color: str):
        if target_color not in BLOCK_PALETTE:
            raise ValueError(f"target_color must be one of {BLOCK_PALETTE}")
        self.target_color = target_color
        self._clear_state()

    def _clear_state(self) -> None:
        self._observations = []
        self._confirmed = None
        self._status = "searching"
        self._reason = None
        self._capture_keys = set()

    def reset(self) -> None:
        """Invalidate the current target and discard all search history."""
        self.target_color = None
        self._clear_state()

    @staticmethod
    def _origin(record):
        pose = np.asarray(record.get("mount_from_cv"), dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return None
        return pose[:3, 3]

    def observe(self, record: dict) -> dict:
        if self._status != "searching":
            return self.result()
        origin = self._origin(record)
        if origin is None:
            return self.result()
        captured = record.get("captured_at_monotonic_s")
        frame = record.get("frame_number")
        capture_key = (record.get("camera"), frame, captured)
        if capture_key in self._capture_keys:
            return self.result()
        self._capture_keys.add(capture_key)
        view_obs = []
        for detection in record.get("detections", []):
            geometry = detection.get("geometry", {})
            if detection.get("color") != self.target_color or geometry.get("status") != "estimated":
                continue
            center = np.asarray(geometry.get("center_xyz_m"), dtype=float)
            raw_vertices = geometry.get("vertices", [])
            vertices = np.asarray([vtx.get("xyz_m") for vtx in raw_vertices], dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                continue
            if vertices.shape != (8, 3) or not np.isfinite(vertices).all():
                continue
            item = {"frame_number": frame, "captured_at_monotonic_s": captured,
                    "camera": record.get("camera"), "mount_from_cv": np.asarray(record["mount_from_cv"]).tolist(),
                    "origin_xyz_m": origin.tolist(),
                    "detection_id": detection.get("detection_id"), "color": detection["color"],
                    "center_xyz_m": center.tolist(), "vertices_xyz_m": vertices.tolist(),
                    "yaw_rad_mod_pi_over_2": geometry.get("yaw_rad_mod_pi_over_2"),
                    "fit_iou": geometry.get("fit_iou"), "geometry": geometry, "detection": detection}
            view_obs.append(item)
            for previous in self._observations:
                center_delta = float(np.linalg.norm(center - np.asarray(previous["center_xyz_m"])))
                corner_delta = self._corner_agreement(vertices, np.asarray(previous["vertices_xyz_m"], float))
                origin_delta = float(np.linalg.norm(origin - np.asarray(previous["origin_xyz_m"])))
                if center_delta <= 0.005 and corner_delta <= 0.005 and origin_delta >= 0.01:
                    self._observations.append(item)
                    selected = item if float(item.get("fit_iou") or 0) >= float(previous.get("fit_iou") or 0) else previous
                    self._confirmed = {"target_color": self.target_color, "status": "confirmed",
                        "center_xyz_m": selected["center_xyz_m"], "vertices_xyz_m": selected["vertices_xyz_m"],
                        "yaw_rad_mod_pi_over_2": selected["yaw_rad_mod_pi_over_2"],
                        "confirmation_criteria": {"same_color": self.target_color,
                            "geometry_status": "estimated", "center_agreement_mm": center_delta * 1000,
                            "corner_agreement_rmse_mm_cube_symmetry": corner_delta * 1000,
                            "minimum_camera_origin_separation_mm": origin_delta * 1000,
                            "distinct_physical_view_threshold_mm": 10.0, "maximum_center_disagreement_mm": 5.0,
                            "maximum_corner_agreement_rmse_mm": 5.0},
                        "observations": [self._public_observation(previous), self._public_observation(item)],
                        "selected_observation": self._public_observation(selected)}
                    self._status = "confirmed"
                    return self.result()
        # Store only view observations; same-pose repeats cannot confirm, but may
        # later be compared with a genuinely displaced camera origin.
        self._observations.extend(view_obs)
        return self.result()

    @staticmethod
    def _corner_agreement(a: np.ndarray, b: np.ndarray) -> float:
        """Eight-corner RMSE with square-index quarter-turn symmetry."""
        errors = []
        for k in range(4):
            permutation = [(i + k) % 4 for i in range(4)] + [4 + (i + k) % 4 for i in range(4)]
            errors.append(float(np.sqrt(np.mean(np.sum((a - b[permutation]) ** 2, axis=1)))))
        return min(errors)

    @staticmethod
    def _public_observation(item):
        return {k: item[k] for k in ("frame_number", "captured_at_monotonic_s", "camera",
                                      "mount_from_cv", "detection_id", "color", "center_xyz_m",
                                      "vertices_xyz_m", "yaw_rad_mod_pi_over_2", "fit_iou")}

    def exhaust(self, reason: str) -> dict:
        if self._status == "searching":
            self._status = "unreachable" if str(reason).startswith("unreachable:") else "exhausted"
            self._reason = str(reason)
        return self.result()

    def result(self) -> dict:
        origins = []
        for item in self._observations:
            origin = np.asarray(item["origin_xyz_m"], float)
            if all(np.linalg.norm(origin - known) >= 0.01 for known in origins):
                origins.append(origin)
        return {"target_color": self.target_color, "status": self._status,
                "reason": self._reason, "observation_n": len(self._observations),
                "distinct_candidate_view_n": len(origins),
                "confirmed": self._confirmed}


def score_target_confirmation(confirmed: dict | None, target_color: str,
                              ground_truth: list[dict], ground_truth_colors: list[str],
                              gate_m: float = 0.10) -> dict:
    """Post-run evaluator only: assess the frozen prediction against settled GT."""
    result = {"ground_truth_n": len(ground_truth), "target_color_exists": target_color in ground_truth_colors,
              "confirmed": confirmed is not None, "matched_gt_index": None,
              "matched_gt_color": None, "center_error_mm": None, "corner_rmse_mm": None,
              "yaw_error_deg_mod90": None, "color_correct": False, "pose_within_5mm": False,
              "correct_target_confirmation": False}
    if confirmed is None or not ground_truth:
        return result
    center = np.asarray(confirmed["center_xyz_m"], float)
    distances = [float(np.linalg.norm(center - np.asarray(row["center_xyz_m"], float))) for row in ground_truth]
    index = int(np.argmin(distances))
    if distances[index] > gate_m:
        return result
    truth = ground_truth[index]
    predicted_vertices = np.asarray(confirmed["vertices_xyz_m"], float)
    truth_vertices = np.asarray(truth["corners_xyz_m"], float)
    corner_rmse_m = TargetColorSearch._corner_agreement(predicted_vertices, truth_vertices)
    center_error_mm, corner_rmse_mm = distances[index] * 1000, corner_rmse_m * 1000
    color_correct = ground_truth_colors[index] == target_color
    pose_within = bool(np.isfinite(center_error_mm) and np.isfinite(corner_rmse_mm)
                       and center_error_mm <= 5 and corner_rmse_mm <= 5)
    period = math.pi / 2
    yaw_error = abs((float(confirmed["yaw_rad_mod_pi_over_2"]) - float(truth["yaw_rad"]) + period/2)
                    % period - period/2)
    result.update({"matched_gt_index": index, "matched_gt_color": ground_truth_colors[index],
                   "center_error_mm": center_error_mm, "corner_rmse_mm": corner_rmse_mm,
                   "yaw_error_deg_mod90": math.degrees(yaw_error), "color_correct": color_correct,
                   "pose_within_5mm": pose_within,
                   "correct_target_confirmation": color_correct and pose_within})
    return result


@dataclass(frozen=True)
class WristSweep:
    points_xyz_m: tuple[tuple[float, float, float], ...]
    transit_z_m: float = 0.24
    observation_z_m: float = 0.18
    max_segment_m: float = 0.01


def plan_wrist_sweep(scene: dict) -> WristSweep:
    """Return a small snake sweep centered from configured tray geometry.

    The fixed XY offset is a robot/camera mounting and reach calibration, not a
    block location. The sweep covers a bounded central portion of the tray; it
    is intentionally conservative and makes no collision-free claim.
    """
    tray = scene["trays"]["storage_pink"]
    center = np.asarray(v(tray["center_xy"]), float)
    size = np.asarray(v(tray["outer_size"]), float)
    yaw = math.radians(float(v(tray.get("yaw_deg", 0.0))))
    # Reachable camera-to-tray mapping for the current SO-101 mount.
    ee_to_tray = np.array([0.134, 0.054], float)
    origin = center - ee_to_tray
    dx = float(size[0]) * 0.25
    dy = float(size[1]) * 0.20
    local = [(-dx, -dy), (0.0, -dy), (dx, -dy),
             (dx, dy), (0.0, dy), (-dx, dy)]
    c, s = math.cos(yaw), math.sin(yaw)
    points = []
    for x, y in local:
        # Tray-relative offsets are rotated by the configured tray yaw.
        xy = origin + np.array([c*x - s*y, s*x + c*y])
        points.append((float(xy[0]), float(xy[1]), 0.18))
    return WristSweep(tuple(points))


def plan_target_search_views(scene: dict) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Bounded base sweep plus four modest higher viewpoints; uses no object data."""
    sweep = plan_wrist_sweep(scene)
    base = [(f"sweep_{i:02d}", point) for i, point in enumerate(sweep.points_xyz_m)]
    # A 20 mm vertical camera displacement changes view geometry while remaining
    # above the previous low-sweep zone. These alternatives are attempted only
    # if the six primary views did not confirm the requested color.
    extra = [(f"elevated_{j:02d}", (p[0], p[1], 0.20))
             for j, p in enumerate((sweep.points_xyz_m[i] for i in (0, 2, 3, 5)))]
    return tuple(base + extra)


@dataclass
class _Cluster:
    cluster_id: int
    best: dict
    observations: list[dict] = field(default_factory=list)
    view_ids: set[str] = field(default_factory=set)


class WristEstimateAccumulator:
    """Greedy spatial-only one-to-one association of rendered wrist predictions."""

    def __init__(self, gate_m: float = 0.025):
        if not math.isfinite(gate_m) or gate_m <= 0:
            raise ValueError("association gate must be finite and positive")
        self.gate_m = float(gate_m)
        self._clusters: list[_Cluster] = []
        self._next_id = 0
        self.observation_n = 0
        self._view_ids: set[str] = set()
        self.latest_view_id: str | None = None

    @staticmethod
    def _center(row):
        return np.asarray(row["geometry"]["center_xyz_m"], float)

    def add_view(self, view_id: str, captured_at_s: float, rows: list[dict]) -> None:
        if not view_id or not math.isfinite(float(captured_at_s)):
            raise ValueError("view id and finite capture timestamp are required")
        if view_id in self._view_ids:
            raise ValueError(f"view id {view_id!r} has already been added")
        self._view_ids.add(view_id)
        self.latest_view_id = view_id
        candidates = []
        for i, row in enumerate(rows):
            if row.get("geometry", {}).get("status") != "estimated":
                continue
            center = self._center(row)
            for j, cluster in enumerate(self._clusters):
                if view_id in cluster.view_ids:
                    continue
                distance = float(np.linalg.norm(center - self._center(cluster.best["row"])))
                if distance <= self.gate_m:
                    candidates.append((distance, i, j))
        assigned_rows, assigned_clusters = set(), set()
        by_index = {}
        for i, row in enumerate(rows):
            if row.get("geometry", {}).get("status") == "estimated":
                by_index[i] = row
        for distance, i, j in sorted(candidates):
            if i in assigned_rows or j in assigned_clusters:
                continue
            assigned_rows.add(i)
            assigned_clusters.add(j)
            self._append(self._clusters[j], view_id, captured_at_s, by_index[i], distance)
        for i, row in by_index.items():
            if i in assigned_rows:
                continue
            cluster = _Cluster(self._next_id, {}, [], set())
            self._next_id += 1
            self._clusters.append(cluster)
            self._append(cluster, view_id, captured_at_s, row, None)

    def _append(self, cluster, view_id, captured_at_s, row, association_distance_m):
        obs = {"view_id": view_id, "captured_at_s": float(captured_at_s),
               "detection_id": row["detection_id"], "color": row.get("color", "unknown"),
               "center_xyz_m": self._center(row).tolist(), "fit_iou": row["geometry"].get("fit_iou"),
               "association_distance_m": association_distance_m, "row": row}
        cluster.observations.append(obs)
        cluster.view_ids.add(view_id)
        self.observation_n += 1
        score = float(row["geometry"].get("fit_iou") or 0.0)
        old_score = float(cluster.best.get("row", {}).get("geometry", {}).get("fit_iou") or -1.0)
        if not cluster.best or score > old_score:
            cluster.best = obs

    def records(self) -> list[dict]:
        """Best-fit observations, with provenance; never selected using GT."""
        output = []
        for c in self._clusters:
            best = c.best
            row = dict(best["row"])
            row["detection_id"] = c.cluster_id
            row["exploration"] = {"cluster_id": c.cluster_id, "selected_by": "maximum_fit_iou_no_GT",
                                  "best_view_id": best["view_id"], "best_captured_at_s": best["captured_at_s"],
                                  "best_source_detection_id": best["detection_id"],
                                  "last_seen_view_id": max(c.observations, key=lambda x: x["captured_at_s"])["view_id"],
                                  "last_seen_at_s": max(x["captured_at_s"] for x in c.observations),
                                  "currently_visible_in_latest_view": self.latest_view_id in c.view_ids,
                                  "source_views": sorted(c.view_ids), "observation_n": len(c.observations),
                                  "observations": [{k: v for k, v in item.items() if k != "row"}
                                                   for item in c.observations]}
            output.append(row)
        return output

    def summary(self):
        return {"association": "predicted_center_xyz_nearest_one_to_one_gate_no_color_no_GT",
                "gate_m": self.gate_m, "observation_n": self.observation_n,
                "cluster_n": len(self._clusters), "merged_observation_n": self.observation_n-len(self._clusters),
                "latest_view_id": self.latest_view_id,
                "currently_visible_n": sum(self.latest_view_id in c.view_ids for c in self._clusters),
                "historical_only_n": sum(self.latest_view_id not in c.view_ids for c in self._clusters),
                "clusters": [{"cluster_id": c.cluster_id, "source_views": sorted(c.view_ids),
                              "observation_n": len(c.observations), "best_view_id": c.best["view_id"],
                              "best_fit_iou": c.best["fit_iou"],
                              "last_seen_at_s": max(x["captured_at_s"] for x in c.observations),
                              "currently_visible_in_latest_view": self.latest_view_id in c.view_ids}
                             for c in self._clusters]}

    def reset(self) -> None:
        """Cancel accumulated history when a scene reset starts a new exploration run."""
        self._clusters.clear()
        self._next_id = 0
        self.observation_n = 0
        self._view_ids.clear()
        self.latest_view_id = None
