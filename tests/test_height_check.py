import math

import numpy as np

from agentic_sim.perception.height import close_pairs, height_check, polygon_gap

E, FLOOR = 0.0295, 0.004
CFG = {"plane_tol_m": 0.0015, "min_points": 30, "tilt_max_deg": 5.0, "side_face_min_deg": 70.0,
       "layer_tol": 0.15, "min_area_faces": 0.5, "merged_area_faces": 1.6}


def _face(center, z, size=E, tilt_deg=0.0, n=30, noise=0.0, seed=0):
    """Grid of points on a square face, tilted about the x axis."""
    s = np.linspace(-size / 2, size / 2, n)
    u, w = np.meshgrid(s, s)
    t = math.radians(tilt_deg)
    pts = np.c_[u.ravel(), w.ravel() * math.cos(t), w.ravel() * math.sin(t)] + [center[0], center[1], z]
    return pts + np.random.default_rng(seed).normal(0, noise, pts.shape)


def test_upright_and_stacked_by_top_height():
    up = height_check(_face((0.2, 0.2), FLOOR + E, noise=0.001), FLOOR, E, CFG)
    assert up["status"] == "upright" and abs(up["layers"] - 1) < 0.05 and up["tilt_deg"] < 2
    st = height_check(_face((0.2, 0.2), FLOOR + 2 * E, noise=0.001), FLOOR, E, CFG)
    assert st["status"] == "stacked" and abs(st["layers"] - 2) < 0.05
    mid = height_check(_face((0.2, 0.2), FLOOR + 1.5 * E), FLOOR, E, CFG)
    assert mid["status"] == "off_layer"


def test_tilted_side_face_merged_and_partial():
    tilted = height_check(_face((0.2, 0.2), FLOOR + E, tilt_deg=20, noise=0.0005), FLOOR, E, CFG)
    assert tilted["status"] == "tilted" and abs(tilted["tilt_deg"] - 20) < 2
    side = height_check(_face((0.2, 0.2), FLOOR + E / 2, tilt_deg=88), FLOOR, E, CFG)
    assert side["status"] == "side_face"
    two = np.r_[_face((0.2, 0.2), FLOOR + E), _face((0.2 + E, 0.2), FLOOR + E)]
    assert height_check(two, FLOOR, E, CFG)["status"] == "merged"
    part = height_check(_face((0.2, 0.2), FLOOR + E, size=E * 0.5), FLOOR, E, CFG)
    assert part["status"] == "partial"
    assert height_check(np.zeros((5, 3)), FLOOR, E, CFG)["status"] == "no_depth"


def test_polygon_gap_handles_small_crossing_squares():
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * E / 2
    diamond = np.array([[0, -1.3], [1.3, 0], [0, 1.3], [-1.3, 0]]) * E / 2   # no vertex inside the other
    assert polygon_gap(sq + 0.25, diamond + 0.25) == 0.0
    assert abs(polygon_gap(sq, sq + [E + 0.01, 0]) - 0.01) < 1e-9


def test_close_pairs_with_rim_margin_and_ignored_side_faces():
    a = height_check(_face((0.20, 0.2), FLOOR + E), FLOOR, E, CFG)
    b = height_check(_face((0.20 + E + 0.012, 0.2), FLOOR + E), FLOOR, E, CFG)
    c = height_check(_face((0.30, 0.2), FLOOR + E), FLOOR, E, CFG)
    pairs = close_pairs([a, b, c], 0.015)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert close_pairs([a, b], 0.015, hull_margin_m=0.002)[0][2] < pairs[0][2]
    side = dict(a, status="side_face")
    assert close_pairs([side, b], 0.015) == []
