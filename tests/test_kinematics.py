"""Numpy-only tests for the SO-101 chain and constrained IK (chain dumped from the workshop USD)."""
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agentic_sim.kinematics import So101Chain  # noqa: E402


def chain():
    c = So101Chain.from_dict(yaml.safe_load(open(os.path.join(ROOT, "configs/so101_chain.yaml"))))
    c.ee_offset = np.array([0.0069, 0.0, -0.085])
    return c


def test_ik_roundtrip_and_speed():
    c = chain()
    q, ok, info = c.solve_ik([0.18, 0.10, 0.06], np.pi / 2, np.array([0.3, 0.0, 0.0, 1.5, 0.0]), iters=200)
    assert ok, info
    t0 = time.perf_counter()
    n = 0
    for dx in np.linspace(0, 0.05, 50):  # warm-started tracking, as in the teleop loop
        q, ok, info = c.solve_ik([0.18 + dx, 0.10, 0.06], np.pi / 2, q)
        assert ok, info
        n += 1
    dt = (time.perf_counter() - t0) / n
    assert dt < 0.01, f"IK too slow for the control loop: {dt*1000:.1f} ms"


def test_unreachable_reports_not_ok():
    c = chain()
    q, ok, info = c.solve_ik([0.6, 0.0, 0.05], 0.0, np.array([0.0, 0.0, 0.0, 1.5, 0.0]))
    assert not ok
    assert np.all(q >= c.lower) and np.all(q <= c.upper)


if __name__ == "__main__":
    test_ik_roundtrip_and_speed()
    test_unreachable_reports_not_ok()
    print("ok")


def test_color_sampler_reproducible_and_independent():
    from agentic_sim.color_dr import ColorSampler, Palette
    pal = Palette.load("configs/palettes/block_colors.yaml")
    assert len(pal.colors) == 8 and [c.name for c in pal.colors][:3] == ["red", "orange", "yellow"]
    a = ColorSampler(pal, seed=7)
    b = ColorSampler(pal, seed=7)
    seq_a = [a.next() for _ in range(20)]
    seq_b = [b.next() for _ in range(20)]
    assert seq_a == seq_b                                   # same seed + same reset order -> same colours
    assert [r["session"] for r in seq_a] == list(range(20))
    blocks = {r["block"]["color_id"] for r in seq_a}
    arms = {r["arm_shell"]["color_id"] for r in seq_a}
    assert blocks <= set(range(8)) and arms <= set(range(8)) and len(blocks) > 3
    assert [r["block"]["color_id"] for r in seq_a] != [r["arm_shell"]["color_id"] for r in seq_a]  # independent draws
    c = ColorSampler(pal, seed=8)
    assert [r["block"]["color_id"] for r in seq_a] != [c.next()["block"]["color_id"] for _ in range(20)]
    only_block = ColorSampler(pal, seed=7, arm_shell=False)
    assert [only_block.next()["block"] for _ in range(20)] == [r["block"] for r in seq_a]  # sequence not shifted
    assert all(r["palette_version"] == pal.version for r in seq_a)


def walk(c, solve, start_q, p0, p1, n=40):
    """Warm-started walk from p0 to p1 like the teleop loop; returns the last reachable point, q, info."""
    q, last = start_q, None
    for s in np.linspace(0, 1, n):
        p = (1 - s) * np.asarray(p0) + s * np.asarray(p1)
        q2, ok, info = solve(p, q)
        if not ok:
            break
        q, last = q2, (p, info)
    return last, q


def test_tilt_ik_equals_vertical_ik_where_vertical_reaches():
    c = chain()
    q0 = np.array([0.3, 0.0, 0.0, 1.5, 0.0])
    for p in ([0.18, 0.10, 0.06], [0.12, -0.05, 0.03], [0.24, 0.0, 0.08]):
        qa, oka, _ = c.solve_ik(p, np.pi / 2, q0, iters=200)
        qb, okb, ib = c.solve_ik_tilt(p, np.pi / 2, qa + 0.03, 60.0)
        assert oka and okb, ib
        assert np.max(np.abs(qa - qb)) < 0.02 and ib["tilt_deg"] < 4.0


def test_tilt_extends_reach_forward_and_up():
    c = chain()
    pan = c.pan_origin_mount()
    q0, _, _ = c.solve_ik([0.15, pan[1], 0.03], np.pi / 2, np.array([0.0, 0.0, 0.0, 1.5, 0.0]), iters=300)

    def vertical(p, q):
        return c.solve_ik(p, np.pi / 2, q)

    def tilted(p, q):
        return c.solve_ik_tilt(p, np.pi / 2, q, 60.0)

    far = ([0.15, pan[1], 0.03], [0.50, pan[1], 0.03])
    (pv, _), _ = walk(c, vertical, q0, *far, n=120)
    (pt, it), _ = walk(c, tilted, q0, *far, n=120)
    assert pt[0] - pv[0] > 0.08, (pv, pt)   # measured: about 0.30 -> 0.42 at grasp height
    assert 20 < it["tilt_deg"] <= 60
    up = ([0.15, pan[1], 0.03], [0.15, pan[1], 0.40])
    (pv, _), _ = walk(c, vertical, q0, *up, n=120)
    (pt, _), _ = walk(c, tilted, q0, *up, n=120)
    assert pt[2] - pv[2] > 0.10, (pv, pt)   # measured: about 0.14 -> 0.28


def test_tilt_never_exceeds_max_and_stays_in_limits():
    c = chain()
    q, ok, info = c.solve_ik_tilt([0.60, 0.0, 0.05], 0.0, np.array([0.0, 0.5, -0.5, 1.5, 0.0]), 30.0)
    assert not ok
    assert np.all(q >= c.lower) and np.all(q <= c.upper)
