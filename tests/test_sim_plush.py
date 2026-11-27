"""Finals plush in the MuJoCo scene (agentic_sim.plush, scene `plush:` section) and the head grasp of
tools/sim_plush_grasp.py. Headless; needs mujoco and the menagerie SO-101 (tools/fetch_mujoco_assets.sh)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from agentic_sim import plush as P  # noqa: E402
from agentic_sim.config import load  # noqa: E402

mujoco = pytest.importorskip("mujoco")
from agentic_sim.mujoco_backend import model as mjm  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(mjm.menagerie_dir(), "robotstudio_so101/so101.xml")),
    reason="menagerie SO-101 not fetched (tools/fetch_mujoco_assets.sh)")

LAB = "configs/scenes/lab_20261125.yaml"
PLUSH = "configs/scenes/lab_20261128_plush.yaml"


def _mjb(m) -> bytes:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mjb") as f:
        mujoco.mj_saveModel(m, f.name, None)
        return open(f.name, "rb").read()


def test_scene_without_plush_is_unchanged():
    """The plush scene minus its section compiles to the same model as the lab scene it copies."""
    lab, plush = load(LAB), load(PLUSH)
    del plush["plush"], plush["name"], lab["name"]
    assert plush == lab
    a, b = mjm.build(lab), mjm.build(plush)
    assert _mjb(a.model) == _mjb(b.model)
    assert a.plush_body is None and a.plush_qpos is None and a.plush_geoms == []


def test_plush_loads_with_its_mass_and_size_and_rests_on_the_table():
    sc = mjm.build(load(PLUSH))
    m, d = sc.model, mujoco.MjData(sc.model)
    assert m.body_mass[sc.plush_body] == pytest.approx(0.0286, abs=0.0005)
    assert m.body_ipos[sc.plush_body][2] < P.BODY_HEIGHT        # centre of mass in the tram body
    mujoco.mj_forward(m, d)
    lo, hi = np.full(3, np.inf), -np.full(3, np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == sc.plush_body:
            R = d.geom_xmat[g].reshape(3, 3)
            c, ext = d.geom_xpos[g] + R @ m.geom_aabb[g][:3], np.abs(R) @ m.geom_aabb[g][3:]
            lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
    size = np.sort(hi[:2] - lo[:2])[::-1].tolist() + [hi[2] - lo[2]]
    assert np.allclose(size, [0.090, 0.060, 0.100], atol=0.003)   # decorations stand 0.6 mm out of the faces
    p0 = d.qpos[sc.plush_qpos:sc.plush_qpos + 3].copy()
    for _ in range(round(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    p1 = d.qpos[sc.plush_qpos:sc.plush_qpos + 3]
    assert np.linalg.norm(p1 - p0) < 0.005
    assert d.xmat[sc.plush_body].reshape(3, 3)[2, 2] > np.cos(np.radians(2))   # still upright


def test_head_grasp_lifts_the_plush():
    from sim_plush_grasp import run_grasp
    r = run_grasp("head")
    assert r["success"], r
    assert r["lift_mm"] > 45 and r["slip_in_hold_mm"] < 5
    assert r["contacts_held"] == {"fixed": ["head"], "moving": ["head"]}
