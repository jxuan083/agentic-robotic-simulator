"""Logic shared by every backend: key map / held keys, scripted-operator core, scene layout (pure Python)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from agentic_sim.config import load  # noqa: E402
from agentic_sim.scene_layout import block_in_tray, block_layout, tray_parts, trays  # noqa: E402
from agentic_sim.teleop.keys import KeyState  # noqa: E402
from agentic_sim.teleop.script_core import ScriptCore, step_kind  # noqa: E402

SCENE = load("configs/scenes/lab_current_minimal.yaml")


# ---- shared pure modules ------------------------------------------------------------------
def test_keystate_hold_release_repeat_actions():
    k = KeyState(actions={"P": "pause", "R": "reset"})
    k.on_key("W", True)
    k.on_key("A", True)
    k.on_key("W", True, repeat=True)  # auto-repeat ignored
    assert k.directions() == ([1, 1, 0], 0, 0)
    k.on_key("W", False)
    assert k.directions()[0] == [0, 1, 0]
    k.on_key("M", True)  # not an action on this backend, not a hold key
    k.on_key("R", True)
    assert k.pop_actions() == ["reset"] and k.pop_actions() == []
    k.on_key("C", True)
    assert k.directions()[2] == -1
    k.clear()
    assert not k.held


class _Recorder(ScriptCore):
    def __init__(self, steps):
        self.events = []
        super().__init__(steps, lambda *_: None)

    def _key(self, name, down):
        self.events.append((name, down))


def test_script_core_keys_checks_and_unsupported_steps():
    s = _Recorder([{"hold": ["W"], "until": "x > 1"}, {"tap": "R"}, {"mark": "a"},
                   {"check": "m['a']['x'] == 2", "label": "mark"}])
    for x in (0, 2, 2, 2, 2, 2):
        s.step({"x": x})
    assert s.done and s.events == [("W", True), ("W", False), ("R", True), ("R", False)]
    assert s.results == [{"label": "mark", "ok": True, "expr": "m['a']['x'] == 2"}]
    assert step_kind({"focus_loss": 1.0, "hold": ["A"]}) == "focus_loss"
    with pytest.raises(ValueError, match="vdrag_push"):
        _Recorder([{"wait": 1}, {"vdrag_push": [1, 0], "seconds": 2}])


def test_scene_layout_matches_scene_numbers():
    parts, info = tray_parts(SCENE["trays"]["storage_pink"])
    assert set(parts) == {"floor", "wall_front", "wall_back", "wall_left", "wall_right"}
    assert parts["floor"][0] == (0.305, 0.23, 0.002)
    lay = block_layout(SCENE, 8)
    assert len(lay.init_pos) == 8 and lay.init_pos[0][:2] == (0.170, 0.120)
    t = trays(SCENE)
    assert all(block_in_tray(p, t["storage_pink"], lay.edge) for p in lay.init_pos)
    assert not any(block_in_tray(p, t["placement_beige"], lay.edge) for p in lay.init_pos)
    with pytest.raises(ValueError):
        block_layout(SCENE, 9)


# MuJoCo panel / 3D handle logic: pure Python (no mujoco import), so it also runs in the Isaac container
def test_panel_settings_pending_and_map_mapping():
    from agentic_sim.mujoco_backend.panel import MjPanel
    p = MjPanel({"blocks": 1, "color_dr": False, "color_dr_seed": 0}, 8, lambda: 0.25, lambda: False, "keys")
    assert p.pending() == {}
    p.set_settings({"color_dr": True, "color_dr_seed": 5})
    assert p.pending() == {"color_dr": True, "color_dr_seed": 5}
    p.mark_applied(p.settings())
    assert p.pending() == {}
    p.map_rect = (100, 50, 380, 380 * 0.5 / 0.72)
    for xy in ((0.2, 0.1), (-0.03, -0.35), (0.45, 0.3)):
        sx, sy = p.world_to_screen(*xy)
        assert np.allclose(p.screen_to_world(sx, sy), xy)
    # top-view drag is relative and Shift-scaled; right click toggles the gripper
    p.rect = (100, 0, 400, 900)
    sx, sy = p.world_to_screen(0.2, 0.0)
    assert p.press(sx, sy, "left") and p.dragging
    p.move(*p.world_to_screen(0.25, 0.0))
    d, times = p.pop_drag()
    assert np.allclose(d, [0.05, 0.0]) and len(times) == 1
    p.release("left")
    p.press(sx, sy, "right")
    assert p.pop_actions() == ["toggle_gripper"]
    assert p.scroll_at(sx, sy, -2) and p.pop_wheel()["coarse"] == -2


def test_handle_drag_is_relative_with_a_projector():
    from agentic_sim.mujoco_backend.handle import Handle

    class TopDown:  # orthographic straight down, 1000 px per metre, origin at pixel (0, 0)
        def project(self, p):
            return p[0] * 1000.0, p[1] * 1000.0

        def ray(self, x, y):
            return np.array([x / 1000.0, y / 1000.0, 1.0]), np.array([0.0, 0.0, -1.0])

    shift = {"on": False}
    h = Handle(lambda: 0.25, lambda: shift["on"])
    h.projector = TopDown()
    h.update([0.15, 0.0, 0.12])
    assert not h.press(400, 400)                       # not on the ring: camera, not the target
    assert h.press(155, 5) and h.dragging              # within the hit radius: no jump on press
    assert np.allclose(h.pop_drag()[0], 0.0)
    h.move(165, 5)
    shift["on"] = True
    h.move(205, 5)                                     # Shift: 40 px -> 10 mm
    assert np.allclose(h.pop_drag()[0], [0.01 + 0.01, 0.0])
    assert h.scroll(-1) and h.pop_wheel()["fine"] == -1  # wheel during a ring drag: height (Shift: fine)
    h.release()
    h.move(400, 400)
    assert h.scroll(1)                                  # within WHEEL_STICKY_S of the last notch: still height
    h.pop_wheel()
    h._last_wheel_t -= 1.0                              # ... and later, away from the ring: the viewer zooms
    assert not h.scroll(1)


def test_check_that_raises_is_a_fail_not_a_crash():
    s = _Recorder([{"check": "dr['seed'] == 5", "label": "needs dr"}, {"check": "1 == 1"}])
    for _ in range(3):
        s.step({"dr": None})
    assert s.done and [r["ok"] for r in s.results] == [False, True] and "TypeError" in s.results[0]["error"]
