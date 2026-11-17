"""EE target handle drawn in the main viewport (omni.ui.scene), the primary mouse input.

- The handle is a screen-size ring at the EE *target* (what the operator commands). The actual EE is a separate solid
  blue sphere in the scene, so the two are never confused.
- Left-drag that STARTS on the handle moves the target on the horizontal plane at the target height. Any other drag
  stays a normal viewport / UI gesture (selection, camera with Alt or right button).
- Mouse wheel over the handle (or during a handle drag) changes height; over empty viewport it still zooms the camera.
  Once the wheel started on the handle it stays on height while notches keep coming (WHEEL_STICKY_S), because the
  ring moves away from the cursor as the height changes.
- Mapping uses the viewport's own view/projection: each event's cursor ray is intersected with the plane, so camera
  moves, window resizes and DPI are handled by the viewport itself.
- Motion is relative: the target moves by the cursor's *change* on the plane, so pressing never jumps the target.
  Shift scales each event's motion as it arrives, so pressing / releasing Shift never jumps either.
"""

from __future__ import annotations

import time

import carb.input
import numpy as np
from omni.kit.viewport.utility import get_active_viewport_window
from omni.ui import scene as sc

COLORS = {  # omni.ui.scene colours are 0xAABBGGRR
    "ok": 0xFF40D040,
    "limit": 0xFF20A0FF,   # amber: clamped by the workspace box (not an error)
    "blocked": 0xFF3030FF,  # red: joint limit / no IK / jump
}
HIT_RADIUS_PX = 18
RING_RADIUS_PX = 11
MIN_RAY_DZ = 0.05          # ignore rays almost parallel to the drag plane (grazing view)
WHEEL_STICKY_S = 0.7       # after a wheel notch on the handle, keep the wheel on height for this long
ANY_MODIFIERS = 0xFFFFFFFF  # fire with or without Shift (Shift = fine); Alt is left to the camera below


def _alt_down() -> bool:
    i = carb.input.acquire_input_interface()
    return (i.get_keyboard_value(None, carb.input.KeyboardInput.LEFT_ALT)
            + i.get_keyboard_value(None, carb.input.KeyboardInput.RIGHT_ALT)) > 0


class _PreventOthers(sc.GestureManager):
    """While the handle drag is active, other viewport gestures (selection marquee, ...) are prevented."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def can_be_prevented(self, gesture):
        return gesture.state not in (sc.GestureState.CHANGED, sc.GestureState.ENDED, sc.GestureState.CANCELED)

    def should_prevent(self, gesture, preventer):
        if (preventer in self._owner.own_gestures and not _alt_down()
                and preventer.state in (sc.GestureState.BEGAN, sc.GestureState.CHANGED)):
            return gesture not in self._owner.own_gestures
        if preventer is self._owner._scroll and self._owner.wheel_captured() and gesture not in self._owner.own_gestures:
            return True  # wheel is on the height: keep the camera from zooming
        return super().should_prevent(gesture, preventer)


class _Drag(sc.DragGesture):
    def __init__(self, owner, manager):
        super().__init__(mouse_button=0, modifiers=ANY_MODIFIERS, manager=manager)
        self._o = owner

    def on_began(self):
        self._o._drag_began(self)

    def on_changed(self):
        self._o._drag_changed(self)

    def on_ended(self):
        self._o._drag_ended()


class _Scroll(sc.ScrollGesture):
    def __init__(self, owner, manager):
        super().__init__(modifiers=ANY_MODIFIERS, manager=manager)
        self._o = owner

    def on_ended(self, *args):
        self._o._scrolled(self)


class EEHandle:
    def __init__(self, fine_scale_fn, shift_fn, log=print):
        self._fine_scale_fn = fine_scale_fn  # () -> drag gain while Shift is held
        self._shift_fn = shift_fn            # () -> bool, Shift currently held (keyboard state)
        self._log = log
        self.target = np.zeros(3)
        self.dragging = False
        self._cancelled = False
        self._plane_z = 0.0
        self._last_hit = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []
        self._logged_payload = False
        self.hover = False
        self._last_wheel_t = -1e9
        self.window = get_active_viewport_window()
        manager = _PreventOthers(self)
        self._drag = _Drag(self, manager)
        self._scroll = _Scroll(self, manager)
        self.own_gestures = (self._drag, self._scroll)
        with self.window.get_frame("ee_teleop_handle"):
            self.scene_view = sc.SceneView()
            with self.scene_view.scene:
                self._root = sc.Transform()
                with self._root:
                    self._guide = sc.Line([0, 0, 0], [0, 0, -0.1], color=0x80FFFFFF, thickness=1)
                    with sc.Transform(look_at=sc.Transform.LookAt.CAMERA, scale_to=sc.Space.SCREEN):
                        self._hit = sc.Arc(HIT_RADIUS_PX, color=0x30FFFFFF, gestures=[
                            self._drag,
                            sc.HoverGesture(on_began_fn=lambda s: self._set_hover(True),
                                            on_ended_fn=lambda s: self._set_hover(False)),
                        ])
                        self._ring = sc.Arc(RING_RADIUS_PX, wireframe=True, thickness=3, color=COLORS["ok"])
                        self._cross = [sc.Line([-6, 0, 0], [6, 0, 0], thickness=2, color=COLORS["ok"]),
                                       sc.Line([0, -6, 0], [0, 6, 0], thickness=2, color=COLORS["ok"])]
                # wheel is watched over the whole viewport; _scrolled decides whether it is ours (height)
                self._screen = sc.Screen(gestures=[self._scroll])
        self.window.viewport_api.add_scene_view(self.scene_view)
        self._patch_wheel_zoom()
        self._hide_selection_marquee()
        try:  # render fills the viewport area (no letterbox when the layout changes); set once, never toggled
            self.window.viewport_widget.fill_frame = True
        except Exception as e:  # noqa: BLE001
            self._log(f"viewport handle: could not set fill_frame ({e})")

    def _hide_selection_marquee(self):
        """No marquee selection in the 3D view (operation UI; prims are unpickable anyway, the Stage panel still
        selects). The viewport's selection item has no name, so it is found by its class in the layer tree."""
        root = getattr(self.window, "_ViewportWindow__viewport_layers", None)
        found, names = [], []

        def walk(layer):
            names.append(f"{getattr(layer, 'name', None)}:{type(layer).__name__}")
            if "SelectionManipulatorItem" in repr(layer):
                found.append(layer)
            for child in getattr(layer, "layers", ()) or ():
                walk(child)

        try:
            walk(root)
            for layer in found:
                layer.visible = False
            if not found:
                self._log(f"viewport handle: selection layer not found in {names}; marquee stays visible")
        except Exception as e:  # noqa: BLE001
            self._log(f"viewport handle: could not hide the selection marquee ({e})")

    def _patch_wheel_zoom(self):
        """The viewport zooms on the wheel through a SceneView wheel callback (not a gesture), which calls
        omni.kit.manipulator.camera.viewport_camera_manipulator._zoom_operation at event time. Skip that zoom while
        the wheel belongs to the handle height; everywhere else the camera zooms as usual."""
        self._vcm = self._orig_zoom = None
        try:
            import omni.kit.manipulator.camera.viewport_camera_manipulator as vcm
            orig = vcm._zoom_operation

            def zoom_unless_captured(x, y, viewport_api):
                if self.wheel_captured():
                    return True
                return orig(x, y, viewport_api)

            vcm._zoom_operation = zoom_unless_captured
            self._vcm, self._orig_zoom = vcm, orig
        except Exception as e:  # noqa: BLE001
            self._log(f"viewport handle: could not hook wheel zoom ({e}); wheel over the handle also zooms")

    # ---- per frame -------------------------------------------------------------------
    def update(self, target, kind: str):
        """kind: ok | limit | blocked (colour only; the reason text lives in the panel)."""
        self.target = np.asarray(target, float).copy()
        self._root.transform = sc.Matrix44.get_translation_matrix(*self.target)
        self._guide.end = [0.0, 0.0, -float(self.target[2])]  # vertical line down to the table (z = 0)
        c = COLORS.get(kind, COLORS["ok"])
        self._ring.color = c
        for ln in self._cross:
            ln.color = c

    def pop_drag(self):
        """(dx, dy) in metres accumulated since the last call (already scaled for Shift), and event times."""
        d, self._delta = self._delta.copy(), np.zeros(2)
        times, self._event_times = self._event_times, []
        return d, times

    def pop_wheel(self):
        w, self._wheel = self._wheel, {"coarse": 0.0, "fine": 0.0}
        return w

    def cancel(self):
        """Stop the current drag (focus loss, pause, reset): later motion of the still-pressed button is ignored."""
        if self.dragging:
            self._cancelled = True
        self.dragging = False
        self._last_hit = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []
        self._last_wheel_t = -1e9

    # ---- geometry --------------------------------------------------------------------
    def _ray(self, gesture):
        """Cursor ray in world space (as omni.kit.manipulator.transform does): ray_closest_point lies on the ray;
        mapping it to NDC and back at the near plane gives a second point on the same ray."""
        b = np.array(gesture.gesture_payload.ray_closest_point, float)
        ndc = gesture.sender.transform_space(sc.Space.WORLD, sc.Space.NDC, b.tolist())
        a = np.array(gesture.sender.transform_space(sc.Space.NDC, sc.Space.WORLD, [ndc[0], ndc[1], 0.0]), float)
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None, None
        return a, d / n

    def _hit_plane(self, gesture, z):
        o, d = self._ray(gesture)
        if o is None or abs(d[2]) < MIN_RAY_DZ:
            return None
        t = (z - o[2]) / d[2]
        if t <= 0:
            return None
        return o + t * d

    def camera_position(self):
        t = self.window.viewport_api.transform
        c = t.ExtractTranslation()
        return np.array([c[0], c[1], c[2]], float)

    def world_to_screen(self, p):
        """Window-space pixel position of world point p (used by scripted tests to find the handle)."""
        sv = self.scene_view
        view = np.array(sv.model.get_as_floats("view")).reshape(4, 4)
        proj = np.array(sv.model.get_as_floats("projection")).reshape(4, 4)
        v = np.array([p[0], p[1], p[2], 1.0]) @ view @ proj
        ndc = v[:3] / v[3]
        w, h = sv.computed_width, sv.computed_height
        x0, y0 = sv.screen_position_x, sv.screen_position_y
        half = w / 2.0  # PRESERVE_ASPECT_HORIZONTAL: ndc x spans the width, square pixels
        return (x0 + w / 2.0 + ndc[0] * half,
                y0 + h / 2.0 - ndc[1] * half * (proj[0, 0] / proj[1, 1]))

    # ---- gesture callbacks -------------------------------------------------------------
    def _set_hover(self, v):
        self.hover = v

    def _drag_began(self, g):
        if not self._logged_payload:
            self._logged_payload = True
            self._log(f"viewport handle: payload fields {[a for a in dir(g.gesture_payload) if not a.startswith('_')]}")
        if _alt_down():  # Alt + drag is camera orbit: never move the arm
            self._cancelled = True
            return
        self._cancelled = False
        self.dragging = True
        self._plane_z = float(self.target[2])
        self._last_hit = self._hit_plane(g, self._plane_z)

    def _drag_changed(self, g):
        if self._cancelled or not self.dragging:
            return
        if abs(self.target[2] - self._plane_z) > 1e-4:  # height changed during the drag: re-anchor, no jump
            self._plane_z = float(self.target[2])
            self._last_hit = self._hit_plane(g, self._plane_z)
            return
        hit = self._hit_plane(g, self._plane_z)
        if hit is None:
            return
        if self._last_hit is not None:
            k = self._fine_scale_fn() if self._shift_fn() else 1.0
            self._delta += (hit[:2] - self._last_hit[:2]) * k
            self._event_times.append(time.monotonic())
        self._last_hit = hit

    def _drag_ended(self):
        try:  # a handle drag never leaves a selection (and its transform gizmo) behind
            import omni.usd
            omni.usd.get_context().get_selection().clear_selected_prim_paths()
        except Exception:  # noqa: BLE001
            pass
        self.dragging = False
        self._cancelled = False
        self._last_hit = None

    def _scrolled(self, g):
        if not getattr(self, "_logged_scroll", False):
            self._logged_scroll = True
            self._log(f"viewport handle: scroll gesture fields {[a for a in dir(g) if not a.startswith('_')]} "
                      f"payload {[a for a in dir(g.gesture_payload) if not a.startswith('_')]} hover={self.hover}")
        self._n_scroll_logs = getattr(self, "_n_scroll_logs", 0) + 1
        if self._n_scroll_logs <= 3:
            self._log(f"viewport handle: scroll={list(g.scroll)} state={g.state} raw={getattr(g, 'raw_input', None)}")
        if not self.wheel_captured():
            return
        s = getattr(g, "scroll", None)
        # omni.ui.scene reports the vertical wheel in scroll[0] (verified with injected events); take the larger
        dy = 0.0 if s is None else (float(s[0]) if abs(s[0]) >= abs(s[1]) else float(s[1]))
        if dy == 0.0:
            return
        self._wheel["fine" if self._shift_fn() else "coarse"] += dy
        self._last_wheel_t = time.monotonic()
        self._event_times.append(self._last_wheel_t)

    def wheel_captured(self) -> bool:
        return self.hover or self.dragging or (time.monotonic() - self._last_wheel_t) < WHEEL_STICKY_S

    def destroy(self):
        if self._vcm is not None:
            self._vcm._zoom_operation = self._orig_zoom
            self._vcm = None
        if self.scene_view is not None:
            self.scene_view.scene.clear()
            self.window.viewport_api.remove_scene_view(self.scene_view)
            self.scene_view = None
