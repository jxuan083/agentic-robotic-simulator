"""EE target handle in the MuJoCo 3D view: the primary mouse input, same rules as the Isaac handle
(agentic_sim.isaac.viewport_handle):

- A screen-size ring at the EE *target*; the actual EE is a separate solid blue sphere.
- A left drag that STARTS on the ring moves the target on the horizontal plane at the target height; any other drag
  moves the camera (Alt + drag on the ring too).
- The wheel over the ring (or during a ring drag) changes height; elsewhere it zooms. After a notch on the ring the
  wheel stays on height for WHEEL_STICKY_S, because the ring moves away from the cursor as the height changes.
- Motion is relative: the target moves by the cursor's change on the plane (each event's cursor ray intersected with
  the plane), so pressing never jumps the target. Shift scales each event's motion as it arrives.

Geometry (cursor rays, projection) comes from a projector the viewer provides; without one (headless) the handle
just never receives events.
"""

from __future__ import annotations

import time

import numpy as np

HIT_RADIUS_PX = 18
RING_RADIUS_PX = 11
MIN_RAY_DZ = 0.05          # ignore rays almost parallel to the drag plane (grazing view)
WHEEL_STICKY_S = 0.7


class Handle:
    def __init__(self, fine_scale_fn, shift_fn):
        self._fine_scale_fn = fine_scale_fn
        self._shift_fn = shift_fn
        self.projector = None            # set by the viewer: ray(x, y), project(p)
        self.target = np.zeros(3)
        self.dragging = False
        self.hover = False
        self._plane_z = 0.0
        self._last_hit = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times: list = []
        self._last_wheel_t = -1e9
        self._cursor = None

    # ---- per frame ----------------------------------------------------------------------
    def update(self, target):
        self.target = np.asarray(target, float).copy()
        if self._cursor is not None:
            self.hover = self.hit(*self._cursor)

    def pop_drag(self):
        d, self._delta = self._delta.copy(), np.zeros(2)
        times, self._event_times = self._event_times, []
        return d, times

    def pop_wheel(self):
        w, self._wheel = self._wheel, {"coarse": 0.0, "fine": 0.0}
        return w

    def cancel(self):
        """Stop the current drag (focus loss, pause, reset): later motion of the still-pressed button is ignored."""
        self.dragging = False
        self._last_hit = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []
        self._last_wheel_t = -1e9

    # ---- geometry ---------------------------------------------------------------------------
    def world_to_screen(self, p):
        return self.projector.project(np.asarray(p, float))

    def hit(self, x, y) -> bool:
        if self.projector is None:
            return False
        s = self.projector.project(self.target)
        return s is not None and np.hypot(s[0] - x, s[1] - y) <= HIT_RADIUS_PX

    def _hit_plane(self, x, y, z):
        r = self.projector.ray(x, y) if self.projector is not None else None
        if r is None:
            return None
        o, d = r
        if abs(d[2]) < MIN_RAY_DZ:
            return None
        t = (z - o[2]) / d[2]
        return o + t * d if t > 0 else None

    # ---- events (window coordinates) --------------------------------------------------------
    def press(self, x, y, alt: bool = False) -> bool:
        """Left press in the 3D view; True if it starts a handle drag."""
        self._cursor = (x, y)
        if alt or not self.hit(x, y):
            return False
        self.dragging = True
        self._plane_z = float(self.target[2])
        self._last_hit = self._hit_plane(x, y, self._plane_z)
        return True

    def move(self, x, y):
        self._cursor = (x, y)
        self.hover = self.hit(x, y)
        if not self.dragging:
            return
        if abs(self.target[2] - self._plane_z) > 1e-4:  # height changed during the drag: re-anchor, no jump
            self._plane_z = float(self.target[2])
            self._last_hit = self._hit_plane(x, y, self._plane_z)
            return
        hit = self._hit_plane(x, y, self._plane_z)
        if hit is None:
            return
        if self._last_hit is not None:
            k = self._fine_scale_fn() if self._shift_fn() else 1.0
            self._delta += (hit[:2] - self._last_hit[:2]) * k
            self._event_times.append(time.monotonic())
        self._last_hit = hit

    def release(self):
        self.dragging = False
        self._last_hit = None

    def leave(self):
        """Cursor left the 3D view."""
        self._cursor = None
        self.hover = False

    def wheel_captured(self) -> bool:
        return self.hover or self.dragging or (time.monotonic() - self._last_wheel_t) < WHEEL_STICKY_S

    def scroll(self, dy: float) -> bool:
        """Wheel in the 3D view; True if it went to the height (otherwise the viewer zooms)."""
        if not self.wheel_captured() or dy == 0.0:
            return False
        self._wheel["fine" if self._shift_fn() else "coarse"] += dy
        self._last_wheel_t = time.monotonic()
        self._event_times.append(self._last_wheel_t)
        return True
