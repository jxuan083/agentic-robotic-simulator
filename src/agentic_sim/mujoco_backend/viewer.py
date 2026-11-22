"""GLFW window for the MuJoCo teleop session: 3D view with the target handle, operation panel, top view.

MuJoCo's bundled passive viewer is not used: its key callback reports presses only (no release, so hold-to-move and
focus-loss handling cannot work) and its own shortcuts take W/S/A/D/E/Q/C. This window passes every key press /
release to agentic_sim.teleop.keys.KeyState, reports the window focus, and routes the mouse like the Isaac app:

  panel (right)   buttons, scene settings, top view (drag = move target, wheel = height, right click = gripper)
  3D view         drag the ring = move the target, wheel on the ring = height (Shift = fine);
                  elsewhere left drag = rotate, right drag = pan, wheel = zoom (Shift = horizontal variants);
                  Alt + drag on the ring = camera

Scripts drive the same handlers through inject_move / inject_button / inject_scroll (only the OS -> GLFW hop is
skipped), with cursor positions in window coordinates.
"""

from __future__ import annotations

import time

import glfw
import mujoco

import numpy as np

from agentic_sim.mujoco_backend import panel as P
from agentic_sim.mujoco_backend.draw2d import Canvas
from agentic_sim.mujoco_backend.handle import HIT_RADIUS_PX, RING_RADIUS_PX
from agentic_sim.mujoco_backend.model import free_camera, sensor_camera_names
from agentic_sim.teleop.keys import KeyState

# sensor camera insets in the top-right corner of the 3D view: (key, label, inset height in window units)
SENSOR_INSETS = (("global", "D455 global", 200), ("wrist", "wrist camera", 180))
INSET_MARGIN = 8
STRIP_RATIO = 0.32      # operator workspace: share of the window height for the bottom camera strip
RING_RGBA = {"ok": (0.25, 0.82, 0.25, 1.0), "limit": (1.0, 0.63, 0.13, 1.0), "blocked": (1.0, 0.25, 0.25, 1.0)}
EXTRA_KEYS = {glfw.KEY_BACKSPACE: "BACKSPACE", glfw.KEY_ENTER: "ENTER", glfw.KEY_KP_ENTER: "KP_ENTER",
              glfw.KEY_ESCAPE: "ESCAPE", glfw.KEY_TAB: "TAB", glfw.KEY_LEFT_SHIFT: "LEFT_SHIFT",
              glfw.KEY_RIGHT_SHIFT: "RIGHT_SHIFT"}


def _key_names():
    names = {getattr(glfw, f"KEY_{c}"): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    names.update({glfw.KEY_1: "KEY_1", glfw.KEY_2: "KEY_2"})   # carb's names for the digit keys (camera cells)
    names.update(EXTRA_KEYS)
    return names


class Projector:
    """Cursor rays and projection for one rendered view (perspective MjvGLCamera + its window-space rectangle)."""

    def __init__(self):
        self.ok = False

    def set(self, glcams, rect):
        """glcams: the scene's two eye cameras; without stereo MuJoCo renders from their average (the eyes sit
        ipd / 2 to either side), so rays and projections use the average too."""
        glcam = glcams[0]
        self.pos = 0.5 * (np.array(glcams[0].pos, float) + np.array(glcams[1].pos, float))
        self.f = np.array(glcam.forward, float)
        self.up = np.array(glcam.up, float)
        self.right = np.cross(self.f, self.up)
        self.near = float(glcam.frustum_near)
        self.bottom, self.top = float(glcam.frustum_bottom), float(glcam.frustum_top)
        self.center = float(glcam.frustum_center)
        self.rect = rect
        aspect = rect[2] / max(rect[3], 1)
        self.halfw = float(glcam.frustum_width) / 2 if glcam.frustum_width > 0 else 0.5 * (self.top - self.bottom) * aspect
        self.ok = self.near > 0

    def ray(self, x, y):
        if not self.ok:
            return None
        vx, vy, vw, vh = self.rect
        nx = (x - vx) / vw * 2 - 1
        ny = 1 - (y - vy) / vh * 2
        p = (self.pos + self.f * self.near + self.right * (self.center + nx * self.halfw)
             + self.up * (self.bottom + (ny + 1) / 2 * (self.top - self.bottom)))
        d = p - self.pos
        return self.pos.copy(), d / np.linalg.norm(d)

    def project(self, p):
        if not self.ok:
            return None
        q = np.asarray(p, float) - self.pos
        zc = float(q @ self.f)
        if zc <= self.near * 0.5:
            return None
        xn, yn = float(q @ self.right) * self.near / zc, float(q @ self.up) * self.near / zc
        nx = (xn - self.center) / self.halfw
        ny = (yn - self.bottom) / (self.top - self.bottom) * 2 - 1
        vx, vy, vw, vh = self.rect
        return vx + (nx + 1) / 2 * vw, vy + (1 - ny) / 2 * vh

    def m_per_px(self, p) -> float:
        zc = max(float((np.asarray(p, float) - self.pos) @ self.f), 1e-6)
        return zc * (self.top - self.bottom) / self.near / self.rect[3]


class Viewer:
    def __init__(self, keys: KeyState, title: str = "SO-101 MuJoCo teleop", size=(1500, 940), log=print,
                 visible: bool = True):
        if not glfw.init():
            raise RuntimeError("GLFW init failed (is DISPLAY set?)")
        # hidden windows (frame captures for checks) skip multisampling: a multisampled default framebuffer
        # cannot be read back with glReadPixels
        glfw.window_hint(glfw.SAMPLES, 4 if visible else 0)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(int(size[0]), int(size[1]), title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create the MuJoCo window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # the session loop paces itself
        self.title = title
        self.keys = keys
        self.log = log
        self._names = _key_names()
        self._buttons = {"left": False, "right": False, "middle": False}
        self._camera_drag = None          # which button moves the camera, while it does
        self._cursor = (0.0, 0.0)
        self._last = None
        self.session = None
        self.model = None
        self.scene = self.scene_top = self.con = None
        self.cam = mujoco.MjvCamera()
        self.cam_top = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.proj = Projector()
        self._cam_set = False
        self.view_rect = (0, 0, 1, 1)
        self.capture_path = None
        self.show_sensor_cameras = True   # scenes with sensor cameras show them as insets in the 3D view
        self._sensor = {}                 # key -> (MjvCamera, MjvScene, aspect)
        self._inset_rects = []
        self._cell_rects = {}             # camera key -> rect: a click switches raw / annotated
        self.workspace = False            # operator workspace (M): fullscreen, cameras in a strip below, status
        self._windowed = None
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_window_focus_callback(self.window, self._on_focus)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_cursor_enter_callback(self.window, self._on_enter)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # ---- session / model -----------------------------------------------------------------
    def attach(self, session, reset_camera: bool = False):
        """Show this session (new experiment or rebuilt scene). The camera view is kept unless reset_camera."""
        glfw.make_context_current(self.window)
        if self.con is not None:
            self.con.free()
        self.session = session
        self.model = session.m
        self.scene = mujoco.MjvScene(self.model, maxgeom=4000)
        self.scene_top = mujoco.MjvScene(self.model, maxgeom=4000)
        # offscreen buffer for the wrist lens resampling (the window is multisampled and cannot be read back)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), 1600)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), 1200)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
        self.cam_top.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_top.fixedcamid = self.model.camera("topview").id
        self._sensor = {}
        from agentic_sim.scene_layout import sensor_cameras
        specs = sensor_cameras(session.scene_cfg)
        for key, cid in sensor_camera_names(self.model).items():
            cam = mujoco.MjvCamera()
            cam.type, cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, cid
            self._sensor[key] = (cam, mujoco.MjvScene(self.model, maxgeom=4000),
                                 specs[key]["width"] / specs[key]["height"], specs[key].get("lens"),
                                 specs[key].get("render_fovy_deg"))
        self._lens_cache = {}
        session.handle.projector = self.proj
        session.camera_position = self.camera_position
        if reset_camera or not self._cam_set:
            self.reset_camera()
            self._cam_set = True

    def reset_camera(self):
        c = free_camera(self.session.scene_cfg)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = c["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = c["distance"], c["azimuth"], c["elevation"]

    def set_camera(self, eye, target):
        c = free_camera({"camera": {"eye": eye, "target": target}})
        self.cam.lookat[:] = c["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = c["distance"], c["azimuth"], c["elevation"]

    def camera_position(self):
        a, e = np.radians(self.cam.azimuth), np.radians(self.cam.elevation)
        fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        return np.array(self.cam.lookat, float) - self.cam.distance * fwd

    def focused(self) -> bool:
        return bool(glfw.get_window_attrib(self.window, glfw.FOCUSED))

    # ---- input: GLFW callbacks and script injection share these handlers -------------------
    def _on_key(self, window, key, scancode, action, mods):
        name = self._names.get(key)
        if name is None or action == glfw.REPEAT:
            return
        s = self.session
        if action == glfw.PRESS and s is not None and s.panel.edit_key(name):
            return
        if name in EXTRA_KEYS.values() and name not in ("LEFT_SHIFT", "RIGHT_SHIFT"):
            return
        self.keys.on_key(name, action == glfw.PRESS)

    def _on_char(self, window, codepoint):
        if self.session is not None:
            self.session.panel.char(chr(codepoint))

    def _on_focus(self, window, focused):
        if not focused:
            self.keys.clear()  # the session also drops input every frame without focus

    def _on_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and self.session is not None:
            x, y = self._cursor
            for key, (rx, ry, rw, rh) in self._cell_rects.items():
                if rx <= x < rx + rw and ry <= y < ry + rh:     # a camera cell: switch raw / annotated
                    self.session.annotate[key] = not self.session.annotate.get(key, True)
                    return
        name = {glfw.MOUSE_BUTTON_LEFT: "left", glfw.MOUSE_BUTTON_RIGHT: "right",
                glfw.MOUSE_BUTTON_MIDDLE: "middle"}.get(button)
        if name is not None:
            alt = bool(mods & glfw.MOD_ALT)
            self.inject_button(name, action == glfw.PRESS, alt=alt)

    def _on_cursor(self, window, x, y):
        self.inject_move(x, y)

    def _on_enter(self, window, entered):
        if not entered and self.session is not None:
            self.session.handle.leave()

    def _on_scroll(self, window, xoff, yoff):
        self.inject_scroll(yoff)

    def in_view(self, x, y) -> bool:
        """3D view, without the sensor camera insets (they take no input)."""
        vx, vy, vw, vh = self.view_rect
        if any(rx <= x < rx + rw and ry <= y < ry + rh for rx, ry, rw, rh in self._inset_rects):
            return False
        return vx <= x < vx + vw and vy <= y < vy + vh

    def inject_button(self, button: str, down: bool, alt: bool = False):
        s = self.session
        x, y = self._cursor
        self._buttons[button] = down
        if s is None:
            return
        if down:
            if s.panel.press(x, y, button):
                return
            if button == "left" and self.in_view(x, y) and s.handle.press(x, y, alt=alt):
                return
            if self.in_view(x, y):
                self._camera_drag = button
                self._last = (x, y)
        else:
            s.panel.release(button)
            if button == "left":
                s.handle.release()
            if self._camera_drag == button:
                self._camera_drag = None

    def inject_move(self, x, y):
        self._cursor = (float(x), float(y))
        s = self.session
        if s is None:
            return
        s.panel.move(x, y)
        if self.in_view(x, y) or s.handle.dragging:
            s.handle.move(x, y)
        else:
            s.handle.leave()
        if self._camera_drag is not None and self._last is not None:
            lx, ly = self._last
            self._last = (x, y)
            shift = self.keys.shift
            if self._camera_drag == "right":
                act = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif self._camera_drag == "left":
                act = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                act = mujoco.mjtMouse.mjMOUSE_ZOOM
            self.move_camera(act, (x - lx) / max(self.view_rect[3], 1), (y - ly) / max(self.view_rect[3], 1))

    def inject_scroll(self, dy: float):
        s = self.session
        x, y = self._cursor
        if s is None or dy == 0:
            return
        if s.panel.scroll_at(x, y, dy):
            return
        if self.in_view(x, y):
            if not s.handle.scroll(dy):
                self.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * dy)

    def move_camera(self, action, dx: float, dy: float):
        """Mouse camera step. MuJoCo 4 dropped the scene argument of mjv_moveCamera (3.x: ..., scene, cam).
        A failure here only logs: an exception inside a GLFW callback would surface in poll_events and end the
        session loop (and the server) on the next frame."""
        if self.model is None:
            return
        try:
            mujoco.mjv_moveCamera(self.model, int(action), float(dx), float(dy), self.cam)
        except Exception as e:  # noqa: BLE001
            self.log(f"camera move failed: {e}")

    # ---- drawing helpers ------------------------------------------------------------------------
    def _geom(self, scn, gtype, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(),
                            np.asarray(rgba, np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scn.ngeom += 1
        return g

    def _connector(self, scn, gtype, width, a, b, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scn.ngeom += 1

    def _ring(self, scn, center, u, v, radius, thick, rgba, n=24):
        pts = [center + radius * (np.cos(t) * u + np.sin(t) * v) for t in np.linspace(0, 2 * np.pi, n + 1)]
        for a, b in zip(pts[:-1], pts[1:]):
            self._connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, thick, a, b, rgba)

    def _handle_markers(self, s):
        """Ring + crosshair facing the camera (screen size), translucent hit disc, guide line, actual EE sphere."""
        scn, pr = self.scene, self.proj
        tgt = s.ctrl.state.pos
        kind = P.kind_of(s.ctrl.state.status)
        rgba = RING_RGBA[kind]
        mpp = pr.m_per_px(tgt)
        u, v = pr.right, pr.up
        mat = np.column_stack([u, v, pr.f])
        hov = s.handle.hover or s.handle.dragging
        self._geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [HIT_RADIUS_PX * mpp, 0.1 * mpp, 0], tgt, mat,
                   (1, 1, 1, 0.32 if hov else 0.16))
        self._ring(scn, tgt, u, v, RING_RADIUS_PX * mpp, (2.2 if hov else 1.6) * mpp, rgba)
        for d in (u, v):
            self._connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, 1.2 * mpp, tgt - 6 * mpp * d, tgt + 6 * mpp * d, rgba)
        self._connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 1.5, tgt, [tgt[0], tgt[1], 0.0], (1, 1, 1, 0.5))
        f = s.last_frame
        if f:
            self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005] * 3, f["ee"], np.eye(3), (0.1, 0.3, 1.0, 1.0))

    def _map_markers(self, s, mpp):
        """Top view: target ring and actual EE drawn above everything (the camera is orthographic)."""
        scn = self.scene_top
        tgt = s.ctrl.state.pos
        z = 0.9
        rgba = RING_RGBA[P.kind_of(s.ctrl.state.status)]
        ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        self._ring(scn, np.array([tgt[0], tgt[1], z]), ex, ey, 8 * mpp, 1.6 * mpp, rgba, n=20)
        f = s.last_frame
        if f:
            self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [4 * mpp] * 3, [f["ee"][0], f["ee"][1], z + 0.01],
                       np.eye(3), (0.1, 0.3, 1.0, 1.0))

    # ---- frame ------------------------------------------------------------------------------------
    def render(self, session, extra: str = ""):
        if self.model is None or session is not self.session:
            glfw.poll_events()
            return
        s = session
        if getattr(s, "workspace_toggle", False):
            s.workspace_toggle = False
            self.set_workspace(not self.workspace)
        # pyGLFW returns a NULL pointer (falsy), not None, for a windowed window
        s.workspace_active, s.fullscreen = self.workspace, bool(glfw.get_window_monitor(self.window))
        if getattr(s, "live_perception", None) is not None:
            had_focus, n_ctx = self.focused(), len(getattr(s.live_perception, "renderers", {}))
            s.live_perception.update(s.d)
            if had_focus and len(getattr(s.live_perception, "renderers", {})) > n_ctx:
                # a new offscreen renderer opens a hidden GLFW window, which takes the keyboard focus from this one
                # on X11: the operator's keys and drags would be dropped as "focus lost" until a click; take it back
                glfw.focus_window(self.window)
        if getattr(s, "reset_view_requested", False):
            s.reset_view_requested = False
            self.reset_camera()
        glfw.make_context_current(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if win_w <= 0 or win_h <= 0:
            glfw.poll_events()
            return
        cv = Canvas(self.con, win_w, win_h, fb_w, fb_h)
        view_w = max(100, win_w - P.WIDTH)
        strip_h = int(win_h * STRIP_RATIO) if (self.workspace and self.show_sensor_cameras and self._sensor) else 0
        self.view_rect = (0, 0, view_w, win_h - strip_h)
        layout, status_rect = (self._strip_layout(view_w, win_h, strip_h) if strip_h
                               else (self._inset_layout(view_w), None))
        lens_images = self._prerender_lens(cv, s, layout)
        # 3D view
        mujoco.mjv_updateScene(self.model, s.d, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self.proj.set(self.scene.camera, self.view_rect)
        s.handle.update(s.ctrl.state.pos)
        self._handle_markers(s)
        mujoco.mjr_render(cv.mjrect(*self.view_rect), self.scene, self.con)
        banner = ("PAUSED" if s.paused else "") or ("" if s.focused() else "focus lost: input dropped")
        if extra:
            banner = (banner + "   " if banner else "") + extra
        if getattr(s, "live_perception", None) is not None and not self.workspace:   # workspace: status column
            pstate = s.live_perception.state()["cameras"]
            parts = []
            for key in ("global", "wrist"):
                item = pstate.get(key)
                if item is None:
                    continue
                age = item["age_s"]
                centers = []
                latest = item.get("latest")
                if latest is not None:
                    for detection in latest.get("detections", []):
                        geometry = detection.get("geometry", {})
                        if geometry.get("status") == "estimated":
                            xyz = geometry["center_xyz_m"]
                            centers.append(f"{detection['detection_id']}=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})")
                center_text = " XYZ " + ";".join(centers[:3]) if centers else " XYZ —"
                parts.append(f"{key}: {'late' if age is None else f'{age:.1f}s'}" + center_text +
                             (f" ERROR {item['error']}" if item["error"] else ""))
            banner = (banner + "   " if banner else "") + "perception " + " | ".join(parts)
        if banner:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               cv.mjrect(*self.view_rect), banner, "", self.con)
        self._draw_sensor_insets(cv, s, layout, lens_images)
        if status_rect is not None:
            self._draw_status(cv, s, status_rect)
        elif getattr(s, "live_perception", None) is not None:
            self._draw_perception_centers(cv, s, view_w, layout)
        # panel + top view
        s.panel.draw(cv, view_w, 0, win_h)
        mr = s.panel.map_rect
        if mr is not None and mr[1] < win_h and mr[1] + mr[3] > 0:
            mujoco.mjv_updateScene(self.model, s.d, self.opt, self.pert, self.cam_top, mujoco.mjtCatBit.mjCAT_ALL,
                                   self.scene_top)
            self._map_markers(s, 1.0 / s.panel.map_scale())
            mujoco.mjr_render(cv.mjrect(*mr), self.scene_top, self.con)
            s.panel.draw_map_overlay(cv, s.sc.trays)
        if self.capture_path:  # read the frame before the swap (the back buffer is undefined afterwards)
            path, self.capture_path = self.capture_path, None
            self._write_frame(path)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _inset_layout(self, view_w) -> list:
        """(key, label, x, y, w, h) of the sensor camera insets, top-right of the 3D view (window units)."""
        out, y = [], INSET_MARGIN
        if not self.show_sensor_cameras:
            return out
        for key, label, h in SENSOR_INSETS:
            if key not in self._sensor:
                continue
            w = h * self._sensor[key][2]
            x = view_w - w - INSET_MARGIN
            if x < view_w * 0.4 or y + h > self.view_rect[3]:
                break
            out.append((key, label, x, y, w, h))
            y += h + INSET_MARGIN
        return out

    def _strip_layout(self, view_w, win_h, strip_h):
        """Operator workspace: global and wrist side by side along the bottom, the status box after them."""
        out, x, m = [], INSET_MARGIN, INSET_MARGIN
        y, h = win_h - strip_h + m, strip_h - 2 * m
        for key, label, _ in SENSOR_INSETS:
            if key not in self._sensor:
                continue
            w = h * self._sensor[key][2]
            out.append((key, label, x, y, w, h))
            x += w + m
        status = (x, y, max(0, view_w - x - m), h)
        return out, (status if status[2] > 120 else None)

    def set_workspace(self, on: bool):
        """Fullscreen on the primary monitor (operator workspace) or back to the window as it was."""
        if on == self.workspace:
            return
        if on:
            self._windowed = (glfw.get_window_pos(self.window), glfw.get_window_size(self.window))
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            glfw.set_window_monitor(self.window, mon, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        elif self._windowed is not None:
            (x, y), (w, h) = self._windowed
            glfw.set_window_monitor(self.window, None, x, y, w, h, 0)
        self.workspace = on
        self.log(f"operator workspace {'on (fullscreen)' if on else 'off'}")

    def _draw_status(self, cv, s, rect):
        from agentic_sim.workspace_status import status_lines
        x, y, w, h = rect
        cv.rect(x, y, w, h, (0.08, 0.09, 0.11, 0.92))
        line_h = max(14, int(cv.line_h)) + 2
        for i, row in enumerate(status_lines(s.status_info(time.monotonic()))[:max(0, int(h // line_h))]):
            cv.text(x + 8, y + 6 + i * line_h, row, (0.95, 0.95, 0.95))

    def _prerender_lens(self, cv, s, layout) -> dict:
        """Wrist camera in "real" lens mode: render the wider pinhole offscreen at 1.5x the inset's framebuffer size,
        read it back and resample it to the raw sensor (agentic_sim.wrist_lens). Returns key -> bottom-up RGB image
        for mjr_drawPixels."""
        out = {}
        for key, _, x, y, w, h in layout:
            cam, scn, _, lens, rfovy = self._sensor[key]
            if lens is None:
                continue
            r = cv.mjrect(x, y, w, h)
            rw = min(int(r.width * 1.5), int(self.model.vis.global_.offwidth))
            rh = min(int(r.height * 1.5), int(self.model.vis.global_.offheight))
            ck = (key, r.width, r.height, rw, rh)
            maps = self._lens_cache.get(ck)
            if maps is None:
                iy, ix = lens.maps(rw, rh, rfovy, r.width, r.height)
                maps = self._lens_cache[ck] = ((rh - 1 - iy)[::-1], ix[::-1])   # GL rows run bottom-up
            mujoco.mjv_updateScene(self.model, s.d, self.opt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            src = mujoco.MjrRect(0, 0, rw, rh)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
            try:
                mujoco.mjr_render(src, scn, self.con)
                buf = np.zeros((rh, rw, 3), np.uint8)
                mujoco.mjr_readPixels(buf, None, src, self.con)
            finally:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
            out[key] = np.ascontiguousarray(buf[maps[0], maps[1]])
        return out

    def _draw_sensor_insets(self, cv, s, layout, lens_images):
        """Sensor cameras (global D455, wrist) from the scene's camera models: insets top-right of the 3D view, or the
        workspace's bottom strip. Each cell shows the raw camera or, switched on (key 1 / 2 or a click), the annotated
        snapshot of the image the estimate came from (agentic_sim.workspace_status.cell_view: a missing or stale
        result leaves the raw camera, with the reason in the cell's label)."""
        from agentic_sim.workspace_status import cell_view, perception_item
        self._inset_rects = []
        self._cell_rects = {}
        perc = getattr(s, "live_perception", None)
        pstate = perc.state()["cameras"] if perc is not None else {}
        now = time.monotonic()
        for key, label, x, y, w, h in layout:
            cam, scn, _, _, _ = self._sensor[key]
            cv.rect(x - 2, y - 2, w + 4, h + 4, (0.1, 0.1, 0.12, 1.0))
            if perc is None or key not in perc.selected:
                mode, text = "raw", "raw (perception off)"
            else:
                mode, text = cell_view(s.annotate.get(key, True), perception_item(pstate.get(key), now))
            s.cells[key] = {"annotate": s.annotate.get(key, True), "text": text, "image_visible": mode == "annotated"}
            if mode == "annotated" and perc.overlay(key) is not None:
                import cv2
                overlay = perc.overlay(key)
                rect = cv.mjrect(x, y, w, h)
                pixels = cv2.resize(overlay, (rect.width, rect.height), interpolation=cv2.INTER_AREA)
                mujoco.mjr_drawPixels(np.flipud(pixels).copy().ravel(), None, rect, self.con)
            elif key in lens_images:
                mujoco.mjr_drawPixels(lens_images[key].ravel(), None, cv.mjrect(x, y, w, h), self.con)
            else:
                mujoco.mjv_updateScene(self.model, s.d, self.opt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
                mujoco.mjr_render(cv.mjrect(x, y, w, h), scn, self.con)
            cv.label(x, y, cv.text_width(label) + 12, cv.line_h + 4, label, (0.1, 0.1, 0.12, 0.8))
            hint = f"[{1 if key == 'global' else 2}] {text}"
            cv.label(x, y + h - cv.line_h - 4, min(w, cv.text_width(hint) + 12), cv.line_h + 4, hint,
                     (0.0, 0.0, 0.0, 0.7))
            self._inset_rects.append((x - 2, y - 2, w + 4, h + 4))
            self._cell_rects[key] = (x, y, w, h)

    def _draw_perception_centers(self, cv, s, view_w, layout):
        """Readable XYZ rows for every estimated candidate, in separate global/wrist sections."""
        manager = s.live_perception
        if not manager.selected:
            return
        line_h = max(14, int(cv.line_h))
        bottom = max((y + h for _, _, _, y, _, h in layout), default=INSET_MARGIN) + 8
        rows = []
        state = manager.state()["cameras"]
        for key in manager.selected:
            camera = state.get(key, {})
            latest = camera.get("latest") or {}
            age = camera.get("age_s")
            suffix = "late" if age is None else f"{age:.1f}s"
            rows.append(f"{key.upper()} perception   {suffix}" + ("  STALE" if camera.get("stale") else ""))
            detections = latest.get("detections", [])
            if not detections:
                rows.append("  no estimates yet")
            for detection in detections:
                geom = detection.get("geometry", {})
                if geom.get("status") == "estimated":
                    x, y, z = geom["center_xyz_m"]
                    rows.append(f"  #{detection['detection_id']} {detection['color']}: {x:+.3f}, {y:+.3f}, {z:+.3f} m")
                else:
                    rows.append(f"  #{detection['detection_id']} rejected: {geom.get('reason', 'unknown')}")
        width = min(520, int(view_w * .55))
        x0 = max(8, int(view_w) - width - 8)
        height = min(len(rows) * line_h + 8, max(0, self.view_rect[3] - bottom))
        if height <= 0:
            return
        cv.rect(x0 - 4, bottom - 4, width + 8, height, (0.04, 0.05, 0.07, .86))
        for i, row in enumerate(rows[:max(0, height // line_h)]):
            cv.text(x0, bottom + i * line_h, row, (0.95, 0.95, 0.95))

    def render_idle(self):
        """No session loaded (server waiting for load_experiment): keep the window responsive."""
        glfw.poll_events()

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.window))

    def resize(self, w: int, h: int):
        glfw.set_window_size(self.window, int(w), int(h))

    def snapshot(self, path: str):
        """Save the next rendered frame of this window to path (PNG); written during the next render()."""
        self.capture_path = path

    def _write_frame(self, path: str):
        import subprocess
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        rgb = np.zeros((fb_h, fb_w, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, fb_w, fb_h), self.con)
        ppm = path + ".ppm"
        with open(ppm, "wb") as f:
            f.write(b"P6 %d %d 255\n" % (fb_w, fb_h) + np.flipud(rgb).tobytes())
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", ppm, path], check=True)
        import os
        os.remove(ppm)

    def close(self):
        if self.con is not None:
            self.con.free()
            self.con = None
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
