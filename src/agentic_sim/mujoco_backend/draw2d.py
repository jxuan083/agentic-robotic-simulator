"""Minimal 2D drawing on a MuJoCo render context, in window coordinates (origin top-left, GLFW units).

MuJoCo's own primitives draw in framebuffer pixels with the origin bottom-left (mjr_rectangle, mjr_label) or relative
to the last viewport (mjr_text); this wrapper hides both and scales for HiDPI framebuffers.
"""

from __future__ import annotations

import mujoco

FONT = mujoco.mjtFont.mjFONT_NORMAL
FONT_BIG = mujoco.mjtFont.mjFONT_BIG


class Canvas:
    def __init__(self, con: mujoco.MjrContext, win_w: int, win_h: int, fb_w: int, fb_h: int):
        self.con = con
        self.w, self.h = win_w, win_h
        self.sx = fb_w / max(win_w, 1)
        self.sy = fb_h / max(win_h, 1)
        self.fb_w, self.fb_h = fb_w, fb_h
        self.line_h = con.charHeight / self.sy          # text line height in window units

    def mjrect(self, x, y, w, h) -> mujoco.MjrRect:
        return mujoco.MjrRect(int(round(x * self.sx)), int(round((self.h - y - h) * self.sy)),
                              max(1, int(round(w * self.sx))), max(1, int(round(h * self.sy))))

    def rect(self, x, y, w, h, rgba):
        mujoco.mjr_rectangle(self.mjrect(x, y, w, h), *rgba)

    def frame(self, x, y, w, h, rgba, t=1.5):
        self.rect(x, y, w, t, rgba)
        self.rect(x, y + h - t, w, t, rgba)
        self.rect(x, y, t, h, rgba)
        self.rect(x + w - t, y, t, h, rgba)

    def label(self, x, y, w, h, text, bg, fg=(0.9, 0.9, 0.9)):
        """Filled box with centred text (buttons, pills)."""
        mujoco.mjr_label(self.mjrect(x, y, w, h), FONT, text, *bg, *fg, self.con)

    def text_width(self, text: str, big: bool = False) -> float:
        cw = self.con.charWidthBig if big else self.con.charWidth
        return sum(int(cw[ord(c)]) if ord(c) < len(cw) else int(cw[63]) for c in text) / self.sx

    def text(self, x, y, text, rgb=(0.9, 0.9, 0.9), big=False):
        """Text with its line box's top-left corner at (x, y)."""
        if not text:
            return
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, self.fb_w, self.fb_h), 0, 0, 0, 0)  # viewport = whole window
        lh = (self.con.charHeightBig if big else self.con.charHeight) / self.sy
        mujoco.mjr_text(FONT_BIG if big else FONT, text, self.con, x / self.w, 1.0 - (y + lh) / self.h, *rgb)

    def wrap(self, text: str, width: float) -> list:
        """Split text into lines that fit `width` (word wrap)."""
        out, line = [], ""
        for word in text.split():
            cand = (line + " " + word).strip()
            if line and self.text_width(cand) > width:
                out.append(line)
                line = word
            else:
                line = cand
        if line:
            out.append(line)
        return out
