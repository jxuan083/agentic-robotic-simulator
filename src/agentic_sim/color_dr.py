"""Colour domain randomization: palette + seeded sampler (pure numpy, testable without Isaac).

One draw per session (start-up = session 0, each reset = next session), always in the same order: block first, then
arm shell, from independent draws of the same palette. The same seed and the same reset sequence therefore give the
same colours. When disabled, nothing is drawn and the scene keeps its fixed baseline colours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic_sim.config import load


@dataclass(frozen=True)
class PaletteColor:
    id: int
    name: str
    rgb: tuple


@dataclass(frozen=True)
class Palette:
    version: str
    colors: tuple

    @classmethod
    def load(cls, path: str) -> "Palette":
        d = load(path)
        cols = tuple(PaletteColor(int(c["id"]), str(c["name"]), tuple(float(x) for x in c["rgb"])) for c in d["colors"])
        ids = [c.id for c in cols]
        if ids != list(range(len(cols))):
            raise ValueError(f"palette ids must be 0..n-1 in order, got {ids}")
        return cls(str(d["version"]), cols)


def srgb_to_linear(rgb):
    """Palette values are sRGB design colours; renderer material colours (OmniPBR / UsdPreviewSurface) are linear."""
    out = []
    for c in rgb:
        c = float(c)
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


class ColorSampler:
    def __init__(self, palette: Palette, seed: int, block: bool = True, arm_shell: bool = True):
        self.palette = palette
        self.seed = int(seed)
        self.block = block
        self.arm_shell = arm_shell
        self._rng = np.random.default_rng(self.seed)
        self.session = -1

    def next(self) -> dict:
        """Colours for the next session. Both draws always happen (fixed order), so turning one target off does not
        shift the other's sequence."""
        self.session += 1
        n = len(self.palette.colors)
        b = self.palette.colors[int(self._rng.integers(n))]
        a = self.palette.colors[int(self._rng.integers(n))]
        rec = {"session": self.session, "seed": self.seed, "palette_version": self.palette.version}
        if self.block:
            rec["block"] = {"color_id": b.id, "name": b.name, "rgb": list(b.rgb)}
        if self.arm_shell:
            rec["arm_shell"] = {"color_id": a.id, "name": a.name, "rgb": list(a.rgb)}
        return rec
