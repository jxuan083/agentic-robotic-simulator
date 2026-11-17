"""Apply colour domain randomization to the Isaac scene (arm shell + block). Visual attributes only.

Disabled (default): nothing on the stage is touched; the scene keeps its baseline colours.
Enabled, once at start-up: make the arm *visual* groups non-instanced, because the renderer keeps its own material
copy for instanced prototypes and would not show the colour change (collisions and physics are separate prims). The
whole shell, gripper fingers included, shares one colour; motors use another material and are never changed.
Enabled, at start-up and every reset: draw the next colours (ColorSampler) and write them to the block material and
the arm shell material, and append one JSON line to the log.
"""

from __future__ import annotations

import json
import os

import omni.usd
from pxr import Gf

from agentic_sim.color_dr import ColorSampler, Palette, srgb_to_linear
from agentic_sim.config import REPO_ROOT, load

BLOCK_SHADER = "/World/Block/geometry/material/Shader"
BLOCK_INPUT = "inputs:diffuseColor"            # UsdPreviewSurface (spawned by Isaac Lab)
SHELL_INPUT = "inputs:diffuse_color_constant"  # OmniPBR (SO-101 USD)


class ColorRandomizer:
    def __init__(self, cfg_path: str, enabled: bool | None = None, seed: int | None = None, log_path: str | None = None,
                 log=print):
        self.cfg = load(cfg_path)
        self.enabled = bool(self.cfg["enabled"] if enabled is None else enabled)
        self.seed = int(self.cfg["seed"] if seed is None else seed)
        self._log = log
        self.records: list[dict] = []
        self.current: dict | None = None
        if not self.enabled:
            return
        self.palette = Palette.load(self.cfg["palette"])
        self.sampler = ColorSampler(self.palette, self.seed, bool(self.cfg["block"]), bool(self.cfg["arm_shell"]))
        p = log_path or self.cfg.get("log")
        self.log_path = None if not p else (p if os.path.isabs(p) else os.path.join(REPO_ROOT, p))
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            open(self.log_path, "w").close()
        self.stage = omni.usd.get_context().get_stage()
        self._shell_attr = None
        if self.cfg["arm_shell"]:
            self._prepare_shell()
        self._block_attr = self.stage.GetPrimAtPath(BLOCK_SHADER).GetAttribute(BLOCK_INPUT)
        if not self._block_attr:
            raise RuntimeError(f"block shader input not found: {BLOCK_SHADER}.{BLOCK_INPUT}")
        self._log(f"colour DR: enabled, seed={self.seed}, palette={self.palette.version}, log={self.log_path}")

    def _prepare_shell(self):
        mat_path = self.cfg["arm_shell_material"]
        shader = self.stage.GetPrimAtPath(f"{mat_path}/Shader")
        self._shell_attr = shader.GetAttribute(SHELL_INPUT)
        if not self._shell_attr:
            raise RuntimeError(f"arm shell shader input not found: {mat_path}/Shader.{SHELL_INPUT}")
        self.shell_baseline = tuple(self._shell_attr.Get())
        # The renderer keeps its own material copy for instanced prototypes, so a colour change would not show on
        # instanced shell meshes: make every arm *visual* group non-instanced (collision groups stay instanced).
        robot = self.stage.GetPrimAtPath(mat_path.split("/Looks/")[0])
        for link in robot.GetChildren():
            vis = link.GetChild("visuals")
            if vis and vis.IsInstanceable():
                vis.SetInstanceable(False)

    def apply_next(self) -> dict | None:
        """Draw and apply the colours for the next session (call at start-up and on every reset)."""
        if not self.enabled:
            return None
        rec = self.sampler.next()
        if "block" in rec:
            rec["block"]["linear"] = [round(x, 6) for x in srgb_to_linear(rec["block"]["rgb"])]
            self._block_attr.Set(Gf.Vec3f(*rec["block"]["linear"]))
        if "arm_shell" in rec and self._shell_attr:
            rec["arm_shell"]["linear"] = [round(x, 6) for x in srgb_to_linear(rec["arm_shell"]["rgb"])]
            self._shell_attr.Set(Gf.Vec3f(*rec["arm_shell"]["linear"]))
        # actual material values read back from the stage (what the renderer uses)
        rec["applied"] = {"block_rgb": [round(float(x), 6) for x in self._block_attr.Get()]}
        if self._shell_attr:
            rec["applied"]["arm_shell_rgb"] = [round(float(x), 6) for x in self._shell_attr.Get()]
        self.records.append(rec)
        self.current = rec
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self._log(f"colour DR: session {rec['session']} block={rec.get('block', {}).get('name')} "
                  f"arm_shell={rec.get('arm_shell', {}).get('name')}")
        return rec

    def block_rgb(self, baseline):
        if self.current and "block" in self.current:
            return tuple(self.current["block"]["rgb"])
        return tuple(baseline)

    def summary(self) -> str:
        if not self.enabled:
            return "Colours: baseline (randomization off)"
        r = self.current or {}
        b, a = r.get("block"), r.get("arm_shell")
        parts = [f"seed {self.seed}"]
        if b:
            parts.append(f"block {b['name']} ({b['color_id']})")
        if a:
            parts.append(f"arm {a['name']} ({a['color_id']})")
        return "Colours: " + ", ".join(parts)

