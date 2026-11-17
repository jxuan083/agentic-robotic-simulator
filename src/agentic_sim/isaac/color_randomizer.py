"""Apply colour domain randomization to the Isaac scene (arm shell + blocks). Visual attributes only.

Disabled: nothing is drawn; the scene shows its baseline colours (restored when randomization is switched off).
The first time it is enabled: make the arm *visual* groups non-instanced, because the renderer keeps its own material
copy for instanced prototypes and would not show the colour change (collisions and physics are separate prims). The
whole shell, gripper fingers included, shares one colour; motors use another material and are never changed.
Enabled, at start-up and every reset: draw the next colours (ColorSampler) and write them to the block materials and
the arm shell material, and append one JSON line to the log.

configure(enabled, seed) switches randomization on / off or restarts it with another seed while the scene is up
(the panel's settings); the change shows from the next apply_next() (reset).
"""

from __future__ import annotations

import json
import os

import omni.usd
from pxr import Gf

from agentic_sim.color_dr import ColorSampler, Palette, srgb_to_linear
from agentic_sim.config import REPO_ROOT, load

BLOCK_INPUT = "inputs:diffuseColor"            # UsdPreviewSurface (spawned by Isaac Lab)
SHELL_INPUT = "inputs:diffuse_color_constant"  # OmniPBR (SO-101 USD)


def block_shader_path(block_prim: str) -> str:
    return f"{block_prim}/geometry/material/Shader"


class ColorRandomizer:
    def __init__(self, cfg_path: str, enabled: bool | None = None, seed: int | None = None, log_path: str | None = None,
                 log=print, block_prims=("/World/Block",)):
        self.cfg = load(cfg_path)
        self._log = log
        self.records: list[dict] = []
        self.current: dict | None = None
        self.enabled = False
        self.seed = int(self.cfg["seed"] if seed is None else seed)
        self.palette = Palette.load(self.cfg["palette"])
        self.sampler = None
        p = log_path or self.cfg.get("log")
        self.log_path = None if not p else (p if os.path.isabs(p) else os.path.join(REPO_ROOT, p))
        self._log_opened = False
        self.stage = omni.usd.get_context().get_stage()
        self._block_attrs = []
        for prim in block_prims:
            attr = self.stage.GetPrimAtPath(block_shader_path(prim)).GetAttribute(BLOCK_INPUT)
            if not attr:
                raise RuntimeError(f"block shader input not found: {block_shader_path(prim)}.{BLOCK_INPUT}")
            self._block_attrs.append(attr)
        self.block_baselines = [tuple(a.Get()) for a in self._block_attrs]
        self._shell_attr = None
        self.shell_baseline = None
        if self.cfg["arm_shell"]:
            mat_path = self.cfg["arm_shell_material"]
            self._shell_attr = self.stage.GetPrimAtPath(f"{mat_path}/Shader").GetAttribute(SHELL_INPUT)
            if not self._shell_attr:
                raise RuntimeError(f"arm shell shader input not found: {mat_path}/Shader.{SHELL_INPUT}")
            self.shell_baseline = tuple(self._shell_attr.Get())
        self._shell_prepared = False
        self.configure(bool(self.cfg["enabled"] if enabled is None else enabled), self.seed)

    def _prepare_shell(self):
        # The renderer keeps its own material copy for instanced prototypes, so a colour change would not show on
        # instanced shell meshes: make every arm *visual* group non-instanced (collision groups stay instanced).
        if self._shell_prepared or self._shell_attr is None:
            return
        robot = self.stage.GetPrimAtPath(self.cfg["arm_shell_material"].split("/Looks/")[0])
        for link in robot.GetChildren():
            vis = link.GetChild("visuals")
            if vis and vis.IsInstanceable():
                vis.SetInstanceable(False)
        self._shell_prepared = True

    def configure(self, enabled: bool, seed: int | None = None) -> bool:
        """Switch randomization on / off or restart it with another seed. Returns True if anything changed."""
        seed = self.seed if seed is None else int(seed)
        enabled = bool(enabled)
        if enabled == self.enabled and (seed == self.seed or not enabled) and (self.sampler is not None or not enabled):
            self.seed = seed
            return False
        self.enabled, self.seed = enabled, seed
        if not self.enabled:
            self.sampler = None
            self.current = None
            for attr, base in zip(self._block_attrs, self.block_baselines):
                attr.Set(Gf.Vec3f(*base))
            if self._shell_attr is not None and self.shell_baseline is not None:
                self._shell_attr.Set(Gf.Vec3f(*self.shell_baseline))
            self._log("colour DR: off (baseline colours)")
            return True
        if self.cfg["arm_shell"]:
            self._prepare_shell()
        self.sampler = ColorSampler(self.palette, self.seed, bool(self.cfg["block"]), bool(self.cfg["arm_shell"]))
        if self.log_path and not self._log_opened:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            open(self.log_path, "w").close()
            self._log_opened = True
        self._log(f"colour DR: enabled, seed={self.seed}, palette={self.palette.version}, log={self.log_path}")
        return True

    def apply_next(self) -> dict | None:
        """Draw and apply the colours for the next session (call at start-up and on every reset)."""
        if not self.enabled:
            return None
        rec = self.sampler.next(len(self._block_attrs))
        blocks = ([rec["block"]] + rec.get("extra_blocks", [])) if "block" in rec else []
        for attr, b in zip(self._block_attrs, blocks):
            b["linear"] = [round(x, 6) for x in srgb_to_linear(b["rgb"])]
            attr.Set(Gf.Vec3f(*b["linear"]))
        if "arm_shell" in rec and self._shell_attr:
            rec["arm_shell"]["linear"] = [round(x, 6) for x in srgb_to_linear(rec["arm_shell"]["rgb"])]
            self._shell_attr.Set(Gf.Vec3f(*rec["arm_shell"]["linear"]))
        # actual material values read back from the stage (what the renderer uses)
        rec["applied"] = {"block_rgb": [round(float(x), 6) for x in self._block_attrs[0].Get()]}
        if len(self._block_attrs) > 1:
            rec["applied"]["blocks_rgb"] = [[round(float(x), 6) for x in a.Get()] for a in self._block_attrs]
        if self._shell_attr:
            rec["applied"]["arm_shell_rgb"] = [round(float(x), 6) for x in self._shell_attr.Get()]
        self.records.append(rec)
        self.current = rec
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self._log(f"colour DR: session {rec['session']} blocks={[b['name'] for b in blocks]} "
                  f"arm_shell={rec.get('arm_shell', {}).get('name')}")
        return rec

    def block_rgbs(self, baselines) -> list:
        """Display colour (sRGB) of every block: the drawn colours, or the given baselines when off."""
        if self.current and "block" in self.current:
            drawn = [self.current["block"]] + self.current.get("extra_blocks", [])
            return [tuple(b["rgb"]) for b in drawn]
        return [tuple(b) for b in baselines]

    def summary(self) -> str:
        if not self.enabled:
            return "baseline colours (randomization off)"
        r = self.current or {}
        b, a = r.get("block"), r.get("arm_shell")
        parts = [f"draw #{r.get('session', 0)}"]
        if b:
            names = [b["name"]] + [x["name"] for x in r.get("extra_blocks", [])]
            parts.append("block " + "/".join(names) if len(names) <= 3 else f"{len(names)} block colours")
        if a:
            parts.append(f"arm {a['name']}")
        return ", ".join(parts)
