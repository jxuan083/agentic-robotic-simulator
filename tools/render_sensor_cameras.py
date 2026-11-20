#!/usr/bin/env python3
"""Render the scene's sensor cameras (MuJoCo, offscreen) at a joint state, optionally next to the real frames.

  uv run tools/render_sensor_cameras.py [--scene configs/scenes/lab_20261120.yaml] [--lerobot p l e wf wr g]
      [--real-global img.png] [--real-wrist img.png] [--out outputs/lab_cameras/view]

--global-k W H FX FY CX CY renders the global camera with other intrinsics (same pose), e.g. the 640 x 480
colour stream of the RGB-D recordings. Writes <out>_global.png / <out>_wrist.png (simulation) and, with real frames, <out>_global_cmp.png /
<out>_wrist_cmp.png: real | simulation | 50 % blend. Joint state: LeRobot readings (agentic_sim.real_joints) or
all joints at zero. Blocks stay at the scene's initial layout. Needs an OpenGL context (MUJOCO_GL=egl by default).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from agentic_sim.config import load  # noqa: E402
from agentic_sim.image_io import read_png, write_png  # noqa: E402
from agentic_sim.mujoco_backend.model import build, sensor_camera_names  # noqa: E402
from agentic_sim.real_joints import lerobot_to_shared  # noqa: E402
from agentic_sim.scene_layout import sensor_cameras  # noqa: E402
from agentic_sim.wrist_lens import apply_wrist_lens  # noqa: E402


def render(scene_cfg, arm, jaw):
    sc = build(scene_cfg)
    m, d = sc.model, mujoco.MjData(sc.model)
    for i, a in enumerate(sc.arm_qpos):
        d.qpos[a] = arm[i] + sc.arm_offset[i]
    d.qpos[sc.jaw_qpos] = jaw
    mujoco.mj_forward(m, d)
    ids, specs, out = sensor_camera_names(m), sensor_cameras(scene_cfg), {}
    for key, cid in ids.items():
        w, h = specs[key].get("render_size", (specs[key]["width"], specs[key]["height"]))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w), max(m.vis.global_.offheight, h)
        with mujoco.Renderer(m, h, w) as r:
            r.update_scene(d, camera=cid)
            img = r.render().copy()
        lens = specs[key].get("lens")
        if lens is not None:   # wrist camera, "real" lens mode: resample the wide pinhole to the raw sensor
            img = lens.apply(img, lens.maps(w, h, specs[key]["render_fovy_deg"]))
        out[key] = img
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    ap.add_argument("--lerobot", type=float, nargs=6, default=None, help="pan lift elbow wrist_flex wrist_roll gripper%%")
    ap.add_argument("--real-global")
    ap.add_argument("--real-wrist")
    ap.add_argument("--global-k", type=float, nargs=6, metavar=("W", "H", "FX", "FY", "CX", "CY"),
                    help="render the global camera with these intrinsics (e.g. the 640 x 480 RGB-D recordings)")
    ap.add_argument("--wrist-lens", choices=["real", "pinhole"], default=None,
                    help="wrist image: raw-looking (distorted) or ideal pinhole (default: the scene's lens_mode)")
    ap.add_argument("--out", default="outputs/lab_cameras/view")
    a = ap.parse_args()
    os.chdir(ROOT)
    scene_cfg = apply_wrist_lens(load(a.scene), a.wrist_lens)
    if a.global_k:
        w, h, fx, fy, cx, cy = a.global_k
        g = scene_cfg["sensor_cameras"]["global"]
        g["resolution"] = {"value": [int(w), int(h)]}
        g["intrinsics"] = {"value": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}}
    if a.lerobot:
        arm, jaw = lerobot_to_shared(a.lerobot)
    else:
        arm, jaw = np.zeros(5), 0.0
    imgs = render(scene_cfg, arm, jaw)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for key, img in imgs.items():
        write_png(f"{a.out}_{key}.png", img)
        real = {"global": a.real_global, "wrist": a.real_wrist}.get(key)
        if real:
            r = read_png(real)
            if r.shape != img.shape:
                raise SystemExit(f"{key}: real frame {r.shape[1]}x{r.shape[0]} != camera {img.shape[1]}x{img.shape[0]}")
            blend = (0.5 * r.astype(float) + 0.5 * img.astype(float)).astype(np.uint8)
            write_png(f"{a.out}_{key}_cmp.png", np.concatenate([r, img, blend], 1))
        print(f"{key}: {img.shape[1]}x{img.shape[0]} -> {a.out}_{key}.png")


if __name__ == "__main__":
    main()
