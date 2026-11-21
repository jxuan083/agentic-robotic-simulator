"""Render a camera RGB image and short video of a reconstructed eight-colour scene.

Run inside the pinned Isaac container, for example:
  docker/isaac/run.sh /isaac-sim/python.sh tools/capture_block_scene.py --output outputs/block-scene-capture

This reconstructs a static, render-only table/tray/block scene from shared geometry and palette.
It contains no SO-101 arm or motion and does not read simulator state to create detections or labels.
Isaac Sim is used only as a local renderer; this does not validate the complete simulator scene.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from agentic_sim.capture_utils import capture_output_paths, validate_capture_options, write_video_frames

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="outputs/block-scene-capture")
parser.add_argument("--frames", type=int, default=48)
parser.add_argument("--fps", type=float, default=16.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()
try:
    validate_capture_options(args.frames, args.fps, args.width, args.height)
except ValueError as exc:
    parser.error(str(exc))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": args.width,
                     "height": args.height, "anti_aliasing": 3})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from agentic_sim.config import load
from agentic_sim.scene_layout import TRAYS, block_layout, table_box, tray_parts


def rgb_material(name: str, rgb: tuple[float, float, float]):
    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def cube(path: str, size: tuple[float, float, float], pos: tuple[float, float, float],
         rgb: tuple[float, float, float], yaw: float = 0.0):
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateZOp().Set(float(math.degrees(yaw)))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI(prim).Bind(rgb_material(path.strip("/").replace("/", "_"), rgb))
    return prim


try:
    image_path, video_path, info_path = capture_output_paths(args.output)
    output = image_path.parent

    scene_cfg = load("configs/scenes/lab_current_minimal.yaml")
    print(f"[capture] output={output} frames={args.frames}", flush=True)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")

    # Neutral studio environment and soft key light.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1100.0)
    key = UsdLux.RectLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(1800.0)
    key.CreateWidthAttr(1.4)
    key.CreateHeightAttr(1.4)
    UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.22, -0.05, 1.25))

    tb = table_box(scene_cfg)
    # The source table is 75 cm thick for physics stability. A camera sees its top surface, so
    # the render-only support uses a thin slab with the same footprint and the same z=0 top.
    surface_size = (tb["size"][0], tb["size"][1], 0.02)
    cube("/World/Table", surface_size, (tb["pos"][0], tb["pos"][1], -0.01), tb["color"])
    for tray_name in TRAYS:
        parts, info = tray_parts(scene_cfg["trays"][tray_name])
        for part_name, (size, pos) in parts.items():
            cube(f"/World/{tray_name}/{part_name}", size, pos, info["color"], info["yaw"])

    layout = block_layout(scene_cfg, n_blocks=8)
    for index, (pos, rgb) in enumerate(zip(layout.init_pos, layout.display_rgb)):
        cube(f"/World/Blocks/block_{index}", (layout.edge,) * 3, pos, rgb)
    print("[capture] rendered scene geometry", flush=True)

    # Render the scene from a camera view that includes the storage tray and the placement tray.
    camera = rep.create.camera(position=(0.76, -0.58, 0.88), look_at=(0.24, 0.0, 0.015), focal_length=32.0)
    render_product = rep.create.render_product(camera, (args.width, args.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    camera_prim = next((p for p in stage.Traverse() if p.GetTypeName() == "Camera"), None)
    if camera_prim is None:
        raise RuntimeError("Replicator did not create a USD camera prim")
    print("[capture] attached RGB annotator", flush=True)

    # Warm up renderer and wait for a valid RGB frame.
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=4)
        app.update()
        data = rgb_annotator.get_data()
        if data is not None and len(data.shape) == 3:
            break
    if data is None or len(data.shape) != 3:
        raise RuntimeError("Isaac Replicator did not produce an RGB frame")
    print(f"[capture] got frame shape={data.shape}", flush=True)

    import cv2
    first = [None]

    def rendered_bgr_frames():
        for frame_index in range(args.frames):
            # Record repeated RGB observations of the static scene from the camera.
            rep.orchestrator.step(rt_subframes=2)
            app.update()
            rgba = np.asarray(rgb_annotator.get_data())
            if rgba.ndim != 3 or rgba.shape[0] != args.height or rgba.shape[1] != args.width:
                raise RuntimeError(f"Unexpected RGB frame shape {rgba.shape}")
            rgb = rgba[:, :, :3].astype(np.uint8, copy=False)
            if first[0] is None:
                first[0] = rgb.copy()
                print(f"[capture] first frame shape={rgb.shape}", flush=True)
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    written_frames = write_video_frames(cv2, video_path, args.fps, args.width, args.height,
                                        rendered_bgr_frames())
    if written_frames != args.frames:
        raise RuntimeError(f"Expected {args.frames} frames but wrote {written_frames}")
    if not cv2.imwrite(str(image_path), cv2.cvtColor(first[0], cv2.COLOR_RGB2BGR)):
        raise RuntimeError("OpenCV could not write block_scene.png")
    info_path.write_text(
        f"Rendered locally with Isaac Sim headless; image/video are RGB camera outputs.\n"
        f"image_size={args.width}x{args.height}\nframes={args.frames}\nfps={args.fps}\n"
        f"scene=reconstructed static render-only scene based on lab_current_minimal; block_count=8; colors=palette design values (provisional)\n"
        f"Contains no SO-101 arm or motion; not full simulator validation. No detections were generated.\n"
    )
    if video_path.stat().st_size < 1024:
        raise RuntimeError(f"Video output is unexpectedly small: {video_path.stat().st_size} bytes")
    print(f"[capture] wrote {output / 'block_scene.png'} and {video_path}", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
