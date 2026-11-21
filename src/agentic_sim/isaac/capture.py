"""RGB read-back of the scene's sensor cameras for demonstration capture (issue #13).

One render product per camera prim (independent of the docked viewer windows) at the camera's own resolution, read
through the Replicator "rgb" annotator after the frame's render. The wrist prim carries the lens model in "real" lens
mode, so the recorded wrist image is the raw-looking (distorted) one, like the lab recordings.
"""

from __future__ import annotations

import time

import numpy as np


class CameraGrabber:
    def __init__(self, cameras: dict, log=print):
        """cameras: key -> (camera prim path, width, height) (SceneHandles.sensor_cameras)."""
        import omni.replicator.core as rep
        self.log = log
        self.items = {}
        for key, (path, w, h) in cameras.items():
            rp = rep.create.render_product(path, (int(w), int(h)))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([rp])
            self.items[key] = (annot, rp, int(w), int(h))
            log(f"capture camera {key}: {path} {w}x{h}")
        self.empty = {k: 0 for k in self.items}

    def grab(self) -> dict:
        """key -> (HxWx3 uint8, wall ns) or None while the annotator has no image of the right size yet."""
        out = {}
        for key, (annot, _rp, w, h) in self.items.items():
            data = annot.get_data()
            arr = None if data is None else np.asarray(data)
            if arr is None or arr.ndim != 3 or arr.shape[0] != h or arr.shape[1] != w:
                self.empty[key] += 1
                out[key] = None
                continue
            out[key] = (arr[..., :3].copy(), time.time_ns())
        return out

    def close(self):
        for annot, rp, _w, _h in self.items.values():
            try:
                annot.detach([rp])
                rp.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.items = {}


def camera_intrinsics(stage, path: str, width: int, height: int) -> dict:
    """The projection a camera prim is actually rendered with, read back from USD (not from the scene file).

    With the RTX OpenCV pinhole lens model applied ("real" wrist lens mode) the image follows that model: its fx, fy,
    cx, cy and k1, k2, p1, p2, k3. Otherwise it is the plain pinhole of the prim: fx = focalLength * width /
    horizontalAperture (fy likewise); Omniverse ignores aperture offsets, so the principal point is the image centre."""
    prim = stage.GetPrimAtPath(path)
    model = prim.GetAttribute("omni:lensdistortion:model")
    if model and model.Get() == "opencvPinhole":
        base = "omni:lensdistortion:opencvPinhole:"
        val = {n: round(float(prim.GetAttribute(base + n).Get()), 6) for n in ("fx", "fy", "cx", "cy", "k1", "k2", "p1",
                                                                                 "p2", "k3")}
        return {"model": "opencv_pinhole", "fx": val["fx"], "fy": val["fy"], "cx": val["cx"], "cy": val["cy"],
                "distortion_opencv": [val[n] for n in ("k1", "k2", "p1", "p2", "k3")],
                "intrinsics_source": "read from the camera prim: RTX OpenCV pinhole lens model attributes"}
    focal = float(prim.GetAttribute("focalLength").Get())
    hap = float(prim.GetAttribute("horizontalAperture").Get())
    vap = float(prim.GetAttribute("verticalAperture").Get())
    return {"model": "pinhole", "fx": round(focal * width / hap, 6), "fy": round(focal * height / vap, 6),
            "cx": width / 2, "cy": height / 2, "distortion_opencv": None,
            "intrinsics_source": "read from the camera prim: focalLength and apertures (no lens distortion)"}
