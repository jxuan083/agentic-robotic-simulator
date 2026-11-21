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
