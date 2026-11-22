"""Pure image processing for the live preview; no simulator object state is read."""
from __future__ import annotations

import numpy as np


from agentic_sim.perception_cli import add_arguments  # noqa: E402,F401  (kept for existing callers)


def rendered_camera(base, view, projection, width, height):
    """Use the captured frame's matrices, rather than joints from a later tick.

    Kit matrices use row vectors and camera axes +X/+Y/-Z. Our geometry uses
    column vectors and OpenCV +X/-Y/+Z. Distorted RTX cameras retain scene K.
    """
    from .vertices import CameraGeometry
    if (width, height) != (base.width, base.height):
        raise ValueError('capture resolution differs from calibrated sensor resolution')
    pose = np.linalg.inv(np.asarray(view, dtype=float).reshape(4, 4).T) @ np.diag([1., -1., -1., 1.])
    k = base.k.copy()
    if not np.any(base.dist):
        p = np.asarray(projection, dtype=float).reshape(4, 4).T
        k = np.array([[p[0, 0]*width/2, 0, (1-p[0, 2])*width/2],
                      [0, p[1, 1]*height/2, (1+p[1, 2])*height/2], [0, 0, 1.]])
    return CameraGeometry(width, height, k, base.dist.copy(), pose, base.name)


class LiveEstimator:
    """One worker owns the detector, including its temporal tracker.

    Wrist camera: close up, blocks of different colours often touch in the image and merge into one rejected
    silhouette, so a rejected silhouette is retried split per hue band (ClassicalConfig.split_by_hue), then with
    light / dark green separated by value (split_green_by_value; splitting greens everywhere cut 5 more Isaac wrist
    fits than it recovered). Splitting
    every silhouette raised accepted wrist fits on the saved rendered evaluation frames from 12 to 44 of 80 (Isaac)
    and 6 to 14 of 24 (MuJoCo) with no false positives or colour errors, but cut yellow blocks whose shaded side
    falls in the green band, and lost 3 of 80 global fits (Isaac), so the global camera is not split."""
    def __init__(self, edge_m, floor_z_m):
        from .classical import ClassicalBlockPerception, ClassicalConfig
        cfg = ClassicalConfig.from_yaml('configs/perception-classical.yaml')
        self.detector = ClassicalBlockPerception(cfg)
        self.split_detectors = []          # finer and finer: hue bands, then light / dark green by value
        for by_value in (False, True):
            split = ClassicalConfig.from_yaml('configs/perception-classical.yaml')
            split.split_by_hue, split.split_green_by_value = True, by_value
            self.split_detectors.append(ClassicalBlockPerception(split))
        self.edge = edge_m
        self.floor = floor_z_m

    def _fits(self, rgb, camera, captured_at):
        """(detection, geometry) per candidate. Wrist: a silhouette that does not fit is replaced by its parts in the
        next finer segmentation that do fit (touching blocks of different colours), level by level; a block whose
        shaded side falls in another band keeps its whole silhouette, which fits."""
        from .vertices import estimate_vertices
        kw = dict(timestamp_s=captured_at, timestamp_source='capture_monotonic', track=False)

        def fit(d):
            return d, estimate_vertices(d.mask, camera, edge_m=self.edge, floor_z_m=self.floor)

        found = [fit(d) for d in self.detector.predict(rgb, **kw).detections]
        if camera.name != 'wrist':
            return found
        for detector in self.split_detectors:
            if all(g['status'] == 'estimated' for _, g in found):
                break
            parts = detector.predict(rgb, **kw).detections
            out = []
            for d, g in found:
                if g['status'] == 'estimated':
                    out.append((d, g))
                    continue
                tried = [fit(p) for p in parts if (p.mask & d.mask).sum() > 0.5 * p.mask.sum()]
                if any(pg[1]['status'] == 'estimated' for pg in tried):
                    out.extend(tried)     # parts that still do not fit stay as rejected (their colour is evidence)
                else:
                    out.append((d, g))
            found = out
        return found

    @staticmethod
    def capture_only(rgb, camera, frame_number, captured_at):
        """--perception-stage capture: the same capture path without estimation (benchmarks the capture cost)."""
        return rgb, {'frame_number': frame_number, 'captured_at_monotonic_s': captured_at, 'camera': camera.name,
                     'image_size': [camera.width, camera.height], 'detections': [], 'method': 'capture_only'}

    def worker(self, stage='estimate', delay_s=0.0, fault=None):
        """The function the perception worker runs per captured image: estimate (default) or capture only; delay_s
        (tests) holds each result back so late results can be checked (they keep their capture time); fault
        (tests, an InjectedFault) makes the estimates of this camera fail for a while."""
        fn = self.predict if stage == 'estimate' else self.capture_only
        if delay_s:
            inner = fn

            def fn(*a):
                import time
                out = inner(*a)
                time.sleep(delay_s)
                return out
        return fn if fault is None else fault.wrap(fn)

    def predict(self, rgb, camera, frame_number, captured_at):
        from .vertices import annotate_vertices
        records, fits = [], []
        for i, (d, geometry) in enumerate(self._fits(rgb, camera, captured_at)):
            fits.append((i, geometry))
            records.append({'detection_id': i, 'color': d.color, 'bbox_xyxy': list(d.bbox_xyxy),
                            'area_px': int(d.mask.sum()), 'geometry': geometry})
        return annotate_vertices(rgb, fits), {
            'frame_number': frame_number, 'captured_at_monotonic_s': captured_at,
            'camera': camera.name, 'image_size': [camera.width, camera.height],
            'coordinate_frame': 'mount', 'units': 'm',
            'pose_source': 'same_render_frame_view_matrix',
            'mount_from_cv': camera.mount_from_cv.tolist(), 'intrinsics': camera.k.tolist(),
            'distortion': camera.dist.tolist(), 'detections': records,
            'id_scope': 'current_frame_only', 'method': 'classical_RGB_and_cube_geometry',
        }


class InjectedFault:
    """Test only (--perception-fault CAMERA:AFTER:DURATION): the estimates of one camera raise for DURATION seconds,
    starting AFTER seconds after its first valid result; afterwards they succeed again. Checks that a failed capture
    or estimate takes the camera cell back to the raw image, and that only a new valid result brings the
    annotation back."""

    def __init__(self, camera: str, after_s: float, duration_s: float):
        self.camera, self.after_s, self.duration_s = camera, float(after_s), float(duration_s)
        self.first_ok = None

    @classmethod
    def parse(cls, text):
        from agentic_sim.perception_cli import parse_fault
        spec = parse_fault(text)
        return None if spec is None else cls(*spec)

    def window(self):
        """(start, end) host monotonic time of the failure, or None before the first valid result."""
        if self.first_ok is None:
            return None
        start = self.first_ok + self.after_s
        return start, start + self.duration_s

    def wrap(self, fn):
        import time

        def run(*a):
            w = self.window()
            if w is not None and w[0] <= time.monotonic() < w[1]:
                raise RuntimeError('injected test fault (--perception-fault)')
            out = fn(*a)
            if self.first_ok is None:
                self.first_ok = time.monotonic()
            return out
        return run
