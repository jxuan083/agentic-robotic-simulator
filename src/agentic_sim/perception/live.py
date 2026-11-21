"""Pure image processing for the live preview; no simulator object state is read."""
from __future__ import annotations

import numpy as np


def add_arguments(parser):
    parser.add_argument('--live-perception', action='store_true', help='show live RGB block corners and coordinates')
    parser.add_argument('--perception-camera', choices=('both', 'wrist', 'global'), default='both')
    parser.add_argument('--perception-hz', type=float, default=2.0, help='maximum inference updates per second (default: 2)')
    parser.add_argument('--perception-json', default=None,
                        help='optional path periodically overwritten with latest full estimates')


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
                fitted = [pg for pg in (fit(p) for p in parts if (p.mask & d.mask).sum() > 0.5 * p.mask.sum())
                          if pg[1]['status'] == 'estimated']
                out.extend(fitted or [(d, g)])
            found = out
        return found

    def predict(self, rgb, camera, frame_number, captured_at):
        from .vertices import annotate_vertices
        records, fits = [], []
        for i, (d, geometry) in enumerate(self._fits(rgb, camera, captured_at)):
            fits.append((i, geometry))
            records.append({'detection_id': i, 'color': d.color, 'bbox_xyxy': list(d.bbox_xyxy),
                            'geometry': geometry})
        return annotate_vertices(rgb, fits), {
            'frame_number': frame_number, 'captured_at_monotonic_s': captured_at,
            'camera': camera.name, 'image_size': [camera.width, camera.height],
            'coordinate_frame': 'mount', 'units': 'm',
            'pose_source': 'same_render_frame_view_matrix',
            'mount_from_cv': camera.mount_from_cv.tolist(), 'intrinsics': camera.k.tolist(),
            'distortion': camera.dist.tolist(), 'detections': records,
            'id_scope': 'current_frame_only', 'method': 'classical_RGB_and_cube_geometry',
        }
