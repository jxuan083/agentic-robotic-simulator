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
    """One worker owns the detector, including its temporal tracker."""
    def __init__(self, edge_m, floor_z_m):
        from .classical import ClassicalBlockPerception, ClassicalConfig
        self.detector = ClassicalBlockPerception(ClassicalConfig.from_yaml('configs/perception-classical.yaml'))
        self.edge = edge_m
        self.floor = floor_z_m

    def predict(self, rgb, camera, frame_number, captured_at):
        from .vertices import estimate_vertices, annotate_vertices
        frame = self.detector.predict(rgb, timestamp_s=captured_at,
                                      timestamp_source='capture_monotonic', track=False)
        records, fits = [], []
        for d in frame.detections:
            geometry = estimate_vertices(d.mask, camera, edge_m=self.edge, floor_z_m=self.floor)
            fits.append((d.detection_id, geometry))
            records.append({'detection_id': d.detection_id, 'color': d.color, 'geometry': geometry})
        return annotate_vertices(rgb, fits), {
            'frame_number': frame_number, 'captured_at_monotonic_s': captured_at,
            'camera': camera.name, 'coordinate_frame': 'mount', 'units': 'm',
            'pose_source': 'same_render_frame_view_matrix',
            'mount_from_cv': camera.mount_from_cv.tolist(), 'intrinsics': camera.k.tolist(),
            'distortion': camera.dist.tolist(), 'detections': records,
            'id_scope': 'current_frame_only', 'method': 'classical_RGB_and_cube_geometry',
        }
