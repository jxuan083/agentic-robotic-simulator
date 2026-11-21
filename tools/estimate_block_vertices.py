#!/usr/bin/env python3
"""Fit and draw eight cube corners from one calibrated camera RGB frame.

Wrist input requires the measured five shared joint angles (radians) from the
same exposure, or a JSON snapshot containing joint_positions_rad. This tool never
loads simulator block poses. Assumes upright blocks on the storage tray floor.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image')
    ap.add_argument('--scene',default='configs/scenes/lab_20261120.yaml')
    ap.add_argument('--camera',choices=('wrist','global'),default='wrist')
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--joints',type=float,nargs=5,metavar=('PAN','LIFT','ELBOW','WRIST','ROLL'))
    group.add_argument('--joint-state',help='JSON with joint_positions_rad from the same exposure')
    ap.add_argument('--wrist-lens',choices=('real','pinhole'),default=None)
    ap.add_argument('--config',default='configs/perception-classical.yaml')
    ap.add_argument('--min-iou',type=float,default=.82)
    ap.add_argument('--output',default='outputs/block-vertices')
    a=ap.parse_args(argv)
    try:
        import cv2
        from agentic_sim.config import load, v
        from agentic_sim.scene_layout import tray_parts
        from agentic_sim.perception.classical import ClassicalBlockPerception, ClassicalConfig
        from agentic_sim.perception.vertices import CameraGeometry, estimate_vertices, annotate_vertices
        from agentic_sim.perception.cli import _write_frame
        joints=a.joints
        joint_meta={}
        if a.joint_state:
            joint_meta=json.loads(Path(a.joint_state).read_text())
            joints=joint_meta['joint_positions_rad']
        scene=load(a.scene)
        camera=CameraGeometry.from_scene(scene,a.camera,joints,a.wrist_lens)
        bgr=cv2.imread(a.image)
        if bgr is None:
            raise ValueError(f'could not read RGB image: {a.image}')
        if bgr.shape[:2]!=(camera.height,camera.width):
            raise ValueError('input image must be the full calibrated camera resolution')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        frame=ClassicalBlockPerception(ClassicalConfig.from_yaml(a.config)).predict(rgb,timestamp_s=joint_meta.get('timestamp_s'))
        edge=float(v(scene['block']['edge']))
        floor=tray_parts(scene['trays']['storage_pink'])[1]['floor_top']
        fits=[(d.detection_id,estimate_vertices(d.mask,camera,edge_m=edge,floor_z_m=floor,min_iou=a.min_iou)) for d in frame.detections]
        output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
        _write_frame(frame,output,rgb,cv2,False)
        record=json.loads((output/'frame_0.json').read_text())
        by_id=dict(fits)
        for d in record['detections']:
            d['geometry']=by_id[d['detection_id']]
        record['camera_geometry']={'camera':a.camera,'mount_from_cv':camera.mount_from_cv.tolist(),
            'intrinsics':camera.k.tolist(),'distortion':camera.dist.tolist(),'joint_positions_rad':joints,
            'joint_sync':'caller_must_supply_same_exposure','calibration':'scene_estimated_or_provisional',
            'source_image':str(Path(a.image).resolve())}
        (output/'vertices.json').write_text(json.dumps(record,indent=2,allow_nan=False))
        overlay=annotate_vertices(rgb,fits)
        if not cv2.imwrite(str(output/'vertices.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)):
            raise OSError('could not write vertex overlay')
        successes=sum(r['status']=='estimated' for _,r in fits)
        print(f'{len(fits)} candidates; {successes} fitted cubes; {len(fits)-successes} rejected. Output: {output}')
        for i,r in fits:
            print(f"  {i}: {r['status']} iou={r.get('fit_iou')} reason={r.get('reason')}")
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,ImportError) as exc:
        print(f'vertex estimation error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
