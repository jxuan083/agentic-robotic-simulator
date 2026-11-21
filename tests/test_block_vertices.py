import math
import cv2
import numpy as np
import pytest

from agentic_sim.config import load
from agentic_sim.perception.vertices import CameraGeometry,cube_vertices,estimate_vertices


def _camera(distorted=False):
    # Camera tilted from vertical, OpenCV +Z forward.
    position=np.array([.38,-.1,.45]);target=np.array([.24,.2,.02])
    z=target-position;z/=np.linalg.norm(z)
    x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack((x,y,z));pose[:3,3]=position
    return CameraGeometry(640,480,np.array([[550,0,320],[0,550,240],[0,0,1.]]),
                          np.array([-.15,0,0,0,0]) if distorted else np.zeros(5),pose)


@pytest.mark.parametrize('yaw',[0,.27,.72,1.3])
@pytest.mark.parametrize('distorted',[False,True])
def test_mask_fit_recovers_cube_center_and_eight_inferred_vertices(yaw,distorted):
    cam=_camera(distorted)
    xyz=cube_vertices([.235,.195],yaw,.0295,.004)
    uv=cam.project(xyz)
    mask=np.zeros((480,640),np.uint8)
    cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
    result=estimate_vertices(mask,cam,edge_m=.0295,floor_z_m=.004)
    assert result['status']=='estimated',result
    assert np.linalg.norm(np.array(result['center_xyz_m'])-xyz.mean(0))<.0015
    assert len(result['vertices'])==8
    estimated=np.array([v['xyz_m'] for v in result['vertices']])
    # Vertex labels have cube 90-degree symmetry: compare nearest geometric corners.
    distance=np.linalg.norm(estimated[:,None,:]-xyz[None,:,:],axis=-1)
    assert distance.min(1).max()<.003
    assert all(v['source']=='geometry_inferred' for v in result['vertices'])
    assert any(not v['self_visible'] for v in result['vertices'])
    assert np.allclose(estimated[4:,2] - estimated[:4,2], .0295)


def test_rejects_bad_or_clipped_masks_and_parallel_rays():
    cam=_camera()
    with pytest.raises(ValueError,match='resolution'):
        estimate_vertices(np.zeros((20,20)),cam,edge_m=.03,floor_z_m=0)
    empty=np.zeros((480,640),np.uint8)
    assert estimate_vertices(empty,cam,edge_m=.03,floor_z_m=0)['reason']=='empty_mask'
    empty[:30,:30]=1
    assert estimate_vertices(empty,cam,edge_m=.03,floor_z_m=0)['reason']=='image_boundary_clipped'
    empty[:]=0;cv2.circle(empty,(320,240),30,1,-1)
    assert estimate_vertices(empty,cam,edge_m=.03,floor_z_m=0,min_iou=.95)['status']=='rejected'


def test_wrist_pose_is_derived_from_joint_state_and_requires_measured_input():
    scene=load('configs/scenes/lab_20261120.yaml')
    with pytest.raises(ValueError,match='synchronized'):
        CameraGeometry.from_scene(scene,'wrist')
    a=CameraGeometry.from_scene(scene,'wrist',[0,0,0,0,0],lens_mode='pinhole')
    b=CameraGeometry.from_scene(scene,'wrist',[.2,0,0,0,0],lens_mode='pinhole')
    assert not np.allclose(a.mount_from_cv,b.mount_from_cv)
    assert np.allclose(a.dist,0)
