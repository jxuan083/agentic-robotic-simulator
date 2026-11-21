"""Upright, known-size cube fitting from a calibrated RGB silhouette.

Inputs are RGB-derived masks and camera calibration / synchronized measured joints.
No simulator block pose or initial layout is used. All eight corners are inferred,
including corners on visible faces; this is not a learned keypoint detector.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from agentic_sim.config import load, v
from agentic_sim.kinematics import So101Chain, quat_wxyz_to_mat, rot_z
from agentic_sim.scene_layout import sensor_cameras
from agentic_sim.wrist_lens import apply_wrist_lens

# Canonical local frame: bottom counterclockwise viewed from +Z, then matching top.
SIGNS = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0],
                  [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float)
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


@dataclass
class CameraGeometry:
    width: int
    height: int
    k: np.ndarray
    dist: np.ndarray
    mount_from_cv: np.ndarray
    name: str = 'camera'

    def __post_init__(self):
        self.k = np.asarray(self.k, float)
        self.dist = np.asarray(self.dist, float)
        self.mount_from_cv = np.asarray(self.mount_from_cv, float)
        if (self.k.shape != (3,3) or self.mount_from_cv.shape != (4,4)
                or self.dist.size not in (4,5,8) or self.width <= 0 or self.height <= 0
                or not all(np.isfinite(a).all() for a in (self.k,self.dist,self.mount_from_cv))
                or self.k[0,0] <= 0 or self.k[1,1] <= 0):
            raise ValueError('invalid camera calibration')
        if not np.allclose(self.k[2], [0,0,1]) or not np.allclose(self.mount_from_cv[3], [0,0,0,1]):
            raise ValueError('camera intrinsics / homogeneous transform have invalid final rows')
        r=self.mount_from_cv[:3,:3]
        if not np.allclose(r.T@r,np.eye(3),atol=.003) or not np.isclose(np.linalg.det(r),1,atol=.003):
            raise ValueError('camera rotation must be right handed and orthonormal')

    @classmethod
    def from_scene(cls, scene, name='wrist', joints_rad=None, lens_mode=None):
        scene=apply_wrist_lens(scene,lens_mode)
        specs=sensor_cameras(scene)
        if name not in specs:
            raise ValueError(f'no {name} camera in scene')
        spec=specs[name]
        pose=np.eye(4)
        if name=='global':
            pose[:3,:3]=spec['rot_cv'];pose[:3,3]=spec['pos']
        else:
            if joints_rad is None or np.shape(joints_rad)!=(5,) or not np.isfinite(joints_rad).all():
                raise ValueError('wrist camera requires five synchronized measured shared joint angles in radians')
            if spec['pos_in_gripper'] is None:
                raise ValueError('wrist camera mount transform is missing')
            chain=So101Chain.from_dict(load('configs/so101_chain.yaml'))
            chain.base_pos=np.array(v(scene['robot']['base_pos']),float)
            pose[:3,:3]=quat_wxyz_to_mat(spec['quat_in_gripper']) @ np.diag([1.,-1.,-1.])
            pose[:3,3]=spec['pos_in_gripper']
            pose=chain.gripper_pose_mount(joints_rad) @ pose
        lens=spec.get('lens')
        fx,fy,cx,cy=(getattr(lens,key) if lens is not None else spec[key] for key in ('fx','fy','cx','cy'))
        return cls(spec['width'],spec['height'],np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),
                   np.array(lens.dist if lens is not None else [0.]*5),pose,name)

    def project(self, xyz, *, distorted=True):
        camera=(np.asarray(xyz)-self.mount_from_cv[:3,3]) @ np.linalg.inv(self.mount_from_cv[:3,:3]).T
        if (camera[:,2] <= .001).any():
            raise ValueError('cube lies behind or too near camera')
        uv,_=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),self.k,
                               self.dist if distorted else np.zeros(5))
        return uv.reshape(-1,2)

    def intersect_plane(self, uv, z):
        # Called with pinhole pixels after mask undistortion.
        ray=np.linalg.solve(self.k,np.array([*uv,1.]))
        direction=self.mount_from_cv[:3,:3] @ ray
        origin=self.mount_from_cv[:3,3]
        if abs(direction[2])<1e-7:
            raise ValueError('view ray is parallel to support plane')
        distance=(z-origin[2])/direction[2]
        if distance<=0:
            raise ValueError('support plane is behind the view ray')
        return origin+distance*direction


def cube_vertices(xy, yaw, edge, floor_z):
    points=SIGNS.copy()
    points[:,:2]*=edge/2
    points[:,2]=points[:,2]*edge+floor_z
    points=points @ rot_z(yaw).T
    points[:,:2]+=xy
    return points


def estimate_vertices(mask, camera, *, edge_m, floor_z_m, min_iou=.82):
    """Fit x/y/yaw by silhouette IoU; reject clipped/poorly explained candidates.

    Assumes a fully visible, unstacked upright cube on the supplied horizontal
    support. A high IoU is fit quality, not calibrated pose confidence.
    """
    def reject(reason, **kw):
        return {'status':'rejected','reason':reason,'vertices':[],**kw}
    if (not math.isfinite(edge_m) or edge_m<=0 or not math.isfinite(floor_z_m)
            or not 0<min_iou<=1):
        raise ValueError('invalid edge, support height or IoU threshold')
    mask=np.asarray(mask,bool)
    if mask.shape!=(camera.height,camera.width):
        raise ValueError('mask resolution must match camera calibration; never use a resized GUI inset')
    if not mask.any():
        return reject('empty_mask')
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        return reject('image_boundary_clipped')
    binary=mask.astype(np.uint8)
    if np.any(camera.dist):
        mx,my=cv2.initUndistortRectifyMap(camera.k,camera.dist,None,camera.k,(camera.width,camera.height),cv2.CV_32FC1)
        binary=cv2.remap(binary,mx,my,cv2.INTER_NEAREST)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=1:
        return reject('mask_is_not_one_component')
    yy,xx=np.nonzero(binary)
    if len(xx)<50:
        return reject('too_few_pixels')
    if binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any():
        return reject('undistorted_boundary_clipped')
    try:
        center=camera.intersect_plane((float(xx.mean()),float(yy.mean())),floor_z_m+edge_m/2)[:2]
    except ValueError as exc:
        return reject(str(exc))
    pad=max(int(np.ptp(xx))+1,int(np.ptp(yy))+1)
    x0,y0=max(0,int(xx.min())-pad),max(0,int(yy.min())-pad)
    x1,y1=min(camera.width,int(xx.max())+pad+1),min(camera.height,int(yy.max())+pad+1)
    target=binary[y0:y1,x0:x1]!=0
    scratch=np.zeros(target.shape,np.uint8)
    def score(p):
        try:
            uv=camera.project(cube_vertices(p[:2],p[2],edge_m,floor_z_m),distorted=False)
        except ValueError:
            return 0.
        if not np.isfinite(uv).all() or (uv.min(0)<[x0,y0]).any() or (uv.max(0)>[x1-1,y1-1]).any():
            return 0.
        hull=cv2.convexHull(np.rint(uv-[x0,y0]).astype(np.int32))
        scratch.fill(0);cv2.fillConvexPoly(scratch,hull,1)
        model=scratch!=0
        return np.count_nonzero(model & target)/max(1,np.count_nonzero(model | target))
    def refine(p,levels):
        best=score(p)
        for step,angle in levels:
            for _ in range(12):
                candidates=[]
                for axis,delta in ((0,step),(1,step),(2,angle)):
                    for sign in (-1,1):
                        q=p.copy();q[axis]+=sign*delta
                        candidates.append((score(q),q))
                value,q=max(candidates,key=lambda c:c[0])
                if value<=best+1e-8:
                    break
                best,p=value,q
        return best,p
    starts=[]
    for yaw in np.arange(0,math.pi/2,math.pi/12):
        starts.append(refine(np.array([*center,yaw]),[(edge_m/8,math.pi/36),(edge_m/24,math.pi/90)]))
    fit,p=max(starts,key=lambda c:c[0])
    fit,p=refine(p,[(edge_m/64,math.pi/180),(edge_m/128,math.pi/360)])
    if fit<min_iou:
        return reject('silhouette_fit_below_threshold',fit_iou=float(fit))
    yaw=float(p[2]%(math.pi/2))
    xyz=cube_vertices(p[:2],yaw,edge_m,floor_z_m)
    uv=camera.project(xyz)
    # Visibility is self-occlusion only; external objects are not accounted for.
    faces=[((4,5,6,7),(0,0,1)),((0,1,5,4),(0,-1,0)),((1,2,6,5),(1,0,0)),
           ((2,3,7,6),(0,1,0)),((3,0,4,7),(-1,0,0))]
    visible=set()
    for ids,normal in faces:
        normal=rot_z(yaw) @ np.array(normal,float)
        if np.dot(normal,camera.mount_from_cv[:3,3]-xyz[list(ids)].mean(0))>0:
            visible.update(ids)
    vertices=[]
    for i,(point,pixel) in enumerate(zip(xyz,uv)):
        vertices.append({'id':i,'xyz_m':point.tolist(),'uv_px':pixel.tolist(),
                         'source':'geometry_inferred','self_visible':i in visible,
                         'in_image':bool(0<=pixel[0]<camera.width and 0<=pixel[1]<camera.height)})
    return {'status':'estimated','frame_id':'mount','units':'m','fit_iou':float(fit),
            'center_xyz_m':[float(p[0]),float(p[1]),floor_z_m+edge_m/2],
            'yaw_rad_mod_pi_over_2':yaw,'edge_m':edge_m,'support_z_m':floor_z_m,
            'assumptions':['upright','known_edge','known_horizontal_support','unoccluded_silhouette'],
            'visibility_scope':'self_occlusion_only','vertices':vertices}


def annotate_vertices(rgb, estimates):
    """Top IDs 4..7 cyan, bottom IDs 0..3 orange; self-hidden vertices hollow."""
    image=rgb.copy()
    for detection_id,result in estimates:
        if result['status']!='estimated':
            continue
        vertices=result['vertices']
        for a,b in EDGES:
            pa,pb=(tuple(np.rint(vertices[i]['uv_px']).astype(int)) for i in (a,b))
            hidden=not (vertices[a]['self_visible'] and vertices[b]['self_visible'])
            if hidden:
                for t in np.arange(0,1,.16):
                    p=np.array(pa)*(1-t)+np.array(pb)*t
                    q=np.array(pa)*(1-min(t+.08,1))+np.array(pb)*min(t+.08,1)
                    cv2.line(image,tuple(p.astype(int)),tuple(q.astype(int)),(160,160,160),1,cv2.LINE_AA)
            else:
                cv2.line(image,pa,pb,(255,255,255),1,cv2.LINE_AA)
        for vertex in vertices:
            pixel=tuple(np.rint(vertex['uv_px']).astype(int))
            color=(40,240,255) if vertex['id']>=4 else (255,150,30)
            cv2.circle(image,pixel,3,color,-1 if vertex['self_visible'] else 1,cv2.LINE_AA)
            label=f"{detection_id}:{vertex['id']}"
            cv2.putText(image,label,(pixel[0]+4,pixel[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(image,label,(pixel[0]+4,pixel[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
    return image
