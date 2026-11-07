"""Experimental bounded sparse RGBD map tracker, inspired by sparse-patch SLAM.

This is an independent geometric prototype, NOT distilled DPVO or a replacement
for cuVSLAM. Metric depth must come from a calibrated, synchronized camera. The
map starts in the first optical frame; relocalization may correct its pose, so
outputs are deliberately NOT connected to PX4 or the flight interface.
"""
from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    features: int = 192
    keyframes: int = 8
    recovery_features: int = 384
    min_inliers: int = 24
    min_ratio: float = .60
    reprojection_px: float = 1.5
    min_depth_m: float = .15
    max_depth_m: float = 6.
    keyframe_translation_m: float = .15
    keyframe_rotation_rad: float = .15
    keyframe_period_s: float = 1.
    max_gap_s: float = .25
    max_speed_m_s: float = 3.
    max_angular_speed_rad_s: float = 3.

    def __post_init__(self):
        if not 24 <= self.min_inliers <= self.features <= 512 or not 2 <= self.keyframes <= 16:
            raise ValueError('invalid feature/keyframe budget')
        if not self.features <= self.recovery_features <= 1024:
            raise ValueError('invalid relocalization budget')
        if not 0 < self.min_ratio <= 1 or not 0 < self.min_depth_m < self.max_depth_m:
            raise ValueError('invalid geometry gates')


def inverse_pose(t):
    result=np.eye(4);result[:3,:3]=t[:3,:3].T
    result[:3,3]=-t[:3,:3].T@t[:3,3]
    return result


class Lost(ValueError):
    pass


class SparseMap:
    def __init__(self, intrinsics, config=Config()):
        self.k=np.asarray(intrinsics,dtype=np.float64)
        if self.k.shape!=(3,3) or not np.isfinite(self.k).all() or min(self.k[0,0],self.k[1,1])<=0:
            raise ValueError('finite calibrated intrinsics required')
        if not np.allclose(self.k[2],[0,0,1]): raise ValueError('invalid pinhole calibration')
        self.kinv=np.linalg.inv(self.k); self.cfg=config
        self.orb=cv2.ORB_create(nfeatures=config.recovery_features*3,edgeThreshold=12,patchSize=21,fastThreshold=12)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.frames=[];self.previous=None;self.pose=np.eye(4)
        self.last_stamp=None;self.last_valid_stamp=None;self.frame_count=0;self.recovery_count=0

    def _depth_ok(self, depth):
        return np.isfinite(depth)&(depth>=self.cfg.min_depth_m)&(depth<=self.cfg.max_depth_m)

    def _detect(self, gray, depth):
        mask=self._depth_ok(depth).astype(np.uint8)*255
        keypoints,descriptors=self.orb.detectAndCompute(gray,mask)
        if descriptors is None or len(keypoints)<self.cfg.min_inliers: raise Lost('insufficient_texture')
        uv=np.array([p.pt for p in keypoints],np.float32)
        # Spread the bounded patch budget over the image. In each cell, prefer
        # sharp corners with reliable nearby metric depth (translation leverage).
        pixels=np.rint(uv).astype(int);z=depth[pixels[:,1],pixels[:,0]]
        valid=self._depth_ok(z)
        uv,descriptors,z=uv[valid],descriptors[valid],z[valid]
        response=np.array([p.response for p in keypoints])[valid]
        cells=np.minimum((uv[:,0]*8/gray.shape[1]).astype(int),7)+8*np.minimum((uv[:,1]*6/gray.shape[0]).astype(int),5)
        score=response/np.maximum(z,.5)
        buckets=[list(np.flatnonzero(cells==cell)[np.argsort(-score[cells==cell])]) for cell in range(48)]
        chosen=[]
        while len(chosen)<self.cfg.recovery_features and any(buckets):
            for bucket in buckets:
                if bucket: chosen.append(bucket.pop(0))
                if len(chosen)==self.cfg.recovery_features:break
        ids=np.array(chosen,dtype=int)
        if len(ids)<self.cfg.min_inliers:raise Lost('insufficient_depth_features')
        rays=np.c_[uv[ids],np.ones(len(ids))]@self.kinv.T
        return uv[ids],rays*z[ids,None],descriptors[ids]

    def _solve(self, world_points, uv, depth):
        c=self.cfg
        if len(uv)<c.min_inliers:raise Lost('insufficient_tracks')
        if not np.isfinite(world_points).all() or not np.isfinite(uv).all():raise Lost('nonfinite_tracks')
        ok,r,t,ids=cv2.solvePnPRansac(world_points,uv,self.k,None,iterationsCount=80,
                                     reprojectionError=2.,confidence=.995,flags=cv2.SOLVEPNP_EPNP)
        if not ok or ids is None or len(ids)<c.min_inliers or len(ids)/len(uv)<c.min_ratio:
            raise Lost('insufficient_inliers')
        ids=ids.ravel()
        support=(uv[ids]-self.k[:2,2])/np.diag(self.k)[:2]
        singular=np.linalg.svd(world_points[ids]-world_points[ids].mean(axis=0),compute_uv=False)
        if min(np.ptp(support,axis=0))<.12 or singular[1]<max(1e-5,singular[0]*.015):
            raise Lost('degenerate_support')
        ok,r,t=cv2.solvePnP(world_points[ids],uv[ids],self.k,None,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(np.r_[r.ravel(),t.ravel()]).all():raise Lost('refinement_failed')
        rotation=cv2.Rodrigues(r)[0]
        optical=world_points@rotation.T+t.ravel()
        prediction=optical@self.k.T
        pixels=np.rint(uv).astype(int)
        measured=depth[pixels[:,1],pixels[:,0]]
        projected=prediction[:,:2]/np.maximum(prediction[:,2:],1e-9)
        errors=np.linalg.norm(projected-uv,axis=1)
        good=(errors<c.reprojection_px)&self._depth_ok(measured)&(optical[:,2]>c.min_depth_m)
        good&=np.abs(optical[:,2]-measured)<.025+.025*measured
        ids=np.flatnonzero(good)
        if len(ids)<c.min_inliers or len(ids)/len(uv)<c.min_ratio:raise Lost('depth_or_reprojection_disagreement')
        camera_from_map=np.eye(4);camera_from_map[:3,:3]=rotation;camera_from_map[:3,3]=t.ravel()
        return inverse_pose(camera_from_map),ids,float(np.sqrt(np.mean(errors[ids]**2)))

    def _track(self,gray,depth):
        old_gray,uv,points=self.previous
        forward,ok,_=cv2.calcOpticalFlowPyrLK(old_gray,gray,uv[:,None,:],None,winSize=(15,15),maxLevel=3)
        if forward is None:raise Lost('flow_failed')
        backward,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,old_gray,forward,None,winSize=(15,15),maxLevel=3)
        if backward is None:raise Lost('flow_failed')
        current=forward.reshape(-1,2)
        keep=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)
        keep&=np.linalg.norm(backward.reshape(-1,2)-uv,axis=1)<.7
        keep&=np.isfinite(current).all(axis=1)
        keep&=(current[:,0]>=1)&(current[:,1]>=1)&(current[:,0]<gray.shape[1]-1)&(current[:,1]<gray.shape[0]-1)
        current,points=current[keep],points[keep]
        pose,ids,rms=self._solve(points,current,depth)
        return pose,current[ids],points[ids],rms

    def _recover(self,gray,depth):
        uv,_,descriptors=self._detect(gray,depth)
        candidates=[]
        for frame in self.frames:
            pairs=self.matcher.knnMatch(descriptors,frame['descriptors'],k=2)
            matches=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
            # Enforce one observation per map landmark.
            unique={}
            for m in sorted(matches,key=lambda m:m.distance):unique.setdefault(m.trainIdx,m)
            matches=list(unique.values())
            candidates.append((len(matches),frame,matches))
        for count,frame,matches in sorted(candidates,key=lambda row:row[0],reverse=True)[:3]:
            if count<self.cfg.min_inliers:continue
            current=uv[[m.queryIdx for m in matches]]
            points=frame['points'][[m.trainIdx for m in matches]]
            try:
                pose,ids,rms=self._solve(points,current,depth)
                return pose,current[ids],points[ids],rms
            except Lost:pass
        raise Lost('relocalization_failed')

    def _insert(self,gray,depth,stamp):
        uv,points,descriptors=self._detect(gray,depth)
        world=points@self.pose[:3,:3].T+self.pose[:3,3]
        self.frames.append({'points':world,'descriptors':descriptors,'pose':self.pose.copy(),'stamp':stamp})
        # Preserve the launch anchor for revisits, bound every other keyframe.
        if len(self.frames)>self.cfg.keyframes:del self.frames[1]
        self.previous=(gray.copy(),uv[:self.cfg.features],world[:self.cfg.features])

    @property
    def map_bytes(self):
        return sum(f['points'].nbytes+f['descriptors'].nbytes+f['pose'].nbytes for f in self.frames)

    def estimate(self,rgb,depth,stamp_s):
        start=time.perf_counter(); recovered=False;reason=None;rms=None;tracks=0
        try:
            if not np.isfinite(stamp_s) or (self.last_stamp is not None and stamp_s<=self.last_stamp):
                raise Lost('nonmonotonic_timestamp')
            self.last_stamp=stamp_s
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]:
                raise Lost('invalid_images')
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            if not self.frames:
                self._insert(gray,depth,stamp_s)
            else:
                try:
                    if self.previous is None or stamp_s-self.last_valid_stamp>self.cfg.max_gap_s:
                        raise Lost('tracking_gap')
                    pose,uv,points,rms=self._track(gray,depth)
                except Lost:
                    pose,uv,points,rms=self._recover(gray,depth);recovered=True
                dt=stamp_s-self.last_valid_stamp
                delta=inverse_pose(self.pose)@pose
                if (np.linalg.norm(delta[:3,3])>self.cfg.max_speed_m_s*dt+.03 or
                    np.linalg.norm(cv2.Rodrigues(delta[:3,:3])[0])>self.cfg.max_angular_speed_rad_s*dt+.03):
                    raise Lost('implausible_motion')
                self.pose=pose;self.previous=(gray.copy(),uv[:self.cfg.features],points[:self.cfg.features]);tracks=min(len(uv),self.cfg.features)
                last=self.frames[-1];change=inverse_pose(last['pose'])@pose
                if (len(uv)<self.cfg.features*.55 or
                    np.linalg.norm(change[:3,3])>self.cfg.keyframe_translation_m or
                    np.linalg.norm(cv2.Rodrigues(change[:3,:3])[0])>self.cfg.keyframe_rotation_rad or
                    stamp_s-last['stamp']>self.cfg.keyframe_period_s):
                    # A failed map expansion must not invalidate a well-supported
                    # tracked pose or discard the existing landmark tracks.
                    try:self._insert(gray,depth,stamp_s)
                    except Lost:pass
            self.last_valid_stamp=stamp_s
            self.recovery_count+=int(recovered)
        except (Lost,cv2.error) as error:
            reason=str(error);self.previous=None
        self.frame_count+=1
        return {'valid':reason is None,'reason':reason,'stamp_s':float(stamp_s) if np.isfinite(stamp_s) else None,
                'map_from_camera':self.pose.tolist() if reason is None else None,
                'relocalized':recovered and reason is None,'recovery_count':self.recovery_count,
                'tracks':tracks,'reprojection_rms_px':rms,'keyframes':len(self.frames),
                'map_bytes':self.map_bytes,'latency_ms':(time.perf_counter()-start)*1000,
                'flight_ready':False}
