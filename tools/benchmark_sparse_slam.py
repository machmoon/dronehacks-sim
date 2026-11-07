#!/usr/bin/env python3
"""Reproducible CPU smoke benchmark; synthetic images are not flight validation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from vision.sparse_slam import Config, SparseMap
from vision.rgbd_odometry import RGBDOdometry


class SyntheticRoom:
    def __init__(self,width=320,height=240,seed=71):
        self.width,self.height=width,height
        self.k=np.array([[240.,0,width/2],[0,240.,height/2],[0,0,1.]])
        y,x=np.mgrid[:height,:width]
        self.rays=np.stack([x,y,np.ones_like(x)],axis=-1)@np.linalg.inv(self.k).T
        self.rng=np.random.default_rng(seed)
        self.textures=[]
        for _ in range(2):
            texture=self.rng.integers(0,256,(640,800,3),dtype=np.uint8)
            self.textures.append(cv2.GaussianBlur(texture,(3,3),.6))

    def render(self,pose,noise=0.):
        ray=self.rays@pose[:3,:3].T;origin=pose[:3,3]
        rgb=np.zeros((self.height,self.width,3),np.uint8)
        depth=np.full((self.height,self.width),np.inf,np.float32)
        for plane_z,texture,bounds in [(2.4,self.textures[0],None),(1.5,self.textures[1],(.30,.65,-.4,.4))]:
            z=(plane_z-origin[2])/ray[:,:,2]
            xyz=origin+ray*z[:,:,None]
            image=cv2.remap(texture,(xyz[:,:,0]*180+400).astype(np.float32),
                            (xyz[:,:,1]*180+320).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            visible=(z>0)&(z<depth)
            if bounds:
                a,b,c,d=bounds
                visible&=(xyz[:,:,0]>a)&(xyz[:,:,0]<b)&(xyz[:,:,1]>c)&(xyz[:,:,1]<d)
            depth[visible]=z[visible];rgb[visible]=image[visible]
        if noise:depth+=self.rng.normal(0,noise,depth.shape).astype(np.float32)
        return rgb,depth


def trajectory(index,count):
    phase=index/max(count-1,1)*2*np.pi
    pose=np.eye(4)
    pose[:3,3]=[.22*np.sin(phase),.035*np.sin(phase*2),.03*(1-np.cos(phase))]
    pose[:3,:3]=cv2.Rodrigues(np.array([.015*np.sin(phase),.04*np.sin(phase),0.]))[0]
    return pose


def run(count=120,features=192,dropout=False,noise=.004,seed=71):
    cv2.setNumThreads(1);cv2.setRNGSeed(42)
    room=SyntheticRoom(seed=seed);small=SparseMap(room.k,Config(features=features))
    baseline=RGBDOdometry((0,0,0),intrinsics=room.k)
    class EqualBudgetVO(RGBDOdometry):
        def _features(self,gray,depth):
            mask=(np.isfinite(depth)&(depth>=self.gates.min_depth)&(depth<=self.gates.max_depth)).astype(np.uint8)*255
            found=cv2.goodFeaturesToTrack(gray,maxCorners=features,qualityLevel=.01,minDistance=6,mask=mask,blockSize=5)
            if found is None or len(found)<self.gates.min_inliers:
                from vision.rgbd_odometry import TrackingFailure
                raise TrackingFailure('insufficient_texture')
            return found
    equal=EqualBudgetVO((0,0,0),intrinsics=room.k)
    baseline_origin=baseline.world_from_optical.copy()
    histories={'sparse_map':[],'frame_to_frame_vo':[],'equal_budget_vo':[]}
    for i in range(count):
        pose=trajectory(i,count);rgb,depth=room.render(pose,noise)
        if dropout and count//3<=i<count//3+3:rgb[:]=0
        stamp=1+i/30
        for name,estimator in [('sparse_map',small),('frame_to_frame_vo',baseline),('equal_budget_vo',equal)]:
            start=time.perf_counter()
            if name=='sparse_map':
                record=estimator.estimate(rgb,depth,stamp)
                prediction=np.array(record['map_from_camera']) if record['valid'] else None
            else:
                record=estimator.estimate(rgb,depth,round(stamp*1e6))
                prediction=np.linalg.inv(baseline_origin)@estimator.world_from_optical if record['valid'] else None
            latency=(time.perf_counter()-start)*1000
            histories[name].append({'frame':i,'valid':record['valid'],'latency_ms':latency,
                                    'translation_error_m':float(np.linalg.norm(prediction[:3,3]-pose[:3,3])) if prediction is not None else None,
                                    'relocalized':record.get('relocalized',False),'reason':record.get('reason')})
    summary={}
    for name,rows in histories.items():
        errors=[r['translation_error_m'] for r in rows if r['valid']]
        latencies=[r['latency_ms'] for r in rows]
        summary[name]={'valid_frames':len(errors),'total_frames':count,
                       'translation_rmse_m':float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                       'latency_p50_ms':float(np.percentile(latencies,50)),
                       'latency_p95_ms':float(np.percentile(latencies,95)),
                       'latency_max_ms':float(max(latencies)),
                       'frames_over_33ms':sum(t>1000/30 for t in latencies),
                       'valid_latency_p50_ms':float(np.median([r['latency_ms'] for r in rows if r['valid']])) if errors else None,
                       'relocalizations':sum(r['relocalized'] for r in rows),
                       'final_valid':rows[-1]['valid']}
    summary['sparse_map'].update(keyframes=len(small.frames),map_array_bytes=small.map_bytes,feature_budget=features)
    return {'seed':seed,'dropout':dropout,'depth_noise_std_m':noise,'summary':summary,'frames':histories}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--seed',type=int,default=71)
    parser.add_argument('--features',type=int,default=192)
    parser.add_argument('--output',type=Path,default=Path('artifacts/slam/sparse-map-benchmark.json'))
    args=parser.parse_args()
    if args.frames<40:parser.error('at least 40 frames required')
    result={'scope':'Synthetic two-plane RGBD, 320x240; CPU only; no Jetson or real-camera claim',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('vision/sparse_slam.py'),Path('vision/rgbd_odometry.py'))},
            'host':platform.platform(),'opencv':cv2.__version__,'numpy':np.__version__,
            'scenarios':[run(args.frames,args.features,dropout=dropout,seed=args.seed) for dropout in (False,True)]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'scenarios':[{k:v for k,v in s.items() if k!='frames'} for s in result['scenarios']]},indent=2))


if __name__=='__main__':main()
