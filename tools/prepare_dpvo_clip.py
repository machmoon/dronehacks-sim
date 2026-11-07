#!/usr/bin/env python3
"""Prepare calibrated rectified image clips for compact upstream DPVO evaluation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def resize_calibration(k, source_wh, target_wh):
    sx,sy=np.array(target_wh)/source_wh
    result=np.array(k,dtype=float).copy()
    result[0,0]*=sx;result[1,1]*=sy
    result[0,2]=(result[0,2]+.5)*sx-.5
    result[1,2]=(result[1,2]+.5)*sy-.5
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--calib',type=Path,required=True,help='rectified fx fy cx cy text file')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    c=np.loadtxt(args.calib).ravel()
    if c.size!=4 or not np.isfinite(c).all() or min(c[:2])<=0:
        parser.error('require four finite rectified pinhole parameters, not distorted input')
    files=sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if not files:parser.error('no images found')
    if args.output.exists():parser.error('output must be a new directory')
    first=cv2.imread(str(files[0]))
    if first is None:parser.error('cannot decode first image')
    height,width=first.shape[:2]
    k=np.array([[c[0],0,c[2]],[0,c[1],c[3]],[0,0,1.]])
    scaled=resize_calibration(k,(width,height),(320,240))
    args.output.mkdir(parents=True);frames=args.output/'images';frames.mkdir()
    for i,path in enumerate(files):
        image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(height,width):raise ValueError(f'inconsistent image: {path}')
        if not cv2.imwrite(str(frames/f'{i:06d}.png'),cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA)):
            raise OSError('image write failed')
    np.savetxt(args.output/'calib.txt',[[scaled[0,0],scaled[1,1],scaled[0,2],scaled[1,2]]])
    # Upstream allocates a 36-frame, 128-channel pyramid at 1/4 and 1/16
    # resolution. This is exact array accounting, NOT measured peak GPU memory.
    cache=lambda w,h:36*128*((w//4)*(h//4)+(w//16)*(h//16))*2
    (args.output/'manifest.json').write_text(json.dumps({
        'source_images':[str(p.resolve()) for p in files], 'target_wh':[320,240],
        'intrinsics':scaled.tolist(),'upstream_commit':'0ac95b656d1fda91c271d2a106460d19ad966fc7',
        'feature_pyramid_fp16_bytes':cache(320,240),'640x480_reference_bytes':cache(640,480),
        'warning':'No CUDA inference or accuracy validation performed by this preparation tool.'},indent=2)+'\n')
    print(f'Prepared {len(files)} frames in {args.output}')


if __name__=='__main__':main()
