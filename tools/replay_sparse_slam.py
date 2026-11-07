#!/usr/bin/env python3
"""Offline synchronized camera replay. Does not publish to ROS/PX4.

NPZ contract (allow_pickle=False): rgb[N,H,W,3] uint8 RGB;
depth[N,H,W] float optical-Z metres; stamps_s[N] increasing capture seconds;
intrinsics[3,3] rectified pinhole K. Depth must align to the RGB optical centre.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vision.sparse_slam import SparseMap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cv2.setNumThreads(1)
    with np.load(args.recording,allow_pickle=False) as recording:
        rgb,depth,stamps=(recording[k] for k in ('rgb','depth','stamps_s'))
        if rgb.ndim!=4 or rgb.shape[-1]!=3 or depth.shape!=rgb.shape[:3] or stamps.shape!=(len(rgb),):
            parser.error('mismatched recording arrays')
        if not len(rgb) or not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):
            parser.error('recording requires increasing finite capture timestamps')
        estimator=SparseMap(recording['intrinsics'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            for image,z,t in zip(rgb,depth,stamps):
                stream.write(json.dumps(estimator.estimate(image,z,float(t)),allow_nan=False)+'\n')
    print(f'Wrote offline map poses to {args.output}; no flight commands sent.')


if __name__=='__main__':main()
