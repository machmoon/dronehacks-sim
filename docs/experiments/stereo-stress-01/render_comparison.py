"""Render two recorded stereo-stress cases, without fitting or issuing commands."""
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from learning.inference import policy_depth

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('stereo_probe', HERE/'probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
protocol=json.loads((HERE/'protocol.json').read_text())
report=json.loads((HERE/'results.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(13.5,7.6),layout='constrained')
for row,index in enumerate((383,395)):
    entry=next(v for v in report['sources'] if v['source']['id']==f'policy-07-window-{index}')
    source=entry['source'];depth=probe.load_depth(source)
    variants,_=probe.transforms(depth,source['intrinsics'][0][0],protocol['stereo_diagnostic'])
    altered=variants['finite_far_edge_mismatch']
    image=cv2.cvtColor(cv2.imread(str(ROOT/source['rgb'])),cv2.COLOR_BGR2RGB)
    baseline=policy_depth(depth).depth; modified=policy_depth(altered).depth
    changed=(baseline >= 1.-1.5/4.) & (modified <= 1.-3./4.)
    axes[row,0].imshow(image)
    axes[row,0].set_title(f'Recorded RGB · frame {index}\n{source["capture_usec"]/1e6:.3f} simulated seconds',fontsize=11)
    im=axes[row,1].imshow(depth,cmap='turbo',vmin=0,vmax=8)
    axes[row,1].set_title('Recorded optical-Z depth',fontsize=11)
    axes[row,2].imshow(altered,cmap='turbo',vmin=0,vmax=8)
    y,x=np.nonzero(changed)
    axes[row,2].scatter(x*16+7.5,y*16+7.5,s=24,marker='s',facecolors='none',edgecolors='white',linewidths=.9)
    axes[row,2].set_title(f'Synthetic finite edge mismatch\n{int(changed.sum())} policy rays changed from ≤1.5 m to ≥3 m',fontsize=11)
    for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=axes[:,1:],label='Optical-Z depth (metres)',fraction=.024,pad=.018)
fig.suptitle('Finite stereo errors can look like valid depth',fontsize=17,weight='bold')
fig.supxlabel('Recorded Gazebo images with controlled depth corruption. White squares mark changed actor rays. Offline diagnostic; no flight commands.',fontsize=10)
fig.savefig(ROOT/'docs/media/stereo-edge-stress.png',dpi=150)
print(ROOT/'docs/media/stereo-edge-stress.png')
