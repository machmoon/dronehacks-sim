#!/usr/bin/env python3
"""Plot the recorded stationary RGBD pair without changing sensor values."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    palette = plt.get_cmap('viridis').copy()
    palette.set_bad('black')
    for row, (condition, title) in enumerate((
        ('control', 'Control: net visuals present'),
        ('treatment', 'Treatment: net visuals removed'),
    )):
        with np.load(args.artifacts / condition / 'capture.npz', allow_pickle=False) as data:
            assert data['rgb'].shape == (32, 256, 384, 3)
            assert data['depth_z_m'].shape == (32, 256, 384)
            rgb = data['rgb'][16]
            depth = np.median(data['depth_z_m'], axis=0)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(title + '\nRGB frame 17 of 32', fontsize=11)
        depth_plot = axes[row, 1].imshow(np.ma.masked_invalid(depth), cmap=palette,
                                        vmin=.1, vmax=8, interpolation='nearest')
        axes[row, 1].set_title('Median optical depth across 32 captures', fontsize=11)
        for axis in axes[row]:
            axis.set_axis_off()
    fig.colorbar(depth_plot, ax=axes[:, 1], shrink=.8, label='Optical depth (m); nonfinite values are black')
    fig.suptitle('Matched stationary views; net collision surfaces retained in both worlds\n'
                 'Rendered net absence is not a physical woven-net model', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
