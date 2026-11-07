# Compact SLAM experiment

This pass implements a small, independently written **sparse metric map tracker**
and prepares a reduced-budget configuration for upstream DPVO. The local tracker
runs without neural weights on NumPy/OpenCV. It is a research prototype, not a
newly validated SLAM algorithm, a distilled neural network, or a replacement for
the drone's cuVSLAM. No flight-control output is connected.

## Research choice

[DPVO](https://arxiv.org/abs/2208.04726) uses sparse learned patches rather than
dense flow, with recurrent matching and differentiable bundle adjustment. Its
[official implementation](https://github.com/princeton-vl/DPVO) is the reference
for the compact neural profile. [MASt3R-SLAM](https://arxiv.org/abs/2412.12392)
uses a learned two-view 3D prior and dense reconstruction; the paper reports 15 FPS,
which is not a Jetson Nano measurement. Sparse tracking is a more direct starting
point for this experiment because OAK-D can provide metric stereo depth already.
[NVIDIA's cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) remains the relevant
hardware baseline. Newer upstream releases are not assumed compatible with the
fleet's pinned Isaac ROS deployment.

The testable hypothesis is: **retain metric landmarks between frames, and spend
extra matching effort only when creating a keyframe or recovering tracking.**
That may reduce both repeated computation and drift caused by repeatedly
reconstructing landmarks from noisy depth. This combines established techniques;
novelty and superiority over frontier SLAM have not been established.

## Implemented mechanism

`vision/sparse_slam.py` bounds live tracking at 192 points, stores at most eight
keyframes with 384 descriptors each, and retains the first keyframe as a launch
anchor. It stores landmarks and descriptors rather than a dense map or image
history. A spatial grid spreads features across the image; corner response divided
by depth is a simple selection heuristic, not a computed Fisher-information score.

For a keyframe K, its measured optical depth z and rectified pixel u give the
fixed map landmark:

`P_map = R_map_from_K · (z · inverse(K_camera) · [u,v,1]) + t_map_from_K`.

Subsequent frames track patches with forward/backward optical flow and estimate
camera pose by robust PnP against these map landmarks. Reprojection, metric-depth
agreement, image coverage, geometric rank and motion plausibility must all pass.
The landmarks are reused until a keyframe is inserted. On loss, ORB descriptor
matching searches the bounded keyframe cache, and the same geometric tests verify
relocalization. Invalid frames return no pose. A relocalization event is explicit.

This is **local mapping with relocalization**, without global pose-graph
optimization, bundle adjustment, IMU fusion, learned correspondence, or a
guarantee against perceptual aliasing. Map poses can be corrected on recovery.
They must not be sent directly as continuous autopilot odometry. A production
integration needs separate map/odom frames, covariance, reset handling and measured
camera-to-IMU extrinsics.

## Local experiment and results

The reproducible benchmark renders a two-plane textured scene at 320×240 with
known camera translation/rotation and 4 mm depth noise. It measures estimator
calls only, excluding image acquisition, stereo-depth production, transport and
rendering. Three seeds change texture and depth noise, not room topology. Each
sequence has 120 frames at nominal 30 Hz. The blackout variant blanks three frames.
There is no trajectory alignment or ground-truth correction inside the estimators.

Native Mac ARM64, OpenCV 5.0.0, one OpenCV thread; other simulator workloads were
running, so latency varies. The clean-sequence results from seeds 71, 72 and 73 are:

| Estimator | Median frame time across runs | Position RMSE across runs | Valid frames |
|---|---:|---:|---:|
| SparseMap, 192 live points | 4.6–7.2 ms | 5.4–7.3 mm | 120/120 |
| Existing VO, 600 corners | 19.5–33.6 ms | 10.1–11.6 mm | 120/120 |
| Existing VO, 192 corners | 8.0–13.1 ms | 9.8–13.5 mm | 120/120 |

SparseMap recovered on the first nonblank frame in all three blackout sequences:
117/120 valid frames, with no pose on the three blanks. Both VO baselines stop at
40/120 because their existing loss latch deliberately requires an explicit reset.
That is a difference in recovery policy, not proof that accepting relocalization is
safer. Errors on the dropout runs cover different valid time spans and should not
be compared as full-trajectory accuracy scores.

The stored map arrays occupied 130–151 KB in these runs; this excludes the current
image, OpenCV/Python overhead and process RSS. Maximum observed clean-sequence
SparseMap latency was 101–290 ms, including cold-start work and host scheduling.
This is not a hard real-time guarantee. There is no Jetson, real OAK-D, EuroC,
TUM-VI, or flight-log result yet. Repeated textures, lighting changes, motion blur,
rolling exposure effects, stereo holes and vibration remain important tests.

The [saved result summaries](sparse-slam-results.json) retain all three runs.
Run and inspect the full traces:

```sh
python -m unittest discover -s tests -p 'test_sparse_slam.py'
python -m tools.benchmark_sparse_slam --seed 71 --output artifacts/slam/sparse-map-host.json
python -m tools.benchmark_sparse_slam --seed 72 --output artifacts/slam/sparse-map-seed-72.json
python -m tools.benchmark_sparse_slam --seed 73 --output artifacts/slam/sparse-map-seed-73.json
python -m tools.replay_sparse_slam capture.npz --output artifacts/slam/real-camera-poses.jsonl
```

The replay tool documents the NPZ schema in its module docstring. It accepts
synchronized RGB and aligned optical-Z depth with calibrated intrinsics and capture
timestamps. It neither reads a simulator pose nor publishes ROS/PX4 commands.
Image-level tests cover metric translation/rotation, loss/recovery, unrelated-scene
rejection, depth inconsistency, memory bounds and camera resizing geometry. They
passed with both the Mac dependencies and the existing simulation image's OpenCV
4.6.0/NumPy 1.26.4.

## Actual DPVO reduction prepared

`vision/dpvo_compact.yaml` targets upstream commit
`0ac95b656d1fda91c271d2a106460d19ad966fc7`:

- Input 320×240 instead of 640×480, with correctly transformed intrinsics.
- 48 patches instead of 80; optimization window 6 instead of 12; patch lifetime 8.
- Keep the existing FP16 mode and learned network. Loop closure is disabled, so
  this profile is **VO**, not full global SLAM.
- Keep the 4096-keyframe buffer: upstream's buffer is not a rolling cache, and
  shrinking it to a tiny number would cause an eventual overflow failure.

`tools/prepare_dpvo_clip.py` resizes rectified image sequences and accounts for pixel
centres in the calibration. From the inspected allocation code, the fixed feature
pyramid arrays shrink from 188,006,400 bytes to 47,001,600 bytes (4×). This is array
accounting, not peak GPU-memory measurement; weights, optimizer state, CUDA workspace
and other caches are additional. Lower input resolution can lose thin features.

```sh
python -m tools.prepare_dpvo_clip --images rectified-left --calib camera.txt --output compact-clip
# In the pinned upstream DPVO environment on a CUDA machine:
python demo.py --imagedir /absolute/path/compact-clip/images \
  --calib /absolute/path/compact-clip/calib.txt \
  --config /absolute/path/dronehacks-sim/vision/dpvo_compact.yaml \
  --network /absolute/path/dpvo.pth --stride 1 --save_trajectory
```

The clip preparation and offline local-map replay were executed successfully.
Upstream DPVO inference was **not** executed here: this Mac has no CUDA device.
The compact configuration is therefore an unvalidated experiment, not a claimed
compression result. The next meaningful comparison is recorded OAK-D data on the
actual Orin Nano, including latency tails, tracking-loss duration, pose error and
power alongside the shipped cuVSLAM baseline.
