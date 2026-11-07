# Marker-free depth policy

The policy outputs bounded heading-aligned FLU velocity from a 24×16 synthetic depth image, estimated body velocity, estimated roll/pitch, and a relative waypoint. It receives no tags, landmark IDs, obstacle coordinates, or absolute position. Training covers the route from hover to landing approach; PX4 handles takeoff and touchdown separately.

## Train for the current x500 experiment

Use a native Python environment with NumPy and PyTorch. Give every run a fresh output directory:

```sh
python -u -m learning.train --profile x500-transfer-delay --device mps --threads 1 \
  --output artifacts/learning/my-run --max-seconds 18000 --eval-interval 300 --seed 45
```

To warm-start, add `--init-checkpoint artifacts/learning/OLD/best.pt`. This loads weights only and creates a fresh optimizer, replay and step counter. Set `--max-seconds` to the remaining budget when continuing an interrupted run. The background supervisor in `runtime/train_overnight.py` provides a separate wall-clock cap and low-disk guard.

The `x500-transfer` profile uses `mission/course-window-x500.json`, the x500 SDF body/gear/rotor envelope with a 15 mm allowance, a 0.34 m forward camera, and 1.8 m heading lookahead. It samples velocity response from 0.50–0.80 seconds and applies 0.8 m/s² command slew. Each 160 ms policy step contains four 40 ms plant steps. Estimated waypoint arrival is 60 mm; truth verification is 80 mm. Window alignment keeps the opening in view.

The explicit `x500-transfer-delay` profile adds a queue of commands after slew limiting, sampled delays of 0.12–0.24 seconds, and a horizontal velocity-gain stress of 1.00–1.08. The teacher predicts from the commands already in that queue. This revision follows a measured braking mismatch in policy-02; those ranges are stress assumptions, not a validated general model of PX4 or the physical drone. The original `x500-transfer` profile remains available unchanged. Checkpoints identify these as v2 and v1 respectively.

Source hashes and the route, camera, airframe and control contract are stored in metadata and checkpoints. The live policy bridge rejects a checkpoint whose contract differs from the current route or vehicle. Do not edit those training sources during a run. The historical `legacy` profile remains the command-line default for reproducibility; it uses the old route and a 0.28 m sphere and is refused by the current x500 flight runner.

## Method and evaluation

Training uses privileged geometry-teacher labels and student-driven rollouts, a DAgger-style supervised imitation procedure. It is not reinforcement learning, Dreamer, or a learned world model. The teacher queries geometry; the actor only receives depth and estimated state. The trainer uses analytic ray casting and approximate dynamics, not Gazebo or PX4.

The supplied CAD bounds and wall vertices define the scene. Straight walls and boulders use proxy boxes, the rotated wall contains the measured opening, and a polygonal shell approximates the hollow tube. Net faces are opaque depth proxies. Sensor stress adds estimator error, depth noise and missing pixels. Geometry changes are capped separately to avoid silently treating impossible x500 clearances as valid demonstrations. Invalid training proposals are counted and resampled within a fixed limit; validation and holdout keep invalid scenarios in their denominator.

Validation seeds and final held-out seeds are disjoint from training and each other. Only validation selects `best.pt`. Metrics compare the policy, the same policy with blank depth, a direct waypoint controller, and the teacher. Blank depth is an out-of-distribution ablation. The x500 profile scores continuous multipart sweeps, full aperture passage, ordered truth waypoints, and traversal time. A conservative unresolved overlap counts as a failed clearance check. The trainer reserves the final 200 seconds, or 30% of a shorter budget, for held-out evaluation.

`status.json`, `latest.pt`, `best.pt` and bounded metrics are updated atomically. Training replay stays in memory. SIGTERM requests checkpoint saving, but an interrupted run may have no final held-out result. Gazebo RGB-D/PX4 integration is tested separately using [the marker-free flight runbook](../docs/markerfree-rgbd.md). Ideal rendered depth, a known static survey, approximate latency and dynamics, and the x500 surrogate do not establish transfer to the physical aircraft.

For Gazebo tests, learned actions pass through a separate [predictive boundary braking guard](../docs/policy-braking.md). The guard is not part of training and its interventions must be reported alongside the learned-policy result.

## Completed training and paired review

Four supervised jobs used 14,141.32 seconds (3.928 hours) cumulatively before the five-hour session cutoff. This is elapsed supervisor time across separate jobs, including initialization and evaluation, not five continuous hours. The last run completed normally and selected the validation checkpoint at step 48,440; its analytic held-out result was 7/8 with one collision. The [training ledger](../docs/training-ledger.json) retains the per-job intervals.

The frozen v1 and v2 checkpoints were compared on the same eight analytic cases. Each completed 7/8 with one collision; v2 had a larger minimum successful window margin (34.25 mm vs. 23.79 mm) but a slower median success time (61.44 s vs. 60.00 s). The paired review did not promote v2. Its public-sanitized package preserves exactly 14 tensors and has never been flown. See the [paired results](../docs/paired-policy-results.json) and [v2 manifest](../checkpoints/window-depth-v2.manifest.json). Separately, the v1 baseline completed policy-07 once under synthetic 10 mm Gaussian depth noise; this Gazebo/PX4 result is not hardware transfer evidence and is described in the [sensor stress report](../docs/sensor-stress.md).
