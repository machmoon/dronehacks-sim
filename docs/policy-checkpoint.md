# Window depth policy checkpoint

[Download the packaged checkpoint](../checkpoints/window-depth-v1.pt) and its [manifest](../checkpoints/window-depth-v1.manifest.json). It contains the same 14 tensors and policy contract as the original policy-03 checkpoint, selected from training run `20261107T120458Z-273a3b` at 13,720 steps. Repackaging removed only the local absolute path in `initialization.checkpoint`; all tensors and all other nested values were checked for exact equality after loading both files with `weights_only=True` on CPU.

The original artifact SHA-256 was `3602a898cad3f3ec91be78786c0811aa1ab0bd2f2d3a6b8f909edcf64c0cdf56`. The sanitized package SHA-256 is `a1919ecdd2149515e4875f4694c52733e269345ef95cad6b074099bfdbff8b26`. Policy-03's 77.000-second flight used the original hash. **Policy-05 separately tested the packaged hash** on the patched PX4 image: 77.000 seconds through landing/disarm, 15/15 ordered truth waypoints, all eight geometry checks, and 91.9 mm minimum modeled window clearance. The [packaged-policy score](packaged-policy-results.json) records its separate run and executable hashes.

The v1 contract is `x500-transfer-v1`: course route SHA-256 `d75381bce0876018544c25e40d10a7bd8864d3ec521b61eb5881a8b3d2acef7a`, x500 airframe SHA-256 `2ac89809b6dca9475521d856f6d75e759cbbb2418a738fb5cc6cc172310f5b5b`, forward camera mount 0.34 m, and heading lookahead 1.8 m. The policy consumes a 16×24 metric-depth image encoded as `1 - range / max_range` (4 m maximum) and a nine-value state vector. The inference service runs on the host CPU:

```sh
python3 -m learning.inference \
  --checkpoint checkpoints/window-depth-v1.pt \
  --exchange artifacts/markerfree/policy-exchange
```

For any flight reproduction, retain the separately implemented policy boundary braking guard; it is enabled by default in the mission runner. The learned actor alone is not the guarded policy-03 controller. The guard is a heuristic with no formal safety guarantee. This package and its simulator results make no hardware-readiness claim.

[Watch the packaged-policy repeat](media/window-policy-repeat.mp4). The clip contains recorded forward RGB and optical-Z depth, including preflight; its 94.48-second encoded duration is separate from the 77.000-second flight timing.

## V2 comparison (not promoted)

The paired eight-case analytic evaluation gave both v1 and v2 7 successes and one collision. V1's median successful episode was 60.00 s with 23.79 mm minimum successful window margin; v2's was 61.44 s with 34.25 mm. The v2 [public-sanitized package](../checkpoints/window-depth-v2.pt) and [manifest](../checkpoints/window-depth-v2.manifest.json) preserve exactly 14 tensors; the manifest records that it was not promoted and has never been flown. The flight baseline remains v1. The [paired evaluation record](paired-policy-results.json) notes that this is only eight analytic cases, without Gazebo/PX4, OAK stereo, or physical transfer.

The separate [policy-07 noisy simulation](noisy-policy-completed-results.json) flew v1, not v2. It completed one 78.000-second arm-to-disarm run with 15/15 ordered truth waypoints and all eight checks, using synthetic 10 mm Gaussian depth noise, FALS7 survey normals and patched PX4. Its video is [available here](media/window-policy-noisy.mp4); details and limits are in the [sensor stress report](sensor-stress.md).
