# Packaged v1 actor response to finite depth mismatches

The [frozen protocol](protocol.json) reused the same six real policy-07 window images and finite far-edge substitutions as the earlier stereo stress replay. Each pair held the estimated state, waypoint and speed cap fixed. The production raw `policy_depth` transform produced both observations. The packaged v1 checkpoint was loaded directly with `torch.load(weights_only=True)` on one CPU thread. No inference service requests, control commands or simulator state were involved. The result measures the actor output before command slew and boundary braking.

All six captures had an exact PX4 attitude publication sample. Two local position and velocity estimates were exact and four had an 8 ms bracket. Reconstruction uses the publication timestamp truncated to MAVLink `time_boot_ms`, which the pinned PX4 stream code uses. The bridge's `PolicyStateHistory` provided the capture-aligned estimates, and mission telemetry agreed on the waypoint and cap on both sides of each capture. No native state bracket crossed an estimator reset. The first archive validation was stricter than the runtime about equal timestamps. The final replay uses the runtime's first-sample rule, dropping 21 duplicate attitude timestamps and no local-position timestamps. The uORB logger did not record the exact throttled MAVLink packet set, so this reconstructs a faithful estimator state without claiming historical byte-identical inputs. [PX4 local-position stream](https://github.com/PX4/PX4-Autopilot/blob/54f0455ffcd755534539a7cf33a09a20bf71d29d/src/modules/mavlink/streams/LOCAL_POSITION_NED.hpp#L59-L68) · [PX4 attitude stream](https://github.com/PX4/PX4-Autopilot/blob/54f0455ffcd755534539a7cf33a09a20bf71d29d/src/modules/mavlink/streams/ATTITUDE.hpp#L61-L71)

All deltas below are finite mismatch minus control, in m/s. Forward means the signed FLU forward axis. Waypoint closing projects the requested velocity onto the current goal direction.

| Frame | Phase | Vector change norm | Forward delta | Speed norm delta | Waypoint closing delta |
|---|---|---:|---:|---:|---:|
| 359 | APPROACH | 0.00290 | +0.00001 | 0.00000 | +0.00001 |
| 371 | APPROACH | 0.00700 | +0.00011 | 0.00000 | −0.00003 |
| 383 | ALIGN | 0.00550 | +0.00419 | 0.00000 | +0.00030 |
| 395 | ALIGN | 0.00470 | −0.00244 | −0.00431 | −0.00445 |
| 407 | CROSS | 0.00431 | −0.00010 | 0.00000 | −0.00002 |
| 419 | CROSS | 0.00624 | +0.00109 | +0.00100 | +0.00100 |

Frame 383 is the image with 14 peripheral near-to-far policy bins. Its forward request rose from 0.13026 to 0.13444 m/s while its speed magnitude stayed at the 0.20475 m/s cap. Frame 371 gave the largest vector change, about 1.8% of its 0.39 m/s cap. These are paired open-loop changes. The two earlier localization warnings do not alter the estimated state in either member of any pair. [Complete state and action vectors](results.json)

The packaged model's archived training simulator independently dropped individual depth bins and sometimes dropped a rectangular patch. Each operation replaced the measured range with `MAX_DEPTH`. The subsequent encoding mapped those replacements to zero, exactly like a maximum-range return. It is an optimistic-return stress that removes apparent obstacles from the affected rays. It carries no preserved unknown-pixel state. The sampled mismatch here instead supplies finite farther measurements and leaves the state fixed. [Current simulator source with the same dropout rule](../../../learning/sim.py)

## Reproduction

Exact preparation requires the original local policy-06/policy-07 camera, telemetry and ULog archives, which are not checked into Git. This folder preserves the protocol, scripts, prepared inputs where available, complete results and input hashes. Restore compressed raw observer files only when a tool requires the original path. The local archive manifests contain byte-verified restoration commands.

Run from the worktree root with a host Python that has OpenCV and PyULog for preparation, then use an environment with PyTorch for inference.

```sh
original=docs/experiments/stereo-actor-sensitivity-01
replay=$(mktemp -d artifacts/markerfree/stereo-actor-replay.XXXXXX)
cp "$original"/protocol.json "$original"/actor_probe.py "$replay/"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$replay/actor_probe.py" prepare
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../research/sotapilot-env/bin/python "$replay/actor_probe.py" infer
```
