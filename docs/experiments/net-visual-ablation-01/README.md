# Net visual ablation protocol

This registered experiment removes the five rendered net surfaces while retaining their collision geometry. It first compares stationary RGBD captures from two generated worlds, then permits a control hover and route followed by a treatment hover and route only if each preceding check passes. The [protocol](protocol.json) defines the checks before any capture or flight in these worlds.

Generate the paired base worlds with:

```sh
python3 -m tools.make_net_visual_ablation \
  --output-dir artifacts/markerfree/net-visual-ablation-NEW
```

Select a generated `control.sdf` or `treatment.sdf` through `SIM_WORLD` when starting the dedicated marker-free simulator. The generator checks that the only control-to-treatment difference is removal of the five named visual elements. It retains every collision and the other world fields. Neither generated world adds a navigation input or a truth-feedback path.

Gazebo uses visual elements for rendering, and RGBD sensors use that rendering scene. Removing a net visual can therefore replace its depth return with background or far-limit depth while its collision remains. The passive comparison must demonstrate a material depth change before a flight ablation claim is made. [Gazebo visual documentation](https://gazebosim.org/api/sim/8/adding_visuals.html), [Gazebo sensor-system source](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/sensors/Sensors.cc).

This is a missing-net-surface-return stress case. It is not a realistic woven-net model, and it changes both RGB and depth. The estimator still uses the original static map, including its net panels, so the sensor/map mismatch is intentional. A successful simulator run would not establish real-camera net detection or physical transfer.

## Passive comparison

The recorded stationary pair passed every preregistered passive gate: 32 matched RGBD captures per condition, maximum capture interval 52 ms, continuous truth alignment, and identical stationary truth pose. The registered depth rule marked 47,657 of 98,304 image pixels as changed (48.4792%); the gate was 1%. This is evidence that the rendered depth input changed for this pair. It does not establish navigation or flight performance. Flight checks remain conditional and are reported separately from this passive result.

The machine-readable report is [results.json](results.json), and [capture-comparison.png](capture-comparison.png) shows the recorded views and 32-frame median depth. The raw capture archives and observer logs are retained locally and are not included in a fresh clone. With those archives present and NumPy installed, reproduce the numerical report from the repository root:

```sh
python3 docs/experiments/net-visual-ablation-01/analyze.py \
  --artifacts artifacts/markerfree/net-visual-ablation-01 \
  --output /tmp/net-visual-ablation-results.json
```

The plot script uses NumPy and Matplotlib; run it in a Python environment with both available:

```sh
python3 docs/experiments/net-visual-ablation-01/plot.py \
  --artifacts artifacts/markerfree/net-visual-ablation-01 \
  --output /tmp/net-visual-ablation-comparison.png
```

### Fresh passive capture

In a fresh checkout, first generate the registered pair at the protocol's fixed evidence root. Each collector invocation handles one condition and refuses an existing condition directory. It does not arm, send vehicle commands, or restart Gazebo. Run these commands only while the dedicated simulator is unarmed and safe to recreate between conditions:

```sh
python3 -m tools.make_net_visual_ablation \
  --output-dir artifacts/markerfree/net-visual-ablation-01

docker_context=${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}
sim_container=dronehacks-markerfree-sim-1
SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2 \
SIM_REAL_TIME_FACTOR=0.3 \
SIM_RGBD_DEPTH_NOISE=gaussian-10mm \
SIM_WORLD=artifacts/markerfree/net-visual-ablation-01/control.sdf \
  ./runtime/run-markerfree.sh
docker --context "$docker_context" exec "$sim_container" \
  python3 -m tools.capture_net_ablation --condition control

# After the control observer completes, recreate the simulator on the treatment world.
SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2 \
SIM_REAL_TIME_FACTOR=0.3 \
SIM_RGBD_DEPTH_NOISE=gaussian-10mm \
SIM_WORLD=artifacts/markerfree/net-visual-ablation-01/treatment.sdf \
  ./runtime/run-markerfree.sh
docker --context "$docker_context" exec "$sim_container" \
  python3 -m tools.capture_net_ablation --condition treatment

python3 \
  docs/experiments/net-visual-ablation-01/analyze.py \
  --artifacts artifacts/markerfree/net-visual-ablation-01 \
  --output /tmp/net-visual-ablation-results.json
```

The collector checks the isolated SITL environment, waits up to 60 wall seconds for the exact sensor diagnostic and at least 2 simulated seconds on the read-only IMU clock, then starts the 30-second observer. It reads the IMU clock again immediately before collecting 32 timestamp-matched RGBD frames. Each condition records `invocation.json` with the collector hash, exact child commands, and planned duration. A failed attempt leaves its directory and logs intact; inspect those before any new attempt. Do not reuse or overwrite failed evidence. The collector does not recreate the simulator: the two explicit `run-markerfree.sh` commands above do, so wait for all activity to end before running them.

## Conditional flight outcome

| Run | Condition | Outcome |
| --- | --- | --- |
| Hover-10 | Net visuals present | Passed all seven applicable hover checks; settled for 3.004 simulated seconds. |
| Policy-11 | Net visuals present | Passed all eight route checks, reached 15/15 waypoints, and landed/disarmed in 77 simulated seconds. Minimum modeled window clearance was 82.57 mm. |
| Hover-11 | Net visuals removed; collisions retained | Tracking failed 5.332 simulated seconds after arming. The hover did not settle. PX4 subsequently confirmed touchdown and disarm after abort. |
| Policy-12 | Net visuals removed | Not run because the hover prerequisite failed. |

The [flight evidence](flight-results.json) preserves the complete sequence, hashes, and configuration checks. Watch the [successful control route](../../media/window-policy-net-control.mp4) and the [failed treatment hover](../../media/net-removed-hover-failure.mp4). Both videos include captures outside the arm-to-flight interval.

The treatment's saved failure frame reports unobservable survey geometry, insufficient overlap for relative depth matching, and insufficient RGB tracking inliers. These are diagnostics from one rejected frame, not aggregate failure counts. The independent scorer certified the modeled abort path from the fault at 24.868 s through confirmed disarm at 29.536 s. Floor contact was permitted during landing. Contact-message coverage was partial, so this geometry result is not proof of zero physical contact.

This controlled visual ablation exposed a failure in the treatment scene. It does not establish repeatability, isolate missing depth constraints from the deliberate map/sensor mismatch and tracker fallback, or predict physical camera behavior. No controller, estimator threshold, or policy was retuned to obtain a passing treatment run.

The saved [rejected capture](first-rejected.npz) and [rank diagnostic](observability.json) are included for a small offline reproduction. At capture time 24.852 s, the fit had 322 matched depth samples and 10.32 mm plane residual, but its recorded linearized matrix had an exactly zero ENU-Y translation row and column. The surfaces supplied no north–south translation constraint in that fit, including its ToF and tilt terms. A numerically tiny negative eigenvalue is roundoff around this singular direction. Lowering the conditioning threshold would not supply the missing information. This statement concerns one recorded fit, not every possible estimator or the physical course.

With NumPy installed, recompute it without Gazebo:

```sh
python3 docs/experiments/net-visual-ablation-01/observability.py \
  --output /tmp/net-visual-observability.json
```

## Startup observation

A [post-hoc comparison of sampled pre-arm telemetry](prearm-comparison.json) found 133/133 valid control-hover samples reporting `depth_survey_tof`, versus 145/145 valid treatment-hover samples reporting `rgbd_odometry`. The later tracked-export control had 134/134 valid samples reporting `depth_survey_tof`. These counts are sampled telemetry, not a complete per-capture ledger; they cannot exclude a short intervening survey success.

The current arming gate requires fresh valid camera pose and PX4 state, but does not separately qualify recent accepted absolute survey updates. A candidate improvement is to record accepted survey updates at capture cadence and require a stable recent qualification before arming. That change is not implemented or validated here, and would not solve loss of geometric constraints later in flight. The failed treatment and its acceptance criteria remain unchanged.
