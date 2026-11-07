# Faster marker-aided simulation run

**Status: experimental; repeatability failed. Do not treat this profile as
crash-safe or ready for hardware.** One recorded run passes the clearance checks;
an identical repeat fails at the window. The default course has not been replaced.

The `course-flowing.json` profile completed the recorded PX4 x500 course in
55.0 simulated seconds, including landing and disarm, compared with the previous
59.0-second recording: 4.0 seconds (6.8%) less time. This is an isolated,
marker-aided simulation experiment, not the real aircraft or marker-free stack.

## Changes

- Complete the last 20 cm of the initial climb while moving down the first
  corridor, after rising high enough to retain camera tracking.
- Descend to window height during the approach instead of waiting until alignment.
- Remove one redundant waypoint in the straight tunnel. All obstacles remain
  on the route; the waypoint count changes from 15 to 14.
- Move maze targets to leave room for measured braking overshoot. Preserve the
  conservative maze speed and acceleration rather than using the rejected faster
  turn settings.
- Optionally fuse velocity derived from timestamped camera poses into PX4 EKF2
  (`SIM_EV_VELOCITY=1`). A short least-squares fit supplies velocity and residual-based
  uncertainty at its middle capture timestamp. Simulator truth is used only by
  the independent offline evaluator, never for navigation.
- Require camera warmup, fresh measurements and a nearly stationary estimated
  vehicle before arming. In-flight stale-data and tracking-loss responses remain.

The velocity path uses MAVLink ODOMETRY in LOCAL_FRD, following the
[PX4 external position documentation](https://docs.px4.io/v1.16/en/ros/external_position_estimation).
It is enabled by the isolated replay script; existing position-only vision remains
the default elsewhere.

## Recorded evidence

Run `flowing-12`, ID `b3dbb0025bc24983b0e37799b7916a5b`:

| Measurement | Result |
| --- | ---: |
| Flight through landing/disarm | 55.000 s |
| Route completion | 49.696 s |
| Waypoints reached | 14 / 14 |
| Minimum sampled obstacle clearance | 4.278 cm |
| Clearance lower bound along interpolated path | 4.265 cm |
| Window aperture clearance | 5.463 cm |
| Modeled envelope overlap samples | 0 |

Evidence is under `artifacts/flowing-isolated/evaluation/flowing-12/`:
`telemetry.summary.json`, `truth.jsonl`, `geometry.json`, `score.json`, the raw camera
recording, and `ekf-flags.txt`. The exact route snapshot is the adjacent
`flowing-12-course.json`. The EKF status capture confirms external visual position,
height, yaw and velocity fusion, with GNSS fusion disabled.

Acceptance requires completed waypoints, landing and disarm, truth recording
covering the entire flight, no recording gap over 0.12 s, no modeled overlap,
at least 4 cm obstacle and window clearance, a complete window crossing, and a
time below 59 s. The obstacle bound subtracts half each interpolated edge length
from the smaller endpoint clearance; checks use at most 5 ms spacing.

More aggressive candidates were rejected for insufficient clearance or modeled
overlap. A 55-second candidate before the final turn adjustment had only 1.7 cm
clearance and was also rejected. Infrastructure/preflight failures are retained
as failed runs, not counted as successful flights.

## Repeat failure

An identical route snapshot in `flowing-13` completed in 56.0 s, reached all 14
waypoints and landed/disarmed, but **failed** the geometric acceptance test:
minimum obstacle clearance was -11.03 cm and window clearance was -13.73 cm.
Its run ID is `91a39f8015604bbd9f73e4bf82a1d9a2`. At simulation time 62.4 s,
the EKF altitude was 0.718 m while independently recorded truth was approximately
0.535 m. The 18.3 cm discrepancy developed near the window; the root cause has
not been established. This run is not accepted as clearing the course.

The two final-profile trials therefore have **one clearance pass and one failure**.
The passing video demonstrates a possible time improvement, not reliable
collision avoidance. Further estimator/vision diagnosis and repeated clearance
passes are required before promotion. Both simulators owned by other work are
running again, and the isolated speed-test simulator is stopped.

## Reproduce

From the repository root, using the existing local image and Docker context:

```sh
bash runtime/replay-flowing.sh flowing-new
```

This snapshots the route and starts an isolated simulator plus camera, truth and
mission recorders. Wait for
`artifacts/flowing-isolated/evaluation/flowing-new/evaluation.json`, then score:

```sh
../Drone008_Dronehacks/.venv/bin/python tools/score_recorded_run.py \
  artifacts/flowing-isolated/evaluation/flowing-new \
  --course artifacts/flowing-isolated/evaluation/flowing-new-course.json
```

The passing run used the available VM resources exclusively; two unrelated,
disarmed preview simulators were temporarily paused. During the repeat, another
task recreated one simulator and started a separate mission. The remaining paused
preview was restored afterward. Resource contention is a confounder, not an
explanation established for the clearance failure. Concurrent renderers can also
cause the existing telemetry watchdog to abort a run.
The 600-second startup timeout is preflight only and is excluded from race time.

Focused validation: 32 tests pass across camera velocity estimation, MAVLink frame
and timestamp handling, preflight checks, acceptance scoring, corner braking,
landing and PX4 frame conversion. Both modified shell scripts pass `bash -n`.

## Limits

The clearance calculation uses static imported CAD and an assumed 0.36 m radius
sphere, with linear interpolation between recorded truth poses. Floor and pad
surfaces are excluded; boulders use the existing convex-hull approximation.
This is evidence of clearance for the recorded simulation and its model, not a
physical contact-sensor result or a guarantee against real-world crashes. The
actual airframe envelope, course measurements, localization errors and flight
conditions still require validation before hardware use. Four centimetres is
the experiment's acceptance threshold, not a certified hardware safety margin.
