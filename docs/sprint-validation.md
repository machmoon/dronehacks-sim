# Marker-free sprint validation

The frozen marker-free sprint profile completed **five consecutive full-course
runs in 57–58 simulated seconds**, with all 15 waypoints, the window, autonomous
landing/disarm and all eight independent score checks in every run. Each run
started from a fresh simulator reset. There were no failed attempts in this
five-run frozen batch. The [machine-readable audit](sprint-repeatability.json)
checks unique run IDs, unchanged controller settings and identical runtime source
hashes. Runtime source comes from `d0dd5f56b8ff943fb6278d5ec9d4e2bf46ab90fe`;
the frozen sprint profile and auditing tools were added in `83e9f3a`.

| Frozen trial | Armed-to-disarm time (s) | Minimum modeled window clearance (mm) | Checks |
|---|---:|---:|---:|
| 1 | 58.0 | 106.3 | 8/8 |
| 2 | 57.0 | 108.8 | 8/8 |
| 3 | 58.0 | 115.6 | 8/8 |
| 4 | 58.0 | 109.6 | 8/8 |
| 5 | 58.0 | 122.1 | 8/8 |

This is repeatability on one rendered course, starting pose and sensor setup.
It does not establish a population success rate or real-drone performance.
Additional synthetic depth-noise trials are recorded separately when complete.

The candidate keeps all 15 checkpoint positions, the 60 mm estimated arrival
radius, the 80 mm independent route-verification radius, and the 15 mm inflated
multipart x500 envelope. It still requires three seconds of settled hover after
takeoff and ends only after PX4 confirms touchdown and disarm on the pad.
Timing uses the entire armed interval in simulation seconds.

## Controller changes

The separate `mission/course-window-sprint.json` profile increases the velocity
request ceiling from 1.0 to 1.25 m/s, command acceleration from 0.8 to 2.0 m/s²,
position feedback gain from 1.5 to 3.2, and measured-velocity damping from 1.0 to
2.0. The yaw-rate limit rises from 0.7 to 1.0 rad/s. Per-leg speed caps remain
explicit: maze cruise is 0.792 m/s, window alignment 0.37 m/s, and crossing
0.52 m/s. These are simulator settings, not measured hardware limits.

The earlier high-speed candidate with weaker damping overshot the first turn
and failed the unchanged course-boundary check. Its aborted run is retained.
Stronger damping anticipates braking from measured motion before tight turns.
The subsequent development run completed in 59 seconds with all eight score
checks, before the frozen candidate added modest speed to open approach legs.
Development attempts are separate from the repeatability batch.

## Acceptance and reproduction

Every accepted trial must finish below 60 seconds, reach all checkpoints in
order, clear the full window and obstacle proxies, land and disarm inside the
pad, and have a complete independently recorded path. The evidence runner and
all recorders must exit cleanly. No simulator truth feeds the flight controller.

The repeatability audit requires at least five unique run IDs and directories,
identical controller/estimator settings and runtime source hashes, matching
score/telemetry run IDs and timing, and all eight named geometry checks. A
failed or missing attempt remains in the report. Repeated simulations do not
establish a real-flight reliability guarantee or unseen-course generalization.

Follow the [Mac setup](mac-quickstart.md), including the patched PX4 image and a
verified marker-free hover. Then run one new, recorded trial:

```sh
bash runtime/run-sprint-trial.sh sprint-repeat-01 artifacts/markerfree/YOUR_HOVER/telemetry.summary.json
```

The script uses the isolated `dronehacks-sprint` Compose project on port 8084.
It does not start a physical aircraft. It refuses an existing run directory and
scores the trial after recording closes. For synthetic depth noise, set
`SIM_RGBD_DEPTH_NOISE=gaussian-10mm` and use a hover proof from that condition.

Audit the exact ordered batch, including every attempt under the frozen profile:

```sh
python3 -m tools.report_sprint \
  artifacts/markerfree/sprint-repeat-01 artifacts/markerfree/sprint-repeat-02 \
  artifacts/markerfree/sprint-repeat-03 artifacts/markerfree/sprint-repeat-04 \
  artifacts/markerfree/sprint-repeat-05 --output artifacts/sprint-repeatability.json
```

## Transfer limits

These tests use rendered RGB-D, the published static survey, and an x500
surrogate. The separate Sky Beta model and physical OAK-D camera are not
validated by the sprint results. Partial rotor contact-topic coverage remains;
the clearance result comes from the independent geometric scorer. The known
missing-net-return localization failure remains documented in the transfer notes.
