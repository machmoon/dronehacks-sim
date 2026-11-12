# dronehacks-sim

Gazebo Harmonic and PX4 SITL for the DroneHacks window route, running locally on an Apple Silicon Mac. Navigation uses rendered forward RGB-D, downward ToF and PX4 EKF2. The active simulator contains no artificial AprilTags and supplies no simulator pose to navigation. A separate observer records truth and contacts for offline scoring.

The public repository is [machmoon/dronehacks-sim](https://github.com/machmoon/dronehacks-sim). The current experiment is on `codex/markerfree-flight`.

## Verified flights

| Run | Controller | Arm to landing and disarm | Verified waypoints | Minimum modeled window clearance |
| --- | --- | ---: | ---: | ---: |
| Course-08 | Waypoint baseline | 83.000 s | 15/15 | 106.3 mm |
| Course-09 | Waypoint baseline repeat | 82.000 s | 15/15 | 109.6 mm |
| Policy-03 | Learned depth policy with boundary braking | 77.000 s | 15/15 | 93.3 mm |
| Policy-05 | Packaged policy repeat, patched PX4 | 77.000 s | 15/15 | 91.9 mm |
| Policy-07 | Packaged v1, Gaussian 10 mm depth noise, FALS7, patched PX4 | 78.000 s | 15/15 | 100.9 mm |
| Policy-08 | Packaged v1, nominal depth, FALS7, 120 ms continuity limit | 78.000 s | 15/15 | 92.3 mm |
| Policy-09 | Packaged v1, Gaussian 10 mm noise, FALS7, 120 ms continuity limit | 78.000 s | 15/15 | 95.3 mm |
| Policy-10 | Same noisy-depth configuration, +40 mm declared launch-position error | 78.000 s | 15/15 | 94.7 mm |
| Policy-11 | Noisy-depth control for the net-visual experiment | 77.000 s | 15/15 | 82.6 mm |
| Clean-export policy-01 | Tracked source export, noisy depth, separate v1 inference service | 78.000 s | 15/15 | 98.2 mm |

All listed runs passed the eight independent route, multipart geometry, window and landing checks. Times use simulation timestamps and include takeoff and touchdown. Window clearances include 15 mm inflation of the modeled x500 envelope. Contact-topic coverage was partial; these geometry results do not establish zero physical collisions or hardware readiness.

Policy-03 accepted 364 learned actions. Its separate braking guard changed 193 of 4,689 outgoing setpoints, or 4.1%. The earlier unguarded policy-02 overshot the north boundary and aborted after 6/15 waypoints. The padded rotor model also overlapped the north-wall proxy during that abort. The successful result belongs to the combined policy and guard.

The [complete trial inventory](docs/flight-trials.md) also includes failed attempts and separate hover and blackout diagnostics. Earlier attempts include a takeoff stop when PX4's periodic battery/power publications froze and policy-06's noisy maze observability failure; both remain documented. Policy-07 subsequently completed a noisy-depth route, passing all eight independent checks. Policy-09 repeated that result with the tighter 120 ms camera-continuity limit in 78.000 simulated seconds and 95.3 mm minimum modeled aperture clearance. It used synthetic zero-mean Gaussian depth noise (10 mm standard deviation), FALS7 survey normals, the separate boundary-braking guard, and patched PX4. These are individual simulator trials, not a reliability estimate or physical-camera results. Contact-topic coverage was partial, so it does not establish zero collisions.

[Watch the learned-policy RGB/depth replay](docs/media/window-policy.mp4), the [baseline replay](docs/media/window-baseline.mp4), or the [noisy policy-09 replay](docs/media/window-policy-noisy-120ms.mp4). The [Policy-03 score](docs/markerfree-policy-results.json) records the checkpoint, guard settings, sensor timestamps and source hashes. Read the [flight evidence and reproduction commands](docs/markerfree-rgbd.md), [baseline score](docs/markerfree-results.json), and [braking model and limits](docs/policy-braking.md). The [section comparison](docs/flight-comparison.md) separates cruise time from takeoff and landing. The matched FALS7 [control](docs/fals7-control-results.json) and [noisy-depth](docs/fals7-noisy-120ms-results.json) records cover Policy-08/09. For locally retained run artifacts, the original three-run timing subset can be regenerated with:

```sh
python3 -m tools.section_timings artifacts/markerfree/course-08 \
  artifacts/markerfree/course-09 artifacts/markerfree/policy-03
```

Separate [contact-sensor controls](docs/contact-sensors.md) observed the exact expected collider pair on all nine body and rotor collision sensors. The displaced-block control also passed. These short isolated drops verify the sensor wiring; they do not replace the flight geometry checks or establish complete contact delivery during flight.

A separate injected camera-aid blackout passed eight fault-response checks. PX4 entered Auto Land, touched down and disarmed without a logged external-vision update after injection. [Fault-test evidence](docs/blackout-results.json) includes the timestamps, hashes and observation limits.

The later [learned-policy camera-aid loss test](docs/experiments/policy-blackout-02/README.md) reached the expected freshness fault, landed and disarmed, but did not pass its full assessment. Eight of nine checks passed; a 204 ms gap in the independent pose recording prevented certification of the entire abort path. The failed result and unchanged scoring threshold are retained.

The [repeat with improved recording](docs/experiments/policy-blackout-03/README.md) certified the modeled abort path but also failed one strict check: a pre-cutoff camera sample reached PX4 4 ms after injection. The send count stayed fixed and no later updates appeared. Both attempts are preserved rather than combined into a single passing result.

The [launch-prior diagnostic](docs/experiments/launch-prior-01/README.md) passed a hover prerequisite and the full route with a 40 mm error in the declared launch position. The simulated spawn stayed fixed. Policy-10 took 78 simulated seconds and passed all eight independent checks; its [RGB/depth replay](docs/media/window-policy-launch-prior.mp4) includes the paired ToF reading. This supports that one initialization-error case, not broader reliability.

The [net-visual ablation](docs/experiments/net-visual-ablation-01/README.md) showed a localization failure in the simulated treatment scene. Its control completed the route in 77 seconds. Removing the rendered net surfaces while retaining their collisions caused a large depth-image change, and the treatment hover lost tracking 5.332 seconds after arming. PX4 confirmed touchdown and disarm after abort; the treatment route was not attempted. The current stack has not demonstrated flight in that missing-net-return scene. This is a simulator stress case, not a physical net optical model.

The [net-removal failure replay](docs/media/net-removed-hover-failure.mp4) and [frame diagnostic](docs/experiments/net-visual-ablation-01/observability.json) preserve the failure. The rejected depth fit lacked a north–south translation constraint. This single control/treatment pair does not establish repeatability or separate missing net returns from the deliberate map/sensor mismatch and tracker fallback. The [OAK stereo handoff](docs/net-transfer-next-steps.md) identifies the existing cuVSLAM localization path and the separate depth-adapter work still required.

An optional [absolute-survey startup check](docs/experiments/survey-startup-01/README.md) subsequently passed the control hover and refused to arm in the net-removed scene. It requires three continuous seconds of accepted map tracking before arming. It remains opt-in and has not been evaluated on a full route; it does not restore navigation after tracking becomes unobservable.

The [tracked-export reproduction](docs/experiments/clean-export-01/README.md) ran the collector, hover, scorer and policy from an export containing only committed files. Its full route passed all eight checks in 78 simulated seconds with 98.2 mm modeled window clearance. It reused the installed simulator image and Python dependencies; it was not a clean-machine build. [Watch that flight](docs/media/window-policy-clean-export.mp4).

## Run locally

For a fresh clone, follow the [Apple Silicon setup guide](docs/mac-quickstart.md) to build the Gazebo/PX4 images and install the host policy dependencies. The commands below assume those images already exist. The Mac setup uses Docker and Colima in a separate Compose project. Start the simulator and viewer in separate terminals:

```sh
SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2 \
  SIM_RGBD_DEPTH_NOISE=gaussian-10mm ./runtime/run-markerfree.sh
python3 -m web.server
```

Open [http://127.0.0.1:8083](http://127.0.0.1:8083). Starting the simulator does not arm the aircraft. Follow the [runbook](docs/markerfree-rgbd.md) for explicit hover, course, frozen-policy and independent-scoring commands. The viewer follows `artifacts/markerfree/current` and identifies the controller and guard intervention count.

The [tested checkpoint package](docs/policy-checkpoint.md) is included for reproduction without retraining. The latest verified flights used the [separate patched PX4 image](docs/px4-hrt-backport.md), selected explicitly with `SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2`.

The lightweight [unit-test workflow](.github/workflows/markerfree-tests.yml) runs on Ubuntu. Torch-dependent tests are skipped in that job; Gazebo flight results are recorded separately.

## Bounded Mac training

The [learning experiment](learning/README.md) trains a 30,003-parameter depth policy using teacher-supervised imitation and student rollouts. It is not Dreamer, reinforcement learning or a learned world model. The actor sees depth and estimated state; privileged geometry is used by the training teacher only. The [bounded training ledger](docs/training-ledger.json) records four supervised jobs totaling 3.928 hours inside the five-hour session window; this was cumulative job time, not five uninterrupted hours. The paired eight-case analytic review found v1 and v2 each at 7/8 with one collision. V2 had more successful-window clearance but was slower (61.44 s vs. 60.00 s median), so it was not promoted or flown; v1 remains the flight baseline. See the [paired results](docs/paired-policy-results.json) and [checkpoint package notes](docs/policy-checkpoint.md).

The current `x500-transfer-delay` profile uses the centered route, multipart x500 envelope, 0.34 m forward camera, queued command delay and randomized velocity response. It trains with analytic depth and dynamics. Policy-03 uses the earlier validation-selected v1 checkpoint; the new v2 training result is evaluated separately.

With NumPy and PyTorch installed in a native Python environment:

```sh
python3 runtime/train_overnight.py --python /path/to/venv/bin/python --seconds 18000 -- \
  --profile x500-transfer-delay --device mps --threads 1 --seed 45 --eval-interval 300
```

Use the remaining wall-time budget when resuming. The supervisor prevents duplicate jobs, caps runtime, monitors free disk and keeps the Mac awake. Keep the Mac plugged in with its lid open. To stop and save:

```sh
python3 runtime/train_overnight.py --stop
```

## Physical transfer and reused work

These runs demonstrate transfer from an analytic training model to Gazebo/PX4. Physical-camera transfer remains unvalidated. Gazebo's ideal depth, opaque net proxies, the known static survey and the x500 surrogate differ from the real OAK-D camera and Skybeta aircraft. The [OAK depth handoff](docs/oak-depth-handoff.md) provides passive capture checks and the required calibration evidence. The [timing audit](docs/timing-budget.md) separates simulated camera age from host policy latency and lists the timing measurements still needed on the real platform.

The [competition references](docs/competition-reuse.md) and [local SAND.O review](docs/sando-reuse.md) explain which projects were investigated. SAND.O's full ROS 2/Gurobi planner is not integrated. Its acceleration-limited graph model is useful as a stopping-distance reference, but does not account for this command delay.

Historical marker-assisted and ground-truth diagnostics remain documented in [PX4 camera diagnostics](docs/px4-camera.md) and [older result metadata](docs/results.json). They are separate from the current marker-free results.
