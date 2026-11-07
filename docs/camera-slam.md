# Camera-driven Sky Beta navigation

Sky Beta now supports synchronized RGBD → camera/body frame conversion → MAVLink
external vision → PX4 EKF2 → Offboard waypoint control. Camera geometry comes from
the CAD model; navigation reads PX4's estimate, never Gazebo vehicle pose.

Two explicit modes are available:

- `--localizer survey`: depth matched against the supplied static course map,
  with ToF and optional capture-aligned inertial roll/pitch. This is map-based
  visual localization, not autonomous SLAM. It completed a Sky Beta takeoff,
  three-second settled hover, landing and disarming in 21 simulated seconds.
- `--localizer sparse`: the bounded 192-feature/eight-keyframe mapper through the
  new `vision/slam_odometry.py` bridge. It builds its own local map without loading
  course geometry. It initializes from the live course camera after skipping
  Gazebo's black startup image. Its flight test supplied 402 vision measurements
  but required relocalization 5.116 simulated seconds after arming; the bridge
  rejected that recovery and PX4 landed/disarmed. Map-free hover is not validated.

Both require a known stationary launch placement to establish course coordinates.
Gazebo RGBD supplies ideal optical depth; this is not DepthAI stereo or a physical
OAK-D connection. No artificial textures or tags were added. Real camera/IMU
calibration and Jetson validation remain outstanding.

## Frames, timing and failure handling

Sky Beta's left optical center is approximately `[0.10514, 0.0375, 0.02279]` metres
in body FLU, with 80° horizontal FOV and 384×256 images at 20 Hz. Its provisional
ToF mount is `[0.10514, 0, -0.03]` m. Scene generation and estimation import the
same calibration selected by `SIM_AIRFRAME`; restart both when changing aircraft.
Camera lever-arm translation is rotated with attitude before ENU/FLU → NED/FRD
conversion. Measurements retain their capture time and carry covariance.

Sparse SLAM waits through unobservable startup frames. After initialization,
tracking loss latches invalid and recovery pose corrections are rejected; restart
requires a newly established stationary launch. Its uncertainty is an uncalibrated
distance/time budget, not a measured statistical model. Raw `flight_ready: false`
mapper diagnostics remain intentional: the adapter is a guarded SITL experiment.

The flight runner rejects invalid observations, verifies aiding parameters, and
requires fresh camera/PX4 estimates before arming. Loss in flight requests Land. Logging failures also abort; a full disk or failed
console write cannot interrupt the Land request and touchdown monitoring.
GPS, barometer, rangefinder and optical-flow EKF aiding are disabled; ToF is used
inside the camera estimator. Optional inertial tilt excludes PX4 position/yaw.
The navigation audit rejects pose/odometry plugins and requires no publisher on
PX4's simulator odometry topic. Only a separate offline evaluation recorder reads
Gazebo's scene poses. The camera adapter consumes RGB, depth, range and clock.

Provisional Gaussian IMU, magnetometer and pressure noise also fixes a concrete
Sky Beta startup problem: perfectly constant simulated values caused PX4 to mark
its sensors stuck. Noise magnitudes match the pinned baseline scale, not measured
physical-aircraft specifications.

## Run

First build the patched image as described in [the Mac guide](mac-quickstart.md).
The new runtime uses `dronehacks-markerfree:hrt-v1.16.2` and a separate compose
project/partition, so it does not restart another simulator. Startup is disarmed:

```sh
./runtime/run-camera-slam.sh
```

The software-rendering real-time factor defaults to 0.15; physics stays at 4 ms.
Capture-age and gap gates are unchanged. `SIM_AIRFRAME=x500` selects the surrogate;
its launch base-link height is 0.24 m, whereas Sky Beta's is 0.07532795197 m.
Hover evidence cannot be reused across airframes or localizers.

A camera-guided Sky Beta hover and landing:

```sh
docker --context colima-dronehacks-sim exec dronehacks-camera-slam-sim-1 \
  bash -lc 'cd /workspace && python3 -m mission.markerfree_run \
    --task hover --localizer survey --use-px4-tilt --require-survey-startup \
    --launch-position -2.5146 0.9144 0.07532795197 --launch-yaw-deg 90 \
    --output artifacts/camera-slam/hover-NEW/telemetry.jsonl'
```

Inspect sparse SLAM without flight commands:

```sh
docker --context colima-dronehacks-sim exec dronehacks-camera-slam-sim-1 \
  bash -lc 'cd /workspace && python3 -m vision.rgbd_odometry \
    --localizer sparse --launch-position -2.5146 0.9144 0.07532795197 \
    --launch-quaternion 0.7071067811865476 0 0 0.7071067811865476 --seconds 20'
```

After a successful hover, use `--task course --hover-summary PATH_TO_HOVER_SUMMARY`
with the same launch/localizer/tilt/startup settings. The existing waypoint route
retains a conservative 0.401 m footprint. Its x500 collision-envelope scorer is
not a Sky Beta clearance certificate. Learned x500 policies are rejected on
Sky Beta rather than silently transferred.

## Evidence and limits

[Recorded results](camera-navigation-results.json) include unsuccessful attempts.
The independent hover recorder measured approximately 1.9 mm camera position RMS
error in ideal rendered depth; this is not a hardware accuracy claim. The first
route attempt exceeded the unchanged 120 ms capture-gap gate with a 148 ms gap
and landed after abort. A slower-rendering retry stopped before arming because
the host ran out of disk space. A complete Sky Beta route is not validated.

All 87 selected tests passed inside the actual PX4 image. The image-to-MAVLink tests cover motion, rotated camera offsets, timestamps,
black frames, tracking loss and correction jumps. Calibration tests compare the
actual generated Sky Beta SDF with estimator parameters.

```sh
python -m unittest tests.test_slam_odometry tests.test_camera_slam_scene \
  tests.test_sparse_slam tests.test_rgbd_odometry tests.test_navigation_audit \
  tests.test_skybeta tests.test_policy_bridge tests.test_px4_contract -q
```

`python -m tools.report_camera_navigation RUN_DIRECTORY --output REPORT.json`
compares estimates against independent recorder truth offline. It neither feeds
control nor certifies clearance/collision freedom. Initialization error before
arming is included in whole-run PX4 statistics; inspect flight-only statistics
separately.
