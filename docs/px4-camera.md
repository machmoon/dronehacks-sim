# PX4 camera localization demo

**Historical diagnostic:** this flight path requires artificial mapped markers that do not exist on the real course. They are now disabled by default. Use `runtime/run-markerfree.sh` and [the current RGB-D runbook](markerfree-rgbd.md) for the active experiment. Set `SIM_TEST_MARKERS=1` only when explicitly reproducing this old fixture. Marker-free hover works; full-course evidence is tracked separately.

## Start and replay

On macOS with Docker Compose and Colima installed, run once from the repository directory to start the dedicated PX4 SITL stack and build its images:

```sh
./runtime/run-px4.sh
```

In a separate terminal, start the local viewer:

```sh
./web/run.sh
```

Open [http://localhost:8081](http://localhost:8081). The viewer selects the PX4 camera telemetry when that log is newer than the ground-truth log. To reset the container and start a background run, use:

```sh
SIM_TEST_MARKERS=1 ./runtime/replay-px4.sh hover
SIM_TEST_MARKERS=1 ./runtime/replay-px4.sh course
```

The helper writes mission output to `artifacts/px4/mission.log`; follow it with `tail -f artifacts/px4/mission.log`. It accepts extra mission arguments after the task. For a headless camera-loss check, cut pose forwarding after 18 simulated seconds:

```sh
SIM_TEST_MARKERS=1 SIM_GUI=0 ./runtime/replay-px4.sh hover --blackout-after 18
```

Replay recreates the simulation container. Use it to begin each run rather than starting another mission process in an active container.

## Navigation path

The stack builds official PX4 v1.16.2 SITL with the Gazebo x500. The generated scene adds a downward-facing RGB camera and surveyed AprilTag 36h11 markers. OpenCV detects mapped marker corners and estimates camera pose with `solvePnP`; a fixed camera-to-body transform produces vehicle position and attitude. The mission forwards fresh image-derived poses over MAVLink to PX4 EKF2, then sends Offboard velocity setpoints using PX4 `LOCAL_POSITION_NED` as feedback. Gazebo ground-truth position is not used for navigation or control.

Verified estimator flags enable external-vision position, height, and yaw. GNSS position and velocity, barometer height, range height, and optical flow are disabled. A 650-response MAVLink time-sync warmup at 50 Hz completes before arming. The simulated camera runs at 30 Hz. With software rendering, simulation may advance slower than wall time; compare measurements by their simulation timestamps.

The Python course-control core is shared with the real system, but the PX4 SITL transport and feedback path use MAVLink. The physical cuVSLAM system uses a ROS 2 bridge. This camera loop does not implement the real OAK-D-S2 stereo cuVSLAM stack.

## Verified results and limits

Recorded results include a headless 640×480 run reaching hover height after 30.040 simulated seconds and a GUI 640×480 course run completing all 15 waypoints in 94.696 simulated seconds. These runs confirm PX4 heartbeat, motor control, and camera-aided EKF operation. The camera-pose-loss test stopped forwarding pose measurements at 18 simulated seconds while climbing at about 0.5 m. The route aborted 0.568 simulated seconds later; PX4 reported landing and disarming, and external-vision fusion stopped.

The native 480×360 camera run also completed all 15 waypoints in 96.376 simulated seconds. Mission timing starts when the bridge confirms armed Offboard mode and excludes the pre-arm camera and clock warmup. The current verified runs end at a position hold above the landing pad, not a landing. Waypoint completion is judged from PX4 estimated position; the runs do not score collisions or swept clearance. See [px4-results.json](px4-results.json) for recorded details.

This is a camera-loop baseline, not complete sim-to-real validation. The demo uses an added downward-facing monocular camera and course markers. The physical aircraft uses a forward-facing OAK-D-S2 stereo camera and cuVSLAM visual-inertial odometry on Jetson; the stock x500 model is not that airframe. No hardware flight, collision-safety, or course-clearance claim follows from these runs.

For marker layout, calibration, frame conventions, and image diagnostics, see [vision.md](vision.md).
