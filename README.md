# dronehacks-sim

A local Gazebo Harmonic simulation of the DroneHacks course. The default world has **no artificial AprilTags**, matching the user's course constraint. Marker-free autonomous PX4 navigation is **not connected yet**. The current live view is a disarmed camera/course preview. The project is public at [machmoon/dronehacks-sim](https://github.com/machmoon/dronehacks-sim).

## Sky Beta aircraft

The default marker-free scene now uses the official Sky Beta CAD, component-based
mass/inertia estimates, matching PX4 motor geometry, and forward stereo camera
proxies. Physical coefficients remain uncalibrated. See [the physics profile and
repeatable Gazebo checks](docs/skybeta-physics.md). Historical marker diagnostics
still select the x500 explicitly; the learning simulator remains separate.

A [compact SLAM experiment](docs/sparse-slam-experiment.md) adds bounded sparse
metric mapping, relocalization, recorded-camera replay, and a reduced-budget DPVO
profile. Local synthetic tests are available; Jetson and real-camera validation
remain pending.

## Current course preview

Install Docker CLI with Compose and [Colima](https://github.com/abiosoft/colima). From this directory, build and start the dedicated PX4 simulation once:

```sh
./runtime/run-px4.sh
```

In another terminal, start the local viewer:

```sh
./web/run.sh
```

Open the viewer at [http://localhost:8081](http://localhost:8081). After the initial build, start the course and raw camera preview with:

```sh
./runtime/preview-marker-free.sh
```

The preview sends no flight commands and publishes no artificial pose. See [current status](docs/marker-free-status.md). The default flight replay refuses to run because its old estimator depends on markers that do not exist on the real course.

## Five-hour Mac learning run

The [learning experiment](learning/README.md) trains a small depth policy using supervised imitation and student rollouts. Its inputs are synthetic forward depth, simulated state estimates and a local waypoint. Training uses course geometry proxies; it does not run Gazebo or PX4 and has no real-camera validation.

With a native Python environment containing NumPy and PyTorch, launch a bounded background run:

```sh
python3 runtime/train_overnight.py --python /path/to/venv/bin/python --seconds 18000 --device mps
```

The viewer shows current training progress separately from the Gazebo preview. Results, source hashes, checkpoints and logs stay in `artifacts/learning/`. The supervisor prevents duplicate jobs, enforces the time cap, and keeps the Mac awake while training. Keep the Mac plugged in with its lid open. To stop and save the current checkpoint:

```sh
python3 runtime/train_overnight.py --stop
```

See the [competition references and SAND-O status](docs/competition-reuse.md) for the reusable projects investigated and their limits.

## Historical marker-assisted diagnostic

The following explicit opt-in recreates the earlier artificial-marker fixture. Its completed flights verify the PX4 external-vision connection, not navigation on the actual course:

```sh
SIM_TEST_MARKERS=1 ./runtime/replay-px4.sh course
SIM_TEST_MARKERS=1 SIM_GUI=0 ./runtime/replay-px4.sh hover --blackout-after 18
```

See [docs/px4-camera.md](docs/px4-camera.md) for the navigation architecture, verified results, and limitations. See [docs/vision.md](docs/vision.md) for camera calibration and marker-map details.

## Historical PX4 diagnostic results and limits

Verified runs include a hover-height acquisition in 30.040 simulated seconds and a GUI run completing all 15 waypoints in 94.696 simulated seconds at 640×480 camera resolution. The camera-loss check stopped forwarding pose measurements at 18 simulated seconds while the vehicle was climbing at about 0.5 m; the route aborted 0.568 simulated seconds later, and PX4 reported landing and disarming. These runs confirm external-vision position, height, and yaw fusion with GNSS, barometer, range, and optical-flow fusion inactive. Waypoint completion uses PX4 estimated position; collisions and swept clearance were not scored. The native 480×360 camera run also completed all 15 waypoints in 96.376 simulated seconds. See [docs/px4-results.json](docs/px4-results.json) for the recorded runs.

This is a camera-loop baseline, not a complete sim-to-real validation. It adds a downward-facing monocular camera and surveyed AprilTag markers to the Gazebo course. The physical aircraft uses a forward-facing OAK-D-S2 stereo camera with cuVSLAM visual-inertial odometry on Jetson. The stock x500 is not the physical airframe. These runs do not validate the real camera stack, real flight, collision safety, or course clearance.

The camera runs at 30 Hz of simulation time. A 650-response MAVLink time-sync warmup at 50 Hz completes before arming. Software rendering may advance simulation slower than wall time, so use simulation timestamps when comparing camera and vehicle data. The Python course-control core is shared, but this SITL path uses MAVLink and PX4 position feedback; the physical cuVSLAM path uses a ROS 2 bridge.

## Gazebo ground-truth comparison mode

The older non-PX4 demo uses the Open Robotics X3 v4 quadrotor and follows the course from Gazebo ground-truth odometry. To run it, start the simulation with `./runtime/run-mac.sh`, then launch the same viewer with `./web/run.sh`. It also uses [http://localhost:8081](http://localhost:8081), with the Gazebo desktop at [http://localhost:8080/vnc.html?autoconnect=1&resize=scale](http://localhost:8080/vnc.html?autoconnect=1&resize=scale).

That separate baseline completed all 15 waypoints in 61.780 simulated seconds with zero obstacle contacts and 0.0905 m clearance at the padded window check. The optional tuned window profile completed in 47.736 simulated seconds with zero contacts and 0.0906 m clearance. These ground-truth results are not directly comparable to PX4 camera runs: vehicle models, navigation sources, and scoring differ. Setting speed scale to 1 on the default ground-truth route failed the conservative window-clearance check. See [docs/results.json](docs/results.json) for details. Isaac Sim and ROS 2 are not required for this comparison demo. Linux Docker Compose may work, but the setup has been tested on Apple Silicon macOS with Colima.
