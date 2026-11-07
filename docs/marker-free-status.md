# Marker-free PX4 course

The default PX4 scene uses the imported course geometry with the official Sky Beta CAD model and ideal forward stereo cameras. See the [physical profile and validation limits](skybeta-physics.md). Synthetic AprilTags are available only as an opt-in diagnostic fixture.

## Stationary camera preview

With the PX4 image built, start a fresh marker-free session from the repository directory and launch the viewer in a separate terminal.

```sh
./runtime/preview-marker-free.sh
./web/run.sh
```

The viewer shows the course and the live camera while Sky Beta is disarmed at launch. The preview reads Gazebo RGB frames and writes the current image and frame metadata to `artifacts/preview/camera.jpg` and `artifacts/preview/latest.json`. The session status is in `artifacts/preview/session.json`. [Open the local viewer](http://127.0.0.1:8081/)

Autonomous PX4 flight on this scene is pending a marker-free localization source. The hardware uses a forward-facing OAK-D-S2 stereo camera and cuVSLAM visual-inertial odometry on Jetson. The stationary preview verifies the scene and the image stream.

## Historical AprilTag diagnostic

Use the explicit opt-in to replay the old mapped-landmark experiment.

```sh
SIM_TEST_MARKERS=1 ./runtime/replay-px4.sh hover
```

The standalone scene generator also accepts `--with-test-markers`. Recorded AprilTag runs apply to that synthetic fixture. [Historical results](px4-results.json)
