# Camera localization in Gazebo

**Historical test fixture only.** Artificial markers are now disabled by default because they do not exist on the real course. This estimator is retained solely for explicit diagnostic opt-in; it is not the current marker-free navigation solution. See [current status](marker-free-status.md).

The PX4 run adds a downward-facing RGB camera to the official x500 model and AprilTag 36h11 landmarks to the imported course. The tags are an experimental addition to the simulated course. Each has a surveyed position and size from [vision/course.py](../vision/course.py), which both the scene generator and pose estimator use. The floor grid gives the camera overlapping coverage. Smaller tags cover the launch and landing pads close to the floor. Four tags sit just above the curved inner bottom of the tunnel, where the tunnel surface would hide floor tags. The landmarks have no collision geometry.

The simulated camera defaults to 480 × 360 RGB at 30 Hz of simulation time with a 1.95 rad horizontal field of view. Gazebo looks along sensor +X. The mount pitches it down by 90° and puts the optical center 0.075 m below the x500 base link. The estimator converts OpenCV optical right, down, and forward to body front, left, and up. `SIM_CAMERA_WIDTH` and `SIM_CAMERA_HEIGHT` set the resolution for the scene generator and the estimator. Use the same values in both processes so the estimator computes matching intrinsics from the field of view and image dimensions. [Gazebo camera coordinate convention](https://gazebosim.org/api/sdformat/14/classsdf_1_1SDF__VERSION__NAMESPACE_1_1Camera.html)

`vision.localize.Vision` subscribes to `/dronehacks/down_camera`. It decodes marker IDs and image corners and solves for camera pose using the mapped 3D corners and pinhole calibration. It applies the fixed camera-to-body transform to return the base link position in world ENU and a body-FLU-to-world-ENU quaternion in WXYZ order. It timestamps each result with the Gazebo **image** header. A separate subscription to the simulation clock supports MAVLink time synchronization. [OpenCV marker detection and pose estimation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)

A frame with no mapped tags of usable size produces no measurement. The pose checks reject points behind the camera, a camera below the observed markers, vehicle tilt above 45°, and reprojection RMS above 2 px. A single tag can give two similar planar fits with different positions. When candidates are within 0.25 px in RMS and more than 0.10 m apart, the estimator rejects the frame. The mission sends only valid, fresh samples to PX4, which fuses the pose with the simulated IMU. PX4 uses external vision for position, height, and yaw and has GNSS fusion disabled. [PX4 external vision position estimation](https://docs.px4.io/v1.16/en/ros/external_position_estimation)

## Inspect the camera

The mission writes an annotated `artifacts/vision/camera.jpg` and matching `artifacts/vision/latest.json`. The JSON reports the image timestamp, decoded tag IDs, validity, reprojection RMS in pixels, and the estimated body pose when valid. Updates are capped at 10 Hz wall time for the preview. The mission polls camera measurements separately and may process fewer than 30 frames per simulated second if the consumer or renderer is overloaded.

To inspect the image path independently inside the PX4 simulation container, run the camera subscriber with a separate output directory:

```sh
python3 -m vision.localize --output-dir artifacts/vision-check --frames 30
```

Each processed frame is printed as JSON. Stop the diagnostic subscriber when finished to free CPU for the simulator.

## Calibration checks

A stationary x500 at the launch mark gave four decoded tags across six consecutive 640 × 480 rendered frames. The estimated body position was `[-2.514601, 0.914395, 0.227211]` m, the quaternion was near identity, and the reprojection RMS was 0.413 px. This is consistent with the known launch XY and the x500 resting height of about 0.227 m. The image timestamps were 32–36 ms apart.

A separate rendering fixture placed the x500 body at `[2.3114, 0.45, 0.95]` m in the tunnel at zero attitude. The estimate used two inside-tunnel tags and returned `[2.31192, 0.43471, 0.95224]` m, a 0.0155 m position error and 0.423 px reprojection RMS. The fixture position was used only to calculate error. Uniform black, gray, and white images returned no estimate.

Reducing both saved preview images to 480 × 360 with matching intrinsics preserved all four launch tags and both tunnel tags, with 0.0012 m and 0.0264 m position error, respectively. The 320 × 240 tunnel preview yielded no usable tags. This resize check used saved JPEG previews with overlays and does not measure native rendering performance.

This baseline assumes ideal pinhole imagery, surveyed landmarks, and enough visibility to decode them. Low altitude, vehicle tilt, partial views, and occlusion can make estimates intermittent. Reprojection error measures how well image corners fit the model, so low error by itself does not establish accuracy. The hardware uses a forward stereo OAK-D-S2 camera and visual-inertial SLAM on Jetson. This monocular landmark setup does not validate that stack or flight on the physical course.
