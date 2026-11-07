# Marker-free RGBD simulation baseline

This is a classic RGBD visual-odometry baseline, not a SOTA claim or a demonstrated autonomous course run. It uses natural image corners, optical flow, metric optical-Z depth, and RANSAC PnP. It adds no markers or course texture. It never subscribes to simulator pose, model state, odometry, or joint state. It uses Gazebo's rendered RGBD camera as its sensor; ideal rendered depth is easier than a physical stereo camera and must be described as such in results.

## What is implemented

`vision.rgbd_scene.generate` wraps the existing scene generator with `with_test_markers=False`, then changes only the downward camera's sensor type and topic. The mounting pose, field of view, dimensions, near/far clipping, course geometry, and PX4 x500 remain unchanged. Even `SIM_TEST_MARKERS=1` does not add markers through this wrapper.

`vision.rgbd_odometry.Vision` implements the existing `poll()`, `sim_usec_now`, and `close()` interface used by `CameraWorker`. It subscribes only to `/dronehacks/rgbd/image`, `/dronehacks/rgbd/depth_image`, and `/world/dronehacks/clock`. Color and depth must have identical nanosecond capture timestamps. At most eight unmatched frames per stream are retained. The RGBD topic convention follows the [official Gazebo bridge example](https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim_demos/config/rgbd_camera_bridge.yaml).

Valid records use `position_enu`, `orientation_enu_flu_wxyz`, `orientation_xyzw`, `sim_usec`, and `stamp_sim_s`, compatible with the current camera/PX4 transform. Optical-flow forward/backward consistency, RANSAC inlier count and ratio, geometry spread, reprojection, current-depth agreement, motion limits, capture order, and capture age gate each update. The optical/body camera lever arm is included when returning body position.

A tracking failure produces `valid=false` and null position/orientation. It stays failed until an explicit reset with a newly established stationary pose. It does not silently move the world origin. The mission must stop on invalid or missing measurements, using its existing watchdog; the adapter itself never issues flight commands. CameraWorker retains latest records, so the consumer must check `valid` and capture age.

## Integration still needed

1. Generate a separate world with `python3 -m vision.rgbd_scene --output artifacts/course-rgbd.sdf`. Start it only through the coordinated simulator launcher after the current run finishes.
2. Use `CameraWorker(vision.rgbd_odometry.Vision(known_launch_position_enu, known_launch_quaternion_wxyz))` in a separate selected backend. Do not pass `output_dir` to this adapter; preview writing is not implemented.
3. Supply a genuinely known stationary **base_link** launch pose, not model-root height or a queried simulator pose. The x500 model-root/base-link height difference matters. Initial pose is emitted only once texture/depth are usable. All subsequent motion comes from images and depth. The caller must ensure the drone is stationary at the first paired capture.
4. First observe static tracking with `python3 -m vision.rgbd_odometry --launch-position X Y Z --seconds 20`. These arguments are numeric measured/surveyed values, not literal placeholders. This command only prints JSON and never connects to PX4.
5. Measure motion accuracy and tracking failures on the actual marker-free course before attempting closed-loop flight. Ground truth may be recorded by a separate evaluator for scoring only. Do not feed it into this estimator.

Blank floor, motion blur, repetitive texture, shadows, and long gaps may make this conservative baseline fail to initialize or latch lost. There is no global relocalization, loop closure, map correction, IMU integration, or calibrated accumulated covariance. Repeated small errors can drift even when local checks pass. Existing fixed camera covariance is not a validated RGBD VO uncertainty estimate. A separate integration must address that before claiming quantified confidence. The physical Jetson cuVSLAM path remains separate; this synthetic downward camera is not an exact replica of the drone's forward stereo sensors.

## Verification

`python3 -m unittest tests.test_rgbd_odometry -v` passes 13 tests in the local Python environment. Tests recover known metric rigid motion with image noise and outliers, allow a planar floor, verify an image translation produces the correct body-axis translation, preserve a stationary launch pose, reject collinear geometry/majority outliers/implausible motion/nonfinite data/inconsistent depth, enforce exact timestamp pairing, and latch tracking loss until explicit reset. Scene generation verifies no markers and preserved mounting.

These are synthetic unit tests. Gazebo RGBD streaming and a complete marker-free closed-loop course run have not yet been validated by this change.
