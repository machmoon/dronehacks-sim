# OAK metric-depth handoff

## What is pinned and what was observed

The Skybeta checkout pins ROS 2 Humble `depthai-ros-driver-v3` **3.3.0** and `depthai-v3` **3.7.1** in `Drone008_Dronehacks/docker/apt-pins.txt`. At the pinned [`v3.3.0-humble` commit](https://github.com/luxonis/depthai-ros/tree/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c), the pipeline factory maps `STEREO` to `pipeline_gen::Stereo` and `DEPTH` to `pipeline_gen::Depth` ([`pipeline_generator.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/pipeline/pipeline_generator.cpp#L20-L35)).

The `Stereo` factory returns left/right sensor wrappers and optional IMU/VIO; it does not create the stereo-depth node ([`base_types.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/pipeline/base_types.cpp#L113-L137)). The `Depth` factory constructs `dai_nodes::Stereo` ([`base_types.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/pipeline/base_types.cpp#L138-L170)); that node creates `dai::node::StereoDepth` and links the left/right sensors ([`sensors/stereo.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/dai_nodes/sensors/stereo.cpp#L61-L86)).

With the stereo-node defaults, topic publication is enabled and output is depth rather than disparity; lazy publication defaults on ([`stereo_param_handler.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/param_handlers/stereo_param_handler.cpp#L53-L60), [`stereo.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/dai_nodes/sensors/stereo.cpp#L159-L185)). Therefore the checked-in `i_pipeline_type: stereo` explains why its nested `stereo` parameters cannot enable a `StereoDepth` output. This is a source-level explanation, not confirmation of the binary/configuration used by the live device.

The pinned v3.7.1 [`StereoDepth.hpp`](https://github.com/luxonis/depthai-core/blob/v3.7.1/include/depthai/pipeline/node/StereoDepth.hpp#L156-L161) declares a RAW8 `confidenceMap` and describes lower values as less confident. Luxonis’s [RVC2 hardware guide](https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth) describes the reverse ordering for its stereo score. The documentation conflicts; no device measurement here resolves the actual OAK-D-S2 output polarity, and the sources do not establish that map and threshold use different encodings. Preserve the raw bytes, device/firmware version and configuration. Do not invert or use the map as a validity mask until its semantics are verified. The pinned API documents depth as RAW16 in millimeters by default, with zero invalid pixels.

The stock ROS depth node at pinned [`depthai-ros` `stereo.cpp`](https://github.com/luxonis/depthai-ros/blob/9e623b3ff9f09bf0930e889f4d9c9028e79cc16c/depthai_ros_driver/src/dai_nodes/sensors/stereo.cpp#L159-L215) does not create a confidence-map publisher in its stereo wiring. Do not assume that a `confidenceMap` node output becomes a ROS topic, or that adding its name to `--record-topic` creates one. First verify a publisher exists in the running graph. The confidenceMap is RAW8 from the stereo node, not the aligned, left-right-checked, postprocessed depth image; do not apply it pixelwise to that depth image as if coordinates and filtering stages matched. Treat it as an independent diagnostic until its geometry and processing stage are established.

There is no OAK-to-policy adapter in this repository. The simulator policy's `+Inf` means a Gazebo far/clear ray; an OAK invalid RAW16 zero must remain invalid and must never be converted to `+Inf`. Conversely, a finite OAK return behind a net means the stereo matcher saw a surface behind it. That is a separate transparency/occlusion problem, not an invalid-depth encoding; confidence, net appearance, and the observed surface must be evaluated separately.

The existing OAK-D-S2 uses its OV9282 stereo pair at 1024×640, 30 Hz for cuVSLAM. The original repo measured the stereo stream at 118 MB/s and cites an approximately 143 MB/s XLink ceiling; that is a prior repo measurement, not a new measurement with depth enabled. Adding full-resolution RAW16 depth at 30 Hz would add about 39 MB/s before confidence, rectified images, or bag overhead, so begin with a lower depth resolution or rate and measure drops and USB throughput. Keep the existing VSLAM stream intact.

## Existing capture limits

The saved local stereo snapshots contain JPEG pixels but no saved pair timestamp manifest, camera intrinsics/distortion, calibrated left-to-right transform, or device calibration export. Some were captured independently rather than as verified synchronized pairs. Existing monocular-depth visualizations are relative estimates, not metric stereo depth. These files cannot reproduce calibrated metric depth without assuming missing geometry. The next passive capture must preserve the camera calibration and message headers alongside the pixels.

## Candidate pipeline change (not yet validated)

The current checked-in YAML selects `pipeline_gen.i_pipeline_type: stereo` and enables IMU. The original repo's field observations show left/right images and IMU without depth or rectified outputs, consistent with the pinned `Stereo` factory path. A candidate source-level change is selecting `pipeline_gen.i_pipeline_type: depth`, which constructs the StereoDepth node and offers depth/disparity and rectification outputs. This is not yet validated on hardware. The `Depth` factory returns the depth node path rather than the standalone left/right sensor-wrapper path, so do not assume it preserves the exact raw topics or cuVSLAM inputs. Before changing the active device config, confirm the deployed driver version and effective parameters, inspect publishers/topic types and live message rates, and test whether the existing VSLAM stream remains intact. Do not start another `dai.Device` owner.

Candidate outputs to validate before capture:

- Existing raw stereo left/right images and their `CameraInfo`, preserving the inputs cuVSLAM already consumes if the selected pipeline continues to publish them.
- Rectified left image and RAW16 metric depth, with matching acquisition timestamps, if the deployed graph publishes them. Prefer same-resolution outputs initially; then reduce depth resolution/rate if transport or compute drops frames.
- RAW8 confidence map and optionally disparity for diagnostics, only if the running graph actually publishes them; do not substitute disparity for metric depth without a calibrated conversion.
- Existing IMU stream and `/tf_static` in the same capture. Record the device calibration from published `CameraInfo`; also record the OAK-to-body static transform. The existing mount launch argument has a zero default, so it must be checked against the real mount before using body-frame results.

Exact ROS topic names depend on the active namespace and driver graph. Discover them from the running node; do not assume a parameter name creates a topic. The official driver source and docs describe a single-device pipeline and custom pipeline plugins, which is the fallback if the default pipeline cannot instantiate the required node: [driver pipeline docs](https://docs.luxonis.com/software-v3/depthai/ros/driver), [ROS parameters](https://docs.luxonis.com/software-v3/depthai/ros/parameters), [StereoDepth node](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth).

## Passive, bounded capture

The pinned container has Humble rosbag2 **0.15.15**, the default storage plugins (including SQLite3), and the MCAP storage plugin. The existing Skybeta `record_bag.sh` explicitly selects MCAP (`-s mcap`). Humble's `--max-bag-duration` and `--max-bag-size` rotate/split files; they do not stop the overall recording. `--max-cache-size` bounds the in-memory cache, not disk use. Version 0.15.15 has no `--max-bag-files` circular-retention option. Use a wall-time-limited wrapper that sends SIGINT for clean shutdown, plus a free-space preflight and hard external disk guard. The exact installed `ros2 bag record` interface should be confirmed with `ros2 bag record --help` in the target container before invoking it.

Record only the device already owned by the Skybeta driver. Select the discovered left/right images, both `CameraInfo` topics, metric depth, confidence (and disparity if useful) only when those topics have publishers, IMU, and `/tf_static`. `--record-topic` selects an existing topic; it cannot expose a device output that the stock pipeline does not publish. Use a bounded writer queue and report dropped/unmatched pairs; do not block the camera callback or synthesize a depth image. Keep camera and depth timestamps in their original clock domain and verify synchronization from message headers. Store calibration metadata alongside the bag. This is sensor-data capture only; it must not publish flight commands or feed simulator/ground-truth pose into localization.

### Passive net/backstop comparison

With the vehicle disarmed and camera fixed, make two short, otherwise identical passive captures of a measured net and a flat backstop: **A**, net in front of the backstop; **B**, the same backstop unobstructed (move the net clear without moving the camera). Capture the stereo pair, matching `CameraInfo`, metric depth, and any confidence topic only if it is already published. Compare a small region over the net and adjacent background, preserving raw values and timestamps. Record finite returns behind net strands separately from zero/invalid pixels. This checks what the matcher returns in this setup; it does not validate a policy adapter or justify changing flight behavior.

### Source references

- Pinned apt package versions: `Drone008_Dronehacks/docker/apt-pins.txt`.
- Measured transport notes: `Drone008_Dronehacks/README.md` and `Drone008_Dronehacks/docs/debugging.md`.
- Bag wrapper and storage selection: `Drone008_Dronehacks/container_scripts/record_bag.sh`.
- Humble rosbag2 0.15.15 recorder options: [`record.py`](https://github.com/ros2/rosbag2/blob/0.15.15/ros2bag/ros2bag/verb/record.py).

## Run the passive check

Discover the actual topic names with `ros2 topic list -t` and inspect their publishers with `ros2 topic info -v`. The check requires an already aligned rectified image, metric depth and matching `CameraInfo` dimensions/frame IDs. Resizing depth alone does not satisfy this contract. Set the variables below to the discovered topics and run inside the existing ROS environment:

```sh
python3 -m tools.check_oak_depth \
  --image-topic "$RECTIFIED_IMAGE_TOPIC" --depth-topic "$DEPTH_TOPIC" \
  --camera-info-topic "$CAMERA_INFO_TOPIC" --imu-topic "$IMU_TOPIC" \
  --output /tmp/oak-depth-check.json
```

This checks relative acquisition-stamp alignment and recent wall-time arrivals; it does not prove absolute clock synchronization, depth accuracy or physical alignment. Missing depth prevents recording. For a bounded recording after a successful check, use a new manifest path and append `--record-seconds 30 --bag-output /data/oak-depth-001`. Add raw stereo, right-camera calibration, or VO topics explicitly with `--record-topic`; add confidence only after confirming that the running graph publishes it. `/tf_static` is always included. Optional ToF must be a `sensor_msgs/msg/Range` topic supplied with `--tof-topic`. The recorder uses the existing MCAP plugin, checks remaining disk space, and flushes with SIGINT at the time limit or on parent SIGTERM. No live ROS graph or hardware test has been performed for this helper.

The preflight manifest reports diagnostic finite-positive depth fractions for the central half of the image and each cell of a 3×3 grid. Bounds are half-open `[row_start, row_stop, col_start, col_stop]` pixel coordinates; integer partitions cover the full image, including odd dimensions. These spatial fractions do not change the existing global valid-pixel acceptance threshold and do not establish that valid depth sees thin net strands. A high valid fraction can coexist with a localized hole, and a finite return behind a net remains a matcher/occlusion result rather than an invalid pixel. Review the spatial fractions and raw captures before defining any downstream policy contract.
