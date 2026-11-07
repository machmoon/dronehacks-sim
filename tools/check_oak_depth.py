"""Passively preflight existing ROS 2 rectified RGB-D streams and optionally bag them."""
import argparse
from bisect import bisect_left
from collections import Counter, deque
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from uuid import uuid4


DEPTH_FORMATS = {'16UC1': (2, 'u2', .001), '32FC1': (4, 'f4', 1.)}
IMAGE_FORMATS = {'mono8': 1, 'mono16': 2, 'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}


def header(msg):
    stamp, frame = msg.header.stamp, msg.header.frame_id
    if (not frame or any(char.isspace() or char == '\0' for char in frame)
            or stamp.sec < 0 or not 0 <= stamp.nanosec < 10**9
            or not (stamp.sec or stamp.nanosec)):
        raise ValueError('missing frame ID or invalid acquisition timestamp')
    return stamp.sec*10**9 + stamp.nanosec, frame


def image(msg, depth=False):
    if depth:
        import numpy as np
        if msg.encoding not in DEPTH_FORMATS:
            raise ValueError('depth must be REP-118 16UC1 (mm) or 32FC1 (m)')
        width_bytes, dtype, scale = DEPTH_FORMATS[msg.encoding]
    else:
        if msg.encoding not in IMAGE_FORMATS:
            raise ValueError('unsupported rectified image encoding')
        width_bytes = IMAGE_FORMATS[msg.encoding]
    if (msg.width <= 0 or msg.height <= 0 or msg.is_bigendian not in (0, 1)
            or msg.step < msg.width*width_bytes or len(msg.data) != msg.height*msg.step):
        raise ValueError('invalid image dimensions, stride, endianness, or data length')
    fields = dict(width=msg.width, height=msg.height, encoding=msg.encoding,
                  step=msg.step, is_bigendian=bool(msg.is_bigendian))
    if not depth:
        return fields, None
    values = np.ndarray((msg.height, msg.width), dtype=('>' if msg.is_bigendian else '<')+dtype,
                        buffer=msg.data, strides=(msg.step, width_bytes))
    valid = np.isfinite(values) & (values > 0)
    finite = values[valid]

    def region(r0, r1, c0, c1):
        count = int(np.count_nonzero(valid[r0:r1, c0:c1]))
        total = (r1-r0)*(c1-c0)
        return dict(pixel_bounds=[r0, r1, c0, c1], valid_pixels=count,
                    total_pixels=total, valid_ratio=count/total if total else None)

    # The center half of each dimension, with integer half-open pixel bounds.
    center = (msg.height//4, (3*msg.height+3)//4,
              msg.width//4, (3*msg.width+3)//4)
    cr0, cr1, cc0, cc1 = center
    if cr1 <= cr0:
        cr1 = min(msg.height, cr0+1)
    if cc1 <= cc0:
        cc1 = min(msg.width, cc0+1)
    tiles = []
    for row in range(3):
        r0, r1 = row*msg.height//3, (row+1)*msg.height//3
        for col in range(3):
            c0, c1 = col*msg.width//3, (col+1)*msg.width//3
            tiles.append(dict(row=row, col=col, **region(r0, r1, c0, c1)))
    fields['metres_per_unit'] = scale
    return fields, dict(valid_ratio=finite.size/values.size,
                        valid_pixels=int(finite.size), total_pixels=int(values.size),
                        min_m=float(finite.min())*scale if finite.size else None,
                        max_m=float(finite.max())*scale if finite.size else None,
                        central_roi=region(cr0, cr1, cc0, cc1),
                        tiles_3x3=tiles)


def calibration(msg):
    matrices = {key: list(getattr(msg, key)) for key in ('k', 'p', 'r', 'd')}
    k, p, r, d = (matrices[key] for key in ('k', 'p', 'r', 'd'))
    if (msg.width <= 0 or msg.height <= 0 or (len(k), len(p), len(r)) != (9, 12, 9) or len(d) > 64
            or not all(math.isfinite(x) for x in (*k, *p, *r, *d))
            or min(k[0], k[4], p[0], p[5]) <= 0
            or not math.isclose(k[8], 1.) or not math.isclose(p[10], 1.)
            or not 0 <= p[2] < msg.width or not 0 <= p[6] < msg.height):
        raise ValueError('invalid or uncalibrated CameraInfo K/P')
    return dict(**matrices, width=msg.width, height=msg.height, distortion_model=msg.distortion_model)


def auxiliary(msg, role):
    if role == 'imu':
        data = (msg.angular_velocity, msg.linear_acceleration)
        if (not all(math.isfinite(x) for v in data for x in (v.x, v.y, v.z))
                or msg.angular_velocity_covariance[0] == -1 or msg.linear_acceleration_covariance[0] == -1):
            raise ValueError('IMU needs finite gyroscope and accelerometer samples')
    elif not (math.isfinite(msg.range) and 0 <= msg.min_range <= msg.range <= msg.max_range < math.inf):
        raise ValueError('ToF range is not finite and within its declared limits')


class Probe:
    def __init__(self, topics, pair_ms=25., auxiliary_ms=50., valid_ratio=.25, minimum_pairs=5):
        self.topics = topics
        self.pair_ns, self.aux_ns = pair_ms*10**6, auxiliary_ms*10**6
        self.valid_ratio, self.minimum_pairs = valid_ratio, minimum_pairs
        self.counts, self.bad = Counter(), Counter()
        self.errors, self.last, self.fields = [], {}, {}
        self.pending = {key: deque(maxlen=32) for key in ('image', 'depth')}
        self.aux_stamps = {key: deque(maxlen=4096) for key in ('imu', 'tof') if key in topics}
        self.pair_stamps = deque(maxlen=128)
        self.pairs = self.low_depth_pairs = self.max_skew = 0
        self.depth = None

    def ingest(self, role, msg, now):
        try:
            stamp, frame = header(msg)
            previous = self.last.get(role)
            if previous and (frame != previous[1] or stamp < previous[0]
                             or (stamp == previous[0] and role != 'camera_info')):
                raise ValueError('changed frame ID or non-increasing timestamp')
            if role in self.pending:
                fields, depth = image(msg, depth=role == 'depth')
            elif role == 'camera_info':
                fields, depth = calibration(msg), None
            else:
                auxiliary(msg, role)
                fields, depth = {}, None
            if role in self.fields and fields != self.fields[role]:
                raise ValueError('stream format or calibration changed during preflight')
            self.fields[role] = fields
            self.last[role] = stamp, frame, now
            self.counts[role] += 1
            if role in self.pending:
                if depth is not None:
                    self.depth = depth
                self.pending[role].append((stamp, depth))
                self.match()
            elif role in self.aux_stamps:
                self.aux_stamps[role].append(stamp)
        except (ValueError, TypeError, IndexError, BufferError, AttributeError) as error:
            self.bad[role] += 1
            if len(self.errors) < 12:
                self.errors.append(f'{role}: {error}')

    def match(self):
        image_queue, depth_queue = self.pending['image'], self.pending['depth']
        while image_queue and depth_queue:
            image_stamp, _ = image_queue[0]
            depth_stamp, depth = depth_queue[0]
            if abs(image_stamp-depth_stamp) <= self.pair_ns:
                image_queue.popleft()
                depth_queue.popleft()
                self.pairs += 1
                self.max_skew = max(self.max_skew, abs(image_stamp-depth_stamp))
                if depth['valid_ratio'] >= self.valid_ratio:
                    self.pair_stamps.append(depth_stamp)
                else:
                    self.low_depth_pairs += 1
            else:
                (image_queue if image_stamp < depth_stamp else depth_queue).popleft()

    def report(self, now):
        problems = self.errors.copy()
        for role in self.topics:
            if role not in self.last:
                problems.append(f'no valid {role} messages on {self.topics[role]}')
            elif role != 'camera_info' and now-self.last[role][2] > 1.:
                problems.append(f'{role} stream stopped before preflight ended')
        camera = [role for role in ('image', 'depth', 'camera_info') if role in self.last]
        if (len({self.last[role][1] for role in camera}) > 1
                or len({(self.fields[role]['width'], self.fields[role]['height']) for role in camera}) > 1):
            problems.append('rectified image, aligned depth and CameraInfo disagree on frame or dimensions')
        stamps = {role: list(items) for role, items in self.aux_stamps.items()}

        def near(items, stamp):
            i = bisect_left(items, stamp)
            return any(abs(value-stamp) <= self.aux_ns for value in items[max(0, i-1):i+1])

        newest = max((self.last[role][0] for role in self.pending if role in self.last), default=0)
        synchronized = sum(stamp >= newest-10**9 and all(near(items, stamp) for items in stamps.values())
                           for stamp in self.pair_stamps)
        if synchronized < self.minimum_pairs:
            problems.append(f'only {synchronized} usable synchronized pairs; need {self.minimum_pairs}')
        if self.depth is not None and self.depth['valid_ratio'] < self.valid_ratio:
            problems.append('latest depth frame has insufficient valid pixels')
        return dict(passed=not problems, problems=problems, topics=self.topics,
                    counts={role: self.counts[role] for role in self.topics}, invalid_counts=dict(self.bad),
                    streams={role: dict(**self.fields[role], frame_id=frame, last_stamp_ns=stamp)
                             for role, (stamp, frame, _) in self.last.items()},
                    matched_pairs=self.pairs, low_valid_depth_pairs=self.low_depth_pairs,
                    synchronized_recent_pairs=synchronized, max_pair_skew_ms=self.max_skew/10**6 if self.pairs else None,
                    latest_depth=self.depth,
                    thresholds=dict(max_image_depth_skew_ms=self.pair_ns/10**6,
                                    max_imu_tof_skew_ms=self.aux_ns/10**6, min_valid_depth_ratio=self.valid_ratio,
                                    min_synchronized_pairs=self.minimum_pairs, pair_evidence_window_s=1.,
                                    max_stream_silence_s=1.))


def observe(probe, seconds, latched_info=False):
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu, Range

    rclpy.init(args=[])
    node = None
    try:
        node = rclpy.create_node('oak_depth_preflight')
        types = dict(image=Image, depth=Image, camera_info=CameraInfo, imu=Imu, tof=Range)
        info_qos = (QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL) if latched_info else qos_profile_sensor_data)
        for role, topic in probe.topics.items():
            node.create_subscription(types[role], topic,
                                     lambda msg, role=role: probe.ingest(role, msg, time.monotonic()),
                                     info_qos if role == 'camera_info' else qos_profile_sensor_data)
        deadline = time.monotonic()+seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=max(0., min(.05, deadline-time.monotonic())))
        if not rclpy.ok():
            raise RuntimeError('ROS preflight interrupted')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def record_bag(output, topics, seconds, qos_path):
    import yaml

    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(output)]
    if qos_path is not None:
        command += ['--qos-profile-overrides-path', str(qos_path)]
    command += topics
    reserve = 512*1024**2
    if shutil.disk_usage(output.parent).free < reserve:
        raise RuntimeError('recording requires at least 512 MiB free disk')
    start = time.monotonic()
    stop_reason = 'recorder_exited'
    terminated = False
    child = None

    def stop_on_term(_signum, _frame):
        nonlocal terminated
        # Defer cleanup to the main loop, including if SIGTERM arrives
        # during Popen before the new process handle has been assigned.
        terminated = True

    previous_term = signal.signal(signal.SIGTERM, stop_on_term)
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        while child.poll() is None:
            if terminated:
                stop_reason = 'terminated'
                break
            remaining = seconds-(time.monotonic()-start)
            if remaining <= 0 or shutil.disk_usage(output.parent).free < reserve:
                stop_reason = 'duration' if remaining <= 0 else 'low_disk'
                break
            time.sleep(min(.25, remaining))
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
    finally:
        try:
            if child is not None and child.poll() is None:
                for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        child.wait(timeout=timeout)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        finally:
            signal.signal(signal.SIGTERM, previous_term)
    if child is None:
        raise RuntimeError('recorder did not start')
    if terminated:
        stop_reason = 'terminated'
    metadata = output/'metadata.yaml'
    entries = (yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']['topics_with_message_count']
               if metadata.is_file() else [])
    counts = {entry['topic_metadata']['name']: entry['message_count'] for entry in entries}
    missing = [topic for topic in topics if not counts.get(topic)]
    return dict(path=str(output), command=command, elapsed_wall_s=time.monotonic()-start,
                stop_reason=stop_reason, exit_code=child.poll(), topic_counts=counts, missing_topics=missing,
                complete=stop_reason == 'duration' and child.returncode in (0, -signal.SIGINT, 130) and not missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('image', 'depth', 'camera-info', 'imu'):
        parser.add_argument('--'+role+'-topic', required=True)
    parser.add_argument('--tof-topic', help='optional sensor_msgs/msg/Range topic')
    parser.add_argument('--latched-camera-info', action='store_true', help='request transient-local CameraInfo')
    parser.add_argument('--output', type=Path, required=True, help='new preflight manifest JSON')
    parser.add_argument('--preflight-seconds', type=float, default=5.)
    parser.add_argument('--max-pair-skew-ms', type=float, default=25.)
    parser.add_argument('--max-aux-skew-ms', type=float, default=50.)
    parser.add_argument('--min-valid-depth-ratio', type=float, default=.25)
    parser.add_argument('--min-pairs', type=int, default=5)
    parser.add_argument('--record-seconds', type=float, default=0.)
    parser.add_argument('--bag-output', type=Path, help='new rosbag2 directory')
    parser.add_argument('--record-topic', action='append', default=[], help='extra explicit bag topic, e.g. VO or /tf')
    parser.add_argument('--qos-overrides', type=Path, help='optional rosbag2 QoS overrides YAML')
    args = parser.parse_args()
    topics = {role: getattr(args, role+'_topic') for role in ('image', 'depth', 'camera_info', 'imu', 'tof')
              if getattr(args, role+'_topic') is not None}
    if (len(set(topics.values())) != len(topics) or any(not re.fullmatch(r'/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*', t)
                                                     for t in (*topics.values(), *args.record_topic))):
        parser.error('supply distinct, fully qualified sensor topics and fully qualified extra topics')
    if not (1 <= args.preflight_seconds <= 60 and 0 < args.max_pair_skew_ms <= 250
            and 0 < args.max_aux_skew_ms <= 250 and 0 < args.min_valid_depth_ratio <= 1 and 1 <= args.min_pairs <= 128):
        parser.error('invalid preflight time, skew, depth ratio, or minimum pair count')
    if (not 0 <= args.record_seconds <= 300 or bool(args.record_seconds) != (args.bag_output is not None)
            or args.output.exists() or (args.bag_output is not None and
                                        (args.bag_output.exists() or args.bag_output.resolve() == args.output.resolve()))):
        parser.error('use new output paths and pair --bag-output with a positive --record-seconds (max 300)')
    if args.qos_overrides is not None and (not args.record_seconds or not args.qos_overrides.is_file()):
        parser.error('--qos-overrides requires recording and an existing YAML file')
    probe = Probe(topics, args.max_pair_skew_ms, args.max_aux_skew_ms, args.min_valid_depth_ratio, args.min_pairs)
    error = None
    try:
        observe(probe, args.preflight_seconds, args.latched_camera_info)
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    report = probe.report(time.monotonic())
    report.update(preflight_seconds=args.preflight_seconds, ros_distro=os.environ.get('ROS_DISTRO'),
                  latched_camera_info=args.latched_camera_info, recording=None,
                  scope='Relative acquisition-stamp alignment and wall stream liveness for declared ROS 2 topics. '
                        'Absolute acquisition age against ROS time, hardware identity, and physical alignment and '
                        'range accuracy remain unverified.')
    if error:
        report['passed'] = False
        report['problems'].insert(0, error)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_name(f'.{args.output.name}.{uuid4().hex}.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)

    save()
    if report['passed'] and args.record_seconds:
        record_topics = list(dict.fromkeys([*topics.values(), '/tf_static', *args.record_topic]))
        try:
            report['recording'] = record_bag(args.bag_output, record_topics, args.record_seconds, args.qos_overrides)
        except Exception as exc:
            report['recording'] = dict(complete=False, error=f'{type(exc).__name__}: {exc}')
        save()
    print(json.dumps(dict(passed=report['passed'], manifest=str(args.output), problems=report['problems'],
                          recording_complete=report['recording']['complete'] if report['recording'] else None)))
    return 0 if report['passed'] and (not report['recording'] or report['recording']['complete']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
