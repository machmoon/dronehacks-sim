"""Bounded, read-only recording of synchronized Gazebo RGB-D camera frames.

This process subscribes to RGB-D images and an optional single-pixel range
sensor only. It does not read vehicle pose, publish messages, or issue flight
commands. Saved depth PNGs encode optical-Z in millimetres (uint16);
latest.jpg and depth.jpg are view-only previews.
"""
import argparse
from collections import OrderedDict, deque
import json
import math
import os
from pathlib import Path
import signal
import struct
from threading import Event, Lock
import time

from .rgbd_scene import (COLOR_TOPIC, DEPTH_TOPIC, TOF_MAX_DEPTH, TOF_MIN_DEPTH,
                         TOF_TOPIC)

MAX_WALL_SECONDS = 3600.0
DEFAULT_MAX_BYTES = 15 * 1024 * 1024
TOF_PAIR_TOLERANCE_NS = 15_000_000
TOF_PAIR_WAIT_S = 0.06


class FrameArchive:
    """Write timestamped RGB/depth pairs under both frame and byte caps."""
    def __init__(self, output, *, max_frames=4500, max_bytes=DEFAULT_MAX_BYTES):
        if int(max_frames) < 1 or int(max_bytes) < 1:
            raise ValueError('frame and byte limits must be positive')
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.frames_path = self.path / 'frames'
        self.frames_path.mkdir()
        self.index = (self.path / 'frames.jsonl').open('x', buffering=1)
        self.max_frames = int(max_frames)
        self.max_bytes = int(max_bytes)
        self.frame_count = 0
        self.bytes_written = 0
        self.last_capture_ns = None
        self.limit_reason = None
        self.tof_subscribed = None
        (self.path / 'metadata.json').write_text(json.dumps({
            'camera_topics': [COLOR_TOPIC, DEPTH_TOPIC],
            'tof': {
                'topic': TOF_TOPIC,
                'optional': True,
                'message_format': 'gz.msgs.Image, 1x1 R_FLOAT32',
                'range_units': 'metres',
                'valid_range_m': [TOF_MIN_DEPTH, TOF_MAX_DEPTH],
                'pair_tolerance_ms': TOF_PAIR_TOLERANCE_NS / 1e6,
            },
            'includes_vehicle_pose': False,
        }, indent=2) + '\n')

    def add(self, capture_ns, rgb_jpg, depth_png, diagnostics=None):
        if self.limit_reason:
            return False
        if self.frame_count >= self.max_frames:
            self.limit_reason = 'max_frames'
            return False
        sample_bytes = len(rgb_jpg) + len(depth_png)
        if self.bytes_written + sample_bytes > self.max_bytes:
            self.limit_reason = 'max_bytes'
            return False
        frame = self.frame_count
        rgb_name, depth_name = f'rgb-{frame:06d}.jpg', f'depth-{frame:06d}.png'
        (self.frames_path / rgb_name).write_bytes(rgb_jpg)
        try:
            (self.frames_path / depth_name).write_bytes(depth_png)
        except BaseException:
            (self.frames_path / rgb_name).unlink(missing_ok=True)
            raise
        self.index.write(json.dumps({
            'frame': frame,
            'capture_ns': int(capture_ns),
            'capture_sim_s': int(capture_ns) * 1e-9,
            'rgb': f'frames/{rgb_name}',
            'depth_z_mm_u16': f'frames/{depth_name}',
            'diagnostics': diagnostics or {},
        }, separators=(',', ':')) + '\n')
        self.frame_count += 1
        self.bytes_written += sample_bytes
        self.last_capture_ns = int(capture_ns)
        return True

    def status(self, state, *, reason=None):
        data = {
            'state': state,
            'reason': reason or self.limit_reason,
            'frames': self.frame_count,
            'encoded_frame_bytes': self.bytes_written,
            'max_frames': self.max_frames,
            'max_bytes': self.max_bytes,
            'last_capture_ns': self.last_capture_ns,
            'frame_index': 'frames.jsonl',
            'camera_topics': [COLOR_TOPIC, DEPTH_TOPIC],
            'tof_topic': TOF_TOPIC,
            'tof_optional': True,
            'tof_subscribed': self.tof_subscribed,
            'tof_pair_tolerance_ms': TOF_PAIR_TOLERANCE_NS / 1e6,
            'includes_vehicle_pose': False,
        }
        tmp = self.path / 'status.json.tmp'
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        os.replace(tmp, self.path / 'status.json')

    def close(self, *, state='stopped', reason=None):
        if not self.index.closed:
            self.index.close()
        self.status(state, reason=reason)


class RGBDPairQueue:
    """Pair sensor images by exact Gazebo capture timestamp, with a small cap."""
    def __init__(self, max_pending=8, max_ready=4):
        self.max_pending = max_pending
        self.color, self.depth = OrderedDict(), OrderedDict()
        self.ready = deque(maxlen=max_ready)

    def push(self, kind, stamp_ns, message):
        if kind not in ('color', 'depth'):
            raise ValueError('kind must be color or depth')
        target = self.color if kind == 'color' else self.depth
        target[int(stamp_ns)] = message
        target.move_to_end(int(stamp_ns))
        while len(target) > self.max_pending:
            target.popitem(last=False)
        common = sorted(self.color.keys() & self.depth.keys())
        for stamp in common:
            pair = (stamp, self.color.pop(stamp), self.depth.pop(stamp))
            if len(self.ready) == self.ready.maxlen:
                self.ready.popleft()
            self.ready.append(pair)

    def take(self):
        return self.ready.popleft() if self.ready else None


def decode_tof_message(message, float_format, *, min_depth_m=TOF_MIN_DEPTH,
                       max_depth_m=TOF_MAX_DEPTH):
    """Read the single float32 range pixel; return (metres, invalid reason)."""
    if message.pixel_format_type != float_format:
        return None, 'unexpected_pixel_format'
    if message.width != 1 or message.height != 1:
        return None, 'unexpected_image_dimensions'
    if message.step != 4 or len(message.data) != 4:
        return None, 'unexpected_float32_payload_size'
    value = float(struct.unpack('<f', message.data)[0])
    if not math.isfinite(value):
        return None, 'nonfinite_range'
    if not min_depth_m <= value <= max_depth_m:
        return None, 'outside_sensor_clip_range'
    return value, None


class ToFBuffer:
    """One-use nearest-neighbor range pairing against RGB-D capture time."""
    def __init__(self, max_samples=64, tolerance_ns=TOF_PAIR_TOLERANCE_NS):
        self.max_samples = max_samples
        self.tolerance_ns = tolerance_ns
        self.samples = OrderedDict()
        self.watermark_ns = None

    def push(self, capture_ns, range_m, invalid_reason=None):
        stamp = int(capture_ns)
        self.samples[stamp] = (range_m, invalid_reason)
        self.samples.move_to_end(stamp)
        self.watermark_ns = stamp if self.watermark_ns is None else max(self.watermark_ns, stamp)
        while len(self.samples) > self.max_samples:
            self.samples.popitem(last=False)

    def match(self, rgbd_capture_ns):
        stamp = int(rgbd_capture_ns)
        if not self.samples:
            return {'source': TOF_TOPIC, 'status': 'missing', 'range_m': None,
                    'capture_ns': None, 'signed_age_ms': None,
                    'tolerance_ms': self.tolerance_ns / 1e6,
                    'reason': 'no_tof_sample'}
        nearest_ns = min(self.samples, key=lambda sample_ns:
                         (self.samples[sample_ns][0] is None,
                          abs(sample_ns - stamp), sample_ns))
        age_ns = nearest_ns - stamp
        age_ms = age_ns / 1e6
        value, invalid_reason = self.samples[nearest_ns]
        if abs(age_ns) > self.tolerance_ns:
            # Discard obsolete readings but retain future samples for later frames.
            for old_ns in list(self.samples):
                if old_ns < stamp - self.tolerance_ns:
                    del self.samples[old_ns]
            return {'source': TOF_TOPIC, 'status': 'missing', 'range_m': None,
                    'capture_ns': None, 'signed_age_ms': None,
                    'nearest_capture_ns': nearest_ns, 'nearest_age_ms': age_ms,
                    'tolerance_ms': self.tolerance_ns / 1e6,
                    'reason': 'source_bad_age'}
        del self.samples[nearest_ns]
        for old_ns in list(self.samples):
            if old_ns < stamp - self.tolerance_ns:
                del self.samples[old_ns]
        if invalid_reason:
            return {'source': TOF_TOPIC, 'status': 'invalid', 'range_m': None,
                    'capture_ns': nearest_ns, 'signed_age_ms': age_ms,
                    'tolerance_ms': self.tolerance_ns / 1e6,
                    'reason': invalid_reason}
        return {'source': TOF_TOPIC, 'status': 'paired', 'range_m': value,
                'capture_ns': nearest_ns, 'signed_age_ms': age_ms,
                'tolerance_ms': self.tolerance_ns / 1e6, 'reason': None}


def _encoded_pair(rgb, depth, cv2, np, *, jpeg_quality=55, depth_max_m=8.0):
    """Encode RGB as bounded JPEG and metric optical depth as uint16 PNG."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('expected uint8 RGB image')
    if depth.dtype != np.float32 or depth.shape != rgb.shape[:2]:
        raise ValueError('expected float32 depth matching RGB dimensions')
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok_rgb, rgb_data = cv2.imencode('.jpg', bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok_rgb:
        raise RuntimeError('could not encode RGB frame')
    valid = np.isfinite(depth) & (depth > 0) & (depth <= depth_max_m)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.rint(depth[valid] * 1000).clip(1, 65535).astype(np.uint16)
    ok_depth, depth_data = cv2.imencode('.png', depth_mm)
    if not ok_depth:
        raise RuntimeError('could not encode depth frame')
    finite = depth[valid]
    preview = np.uint8(np.clip(np.nan_to_num(depth, nan=depth_max_m,
                                              posinf=depth_max_m) / depth_max_m,
                               0, 1) * 255)
    preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    return (rgb_data.tobytes(), depth_data.tobytes(), preview,
            {'depth_valid_fraction': float(valid.mean()),
             'depth_min_m': float(finite.min()) if finite.size else None,
             'depth_max_m': float(finite.max()) if finite.size else None})


def main():
    import cv2
    import numpy as np
    from gz.msgs10.image_pb2 import Image, RGB_INT8, R_FLOAT32
    from gz.transport13 import Node

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='new output directory; it must not already exist')
    parser.add_argument('--duration', type=float, default=900,
                        help='wall-clock recording duration, at most 3600 seconds')
    parser.add_argument('--fps-sim', type=float, default=5,
                        help='maximum recorded sample rate by simulator time')
    parser.add_argument('--max-frames', type=int, default=4500)
    parser.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument('--jpeg-quality', type=int, default=55)
    args = parser.parse_args()
    if not 0 < args.duration <= MAX_WALL_SECONDS:
        parser.error('duration must be between 0 and 3600 wall seconds')
    if not 0 < args.fps_sim <= 30 or not 1 <= args.jpeg_quality <= 95:
        parser.error('fps-sim must be in (0,30] and JPEG quality in [1,95]')

    archive = FrameArchive(args.output, max_frames=args.max_frames,
                           max_bytes=args.max_bytes)
    node, lock, pairs = Node(), Lock(), RGBDPairQueue()
    tof_buffer, pending_frames = ToFBuffer(), deque(maxlen=4)
    stop = Event()
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())

    def receive(kind, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
        with lock:
            pairs.push(kind, stamp, message)

    def receive_tof(message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
        value, invalid_reason = decode_tof_message(message, R_FLOAT32)
        with lock:
            tof_buffer.push(stamp, value, invalid_reason)

    subscriptions = []
    state, reason = 'stopped', None
    started = time.monotonic()
    try:
        for msg_type, topic, kind in ((Image, COLOR_TOPIC, 'color'),
                                      (Image, DEPTH_TOPIC, 'depth')):
            if not node.subscribe(msg_type, topic,
                                  lambda msg, kind=kind: receive(kind, msg)):
                raise RuntimeError('cannot subscribe to ' + topic)
            subscriptions.append(topic)
        archive.tof_subscribed = bool(node.subscribe(Image, TOF_TOPIC, receive_tof))
        if archive.tof_subscribed:
            subscriptions.append(TOF_TOPIC)
        state = 'recording'
        last_status = started
        last_selected_capture = None
        while not stop.is_set() and time.monotonic() - started < args.duration:
            now = time.monotonic()
            with lock:
                while True:
                    pair = pairs.take()
                    if pair is None:
                        break
                    stamp_ns = pair[0]
                    stamp_s = stamp_ns * 1e-9
                    if (last_selected_capture is not None and
                            stamp_s - last_selected_capture < 1 / args.fps_sim - 1e-9):
                        continue
                    pending_frames.append((pair, now))
                    last_selected_capture = stamp_s
                ready = (pending_frames and
                         (now - pending_frames[0][1] >= TOF_PAIR_WAIT_S or stop.is_set() or
                          (tof_buffer.watermark_ns is not None and
                           tof_buffer.watermark_ns >= pending_frames[0][0][0] +
                           TOF_PAIR_TOLERANCE_NS)))
                if ready:
                    (pair_stamp_ns, color, depth_msg), _ = pending_frames.popleft()
                    tof_diagnostic = tof_buffer.match(pair_stamp_ns)
                else:
                    pair_stamp_ns = color = depth_msg = tof_diagnostic = None
            if pair_stamp_ns is None:
                if time.monotonic() - last_status >= 1:
                    archive.status(state)
                    last_status = time.monotonic()
                time.sleep(.005)
                continue
            stamp_ns = pair_stamp_ns
            if (color.pixel_format_type != RGB_INT8 or depth_msg.pixel_format_type != R_FLOAT32
                    or color.width != depth_msg.width or color.height != depth_msg.height
                    or color.step != color.width * 3 or depth_msg.step != depth_msg.width * 4):
                raise ValueError('unexpected RGB-D dimensions, format, or row stride')
            stamp_s = stamp_ns * 1e-9
            rgb = np.frombuffer(color.data, dtype=np.uint8).reshape(color.height, color.width, 3)
            depth = np.frombuffer(depth_msg.data, dtype='<f4').reshape(depth_msg.height,
                                                                       depth_msg.width)
            rgb_bytes, depth_bytes, depth_preview, stats = _encoded_pair(
                rgb, depth, cv2, np, jpeg_quality=args.jpeg_quality)
            accepted = archive.add(stamp_ns, rgb_bytes, depth_bytes,
                                  {**stats, 'tof': tof_diagnostic})
            # Preview writes are atomic and always represent the same paired capture.
            ok_preview, preview_bytes = cv2.imencode(
                '.jpg', depth_preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok_preview:
                raise RuntimeError('could not encode depth preview')
            for filename, encoded in (('latest.jpg', rgb_bytes),
                                      ('depth.jpg', preview_bytes.tobytes())):
                temp = args.output / (filename + '.tmp')
                temp.write_bytes(encoded)
                os.replace(temp, args.output / filename)
            latest_meta = args.output / 'latest.json.tmp'
            latest_meta.write_text(json.dumps({
                'capture_ns': int(stamp_ns), 'capture_sim_s': stamp_s,
                'depth_z_units_in_saved_frames': 'millimetres, uint16; 0=invalid',
                **stats, 'tof': tof_diagnostic,
            }, indent=2) + '\n')
            os.replace(latest_meta, args.output / 'latest.json')
            if archive.limit_reason:
                state = 'preview_only_archive_limit'
                reason = archive.limit_reason
            if time.monotonic() - last_status >= 1:
                archive.status(state)
                last_status = time.monotonic()
        else:
            state = 'duration_complete' if not stop.is_set() else 'stopped'
        if archive.frame_count == 0 and reason is None:
            reason = 'no synchronized RGB-D frames received'
    except KeyboardInterrupt:
        state = 'stopped'
    except BaseException as error:
        state, reason = 'error', f'{type(error).__name__}: {error}'
        raise
    finally:
        for topic in subscriptions:
            node.unsubscribe(topic)
        archive.close(state=state, reason=reason)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'output': str(args.output), 'frames': archive.frame_count,
                      'bytes': archive.bytes_written, 'state': state,
                      'reason': reason,
                      'source_topics': [COLOR_TOPIC, DEPTH_TOPIC, TOF_TOPIC],
                      'tof_subscribed': archive.tof_subscribed}))


if __name__ == '__main__':
    main()
