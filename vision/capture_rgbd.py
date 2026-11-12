"""Record a bounded synchronized RGB/depth sample; no vehicle-state access."""
import argparse
import json
from pathlib import Path
from threading import Lock
import time


def main():
    import cv2
    import numpy as np
    from gz.msgs10.image_pb2 import Image, RGB_INT8, R_FLOAT32
    from gz.transport13 import Node
    from .rgbd_scene import COLOR_TOPIC, DEPTH_TOPIC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    if not 1 <= args.frames <= 200 or not 0 < args.timeout <= 300:
        parser.error('require 1–200 frames and a timeout of at most 300 wall seconds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('output already exists')
    node, lock, buffers = Node(), Lock(), {'color': {}, 'depth': {}}

    def receive(kind, msg):
        expected = RGB_INT8 if kind == 'color' else R_FLOAT32
        if msg.pixel_format_type != expected:
            return
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
        dtype, channels = (np.uint8, 3) if kind == 'color' else (np.float32, 1)
        row_bytes = msg.width * channels * np.dtype(dtype).itemsize
        if msg.step not in (0, row_bytes) or len(msg.data) != msg.height * row_bytes:
            return
        frame = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
        if channels == 1:
            frame = frame[..., 0]
        with lock:
            target = buffers[kind]
            target[stamp] = frame.copy()
            while len(target) > 12:
                del target[min(target)]

    for kind, topic in (('color', COLOR_TOPIC), ('depth', DEPTH_TOPIC)):
        if not node.subscribe(Image, topic, lambda msg, kind=kind: receive(kind, msg)):
            raise RuntimeError('could not subscribe to ' + topic)
    frames, started = [], time.monotonic()
    while len(frames) < args.frames and time.monotonic() - started < args.timeout:
        with lock:
            paired = sorted(buffers['color'].keys() & buffers['depth'].keys())
            for stamp in paired:
                frames.append((stamp, buffers['color'].pop(stamp), buffers['depth'].pop(stamp)))
        time.sleep(.01)
    for topic in (COLOR_TOPIC, DEPTH_TOPIC):
        node.unsubscribe(topic)
    if len(frames) < args.frames:
        raise RuntimeError(f'only {len(frames)} of {args.frames} synchronized RGBD captures received')
    frames = frames[:args.frames]
    stamps, rgb, depth = zip(*frames)
    np.savez_compressed(args.output, stamp_ns=np.array(stamps), rgb=np.stack(rgb), depth_z_m=np.stack(depth))
    cv2.imwrite(str(args.output.with_suffix('.rgb.png')), cv2.cvtColor(rgb[-1], cv2.COLOR_RGB2BGR))
    scaled = np.uint8(np.clip(np.nan_to_num(depth[-1], nan=8., posinf=8.) / 8, 0, 1) * 255)
    cv2.imwrite(str(args.output.with_suffix('.depth.png')), cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO))
    print(json.dumps({'frames': len(frames), 'output': str(args.output), 'shape': list(rgb[-1].shape),
                      'depth_finite_fraction': float(np.isfinite(depth[-1]).mean()),
                      'sensor_topics': [COLOR_TOPIC, DEPTH_TOPIC], 'vehicle_state_subscriptions': []}))


if __name__ == '__main__':
    main()
