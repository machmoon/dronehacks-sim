"""Read-only Gazebo camera recording with capture timestamps for every frame."""
import argparse
import json
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np

from .course import CAMERA_TOPIC


class VideoLog:
    def __init__(self, output, fps=10):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.fps = float(fps)
        if not 1 <= self.fps <= 30:
            raise ValueError('fps must be between 1 and 30')
        self.writer = None
        self.index = (self.path/'frames.jsonl').open('x', buffering=1)
        self.frame_count = 0
        self.last_stamp = None
        self.size = None

    def add(self, bgr, stamp_s):
        if self.last_stamp is not None:
            if stamp_s <= self.last_stamp:
                return
            if stamp_s-self.last_stamp < 1/self.fps-1e-6:
                return
        size = (bgr.shape[1], bgr.shape[0])
        if self.writer is None:
            self.size = size
            self.writer = cv2.VideoWriter(str(self.path/'camera.avi'),
                                          cv2.VideoWriter_fourcc(*'MJPG'), self.fps, size)
            if not self.writer.isOpened():
                raise RuntimeError('OpenCV cannot open the MJPG recording')
        if size != self.size:
            raise ValueError('camera dimensions changed during recording')
        self.writer.write(bgr)
        self.index.write(json.dumps({'frame': self.frame_count, 'capture_sim_s': stamp_s})+'\n')
        self.frame_count += 1
        self.last_stamp = stamp_s

    def close(self):
        if self.writer:
            self.writer.release()
        self.index.close()
        (self.path/'video.json').write_text(json.dumps({
            'frames': self.frame_count, 'playback_fps': self.fps, 'size': self.size,
            'timestamps': 'frames.jsonl', 'source': CAMERA_TOPIC,
            'note': 'Playback is constant rate; use capture timestamps for timing and gaps.'},
            indent=2)+'\n')


def main():
    from gz.msgs10.image_pb2 import Image, RGB_INT8
    from gz.transport13 import Node
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=300, help='Wall-clock recording seconds')
    parser.add_argument('--fps', type=float, default=10)
    options = parser.parse_args()
    if not 0 < options.duration <= 3600:
        parser.error('duration must be between 0 and 3600')
    node, lock = Node(), Lock()
    latest = [None]

    def receive(message):
        with lock:
            latest[0] = message

    log = VideoLog(options.output, options.fps)
    if not node.subscribe(Image, CAMERA_TOPIC, receive):
        log.close()
        raise RuntimeError('cannot subscribe to camera')
    deadline = time.monotonic()+options.duration
    try:
        while time.monotonic() < deadline:
            with lock:
                msg, latest[0] = latest[0], None
            if msg is None:
                time.sleep(.005)
                continue
            if msg.pixel_format_type != RGB_INT8 or msg.step != msg.width*3:
                raise ValueError('expected contiguous RGB camera pixels')
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            stamp = msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
            log.add(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), stamp)
    except KeyboardInterrupt:
        pass
    finally:
        node.unsubscribe(CAMERA_TOPIC)
        log.close()
    if log.frame_count == 0:
        raise RuntimeError('no camera frames received')
    print(json.dumps({'frames': log.frame_count, 'output': str(log.path)}))


if __name__ == '__main__':
    main()
