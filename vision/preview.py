"""Display raw Gazebo camera pixels without estimating or publishing a pose."""
import argparse
import json
import os
from pathlib import Path
import signal
from threading import Lock
import time

from .course import CAMERA_TOPIC


def atomic_json(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    import cv2
    import numpy as np
    from gz.msgs10.image_pb2 import Image, RGB_INT8
    from gz.transport13 import Node

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/preview'))
    parser.add_argument('--duration', type=float, default=18000)
    args = parser.parse_args()
    if not 0 < args.duration <= 18000:
        parser.error('duration must be positive and at most five hours')
    args.output.mkdir(parents=True, exist_ok=True)
    running, latest, lock = [True], [None], Lock()
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__(0, False))
    session = {'mode': 'marker_free_preview', 'started_at': time.time(),
               'source': CAMERA_TOPIC, 'navigation': 'not connected',
               'description': 'No artificial tags. Camera preview only; no flight commands.'}
    atomic_json(args.output / 'session.json', session)
    node = Node()

    def receive(message):
        with lock:
            latest[0] = message

    if not node.subscribe(Image, CAMERA_TOPIC, receive):
        raise RuntimeError('could not subscribe to the raw Gazebo camera')
    deadline, count, previous = time.monotonic() + args.duration, 0, -1
    try:
        while running[0] and time.monotonic() < deadline:
            with lock:
                msg, latest[0] = latest[0], None
            if msg is not None:
                stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
                if stamp > previous:
                    if msg.pixel_format_type != RGB_INT8 or msg.step != msg.width * 3:
                        raise ValueError('expected contiguous RGB pixels')
                    rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    encoded, image = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if not encoded:
                        raise RuntimeError('JPEG encoding failed')
                    temp = args.output / 'camera.jpg.tmp'
                    temp.write_bytes(image.tobytes())
                    os.replace(temp, args.output / 'camera.jpg')
                    count += 1
                    atomic_json(args.output / 'latest.json', {
                        'stamp_sim_s': stamp, 'updated_at': time.time(), 'frames': count,
                        'width': msg.width, 'height': msg.height, 'source': CAMERA_TOPIC,
                        'pose_available': False, 'artificial_tags': False,
                        'flight_commands_sent': 0, 'session_started_at': session['started_at'],
                    })
                    previous = stamp
            time.sleep(.2)
    finally:
        node.unsubscribe(CAMERA_TOPIC)
        session.update(stopped_at=time.time(), frames=count)
        atomic_json(args.output / 'session.json', session)


if __name__ == '__main__':
    main()
