"""Read-only simulator truth log for diagnosis. Never supplies flight feedback."""
import argparse
import json
from pathlib import Path
from threading import Lock
import time

from mission.frames import rotate_body_velocity


def base_position(model_position, model_quaternion_xyzw, link_position):
    offset = rotate_body_velocity(model_quaternion_xyzw, link_position)
    return [a+b for a, b in zip(model_position, offset)]


def main():
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=300)
    args = parser.parse_args()
    if not 0 < args.duration <= 3600:
        parser.error('duration must be between 0 and 3600 seconds')
    topic = '/world/dronehacks/dynamic_pose/info'
    node, lock, latest = Node(), Lock(), [None]

    def receive(message):
        with lock:
            latest[0] = message

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', buffering=1) as stream:
        if not node.subscribe(Pose_V, topic, receive):
            raise RuntimeError('could not subscribe to simulator truth')
        deadline, previous, count = time.monotonic()+args.duration, -1.0, 0
        try:
            while time.monotonic() < deadline:
                with lock:
                    msg, latest[0] = latest[0], None
                if msg is None:
                    time.sleep(.005)
                    continue
                stamp = msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
                if stamp-previous < .02:
                    continue
                poses = {p.name: p for p in msg.pose}
                if 'course_quad' not in poses or 'base_link' not in poses:
                    continue
                model, link = poses['course_quad'], poses['base_link']
                p = [model.position.x, model.position.y, model.position.z]
                q = [model.orientation.x, model.orientation.y, model.orientation.z, model.orientation.w]
                local = [link.position.x, link.position.y, link.position.z]
                stream.write(json.dumps({'sim_time': stamp, 'position': base_position(p, q, local),
                                         'model_orientation_xyzw': q,
                                         'source': 'Gazebo truth; evaluation only'})+'\n')
                previous, count = stamp, count+1
        except KeyboardInterrupt:
            pass
        finally:
            node.unsubscribe(topic)
    if not count:
        raise RuntimeError('no simulator truth samples received')
    print(json.dumps({'samples': count, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
