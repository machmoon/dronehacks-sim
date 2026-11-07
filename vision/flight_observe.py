"""Independent Gazebo truth/contact observer for scoring only.

This observer is deliberately separate from the controller and camera
recorder. It subscribes to truth and collision topics, writes evidence files,
and never publishes flight, MAVLink, or estimator messages.
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
from threading import Event, Lock
import time

TRUTH_TOPIC = '/world/dronehacks/dynamic_pose/info'
CONTACT_TOPICS = tuple(
    f'/model/course_quad/contact/{link}'
    for link in ('base_link', 'rotor_0', 'rotor_1', 'rotor_2', 'rotor_3')
)
MAX_WALL_SECONDS = 3600.0
TRUTH_QUEUE_CAPACITY = 512
CONTACT_QUEUE_CAPACITY = 2048


def _xyz(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def _quaternion_xyzw(value):
    q = (float(value.x), float(value.y), float(value.z), float(value.w))
    norm = math.sqrt(sum(x*x for x in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid pose quaternion')
    return tuple(x / norm for x in q)


def rotate_xyzw(quaternion, vector):
    """Rotate a vector by an xyzw quaternion, returning ENU coordinates."""
    x, y, z, w = quaternion
    vx, vy, vz = vector
    return (
        (1 - 2*(y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    )


def multiply_xyzw(first, second):
    """Quaternion composition: result rotates by second, then first."""
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    result = (aw*bx + ax*bw + ay*bz - az*by,
              aw*by - ax*bz + ay*bw + az*bx,
              aw*bz + ax*by - ay*bx + az*bw,
              aw*bw - ax*bx - ay*by - az*bz)
    norm = math.sqrt(sum(v*v for v in result))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid composed quaternion')
    return tuple(v / norm for v in result)


def base_link_pose(model_pose, link_pose):
    """Transform a link-local pose into world ENU using model pose."""
    model_q = _quaternion_xyzw(model_pose.orientation)
    link_q = _quaternion_xyzw(link_pose.orientation)
    offset = rotate_xyzw(model_q, _xyz(link_pose.position))
    model_p = _xyz(model_pose.position)
    position = tuple(p + d for p, d in zip(model_p, offset))
    orientation = multiply_xyzw(model_q, link_q)
    return position, orientation


def contact_topic_audit(node, topics=CONTACT_TOPICS):
    """Return transport discovery counts; discovery is not a sensor health test."""
    discovered = set(node.topic_list())
    result = {}
    for topic in topics:
        publishers = None
        if topic in discovered:
            try:
                info = node.topic_info(topic)
                # gz.transport Node.topic_info returns (publishers, subscribers).
                publishers = len(info[0]) if info is not None else None
            except Exception:
                publishers = None
        result[topic] = publishers
    return result


def contact_topic_interpretation(publisher_start, publisher_end, messages):
    """Describe observed coverage without treating advertisement as health."""
    if messages:
        return 'messages_received'
    if publisher_start or publisher_end:
        return 'publisher_advertised_no_messages_observed'
    return 'no_publisher_observed'


def _stamp_range_add(current, stamp_ns):
    if current is None:
        return [int(stamp_ns), int(stamp_ns)]
    return [min(current[0], int(stamp_ns)), max(current[1], int(stamp_ns))]


def enqueue_truth(log, queue, item):
    """Queue a received truth message, counting any explicitly dropped oldest item."""
    log.truth_messages_received += 1
    dropped = queue.popleft() if len(queue) == queue.maxlen else None
    if dropped is not None:
        log.truth_queue_overflow += 1
        log.truth_queue_overflow_timestamp_range_ns = _stamp_range_add(
            log.truth_queue_overflow_timestamp_range_ns, dropped[0])
    queue.append(item)
    log.truth_queue_pending = len(queue)


def enqueue_contact(log, queue, topic, item):
    """Queue a contact message containing pairs, counting any dropped item."""
    log.contact_messages_queued += 1
    dropped = queue.popleft() if len(queue) == queue.maxlen else None
    if dropped is not None:
        dropped_topic, dropped_stamp_ns, _ = dropped
        log.contact_queue_overflow += 1
        log.contact_queue_overflow_timestamp_range_ns = _stamp_range_add(
            log.contact_queue_overflow_timestamp_range_ns, dropped_stamp_ns)
        log.contact_topic_queue_overflow[dropped_topic] += 1
        log.contact_topic_overflow_timestamp_range_ns[dropped_topic] = _stamp_range_add(
            log.contact_topic_overflow_timestamp_range_ns[dropped_topic], dropped_stamp_ns)
    queue.append(item)
    log.contact_queue_pending = len(queue)


def receive_contact_evidence(log, queue, topic, stamp_ns, pairs):
    """Count every received contact message and queue only messages with pairs."""
    log.contact_messages_received += 1
    log.contact_topic_messages[topic] += 1
    if pairs:
        log.contact_topic_pairs[topic] += 1
        enqueue_contact(log, queue, topic, (topic, stamp_ns, pairs))


def accept_truth(log, lock, accepting, queue, item):
    with lock:
        if not accepting.is_set():
            return False
        enqueue_truth(log, queue, item)
        return True


def accept_contact(log, lock, accepting, queue, topic, stamp_ns, pairs):
    with lock:
        if not accepting.is_set():
            return False
        receive_contact_evidence(log, queue, topic, stamp_ns, pairs)
        return True


def process_batches(log, pose_batch, contact_batch, last_stamp_ns):
    """Write drained evidence in arrival order and return the truth watermark."""
    log.truth_messages_dequeued += len(pose_batch)
    for stamp, message in pose_batch:
        if stamp <= last_stamp_ns:
            log.truth_messages_ignored += 1
            continue
        by_name = {pose.name: pose for pose in message.pose}
        model, link = by_name.get('course_quad'), by_name.get('base_link')
        if model is None or link is None:
            log.truth_messages_ignored += 1
            continue
        log.add_truth(stamp, model, link)
        last_stamp_ns = stamp
    log.contact_messages_dequeued += len(contact_batch)
    for topic, stamp, pairs in contact_batch:
        log.add_contacts(topic, stamp, pairs)
    return last_stamp_ns


def drain_pending(log, lock, poses, contacts, last_stamp_ns):
    """Drain all queued evidence; intended for the post-unsubscribe final flush."""
    while True:
        with lock:
            pose_batch, contact_batch = list(poses), list(contacts)
            poses.clear()
            contacts.clear()
            log.truth_queue_pending = len(poses)
            log.contact_queue_pending = len(contacts)
        if not pose_batch and not contact_batch:
            return last_stamp_ns
        last_stamp_ns = process_batches(log, pose_batch, contact_batch, last_stamp_ns)


def unsubscribe_and_drain(node, subscribed, log, lock, poses, contacts, last_stamp_ns, accepting):
    # Establish a capture cutoff under the callback lock. Already-dispatched
    # callbacks acquiring it later cannot mutate queues or counters, regardless
    # of whether Transport waits for callbacks during unsubscribe.
    with lock:
        accepting.clear()
        log.callback_acceptance_closed = True
    for topic in subscribed:
        node.unsubscribe(topic)
    return drain_pending(log, lock, poses, contacts, last_stamp_ns)


class ObserverLog:
    def __init__(self, output, truth_queue_capacity=TRUTH_QUEUE_CAPACITY,
                 contact_queue_capacity=CONTACT_QUEUE_CAPACITY):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.truth_queue_capacity = truth_queue_capacity
        self.contact_queue_capacity = contact_queue_capacity
        self.callback_acceptance_closed = False
        self.truth = (self.path / 'truth.jsonl').open('x', buffering=1)
        self.contacts = (self.path / 'contacts.jsonl').open('x', buffering=1)
        self.truth_samples = 0
        self.truth_messages_received = 0
        self.truth_messages_dequeued = 0
        self.truth_messages_ignored = 0
        self.truth_queue_overflow = 0
        self.truth_queue_overflow_timestamp_range_ns = None
        self.truth_queue_pending = 0
        self.contact_messages = 0
        self.contact_messages_received = 0
        self.contact_messages_queued = 0
        self.contact_messages_dequeued = 0
        self.contact_queue_overflow = 0
        self.contact_queue_overflow_timestamp_range_ns = None
        self.contact_queue_pending = 0
        self.contact_pairs = 0
        self.contact_topic_messages = {topic: 0 for topic in CONTACT_TOPICS}
        self.contact_topic_pairs = {topic: 0 for topic in CONTACT_TOPICS}
        self.contact_topic_queue_overflow = {topic: 0 for topic in CONTACT_TOPICS}
        self.contact_topic_overflow_timestamp_range_ns = {topic: None for topic in CONTACT_TOPICS}
        self.first_sim_s = None
        self.last_sim_s = None
        self.subscribed_truth = False
        self.contact_subscribed = []
        self.contact_unavailable = []
        self.contact_publishers_start = {}
        self.contact_publishers_end = {}

    def add_truth(self, stamp_ns, model_pose, link_pose):
        position, orientation = base_link_pose(model_pose, link_pose)
        stamp_s = int(stamp_ns) * 1e-9
        self.truth.write(json.dumps({
            'sim_time_s': stamp_s,
            'base_link_position_enu_m': position,
            'base_link_orientation_xyzw': orientation,
            'source': 'Gazebo dynamic_pose/info; evaluation only',
        }, separators=(',', ':')) + '\n')
        self.truth_samples += 1
        if self.first_sim_s is None:
            self.first_sim_s = stamp_s
        self.last_sim_s = stamp_s

    def add_contacts(self, topic, stamp_ns, pairs):
        if not pairs:
            return
        stamp_s = int(stamp_ns) * 1e-9
        self.contacts.write(json.dumps({
            'sim_time_s': stamp_s,
            'topic': topic,
            'pairs': pairs,
            'source': 'Gazebo contact sensor; raw observations only',
        }, separators=(',', ':')) + '\n')
        self.contact_messages += 1
        self.contact_pairs += len(pairs)

    def status(self, state, reason=None):
        terminal = state in ('stopped', 'duration_complete')
        truth_accounted = (self.truth_messages_received == self.truth_queue_overflow
                           + self.truth_messages_dequeued + self.truth_queue_pending)
        contacts_accounted = (self.contact_messages_queued == self.contact_queue_overflow
                              + self.contact_messages_dequeued + self.contact_queue_pending)
        contact_topics_accounted = (self.contact_messages_received
                                     == sum(self.contact_topic_messages.values()))
        contact_pairs_accounted = (self.contact_messages_queued
                                   == sum(self.contact_topic_pairs.values()))
        truth_dequeued_accounted = self.truth_messages_dequeued == self.truth_samples + self.truth_messages_ignored
        evidence_loss = bool(self.truth_queue_overflow or self.contact_queue_overflow)
        recording_complete = (terminal and reason is None and self.subscribed_truth
                              and self.truth_samples > 0 and not evidence_loss
                              and self.truth_queue_pending == 0 and self.contact_queue_pending == 0
                              and truth_accounted and contacts_accounted
                              and truth_dequeued_accounted and contact_topics_accounted
                              and contact_pairs_accounted and self.callback_acceptance_closed)
        data = {
            'state': state,
            'reason': reason,
            'truth_topic': TRUTH_TOPIC,
            'truth_subscribed': self.subscribed_truth,
            'truth_samples': self.truth_samples,
            'truth_messages_received': self.truth_messages_received,
            'truth_messages_dequeued': self.truth_messages_dequeued,
            'truth_messages_ignored': self.truth_messages_ignored,
            'truth_queue_capacity': self.truth_queue_capacity,
            'truth_queue_pending': self.truth_queue_pending,
            'truth_queue_overflow': self.truth_queue_overflow,
            'truth_queue_overflow_timestamp_range_ns': self.truth_queue_overflow_timestamp_range_ns,
            'truth_queue_overflow_timestamp_range_scope':
                'Inclusive min/max simulation-time stamps of explicitly discarded oldest queue entries.',
            'truth_queue_accounting_consistent': truth_accounted,
            'truth_dequeue_accounting_consistent': truth_dequeued_accounted,
            'contacts_expected_topics': list(CONTACT_TOPICS),
            'contacts_subscribed_topics': self.contact_subscribed,
            'contacts_unavailable_topics': self.contact_unavailable,
            'contact_topic_publishers_at_start': self.contact_publishers_start,
            'contact_topic_publishers_at_end': self.contact_publishers_end,
            'contact_topic_observation': {
                topic: {
                    'publisher_count_start': self.contact_publishers_start.get(topic),
                    'publisher_count_end': self.contact_publishers_end.get(topic),
                    'transport_messages_received': self.contact_topic_messages[topic],
                    'messages_with_contact_pairs': self.contact_topic_pairs[topic],
                    'observer_queue_overflow': self.contact_topic_queue_overflow[topic],
                    'observer_queue_overflow_timestamp_range_ns':
                        self.contact_topic_overflow_timestamp_range_ns[topic],
                    'interpretation': contact_topic_interpretation(
                        self.contact_publishers_start.get(topic),
                        self.contact_publishers_end.get(topic),
                        self.contact_topic_messages[topic]),
                } for topic in CONTACT_TOPICS
            },
            'contacts_topics_with_messages': [topic for topic, count in
                                              self.contact_topic_messages.items() if count],
            'contacts_topics_without_messages': [topic for topic, count in
                                                 self.contact_topic_messages.items() if not count],
            'contact_messages_with_pairs': self.contact_messages,
            'contact_messages_received': self.contact_messages_received,
            'contact_messages_queued': self.contact_messages_queued,
            'contact_messages_dequeued': self.contact_messages_dequeued,
            'contact_queue_capacity': self.contact_queue_capacity,
            'contact_queue_pending': self.contact_queue_pending,
            'contact_queue_overflow': self.contact_queue_overflow,
            'contact_queue_overflow_timestamp_range_ns': self.contact_queue_overflow_timestamp_range_ns,
            'contact_queue_overflow_timestamp_range_scope':
                'Inclusive min/max simulation-time stamps of explicitly discarded queued contact messages.',
            'contact_queue_accounting_consistent': contacts_accounted,
            'contact_topic_receive_accounting_consistent': contact_topics_accounted,
            'contact_topic_pair_queue_accounting_consistent': contact_pairs_accounted,
            'raw_contact_pair_observations': self.contact_pairs,
            'recording_complete': recording_complete,
            'callback_acceptance_closed': self.callback_acceptance_closed,
            'recording_complete_scope': ('Observer queues were drained on an orderly stop with no counted '
                                         'observer-queue overflow, after closing callback acceptance. '
                                         'Callbacks arriving after that capture cutoff are ignored. '
                                         'This does not prove Gazebo Transport '
                                         'delivered every generated sample.'),
            'evidence_loss': evidence_loss,
            'evidence_loss_scope': ('True when this observer explicitly dropped queued messages on overflow; '
                                    'upstream Gazebo Transport loss is not observable here.'),
            'first_sim_time_s': self.first_sim_s,
            'last_sim_time_s': self.last_sim_s,
            'truth_file': 'truth.jsonl',
            'contacts_file': 'contacts.jsonl',
            'computed_mission_score': False,
            'includes_latest_pose': False,
        }
        temp = self.path / 'status.json.tmp'
        temp.write_text(json.dumps(data, indent=2) + '\n')
        os.replace(temp, self.path / 'status.json')
        return data

    def close(self, state, reason=None):
        self.truth.close()
        self.contacts.close()
        return self.status(state, reason)


def main():
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='new output directory; it must not already exist')
    parser.add_argument('--duration', type=float, default=900,
                        help='wall-clock observation duration, at most 3600 seconds')
    args = parser.parse_args()
    if not 0 < args.duration <= MAX_WALL_SECONDS:
        parser.error('duration must be between 0 and 3600 wall seconds')

    log = ObserverLog(args.output)
    node, lock = Node(), Lock()
    poses, contacts = deque(maxlen=TRUTH_QUEUE_CAPACITY), deque(maxlen=CONTACT_QUEUE_CAPACITY)
    stop = Event()
    accepting = Event()
    accepting.set()
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())

    def receive_truth(message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
        accept_truth(log, lock, accepting, poses, (stamp, message))

    def receive_contact(topic, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
        pairs = [(c.collision1.name, c.collision2.name) for c in message.contact]
        accept_contact(log, lock, accepting, contacts, topic, stamp, pairs)

    subscribed = []
    state, reason = 'observing', None
    last_stamp_ns, started = -1, time.monotonic()
    try:
        log.subscribed_truth = bool(node.subscribe(Pose_V, TRUTH_TOPIC, receive_truth))
        if log.subscribed_truth:
            subscribed.append(TRUTH_TOPIC)
        else:
            reason = f'cannot subscribe to {TRUTH_TOPIC}'
        for topic in CONTACT_TOPICS:
            if node.subscribe(Contacts, topic,
                              lambda msg, topic=topic: receive_contact(topic, msg)):
                subscribed.append(topic)
                log.contact_subscribed.append(topic)
            else:
                log.contact_unavailable.append(topic)
        # Discovery is only evidence that Gazebo Transport currently advertises
        # a publisher. Contact sensors may be silent until contact occurs.
        log.contact_publishers_start = contact_topic_audit(node)
        deadline = started + args.duration
        last_status = started
        while not stop.is_set() and time.monotonic() < deadline:
            with lock:
                pose_batch, contact_batch = list(poses), list(contacts)
                poses.clear()
                contacts.clear()
                log.truth_queue_pending = len(poses)
                log.contact_queue_pending = len(contacts)
            last_stamp_ns = process_batches(log, pose_batch, contact_batch, last_stamp_ns)
            if time.monotonic() - last_status >= 1:
                log.status(state, reason)
                last_status = time.monotonic()
            time.sleep(.005)
        if stop.is_set():
            state = 'stopped'
        else:
            state = 'duration_complete'
        if log.truth_samples == 0:
            reason = reason or 'truth topic subscribed but no course_quad/base_link samples received'
    except KeyboardInterrupt:
        state = 'stopped'
    except BaseException as error:
        state, reason = 'error', f'{type(error).__name__}: {error}'
        raise
    finally:
        log.contact_publishers_end = contact_topic_audit(node)
        last_stamp_ns = unsubscribe_and_drain(node, subscribed, log, lock, poses, contacts,
                                             last_stamp_ns, accepting)
        final_status = log.close(state, reason)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'output': str(args.output), 'truth_samples': log.truth_samples,
                      'raw_contact_pair_observations': log.contact_pairs,
                      'truth_queue_overflow': log.truth_queue_overflow,
                      'contact_queue_overflow': log.contact_queue_overflow,
                      'recording_complete': final_status['recording_complete'],
                      'evidence_loss': final_status['evidence_loss'],
                      'state': state, 'reason': reason,
                      'topics_subscribed': subscribed,
                      'computed_mission_score': False}))


if __name__ == '__main__':
    main()
