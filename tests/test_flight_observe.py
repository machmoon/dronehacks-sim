import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

from vision import flight_observe as observer


def _pose(name, position):
    return SimpleNamespace(
        name=name,
        position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _truth_message():
    return SimpleNamespace(pose=[_pose('course_quad', (1., 2., 0.)),
                                 _pose('base_link', (0., 0., .24))])


class FlightObserverQueueTests(unittest.TestCase):
    def test_short_writer_stall_preserves_pose_burst_at_default_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            log = observer.ObserverLog(Path(temp) / 'observer')
            log.subscribed_truth = True
            poses = deque(maxlen=observer.TRUTH_QUEUE_CAPACITY)
            contacts = deque(maxlen=observer.CONTACT_QUEUE_CAPACITY)
            # 32 frames at the observed 8ms simulation cadence: a burst that
            # the old four-message queue would silently shorten to 24ms.
            stamps = [1_000_000_000 + i * 8_000_000 for i in range(32)]
            for stamp in stamps:
                observer.enqueue_truth(log, poses, (stamp, _truth_message()))
            accepting = Event()
            accepting.set()
            observer.unsubscribe_and_drain(None, [], log, Lock(), poses, contacts, -1, accepting)
            status = log.close('stopped')
            self.assertTrue(status['recording_complete'])
            self.assertEqual(status['truth_queue_overflow'], 0)
            rows = [json.loads(line) for line in (log.path / 'truth.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            self.assertEqual([round(row['sim_time_s'] * 1e9) for row in rows], stamps)

    def test_bounded_overflow_counts_and_records_discarded_timestamp_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'observer'
            log = observer.ObserverLog(output, truth_queue_capacity=2,
                                       contact_queue_capacity=2)
            poses = deque(maxlen=2)
            contacts = deque(maxlen=2)
            topic = observer.CONTACT_TOPICS[0]
            lock, accepting = Lock(), Event()
            accepting.set()
            for stamp in (100, 200, 300):
                observer.enqueue_truth(log, poses, (stamp, _truth_message()))
                observer.receive_contact_evidence(log, contacts, topic, stamp,
                                                  [('course_quad::base_link', 'wall')])
            self.assertEqual(log.truth_messages_received, 3)
            self.assertEqual(log.truth_queue_overflow, 1)
            self.assertEqual(log.truth_queue_overflow_timestamp_range_ns, [100, 100])
            self.assertEqual(log.contact_messages_received, 3)
            self.assertEqual(log.contact_queue_overflow, 1)
            self.assertEqual(log.contact_queue_overflow_timestamp_range_ns, [100, 100])
            self.assertEqual(log.contact_topic_queue_overflow[topic], 1)
            self.assertEqual(log.contact_topic_overflow_timestamp_range_ns[topic], [100, 100])
            status = log.close('stopped')
            self.assertFalse(status['recording_complete'])
            self.assertTrue(status['evidence_loss'])
            self.assertTrue(status['truth_queue_accounting_consistent'])
            self.assertTrue(status['contact_queue_accounting_consistent'])
            self.assertTrue(status['contact_topic_receive_accounting_consistent'])
            self.assertTrue(status['contact_topic_pair_queue_accounting_consistent'])
            self.assertTrue(status['truth_dequeue_accounting_consistent'])
            self.assertEqual(status['truth_queue_pending'], 2)
            self.assertEqual(status['contact_queue_pending'], 2)

    def test_unsubscribe_then_final_drain_writes_all_queued_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'observer'
            log = observer.ObserverLog(output, truth_queue_capacity=2,
                                       contact_queue_capacity=2)
            log.subscribed_truth = True
            poses = deque(maxlen=2)
            contacts = deque(maxlen=2)
            topic = observer.CONTACT_TOPICS[0]
            lock, accepting = Lock(), Event()
            accepting.set()
            observer.enqueue_truth(log, poses, (1_000_000_000, _truth_message()))
            observer.receive_contact_evidence(log, contacts, topic, 1_000_000_000,
                                              [('course_quad::base_link', 'floor')])

            class FakeNode:
                def __init__(self):
                    self.calls = []

                def unsubscribe(self, name):
                    self.calls.append(name)
                    self.assert_queued()
                    # A callback dispatched before unsubscribe may reach the
                    # acceptance lock after cutoff. It must leave no late data.
                    if observer.accept_truth(log, lock, accepting, poses,
                                             (2_000_000_000, _truth_message())):
                        raise AssertionError('truth callback accepted after cutoff')
                    if observer.accept_contact(log, lock, accepting, contacts, topic,
                                               2_000_000_000, [('drone', 'wall')]):
                        raise AssertionError('contact callback accepted after cutoff')

                def assert_queued(self):
                    if len(poses) != 1 or len(contacts) != 1:
                        raise AssertionError('final drain must follow unsubscribe while evidence is queued')

            node = FakeNode()
            last_stamp = observer.unsubscribe_and_drain(
                node, [observer.TRUTH_TOPIC, topic], log, lock, poses, contacts, -1, accepting)
            self.assertEqual(node.calls, [observer.TRUTH_TOPIC, topic])
            self.assertEqual(last_stamp, 1_000_000_000)
            self.assertEqual(len(poses), 0)
            self.assertEqual(len(contacts), 0)
            status = log.close('stopped')
            self.assertTrue(status['recording_complete'])
            self.assertFalse(status['evidence_loss'])
            self.assertEqual(status['truth_samples'], 1)
            self.assertEqual(status['truth_messages_received'], 1)
            self.assertEqual(status['truth_messages_dequeued'], 1)
            self.assertEqual(status['contact_messages_received'], 1)
            self.assertEqual(status['contact_messages_queued'], 1)
            self.assertEqual(status['contact_messages_dequeued'], 1)
            self.assertEqual(status['contact_messages_with_pairs'], 1)
            truth = [json.loads(line) for line in (output / 'truth.jsonl').read_text().splitlines()]
            contacts_written = [json.loads(line) for line in (output / 'contacts.jsonl').read_text().splitlines()]
            self.assertEqual(truth[0]['sim_time_s'], 1.0)
            self.assertEqual(contacts_written[0]['sim_time_s'], 1.0)
            self.assertTrue(status['truth_queue_accounting_consistent'])
            self.assertTrue(status['contact_queue_accounting_consistent'])
            self.assertTrue(status['contact_topic_receive_accounting_consistent'])
            self.assertTrue(status['contact_topic_pair_queue_accounting_consistent'])
            self.assertTrue(status['truth_dequeue_accounting_consistent'])
            self.assertTrue(status['callback_acceptance_closed'])
            self.assertFalse(accepting.is_set())
            # Inconsistent accounting must never coexist with a complete flag.
            log.truth_messages_ignored += 1
            self.assertFalse(log.status('stopped')['recording_complete'])
            log.truth_messages_ignored -= 1
            log.contact_topic_messages[topic] += 1
            self.assertFalse(log.status('stopped')['recording_complete'])
            log.contact_topic_messages[topic] -= 1
            log.contact_topic_pairs[topic] += 1
            self.assertFalse(log.status('stopped')['recording_complete'])


if __name__ == '__main__':
    unittest.main()
