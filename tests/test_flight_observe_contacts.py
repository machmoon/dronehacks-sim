import unittest

from vision.flight_observe import (
    CONTACT_TOPICS,
    contact_topic_audit,
    contact_topic_interpretation,
)


class FakeNode:
    def __init__(self, topics, info):
        self.topics = topics
        self.info = info

    def topic_list(self):
        return self.topics

    def topic_info(self, topic):
        return self.info[topic]


class ContactObserverTests(unittest.TestCase):
    def test_audit_counts_publishers_without_claiming_sensor_health(self):
        topic = CONTACT_TOPICS[0]
        node = FakeNode([topic], {topic: ([object()] * 5, [object()])})
        audit = contact_topic_audit(node, (topic, CONTACT_TOPICS[1]))
        self.assertEqual(audit[topic], 5)
        self.assertIsNone(audit[CONTACT_TOPICS[1]])
        self.assertEqual(contact_topic_interpretation(audit[topic], audit[topic], 0),
                         'publisher_advertised_no_messages_observed')

    def test_missing_publisher_and_observed_message_are_distinct(self):
        self.assertEqual(contact_topic_interpretation(None, 0, 0),
                         'no_publisher_observed')
        self.assertEqual(contact_topic_interpretation(None, None, 1),
                         'messages_received')


if __name__ == '__main__':
    unittest.main()
