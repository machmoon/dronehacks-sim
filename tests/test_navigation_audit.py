import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mission.navigation_audit import TRUTH_TOPIC, audit_navigation_sources


class NavigationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.resources = self.root / 'resources'
        self.resources.mkdir()
        self.world = self.root / 'world.sdf'

    def write_world(self, body):
        self.world.write_text(f'<sdf version="1.9"><world name="test">{body}</world></sdf>')
        return self.world

    def add_model(self, name, body=''):
        folder = self.resources / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'model.sdf'
        path.write_text(f'<sdf version="1.9"><model name="{name}">{body}</model></sdf>')
        return path

    def isolated_environment(self, **extra):
        return patch.dict(os.environ, {
            'DRONEHACKS_SITL': '1',
            'GZ_PARTITION': 'dronehacks_markerfree',
            'GZ_SIM_RESOURCE_PATH': str(self.resources),
            **extra,
        }, clear=True)

    @staticmethod
    def no_publisher_result():
        return subprocess.CompletedProcess(
            args=['gz', 'topic'], returncode=0,
            stdout=f'No publishers on topic [{TRUTH_TOPIC}]\n', stderr='')

    def test_accepts_clean_model_graph_and_verified_no_publisher(self):
        self.add_model('x500')
        self.write_world('<include><uri>model://x500</uri></include>')
        with self.isolated_environment(), patch(
                'mission.navigation_audit.subprocess.run',
                return_value=self.no_publisher_result()) as run:
            result = audit_navigation_sources(str(self.world))
        self.assertEqual(result['truth_topic_publishers'], 0)
        self.assertEqual(result['artificial_landmark_models'], 0)
        self.assertFalse(result['live_simulator_position_used'])
        run.assert_called_once()

    def test_rejects_landmarks_publish_plugins_and_unreviewed_includes(self):
        fixtures = (
            ('<model name="vision_landmarks_fixture"/>', 'landmark'),
            ('<plugin name="truth_odometry" filename="publisher.so"/>', 'odometry'),
            ('<plugin name="pose" filename="PosePublisher.so"/>', 'pose publisher'),
            ('<include><uri>file://external/model.sdf</uri></include>', 'file include'),
            ('<include><uri>model://missing</uri></include>', 'missing model'),
        )
        for body, label in fixtures:
            with self.subTest(label=label):
                self.write_world(body)
                with self.isolated_environment(), patch(
                        'mission.navigation_audit.subprocess.run') as run:
                    with self.assertRaises(RuntimeError):
                        audit_navigation_sources(str(self.world))
                    run.assert_not_called()

    def test_rejects_truth_publisher_and_unrecognized_topic_output(self):
        self.write_world('')
        outputs = (
            f'Publishers [Address, Message Type]:\n  tcp://127.0.0.1, gz.msgs.OdometryWithCovariance\nTopic [{TRUTH_TOPIC}]',
            'topic not found',
        )
        for output in outputs:
            with self.subTest(output=output[:24]):
                result = subprocess.CompletedProcess(args=['gz', 'topic'], returncode=0,
                                                     stdout=output, stderr='')
                with self.isolated_environment(), patch(
                        'mission.navigation_audit.subprocess.run', return_value=result):
                    with self.assertRaises(RuntimeError):
                        audit_navigation_sources(str(self.world))

    def test_rejects_wrong_partition_before_audit(self):
        self.write_world('')
        with self.isolated_environment(GZ_PARTITION='another_world'), patch(
                'mission.navigation_audit.subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'isolated marker-free SITL partition'):
                audit_navigation_sources(str(self.world))
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
