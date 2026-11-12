import json
import ast
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from mission import px4_run


class FakeMavlinkConstants:
    MAV_MODE_FLAG_SAFETY_ARMED = 128


class FakeHeartbeat:
    def __init__(self, base_mode=0, custom_mode=0):
        self.base_mode = base_mode
        self.custom_mode = custom_mode


class Px4ContractTests(unittest.TestCase):
    def make_link(self, heartbeat):
        link = px4_run.Link.__new__(px4_run.Link)
        link.m = FakeMavlinkConstants
        link.heartbeat = heartbeat
        return link

    def test_offboard_mode_decoder_returns_boolean_for_px4_main_mode_six(self):
        self.assertTrue(self.make_link(FakeHeartbeat(custom_mode=6 << 16)).offboard)
        self.assertFalse(self.make_link(FakeHeartbeat(custom_mode=5 << 16)).offboard)
        self.assertFalse(self.make_link(None).offboard)

    def test_arm_decoder_returns_boolean(self):
        self.assertTrue(self.make_link(FakeHeartbeat(base_mode=128)).armed)
        self.assertFalse(self.make_link(FakeHeartbeat(base_mode=0)).armed)
        self.assertFalse(self.make_link(None).armed)

    def test_telemetry_freshness_uses_sim_heartbeat_and_wall_position_age(self):
        link = px4_run.Link.__new__(px4_run.Link)
        link.position = object()
        link.position_wall = 100.1
        link.heartbeat_wall = 0.0  # Wall age is intentionally irrelevant here.
        link.boot_s = 10.5
        link.heartbeat_sim_s = 10.0

        self.assertTrue(link.telemetry_fresh(now=100.2))
        link.boot_s = 14.0  # Four seconds since the last PX4 heartbeat in sim time.
        self.assertFalse(link.telemetry_fresh(now=100.2))
        link.boot_s = 10.5
        link.position_wall = 97.2
        self.assertFalse(link.telemetry_fresh(now=100.2))

    def test_px4_bytewise_int32_parameter_decodes_float_field_as_integer(self):
        encoded_11 = struct.unpack('<f', struct.pack('<i', 11))[0]
        message = SimpleNamespace(param_type=6, param_value=encoded_11)
        self.assertEqual(px4_run.px4_parameter_value(message), 11)

    def test_runner_has_no_simulator_pose_subscription_or_feedback_path(self):
        source = Path(px4_run.__file__).read_text().lower()
        tree = ast.parse(source)
        called_attributes = {node.func.attr.lower() for node in ast.walk(tree)
                             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("model_states", called_attributes)
        self.assertNotIn("subscribe", called_attributes)
        self.assertIn("local_position_ned", source)
        self.assertIn("vision_position_estimate_send", source)

    def test_failure_summary_keeps_original_fault_and_event_in_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            log = px4_run.Log(path, {'waypoints': []}, 'hover')
            cause = 'camera_tracking_lost'
            log.event(f'Mission stopped: {cause}; requested PX4 Land')
            log.finish('failed', fault=cause, route_completed=False)
            log.file.close()

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[-1]['type'], 'summary')
            self.assertEqual(rows[-1]['fault'], cause)
            self.assertTrue(any(row.get('message', '').endswith(cause + '; requested PX4 Land')
                                for row in rows))
            summary = json.loads(path.with_suffix('.summary.json').read_text())
            self.assertEqual(summary['fault'], cause)

    def test_unexpected_loop_exception_is_summarized_with_original_detail(self):
        class FakeLink:
            def __init__(self, log):
                self.armed = False
                self.boot_s = 0.0

            def drain(self):
                raise RuntimeError('injected bridge fault')

        class FakeVision:
            def __init__(self, **kwargs):
                pass

            def poll(self):
                return None

        localize = ModuleType('vision.localize')
        localize.Vision = FakeVision
        args = SimpleNamespace(course=Path(px4_run.__file__).parent / 'course.json',
                               output=None, task='course', speed_scale=.45,
                               startup_timeout=20, max_sim_seconds=20,
                               blackout_after=None, exit_on_finish=False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'unexpected.jsonl'
            args.output = output
            with patch.dict('os.environ', {'DRONEHACKS_SITL': '1'}), \
                    patch.dict(sys.modules, {'vision.localize': localize}), \
                    patch.object(px4_run, 'Link', FakeLink):
                result = px4_run.run(args)

            self.assertEqual(result, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            events = [row for row in rows if row.get('type') == 'event']
            summary = json.loads(output.with_suffix('.summary.json').read_text())
            self.assertEqual(summary['fault'], 'bridge_exception')
            exception_event = next(row for row in events if row.get('message') == 'Bridge exception')
            self.assertEqual(exception_event['error'], 'RuntimeError: injected bridge fault')
            self.assertTrue(any(row.get('message', '').endswith(
                'bridge_exception; requested PX4 Land') for row in events))

    def test_initialization_exception_is_logged_before_it_is_reraised(self):
        class BrokenLink:
            def __init__(self, log):
                raise OSError('injected socket setup failure')

        class UnusedVision:
            def __init__(self, **kwargs):
                raise AssertionError('vision should not initialize after link failure')

        localize = ModuleType('vision.localize')
        localize.Vision = UnusedVision
        args = SimpleNamespace(course=Path(px4_run.__file__).parent / 'course.json',
                               output=None, task='course', speed_scale=.45)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'initialization.jsonl'
            args.output = output
            with patch.dict('os.environ', {'DRONEHACKS_SITL': '1'}), \
                    patch.dict(sys.modules, {'vision.localize': localize}), \
                    patch.object(px4_run, 'Link', BrokenLink):
                with self.assertRaisesRegex(OSError, 'injected socket setup failure'):
                    px4_run.run(args)

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            summary = json.loads(output.with_suffix('.summary.json').read_text())
            self.assertEqual(summary['fault'], 'bridge_initialization_failed')
            self.assertTrue(any(row.get('error') ==
                                'OSError: injected socket setup failure' for row in rows))


if __name__ == '__main__':
    unittest.main()
