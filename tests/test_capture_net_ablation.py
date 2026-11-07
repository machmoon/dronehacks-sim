import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import capture_net_ablation as capture


class FakeObserver:
    def __init__(self, command, **kwargs):
        self.output = Path(command[command.index('--output') + 1])
        self.output.mkdir()
        self.status = {'state': 'observing', 'truth_samples': 1}
        self._write_status()
        self.terminated = False
        self.killed = False
        self.returncode = None

    def _write_status(self):
        (self.output / 'status.json').write_text(json.dumps(self.status))

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.status['state'] = 'stopped'
        self._write_status()
        self.returncode = -15

    def wait(self, timeout=None):
        self.returncode = self.wait_returncode if hasattr(self, 'wait_returncode') else (self.returncode or 0)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def sample_protocol(root):
    return {
        'passive_capture': {
            'capture_paths': {
                'control': 'artifacts/markerfree/net-visual-ablation-01/control/capture.npz',
                'treatment': 'artifacts/markerfree/net-visual-ablation-01/treatment/capture.npz',
            },
            'minimum_sim_clock_before_capture_s': 2.0,
            'per_condition_frames': 32,
            'observer_seconds_wall': 30,
        },
        'expected_runtime_worlds': {'control': {'world_sha256': 'control-world'}},
    }


class CaptureNetAblationTests(unittest.TestCase):
    def test_environment_requires_both_isolation_markers(self):
        with self.assertRaisesRegex(RuntimeError, 'isolated marker-free'):
            capture.validate_environment({'DRONEHACKS_SITL': '1'})
        capture.validate_environment({'DRONEHACKS_SITL': '1',
                                      'GZ_PARTITION': 'dronehacks_markerfree'})

    def test_registered_root_and_existing_condition_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = sample_protocol(root)
            self.assertEqual(capture.registered_root(protocol, root),
                             (root / 'artifacts/markerfree/net-visual-ablation-01').resolve())
            output = capture.registered_root(protocol, root)
            (output / 'control').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                capture.validate_condition_output('control', output)
            with self.assertRaises(ValueError):
                capture.validate_condition_output('both', output)

    def test_sensor_configuration_mismatch_refuses_before_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(capture.Path, 'read_text', return_value='{"world_sha256":"wrong"}'):
                with self.assertRaisesRegex(RuntimeError, 'does not exactly match'):
                    capture.preflight({'world_sha256': 'expected'}, {}, {
                        'DRONEHACKS_SITL': '1', 'GZ_PARTITION': 'dronehacks_markerfree'})

    def test_diagnostic_waits_through_missing_and_partial_startup_file(self):
        responses = iter((FileNotFoundError(), json.JSONDecodeError('partial', '{', 1),
                          json.dumps({'ready': True})))
        class DiagnosticPath:
            def read_text(self, encoding=None):
                response = next(responses)
                if isinstance(response, Exception):
                    raise response
                return response
        ticks = iter((0., .1, .2, .3, .4, .5, .6, .7))
        result = capture.wait_for_sensor_diagnostic(
            {'ready': True}, wall_timeout_s=2., path=DiagnosticPath(),
            monotonic=lambda: next(ticks), sleep=lambda _: None)
        self.assertEqual(result, {'ready': True})

    def test_completed_wrong_diagnostic_fails_without_waiting_for_rewrite(self):
        class DiagnosticPath:
            def read_text(self, encoding=None):
                return '{"ready": false}'
        with self.assertRaisesRegex(RuntimeError, 'does not exactly match'):
            capture.wait_for_sensor_diagnostic({'ready': True}, path=DiagnosticPath())

    def test_capture_failure_terminates_only_started_observer_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = sample_protocol(root)
            output_root = capture.registered_root(protocol, root)
            process_box = []

            def fake_popen(command, **kwargs):
                process = FakeObserver(command, **kwargs)
                process_box.append(process)
                return process

            def failed_capture(*args, **kwargs):
                return subprocess.CompletedProcess(args[0], 3)

            with patch.object(capture, 'wait_for_sim_clock', return_value=2.5):
                with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                    capture.collect_condition(
                        'control', protocol,
                        env={'DRONEHACKS_SITL': '1', 'GZ_PARTITION': 'dronehacks_markerfree'},
                        output_root=output_root,
                        preflight_fn=lambda *_, **__: {'sensor_diagnostic': {'ok': True}},
                        popen=fake_popen, run=failed_capture)

            condition = output_root / 'control'
            self.assertEqual(len(process_box), 1)
            self.assertTrue(process_box[0].terminated)
            self.assertTrue((condition / 'runtime.json').is_file())
            self.assertTrue((condition / 'observer.stdout.log').is_file())
            self.assertTrue((condition / 'capture.stdout.log').is_file())
            self.assertEqual(json.loads((condition / 'observer/status.json').read_text())['state'],
                             'stopped')

    def test_success_keeps_duration_complete_observer_running_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = sample_protocol(root)
            output_root = capture.registered_root(protocol, root)
            process_box = []
            order = []
            def clock(*args, **kwargs):
                order.append('read_imu')
                return 2.5 + len(order) * .1
            def fake_popen(command, **kwargs):
                order.append('start_observer')
                process = FakeObserver(command, **kwargs)
                process_box.append(process)
                return process
            def successful_capture(command, **kwargs):
                order.append('capture')
                output = Path(command[command.index('--output') + 1])
                output.write_bytes(b'fake archive')
                process_box[0].status.update(state='duration_complete', recording_complete=True)
                process_box[0]._write_status()
                return subprocess.CompletedProcess(command, 0)
            with patch.object(capture, 'wait_for_sim_clock', side_effect=clock):
                result = capture.collect_condition(
                    'control', protocol,
                    env={'DRONEHACKS_SITL': '1', 'GZ_PARTITION': 'dronehacks_markerfree'},
                    output_root=output_root,
                    preflight_fn=lambda *_, **__: {'sensor_diagnostic': {'ok': True}},
                    popen=fake_popen, run=successful_capture)
            self.assertEqual(result['condition'], 'control')
            self.assertFalse(process_box[0].terminated)
            self.assertTrue((output_root / 'control/capture.npz').is_file())
            self.assertEqual(order, ['read_imu', 'start_observer', 'read_imu', 'capture'])
            invocation = json.loads((output_root / 'control/invocation.json').read_text())
            self.assertEqual(invocation['observer_duration_wall_s'], 30)
            self.assertEqual(invocation['observer_command'][-1], '30')
            runtime = json.loads((output_root / 'control/runtime.json').read_text())
            self.assertEqual(runtime['readiness_imu_sim_s'], 2.6)
            self.assertEqual(runtime['pre_capture_imu_sim_s'], 2.8)

    def test_nonzero_observer_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = sample_protocol(root)
            output_root = capture.registered_root(protocol, root)
            process_box = []
            def fake_popen(command, **kwargs):
                process = FakeObserver(command, **kwargs)
                process_box.append(process)
                return process
            def successful_capture(command, **kwargs):
                Path(command[command.index('--output') + 1]).write_bytes(b'fake archive')
                process_box[0].status.update(state='duration_complete', recording_complete=True)
                process_box[0]._write_status()
                process_box[0].wait_returncode = 7
                return subprocess.CompletedProcess(command, 0)
            with patch.object(capture, 'wait_for_sim_clock', return_value=2.5):
                with self.assertRaisesRegex(RuntimeError, 'observer exited with status 7'):
                    capture.collect_condition(
                        'control', protocol,
                        env={'DRONEHACKS_SITL': '1', 'GZ_PARTITION': 'dronehacks_markerfree'},
                        output_root=output_root,
                        preflight_fn=lambda *_, **__: {'sensor_diagnostic': {'ok': True}},
                        popen=fake_popen, run=successful_capture)


if __name__ == '__main__':
    unittest.main()
