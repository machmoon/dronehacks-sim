import io
from collections import defaultdict
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from pathlib import Path
import tempfile
import time
import unittest

from tools.run_contact_sensor_controls import (
    MAX_LOG_BYTES, CLI_CAPTURE_LIMIT, _bounded_drain, partition_name, _build_parser, _validate_args,
    _clock_callback, _contact_callback, _request_steps, _control_command,
    _run_control_command, _pause_readiness,
    _is_exact_setup_step, _summary_exit_code, _case_passed, _verify_inputs,
    _sha256, _worker_command, _cleanup_process_group,
)


class ContactSensorRunnerTests(unittest.TestCase):
    def test_partition_names_are_unique_and_never_use_flight_partition(self):
        names = {partition_name() for _ in range(50)}
        self.assertEqual(len(names), 50)
        for name in names:
            self.assertTrue(name.startswith('dronehacks_contact_probe_'))
            self.assertNotEqual(name, 'markerfree')

    def test_log_drain_is_capped_while_consuming_entire_pipe(self):
        payload = b'x'*(MAX_LOG_BYTES+2048)
        stream = io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bounded.log'
            written, truncated = _bounded_drain(stream, path, limit=1024)
            self.assertEqual(written, 1024)
            self.assertTrue(truncated)
            self.assertEqual(path.stat().st_size, 1024)
            self.assertEqual(stream.read(), b'')

    def test_worker_cli_does_not_require_parent_only_paths(self):
        parser = _build_parser()
        args = parser.parse_args(['--worker', '--world', 'case.sdf', '--fixture-json', '{}',
                                  '--case-dir', '/tmp/case', '--partition', 'isolated'])
        _validate_args(parser, args)
        self.assertIsNone(args.fixtures)
        self.assertIsNone(args.output)

    def test_worker_command_uses_module_entry_and_repository_cwd(self):
        root = Path(__file__).resolve().parents[1]
        command, cwd = _worker_command(sys.executable, root, root/'case.sdf', {'topic': '/t'},
                                        root/'case', 'partition', 40, 30)
        self.assertEqual(command[1:3], ['-m', 'tools.run_contact_sensor_controls'])
        self.assertEqual(Path(cwd), root)
        self.assertEqual(command[command.index('--world')+1], str(root/'case.sdf'))

    def test_clock_and_contacts_use_distinct_timestamp_shapes(self):
        lock = __import__('threading').Lock()
        clocks, rows, counts = [], [], defaultdict(int)
        _clock_callback(SimpleNamespace(sim=SimpleNamespace(sec=4, nsec=123)), clocks, lock)
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=5, nsec=456)),
            contact=[SimpleNamespace(collision1=SimpleNamespace(name='quad::body'),
                                     collision2=SimpleNamespace(name='probe::box'))])
        _contact_callback('/topic', message, rows, counts, lock)
        self.assertEqual(clocks, [4_000_000_123])
        self.assertEqual(rows, [{'topic': '/topic', 'sim_time_ns': 5_000_000_456,
                                 'pairs': [['quad::body', 'probe::box']]}])
        self.assertEqual(counts['/topic'], 1)

    def test_cli_command_contains_typed_service_and_bounded_request(self):
        command = _control_command('/world/x/control', pause=True, multi_step=40,
                                   timeout_ms=3000)
        self.assertEqual(command[:4], ['gz', 'service', '-s', '/world/x/control'])
        self.assertEqual(command[command.index('--reqtype')+1], 'gz.msgs.WorldControl')
        self.assertEqual(command[command.index('--reptype')+1], 'gz.msgs.Boolean')
        self.assertEqual(command[command.index('--timeout')+1], '3000')
        self.assertEqual(command[command.index('--req')+1], 'pause: true, multi_step: 40')
        self.assertEqual(_control_command('/world/x/control', pause=True)[-1], 'pause: true')

    def test_pause_readiness_retries_only_idempotent_pause(self):
        sent = []

        def request(service, *, pause, multi_step=0, timeout_s=4.):
            sent.append((service, pause, multi_step, timeout_s))
            accepted = len(sent) == 2
            return {'accepted': accepted, 'transport_ok': accepted,
                    'response_data': accepted, 'elapsed_ms': 1.}

        attempts = _pause_readiness('/world/x/control', time.monotonic()+1,
                                    retry_interval=0, request_func=request)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]['accepted'])
        self.assertTrue(attempts[1]['accepted'])
        self.assertEqual([item[:3] for item in sent], [('/world/x/control', True, 0)]*2)
        self.assertTrue(all(0 < item[3] <= 1 for item in sent))

    def test_cli_response_requires_success_exit_and_exact_true_text(self):
        def run(fake_process):
            def factory(command, **kwargs):
                self.assertEqual(command[0:2], ['gz', 'service'])
                return fake_process
            return _run_control_command(['gz', 'service'], popen_factory=factory)

        class FakeProcess:
            def __init__(self, stdout, stderr=b'', returncode=0, timeout_once=False):
                self.stdout = stdout if hasattr(stdout, 'read') else io.BytesIO(stdout)
                self.stderr = stderr if hasattr(stderr, 'read') else io.BytesIO(stderr)
                self.returncode = returncode
                self.timeout_once = timeout_once

            def wait(self, timeout=None):
                if self.timeout_once:
                    self.timeout_once = False
                    raise subprocess.TimeoutExpired('gz service', timeout)
                return self.returncode

            def send_signal(self, sig):
                self.returncode = -sig

            def terminate(self):
                self.returncode = -signal.SIGTERM

            def kill(self):
                self.returncode = -signal.SIGKILL

        accepted = run(FakeProcess(b'data: true\n\n'))
        self.assertTrue(accepted['accepted'])
        self.assertEqual(accepted['response_data'], True)
        nonzero = run(FakeProcess(b'data: true\n', returncode=1))
        self.assertFalse(nonzero['accepted'])
        malformed = run(FakeProcess(b'ok: true\n'))
        self.assertFalse(malformed['accepted'])
        false_response = run(FakeProcess(b'data: false\n'))
        self.assertFalse(false_response['accepted'])
        timed_out = run(FakeProcess(b'data: true\n', timeout_once=True))
        self.assertTrue(timed_out['timed_out'])
        self.assertFalse(timed_out['accepted'])

        truncated = run(FakeProcess(b'data: true\n'+b' '*(CLI_CAPTURE_LIMIT+1)))
        self.assertTrue(truncated['stdout_truncated'])
        self.assertTrue(truncated['stdout_drain_complete'])
        self.assertFalse(truncated['accepted'])

        class BlockingStream:
            def __init__(self):
                self.release = threading.Event()

            def read(self, _size):
                self.release.wait(timeout=3)
                return b''

        blocked_stderr = BlockingStream()
        incomplete = run(FakeProcess(b'data: true\n', stderr=blocked_stderr))
        self.assertFalse(incomplete['stderr_drain_complete'])
        self.assertFalse(incomplete['accepted'])
        blocked_stderr.release.set()

    def test_setup_clock_must_advance_exactly_one_step(self):
        self.assertTrue(_is_exact_setup_step(1_000_000, 5_000_000))
        self.assertFalse(_is_exact_setup_step(None, 4_000_000))
        self.assertFalse(_is_exact_setup_step(1_000_000, 1_000_000))
        self.assertFalse(_is_exact_setup_step(1_000_000, 9_000_000))

    def test_runner_exit_is_nonzero_for_any_failed_or_inconclusive_case(self):
        self.assertEqual(_summary_exit_code([{'passed': True}]*10), 0)
        self.assertEqual(_summary_exit_code([{'passed': True}]*9+[{'passed': False}]), 1)

    def test_pass_requires_runtime_clock_hashes_and_expected_score(self):
        fixture = {'kind': 'positive'}
        good = {'state': 'complete', 'clock_advanced': True, 'clock_advance_sufficient': True,
                'worker_returncode': 0, 'timed_out': False,
                'worker_log_drain_complete': True, 'gz_log_drain_complete': True}
        good_score = {'outcome': 'positive_pair_observed'}
        self.assertTrue(_case_passed(fixture, good, good_score, True))
        for runtime, verified, score in (
                (dict(good, state='worker_failed'), True, good_score),
                (dict(good, clock_advance_sufficient=False), True, good_score),
                (dict(good, worker_returncode=-signal.SIGKILL), True, good_score),
                (dict(good, timed_out=True), True, good_score),
                (dict(good, worker_log_drain_complete=False), True, good_score),
                (good, False, good_score),
                (good, True, {'outcome': 'expected_pair_not_observed'})):
            self.assertFalse(_case_passed(fixture, runtime, score, verified))

    @unittest.skipUnless(os.name == 'posix', 'POSIX process groups required')
    def test_group_cleanup_stops_descendant_after_worker_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'term-received'
            ready = Path(tmp)/'child-ready'
            child = (
                'import signal,pathlib,time,sys; '
                'marker=pathlib.Path(sys.argv[1]); ready=pathlib.Path(sys.argv[2]); '
                'signal.signal(signal.SIGINT,signal.SIG_IGN); '
                "signal.signal(signal.SIGTERM,lambda *_:(marker.write_text('done'),sys.exit(0))); "
                "ready.write_text('ready'); time.sleep(30)"
            )
            parent = subprocess.Popen(
                [sys.executable, '-c',
                 'import subprocess,sys; subprocess.Popen([sys.executable,"-c",sys.argv[1], '
                 'sys.argv[2],sys.argv[3]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);',
                 child, str(marker), str(ready)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            parent.wait(timeout=2)
            deadline = time.monotonic()+2
            while time.monotonic() < deadline and not ready.exists():
                time.sleep(.01)
            self.assertTrue(ready.exists())
            _cleanup_process_group(parent.pid, grace_seconds=.3)
            self.assertTrue(marker.exists())

    def test_source_and_world_hashes_are_checked_before_launch(self):
        source = Path(__file__).resolve().parents[1]/'tools/check_contact_sensors.py'
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            world = fixture_dir/'case.sdf'
            world.write_text('<sdf/>')
            manifest = {
                'source_sha256': {'tools/check_contact_sensors.py': _sha256(source)},
                'fixtures': [{'world_file': 'case.sdf', 'world_sha256': _sha256(world)}],
            }
            self.assertTrue(_verify_inputs(fixture_dir, manifest))
            world.write_text('<sdf changed="true"/>')
            with self.assertRaisesRegex(ValueError, 'world hash mismatch'):
                _verify_inputs(fixture_dir, manifest)


if __name__ == '__main__':
    unittest.main()
