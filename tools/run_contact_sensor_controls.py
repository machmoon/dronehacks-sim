"""Run bounded, paused-step contact-sensor controls in isolated Gazebo partitions.

Run inside the Gazebo container where gz Python bindings are installed. Each
case gets a unique partition and its own gz sim server process group.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from tools.check_contact_sensors import score_fixture

MAX_CASE_SECONDS = 30.0
MAX_LOG_BYTES = 1_000_000
DEFAULT_STEPS = 40
STEP_NS = 4_000_000
CLOCK_TOPIC = '/clock'
PAUSE_READY_ATTEMPTS = 3
CONTROL_REQUEST_TIMEOUT_MS = 3000
CLI_CAPTURE_LIMIT = 65536


def partition_name():
    return f'dronehacks_contact_probe_{uuid.uuid4().hex}'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_drain(stream, path, limit=MAX_LOG_BYTES):
    written, truncated = 0, False
    with Path(path).open('wb') as output:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            remaining = limit-written
            if remaining > 0:
                output.write(chunk[:remaining])
                written += min(remaining, len(chunk))
            if len(chunk) > remaining:
                truncated = True
    return written, truncated


def _clock_ns(message):
    value = message.sim
    return int(value.sec)*1_000_000_000 + int(value.nsec)


def _contact_stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec)*1_000_000_000 + int(stamp.nsec)


def _pairs(message):
    return [[contact.collision1.name, contact.collision2.name]
            for contact in message.contact]


def _world_name(world_path):
    return ET.parse(world_path).getroot().find('world').get('name')


def _control_command(service, *, pause, multi_step=0, timeout_ms=CONTROL_REQUEST_TIMEOUT_MS):
    fields = [f'pause: {str(bool(pause)).lower()}']
    if multi_step:
        fields.append(f'multi_step: {int(multi_step)}')
    return ['gz', 'service', '-s', service,
            '--reqtype', 'gz.msgs.WorldControl', '--reptype', 'gz.msgs.Boolean',
            '--timeout', str(int(timeout_ms)), '--req', ', '.join(fields)]


def _capture_limited(stream, bytearray_out, truncated_out, limit=CLI_CAPTURE_LIMIT):
    while True:
        chunk = stream.read(8192)
        if not chunk:
            return
        remaining = limit-len(bytearray_out)
        if remaining > 0:
            bytearray_out.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated_out.append(True)


def _run_control_command(command, timeout_s=4.0, popen_factory=subprocess.Popen):
    process = popen_factory(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=os.environ.copy(), start_new_session=False)
    stdout_bytes, stderr_bytes = bytearray(), bytearray()
    stdout_truncated, stderr_truncated = [], []
    out_thread = threading.Thread(target=_capture_limited,
                                  args=(process.stdout, stdout_bytes, stdout_truncated), daemon=True)
    err_thread = threading.Thread(target=_capture_limited,
                                  args=(process.stderr, stderr_bytes, stderr_truncated), daemon=True)
    out_thread.start()
    err_thread.start()
    started = time.monotonic()
    timed_out = False
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=.5)
    finally:
        out_thread.join(timeout=1)
        err_thread.join(timeout=1)
    stdout = stdout_bytes.decode('utf-8', errors='replace')
    stderr = stderr_bytes.decode('utf-8', errors='replace')
    valid_response = stdout.strip() == 'data: true'
    return {'returncode': process.returncode, 'timed_out': timed_out,
            'response_data': True if valid_response else (False if stdout.strip() == 'data: false' else None),
            'response_valid': valid_response,
            'accepted': (process.returncode == 0 and not timed_out and valid_response
                         and not stdout_truncated and not stderr_truncated
                         and not out_thread.is_alive() and not err_thread.is_alive()),
            'elapsed_ms': (time.monotonic()-started)*1000,
            'stdout': stdout, 'stderr': stderr,
            'stdout_truncated': bool(stdout_truncated),
            'stderr_truncated': bool(stderr_truncated),
            'stdout_drain_complete': not out_thread.is_alive(),
            'stderr_drain_complete': not err_thread.is_alive(),
            'command': command}


def _request_control(service, *, pause, multi_step=0, timeout_s=4.0):
    timeout_ms = max(1, min(CONTROL_REQUEST_TIMEOUT_MS, int(timeout_s*1000)))
    command = _control_command(service, pause=pause, multi_step=multi_step,
                               timeout_ms=timeout_ms)
    return _run_control_command(command, timeout_s=timeout_s)


def _request_steps(service, count):
    return _request_control(service, pause=True, multi_step=count)


def _pause_readiness(service, deadline,
                     max_attempts=PAUSE_READY_ATTEMPTS, retry_interval=.05,
                     request_func=None):
    attempts = []
    request_func = request_func or _request_control
    for attempt in range(1, max_attempts+1):
        if time.monotonic() >= deadline:
            break
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            break
        result = request_func(service, pause=True, timeout_s=min(remaining, 4.0))
        result['attempt'] = attempt
        attempts.append(result)
        if result['accepted']:
            break
        if attempt < max_attempts:
            time.sleep(min(retry_interval, max(0., deadline-time.monotonic())))
    return attempts


def _is_exact_setup_step(before_ns, after_ns):
    return before_ns is not None and after_ns is not None and after_ns-before_ns == STEP_NS


def _summary_exit_code(run_rows):
    return 0 if all(row.get('passed') is True for row in run_rows) else 1


def _clock_callback(message, samples, lock):
    with lock:
        samples.append(_clock_ns(message))


def _contact_callback(topic, message, rows, counts, lock):
    pairs = _pairs(message)
    stamp_ns = _contact_stamp_ns(message)
    with lock:
        counts[topic] += 1
        if pairs:
            rows.append({'topic': topic, 'sim_time_ns': stamp_ns, 'pairs': pairs})


def _worker(world_path, fixture, case_dir, partition, steps, case_timeout):
    os.environ['GZ_PARTITION'] = partition
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.transport13 import Node

    case_dir = Path(case_dir)
    lifecycle, contacts, clock_samples = [], [], []
    counts, lock = defaultdict(int), threading.Lock()
    world_name = _world_name(world_path)
    control_service = f'/world/{world_name}/control'
    started, deadline = time.monotonic(), time.monotonic()+case_timeout
    sim = None
    sim_log_thread = None
    log_state = {}
    result = None
    try:
        lifecycle.append({'event': 'server_start', 'paused': True, 'partition': partition})
        sim = subprocess.Popen(['gz', 'sim', '-v', '4', '-s', str(world_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=os.environ.copy(), start_new_session=False)
        sim_log_thread = threading.Thread(
            target=lambda: log_state.update(zip(('bytes', 'truncated'),
                _bounded_drain(sim.stdout, case_dir/'gz-server.log'))), daemon=True)
        sim_log_thread.start()

        node = Node()
        if not node.subscribe(Clock, CLOCK_TOPIC,
                              lambda msg: _clock_callback(msg, clock_samples, lock)):
            raise RuntimeError('cannot subscribe to Gazebo /clock')
        topic = fixture['topic']
        if not node.subscribe(Contacts, topic,
                              lambda msg: _contact_callback(topic, msg, contacts, counts, lock)):
            raise RuntimeError(f'cannot subscribe to {topic}')
        lifecycle.append({'event': 'subscribed', 'clock_topic': CLOCK_TOPIC,
                          'contact_topics': [topic]})

        discovery = {'contact_publishers_before_setup': None,
                     'contact_publishers_after_setup': None,
                     'clock_publishers': None, 'control_service': False}
        # Contact publisher advertisement can be delayed until sensors initialize;
        # require clock + control service, then use one safe 4 ms setup step.
        while time.monotonic() < deadline:
            if sim.poll() is not None:
                raise RuntimeError(f'gz sim exited early with code {sim.returncode}')
            try:
                discovery['clock_publishers'] = len(node.topic_info(CLOCK_TOPIC)[0])
            except Exception:
                pass
            discovery['control_service'] = control_service in node.service_list()
            try:
                discovery['contact_publishers_before_setup'] = len(node.topic_info(topic)[0])
            except Exception:
                pass
            if discovery['clock_publishers'] and discovery['control_service']:
                break
            time.sleep(.05)
        else:
            raise TimeoutError(f'clock/control discovery timeout: {discovery}')

        pause_attempts = _pause_readiness(control_service, deadline)
        lifecycle.append({'event': 'pause_readiness', 'attempts': pause_attempts,
                          'accepted': bool(pause_attempts and pause_attempts[-1]['accepted'])})
        if not pause_attempts or not pause_attempts[-1]['accepted']:
            raise RuntimeError(f'pause readiness failed: {pause_attempts}')

        # A freshly loaded, paused world starts at zero; Gazebo may not
        # publish /clock until its first iteration. This is declared setup
        # state, not a fabricated received timestamp. The measured post-step
        # clock must be exactly 4 ms, so accidental early advancement fails.
        with lock:
            observed_baseline = clock_samples[-1] if clock_samples else None
        if observed_baseline not in (None, 0):
            raise RuntimeError(f'fresh paused world already advanced: {observed_baseline}')
        before_setup = 0
        lifecycle.append({'event': 'initial_clock', 'observed_clock_ns': observed_baseline,
                          'expected_fresh_world_origin_ns': 0})
        setup_request = _request_steps(control_service, 1)
        lifecycle.append({'event': 'setup_step_request', **setup_request})
        if not setup_request['accepted']:
            raise RuntimeError(f'one-step setup request failed: {setup_request}')
        setup_clock = None
        setup_deadline = min(deadline, time.monotonic()+1.)
        while time.monotonic() < setup_deadline:
            with lock:
                setup_clock = clock_samples[-1] if clock_samples else None
            if setup_clock is not None and setup_clock != before_setup:
                break
            time.sleep(.01)
        if not _is_exact_setup_step(before_setup, setup_clock):
            raise RuntimeError(
                f'setup step did not advance exactly {STEP_NS} ns: '
                f'before={before_setup}, after={setup_clock}')
        lifecycle.append({'event': 'setup_step', 'clock_ns': setup_clock, 'steps': 1,
                          'initial_clearance_m': fixture.get('initial_clearance_m')})

        while time.monotonic() < deadline:
            try:
                discovery['contact_publishers_after_setup'] = len(node.topic_info(topic)[0])
            except Exception:
                discovery['contact_publishers_after_setup'] = 0
            if discovery['contact_publishers_after_setup']:
                break
            time.sleep(.05)
        if not discovery['contact_publishers_after_setup']:
            raise TimeoutError(f'contact publisher absent after safe setup step: {discovery}')
        lifecycle.append({'event': 'discovered', **discovery})

        before = setup_clock
        drop_request = _request_steps(control_service, steps)
        lifecycle.append({'event': 'bounded_step_request', **drop_request})
        if not drop_request['accepted']:
            raise RuntimeError(f'{steps}-step request failed: {drop_request}')
        required_after = before + steps*STEP_NS
        after = None
        while time.monotonic() < deadline:
            with lock:
                after = clock_samples[-1] if clock_samples else None
            if after is not None and after >= required_after:
                break
            time.sleep(.01)
        advanced = after is not None and after > before
        enough = after is not None and after == required_after
        lifecycle.append({'event': 'bounded_steps', 'requested_steps': steps,
                          'expected_sim_advance_ns': steps*STEP_NS,
                          'clock_before_ns': before, 'clock_after_ns': after,
                          'clock_advanced': advanced, 'advance_sufficient': enough, 'requires_exact_step_count': True})
        # Let queued callbacks drain after simulation remains paused.
        time.sleep(.05)
        with lock:
            rows = list(contacts)
            clock_sample_rows = list(clock_samples)
        result = {'state': 'complete' if advanced and enough else 'inconclusive_clock',
                  'partition': partition, 'world_name': world_name,
                  'contact_topic_discovery': discovery, 'clock_advanced': advanced,
                  'clock_advance_sufficient': enough, 'rows': rows,
                  'clock_samples_ns': clock_sample_rows,
                  'contact_message_counts': dict(counts), 'lifecycle': lifecycle}
    except Exception as error:
        with lock:
            rows = list(contacts)
            partial_clock_samples = list(clock_samples)
        lifecycle.append({'event': 'worker_error',
                          'error': f'{type(error).__name__}: {error}'})
        result = {'state': 'worker_error', 'partition': partition,
                  'world_name': world_name, 'rows': rows,
                  'clock_samples_ns': partial_clock_samples,
                  'contact_message_counts': dict(counts), 'lifecycle': lifecycle,
                  'error': f'{type(error).__name__}: {error}'}
    finally:
        if sim is not None:
            # Signal the direct server process; the parent owns group cleanup,
            # including any server descendants left behind.
            try:
                sim.send_signal(signal.SIGINT)
                lifecycle.append({'event': 'server_sigint'})
            except ProcessLookupError:
                pass
            if sim.poll() is None:
                try:
                    sim.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    lifecycle.append({'event': 'server_terminate'})
                    sim.terminate()
                    try:
                        sim.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        lifecycle.append({'event': 'server_kill'})
                        sim.kill()
                        sim.wait(timeout=1)
        if sim_log_thread is not None:
            sim_log_thread.join(timeout=2)
        gz_log_drain_complete = sim_log_thread is None or not sim_log_thread.is_alive()
        lifecycle.append({'event': 'worker_teardown_complete',
                          'elapsed_wall_s': time.monotonic()-started,
                          'gz_log_bytes': log_state.get('bytes'),
                          'gz_log_truncated': log_state.get('truncated', False),
                          'gz_log_drain_complete': gz_log_drain_complete})
        (case_dir/'worker_lifecycle.json').write_text(json.dumps(lifecycle, indent=2)+'\n')
    if result is None:
        raise RuntimeError('worker ended without result')
    result['lifecycle'] = lifecycle
    result['gz_log_bytes'] = log_state.get('bytes')
    result['gz_log_truncated'] = log_state.get('truncated', False)
    result['gz_log_drain_complete'] = gz_log_drain_complete
    return result


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, help='generated fixture directory')
    parser.add_argument('--output', type=Path, help='new results directory')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--case-timeout', type=float, default=MAX_CASE_SECONDS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--world', type=Path)
    parser.add_argument('--fixture-json')
    parser.add_argument('--case-dir', type=Path)
    parser.add_argument('--partition')
    return parser


def _validate_args(parser, args):
    if not 1 <= args.steps <= 50 or not 1 <= args.case_timeout <= MAX_CASE_SECONDS:
        parser.error('steps must be 1..50 and case timeout 1..30 seconds')
    if args.worker:
        missing = [name for name in ('world', 'fixture_json', 'case_dir', 'partition')
                   if getattr(args, name) is None]
        if missing:
            parser.error(f'worker mode requires: {", ".join(missing)}')
    elif args.fixtures is None or args.output is None:
        parser.error('parent mode requires --fixtures and --output')


def _run_case(world_path, fixture, case_dir, steps, case_timeout):
    case_dir.mkdir(parents=True, exist_ok=False)
    partition = partition_name()
    repo_root = Path(__file__).resolve().parents[1]
    command, cwd = _worker_command(sys.executable, repo_root, world_path, fixture,
                                   case_dir, partition, steps, case_timeout)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={**os.environ, 'GZ_PARTITION': partition},
                               cwd=cwd, start_new_session=True)
    drain_result = {}
    drain = threading.Thread(target=lambda: drain_result.update(zip(('bytes', 'truncated'),
        _bounded_drain(process.stdout, case_dir/'worker.log', limit=MAX_LOG_BYTES))), daemon=True)
    drain.start()
    timed_out = False
    interrupted = False
    try:
        process.wait(timeout=case_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Always clean this worker's private process group, including after a
        # normal worker exit that may have left a server descendant behind.
        _cleanup_process_group(process.pid)
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        drain.join(timeout=2)
        worker_log_drain_complete = not drain.is_alive()
    if interrupted:
        raise KeyboardInterrupt
    result_file = case_dir/'worker_result.json'
    if result_file.exists():
        result = json.loads(result_file.read_text())
    else:
        result = {'state': 'worker_timeout' if timed_out else 'worker_failed'}
        lifecycle_file = case_dir/'worker_lifecycle.json'
        if lifecycle_file.exists():
            result['lifecycle'] = json.loads(lifecycle_file.read_text())
    result.update(worker_log_bytes=drain_result.get('bytes'),
                  worker_log_truncated=drain_result.get('truncated', False), partition=partition,
                  worker_log_drain_complete=worker_log_drain_complete,
                  worker_returncode=process.returncode, timed_out=timed_out)
    return result


def _worker_command(python, repo_root, world_path, fixture, case_dir,
                    partition, steps, case_timeout):
    command = [python, '-m', 'tools.run_contact_sensor_controls', '--worker',
               '--world', str(world_path), '--fixture-json', json.dumps(fixture),
               '--case-dir', str(case_dir), '--partition', partition,
               '--steps', str(steps), '--case-timeout', str(case_timeout)]
    return command, str(Path(repo_root).resolve())


def _verify_inputs(fixtures_dir, manifest):
    repo_root = Path(__file__).resolve().parents[1]
    mismatches = []
    for relative, expected in manifest['source_sha256'].items():
        path = repo_root/relative
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(f'source hash mismatch: {relative}')
    for fixture in manifest['fixtures']:
        world_path = fixtures_dir/fixture['world_file']
        if not world_path.is_file() or _sha256(world_path) != fixture['world_sha256']:
            mismatches.append(f'world hash mismatch: {fixture["world_file"]}')
    if mismatches:
        raise ValueError('; '.join(mismatches))
    return True


def _case_passed(fixture, runtime, score, hashes_verified):
    if not hashes_verified or runtime.get('state') != 'complete':
        return False
    if (runtime.get('worker_returncode') != 0 or runtime.get('timed_out') is not False
            or runtime.get('worker_log_drain_complete') is not True
            or runtime.get('gz_log_drain_complete') is not True):
        return False
    if runtime.get('clock_advanced') is not True or runtime.get('clock_advance_sufficient') is not True:
        return False
    return score['outcome'] == ('positive_pair_observed' if fixture['kind'] == 'positive'
                                else 'negative_control_passed_reference_only')


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.worker:
        case_dir = args.case_dir
        result = _worker(args.world, json.loads(args.fixture_json), case_dir,
                         args.partition, args.steps, args.case_timeout)
        (case_dir/'worker_result.json').write_text(json.dumps(result, indent=2)+'\n')
        return 0

    fixtures_dir = args.fixtures.resolve()
    manifest = json.loads((fixtures_dir/'manifest.json').read_text())
    if len(manifest.get('fixtures', ())) != 10:
        raise ValueError('expected exactly 10 generated contact-control fixtures')
    ids = [fixture['fixture_id'] for fixture in manifest['fixtures']]
    if len(ids) != len(set(ids)):
        raise ValueError('fixture manifest contains duplicate IDs')
    hashes_verified = _verify_inputs(fixtures_dir, manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_rows = []
    for fixture in manifest['fixtures']:
        case_dir = output/fixture['fixture_id']
        result = _run_case(fixtures_dir/fixture['world_file'], fixture, case_dir,
                           args.steps, args.case_timeout)
        rows = result.pop('rows', [])
        (case_dir/'contacts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        score = score_fixture(fixture, rows)
        world_actual = _sha256(fixtures_dir/fixture['world_file'])
        case_record = {
            'fixture_id': fixture['fixture_id'], 'source_fixture_id': fixture.get('source_fixture_id'),
            'kind': fixture['kind'], 'source_sha256': manifest['source_sha256'],
            'source_hashes_verified': hashes_verified,
            'world_file': fixture['world_file'], 'world_sha256_manifest': fixture['world_sha256'],
            'world_sha256_actual': world_actual,
            'topic_discovery': result.get('contact_topic_discovery'),
            'clock_advanced': result.get('clock_advanced'),
            'clock_advance_sufficient': result.get('clock_advance_sufficient'),
            'clock_samples_ns': result.get('clock_samples_ns'),
            'contact_message_counts': result.get('contact_message_counts'),
            'runner_state': result.get('state'), 'worker_lifecycle': result.get('lifecycle'),
            'worker_returncode': result.get('worker_returncode'),
            'timed_out': result.get('timed_out'),
            'worker_log_bytes': result.get('worker_log_bytes'),
            'worker_log_truncated': result.get('worker_log_truncated'),
            'worker_log_drain_complete': result.get('worker_log_drain_complete'),
            'gz_log_bytes': result.get('gz_log_bytes'), 'gz_log_truncated': result.get('gz_log_truncated'),
            'gz_log_drain_complete': result.get('gz_log_drain_complete'),
            'score': score,
        }
        case_record['passed'] = _case_passed(
            fixture, result, score, hashes_verified and world_actual == fixture['world_sha256'])
        (case_dir/'result.json').write_text(json.dumps(case_record, indent=2)+'\n')
        run_rows.append(case_record)
    summary = {'schema_version': 1, 'fixture_count': len(run_rows),
               'passed_positive': sum(row['passed'] and row['kind'] == 'positive' for row in run_rows),
               'passed_negative': sum(row['passed'] and row['kind'] != 'positive' for row in run_rows),
               'cases': run_rows, 'runtime_started_by_this_command': True}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return _summary_exit_code(run_rows)


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _cleanup_process_group(pgid, grace_seconds=.25):
    """Escalate only within one case-owned POSIX process group."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if not _group_exists(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic()+grace_seconds
        while time.monotonic() < deadline and _group_exists(pgid):
            time.sleep(.01)


if __name__ == '__main__':
    raise SystemExit(main())
