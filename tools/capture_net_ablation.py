#!/usr/bin/env python3
"""Collect one preregistered passive net-ablation RGBD condition in marker-free SITL."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'docs/experiments/net-visual-ablation-01/protocol.json'
TRUTH_WAIT_WALL_S = 15.0
READINESS_TIMEOUT_S = 60.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path=PROTOCOL_PATH):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def validate_environment(env):
    if (env.get('DRONEHACKS_SITL') != '1'
            or env.get('GZ_PARTITION') != 'dronehacks_markerfree'):
        raise RuntimeError('requires the isolated marker-free SITL container and partition')


def registered_root(protocol, root=ROOT):
    paths = protocol['passive_capture']['capture_paths']
    roots = {str(Path(path).parent.parent) for path in paths.values()}
    if len(roots) != 1:
        raise ValueError('protocol capture paths do not share one output root')
    return (Path(root) / next(iter(roots))).resolve()


def validate_condition_output(condition, output_root):
    if condition not in ('control', 'treatment'):
        raise ValueError('condition must be control or treatment')
    if (output_root / condition).exists():
        raise FileExistsError(f'refusing to reuse existing condition output: {output_root / condition}')


def preflight(expected_world, protocol, env, *, deadline=None,
              monotonic=time.monotonic, sleep=time.sleep):
    """Read and verify current world configuration, PX4 provenance, and navigation audit."""
    validate_environment(env)
    timeout = READINESS_TIMEOUT_S if deadline is None else deadline - monotonic()
    if timeout <= 0:
        raise TimeoutError('total startup readiness exceeded 60 wall seconds')
    sensor_diagnostic = wait_for_sensor_diagnostic(expected_world, timeout,
                                                    monotonic=monotonic, sleep=sleep)

    from mission.navigation_audit import audit_navigation_sources
    from mission.runtime_provenance import px4_runtime_provenance

    px4_runtime = px4_runtime_provenance()
    fixed = protocol['fixed_configuration']
    manifest = px4_runtime.get('build_manifest') or {}
    expected_binary = fixed['px4_binary_sha256']
    if (px4_runtime.get('px4_binary_sha256') != expected_binary
            or px4_runtime.get('manifest_verified') is not True
            or manifest.get('px4_binary_sha256') != expected_binary):
        raise RuntimeError('PX4 executable/build manifest does not match registered provenance')

    navigation_audit = audit_navigation_sources(expected_world['world_path'])
    sources = navigation_audit.get('world_and_model_sources') or []
    world_source = next((row for row in sources
                         if row.get('path') == expected_world['world_path']), None)
    if (navigation_audit.get('truth_topic_publishers') != 0
            or navigation_audit.get('artificial_landmark_models') != 0
            or navigation_audit.get('live_simulator_position_used') is not False
            or world_source is None
            or world_source.get('sha256') != expected_world['world_sha256']):
        raise RuntimeError('navigation source audit does not match protocol or has truth aiding')
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError('total startup readiness exceeded 60 wall seconds')

    source_hashes = {
        'vision/capture_rgbd.py': sha256(ROOT / 'vision/capture_rgbd.py'),
        'vision/flight_observe.py': sha256(ROOT / 'vision/flight_observe.py'),
    }
    expected_sources = {
        'vision/capture_rgbd.py': protocol['passive_capture']['capture_source_sha256'],
        'vision/flight_observe.py': fixed['flight_source_sha256']['vision/flight_observe.py'],
    }
    if source_hashes != expected_sources:
        raise RuntimeError('capture or independent observer source differs from protocol')
    return {'sensor_diagnostic': sensor_diagnostic, 'px4_runtime': px4_runtime,
            'navigation_audit': navigation_audit, 'source_sha256': source_hashes}


def wait_for_sensor_diagnostic(expected_world, wall_timeout_s=READINESS_TIMEOUT_S, *,
                               path=Path('/tmp/dronehacks-rgbd-depth-noise.json'),
                               monotonic=time.monotonic, sleep=time.sleep):
    """Wait briefly for startup's diagnostic file; reject any completed mismatch."""
    deadline = monotonic() + wall_timeout_s
    last_problem = 'not written yet'
    while monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            last_problem = 'not written yet'
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            last_problem = f'incomplete/unreadable: {type(error).__name__}'
        else:
            if value != expected_world:
                raise RuntimeError('completed RGBD depth-noise diagnostic does not exactly match protocol')
            return value
        sleep(min(.25, max(0., deadline - monotonic())))
    raise TimeoutError(f'RGBD depth-noise diagnostic not ready within {wall_timeout_s:.1f}s ({last_problem})')


def imu_timestamp_usec(text):
    from mission.sitl_health import _fields, _timestamp
    return _timestamp(_fields(text, 'sensor_combined', ('timestamp',)), 'sensor_combined')


def wait_for_sim_clock(minimum_s, wall_timeout_s=READINESS_TIMEOUT_S, *, run=subprocess.run,
                       monotonic=time.monotonic, sleep=time.sleep):
    """Wait read-only for the PX4 IMU simulation timestamp to reach the requested time."""
    deadline = monotonic() + wall_timeout_s
    last = None
    while monotonic() < deadline:
        remaining = deadline - monotonic()
        try:
            result = run(['px4-listener', 'sensor_combined', '-n', '1'],
                         capture_output=True, text=True, check=True,
                         timeout=max(.1, min(2.0, remaining)))
            last = imu_timestamp_usec(result.stdout) / 1e6
            if last >= minimum_s:
                return last
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            pass
        sleep(min(1.0, max(0.0, deadline - monotonic())))
    raise TimeoutError(f'sensor_combined did not reach {minimum_s:.3f} sim s; last={last}')


def write_new_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _status(path):
    try:
        with (path / 'status.json').open(encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _stop_own_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def collect_condition(condition, protocol, *, env=None, output_root=None,
                      preflight_fn=preflight, popen=subprocess.Popen, run=subprocess.run,
                      monotonic=time.monotonic, sleep=time.sleep):
    env = dict(os.environ if env is None else env)
    validate_environment(env)
    output_root = Path(output_root or registered_root(protocol)).resolve()
    validate_condition_output(condition, output_root)
    directory = output_root / condition
    directory.mkdir(parents=True, exist_ok=False)
    expected_world = protocol['expected_runtime_worlds'][condition]
    readiness_deadline = monotonic() + READINESS_TIMEOUT_S
    context = preflight_fn(expected_world, protocol, env, deadline=readiness_deadline,
                           monotonic=monotonic, sleep=sleep)
    minimum_clock = protocol['passive_capture']['minimum_sim_clock_before_capture_s']
    readiness_remaining = readiness_deadline - monotonic()
    if readiness_remaining <= 0:
        raise TimeoutError('total startup readiness exceeded 60 wall seconds')
    readiness_imu_sim_s = wait_for_sim_clock(minimum_clock,
                                             wall_timeout_s=readiness_remaining,
                                             run=run, monotonic=monotonic, sleep=sleep)

    observer_dir = directory / 'observer'
    observer_stdout = (directory / 'observer.stdout.log').open('x')
    observer_stderr = (directory / 'observer.stderr.log').open('x')
    observer = None
    observer_reaped_successfully = False
    try:
        observer_command = [sys.executable, '-m', 'vision.flight_observe',
                            '--output', str(observer_dir), '--duration',
                            str(protocol['passive_capture']['observer_seconds_wall'])]
        capture_command = [sys.executable, '-m', 'vision.capture_rgbd',
                           '--output', str(directory / 'capture.npz'),
                           '--frames', str(protocol['passive_capture']['per_condition_frames']),
                           '--timeout', '45']
        invocation = {
            'schema_version': 1,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'condition': condition,
            'collector_sha256': sha256(Path(__file__)),
            'observer_command': observer_command,
            'observer_duration_wall_s': protocol['passive_capture']['observer_seconds_wall'],
            'capture_command': capture_command,
            'required_pre_capture_sim_clock_s': minimum_clock,
            'readiness_wall_deadline_s': READINESS_TIMEOUT_S,
            'readiness_imu_sim_s': readiness_imu_sim_s,
        }
        write_new_json(directory / 'invocation.json', invocation)
        observer = popen(observer_command,
                         cwd=ROOT, env=env, stdout=observer_stdout, stderr=observer_stderr)
        truth_deadline = monotonic() + TRUTH_WAIT_WALL_S
        observed_status = None
        while monotonic() < truth_deadline:
            observed_status = _status(observer_dir)
            if observed_status and observed_status.get('truth_samples', 0) > 0:
                break
            if observer.poll() is not None:
                raise RuntimeError(f'observer exited before truth arrived: {observed_status}')
            sleep(min(.25, max(0., truth_deadline - monotonic())))
        else:
            raise TimeoutError('observer did not record truth within bounded startup deadline')

        readiness_remaining = readiness_deadline - monotonic()
        if readiness_remaining <= 0:
            raise TimeoutError('total startup readiness exceeded 60 wall seconds')
        pre_capture_imu_sim_s = wait_for_sim_clock(minimum_clock,
                                                   wall_timeout_s=readiness_remaining,
                                                   run=run, monotonic=monotonic, sleep=sleep)
        observed_status = _status(observer_dir)
        if (observer.poll() is not None or not observed_status
                or observed_status.get('state') != 'observing'):
            raise RuntimeError(f'observer is not active at capture start: {observed_status}')
        runtime = dict(context, readiness_imu_sim_s=readiness_imu_sim_s,
                       pre_capture_imu_sim_s=pre_capture_imu_sim_s)
        write_new_json(directory / 'runtime.json', runtime)
        capture_stdout = (directory / 'capture.stdout.log').open('x')
        capture_stderr = (directory / 'capture.stderr.log').open('x')
        try:
            result = run(capture_command, cwd=ROOT, env=env,
                         stdout=capture_stdout, stderr=capture_stderr,
                         timeout=60, check=False)
        finally:
            capture_stdout.close()
            capture_stderr.close()
        if result.returncode != 0:
            raise RuntimeError(f'RGBD capture failed with exit status {result.returncode}')
        observer_deadline = monotonic() + protocol['passive_capture']['observer_seconds_wall'] + 15
        while monotonic() < observer_deadline:
            final_status = _status(observer_dir)
            if final_status and final_status.get('state') == 'duration_complete':
                if final_status.get('recording_complete') is not True:
                    raise RuntimeError('observer duration completed but evidence is incomplete')
                break
            if observer.poll() is not None:
                raise RuntimeError(f'observer exited without duration_complete status: {final_status}')
            sleep(min(.25, max(0., observer_deadline - monotonic())))
        else:
            raise TimeoutError('observer did not complete its registered duration')
        observer_return_code = observer.wait(timeout=10)
        if observer_return_code != 0:
            raise RuntimeError(f'observer exited with status {observer_return_code}')
        observer_reaped_successfully = True
        capture_path = directory / 'capture.npz'
        if not capture_path.is_file() or capture_path.stat().st_size == 0:
            raise RuntimeError('capture command succeeded but did not create capture.npz')
        return {'condition': condition, 'directory': str(directory),
                'pre_capture_imu_sim_s': pre_capture_imu_sim_s,
                'observer_status': final_status}
    finally:
        if observer is not None and not observer_reaped_successfully:
            _stop_own_process(observer)
        observer_stdout.close()
        observer_stderr.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('control', 'treatment'), required=True)
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol()
        result = collect_condition(args.condition, protocol)
    except Exception as error:
        parser.exit(1, f'capture failed; outputs preserved where created: {type(error).__name__}: {error}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
