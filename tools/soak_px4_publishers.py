"""Passively check periodic SITL publishers against the advancing IMU clock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from mission.runtime_provenance import px4_runtime_provenance
from mission.sitl_health import (_fields, _timestamp, parse_battery_status,
                                 parse_system_power, MAX_AGE_SIM_S)


def stamp(text, topic):
    return _timestamp(_fields(text, topic, ('timestamp',)), topic)


def assess(samples, duration_s):
    errors = []
    for index, row in enumerate(samples):
        if row.get('error'):
            errors.append(f'sample_{index}:{row["error"]}')
            continue
        before, after = row['imu_before_usec'], row['imu_after_usec']
        if after < before or (index and before < samples[index-1].get('imu_after_usec', 0)):
            errors.append(f'sample_{index}:imu_clock_regressed')
        for topic in ('battery_status', 'system_power', 'cpuload'):
            value = row[f'{topic}_timestamp_usec']
            if not before-800000 <= value <= after+200000:
                errors.append(f'sample_{index}:{topic}_stale_or_future')
            if index and value < samples[index-1].get(f'{topic}_timestamp_usec', 0):
                errors.append(f'sample_{index}:{topic}_clock_regressed')
    good = [row for row in samples if not row.get('error')]
    span = (good[-1]['imu_after_usec']-good[0]['imu_before_usec'])/1e6 if good else 0.
    advanced = {topic: bool(len(good) >= 3 and good[-1][f'{topic}_timestamp_usec'] >
                            good[0][f'{topic}_timestamp_usec'])
                for topic in ('battery_status', 'system_power', 'cpuload')}
    return dict(passed=not errors and span >= duration_s and all(advanced.values()),
                sample_count=len(samples), observed_sim_seconds=span,
                requested_sim_seconds=duration_s, publishers_advanced=advanced, errors=errors)


def read_topic(topic):
    return subprocess.run(['px4-listener', topic, '-n', '1'], capture_output=True,
                          text=True, check=True, timeout=1.5).stdout


def snapshot():
    start = time.monotonic()
    before = stamp(read_topic('sensor_combined'), 'sensor_combined')
    battery, _ = parse_battery_status(read_topic('battery_status'))
    power, _ = parse_system_power(read_topic('system_power'))
    load = stamp(read_topic('cpuload'), 'cpuload')
    after = stamp(read_topic('sensor_combined'), 'sensor_combined')
    return dict(imu_before_usec=before, imu_after_usec=after,
                battery_status_timestamp_usec=battery, system_power_timestamp_usec=power,
                cpuload_timestamp_usec=load, read_wall_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sim-seconds', type=float, default=60.)
    parser.add_argument('--wall-seconds', type=float, default=400.)
    args = parser.parse_args()
    if (os.environ.get('DRONEHACKS_SITL') != '1'
            or os.environ.get('GZ_PARTITION') != 'dronehacks_markerfree'):
        parser.error('requires the isolated marker-free SITL container')
    if not 1 <= args.sim_seconds <= 600 or not 5 <= args.wall_seconds <= 2400:
        parser.error('use 1–600 simulation seconds and 5–2400 wall seconds')
    if args.output.exists() or args.output.with_suffix('.summary.json').exists():
        parser.error('use new output paths')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(type='metadata', px4_runtime=px4_runtime_provenance(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source='read-only px4-listener; no arm, setpoint, parameter or actuator commands',
                    max_publication_age_sim_s=MAX_AGE_SIM_S,
                    timing='Publication stamps bracketed by IMU reads; ±200 ms future tolerance.')
    samples = []
    started = time.monotonic()
    with args.output.open('x') as output:
        output.write(json.dumps(metadata)+'\n')
        while time.monotonic()-started < args.wall_seconds:
            try:
                row = snapshot()
            except (ValueError, subprocess.SubprocessError, OSError) as error:
                row = {'error': f'{type(error).__name__}:{error}'}
            row['wall_seconds'] = time.monotonic()-started
            samples.append(row)
            output.write(json.dumps(dict(type='sample', **row))+'\n')
            output.flush()
            report = assess(samples, args.sim_seconds)
            if report['errors'] or report['passed']:
                break
            time.sleep(1.)
    report = dict(**assess(samples, args.sim_seconds), wall_seconds=time.monotonic()-started,
                  log_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  scope='Finite passive SITL publisher soak. No arming-transition test or flight proof.')
    args.output.with_suffix('.summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
