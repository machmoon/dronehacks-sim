#!/usr/bin/env python3
"""Audit explicitly supplied marker-free course runs for sprint repeatability.

This report is an offline artifact audit. It does not run simulations and it
does not infer missing evidence. Every supplied directory gets an attempt row,
including malformed and failed runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REQUIRED_SCORE_CHECKS = (
    'controller_reports_all_waypoints',
    'truth_complete_during_flight',
    'truth_reaches_all_waypoints_in_order',
    'truth_crosses_window_with_full_envelope',
    'truth_multipart_path_certified_clear',
    'truth_transit_floor_checked',
    'px4_ground_and_disarm_confirmed',
    'truth_touchdown_inside_pad',
)
REQUIRED_FILES = ('telemetry.summary.json', 'telemetry.jsonl', 'score.json', 'recording.json')
REPEAT_COUNT = 5
TIME_LIMIT_S = 60.0


def _read_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get('type') == 'metadata':
                return row
    raise ValueError('telemetry.jsonl has no metadata row')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _controller_profile(metadata: dict[str, Any]) -> dict[str, Any] | None:
    waypoints = metadata.get('waypoints')
    limits = metadata.get('course_limits')
    if not isinstance(waypoints, list) or not waypoints or not isinstance(limits, dict):
        return None
    px4_runtime = metadata.get('px4_runtime')
    if px4_runtime is not None and not isinstance(px4_runtime, dict):
        return None
    # speed_scale changes each effective waypoint speed and is part of the
    # controller profile even though the raw logged waypoints remain unchanged.
    profile = {
        'waypoints': waypoints,
        'course_limits': limits,
        'speed_scale': metadata.get('speed_scale', 1.0),
        'task': metadata.get('task'),
        'policy_checkpoint_sha256': metadata.get('policy_checkpoint_sha256'),
        'heading_lookahead_m': metadata.get('heading_lookahead_m'),
        'policy_boundary_brake_enabled': metadata.get('policy_boundary_brake_enabled'),
        'localizer': metadata.get('localizer', 'survey'),
        'inertial_tilt_prior_enabled': metadata.get('inertial_tilt_prior_enabled'),
        'survey_normals': metadata.get('survey_normals'),
        'vision_max_capture_gap_s': metadata.get('vision_max_capture_gap_s'),
        'vision_max_capture_age_s': metadata.get('vision_max_capture_age_s'),
        'camera_resolution_px': metadata.get('camera_resolution_px'),
        'camera_translation_flu_m': metadata.get('camera_translation_flu_m'),
        'px4_binary_sha256': (px4_runtime or {}).get('px4_binary_sha256'),
    }
    try:
        _canonical_sha256(profile)
    except (TypeError, ValueError):
        return None
    return profile


def _clean_recorder_exit(recording: dict[str, Any]) -> bool:
    process_codes = recording.get('process_exit_codes')
    return (recording.get('mission_exit_code') == 0
            and recording.get('runner_error') is None
            and isinstance(process_codes, list) and bool(process_codes)
            and all(code == 0 for code in process_codes))


def audit_attempt(directory: str | Path) -> dict[str, Any]:
    """Audit one explicitly named run directory, preserving failures."""
    run_dir = Path(directory).expanduser()
    row: dict[str, Any] = {
        'run_dir': str(run_dir),
        'run_id': None,
        'passed': False,
        'failures': [],
        'elapsed_sim_s': None,
        'profile_sha256': None,
        'profile': None,
        'input_sha256': {},
        'runtime_source_sha256': None,
        'score_checks': {},
    }
    errors: list[str] = []
    paths = {name: run_dir / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        errors.extend('missing_' + name for name in missing)
    for name, path in paths.items():
        if path.is_file():
            try:
                row['input_sha256'][name] = _sha256(path)
            except OSError:
                errors.append('unreadable_' + name)

    summary = score = recording = metadata = None
    if paths['telemetry.summary.json'].is_file():
        try:
            summary = _read_json(paths['telemetry.summary.json'])
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            errors.append('invalid_telemetry_summary')
        else:
            if not isinstance(summary, dict):
                summary = None
                errors.append('invalid_telemetry_summary')
    if paths['score.json'].is_file():
        try:
            score = _read_json(paths['score.json'])
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            errors.append('invalid_score')
        else:
            if not isinstance(score, dict):
                score = None
                errors.append('invalid_score')
    if paths['recording.json'].is_file():
        try:
            recording = _read_json(paths['recording.json'])
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            errors.append('invalid_recording')
        else:
            if not isinstance(recording, dict):
                recording = None
                errors.append('invalid_recording')
    if paths['telemetry.jsonl'].is_file():
        try:
            metadata = _read_metadata(paths['telemetry.jsonl'])
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            errors.append('invalid_telemetry_metadata')

    if isinstance(summary, dict):
        row['run_id'] = summary.get('run_id')
        if not isinstance(summary.get('run_id'), str) or not summary['run_id'].strip():
            errors.append('summary_run_id_missing_or_invalid')
        elapsed = summary.get('elapsed_sim_s')
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed):
            row['elapsed_sim_s'] = float(elapsed)
        else:
            errors.append('elapsed_time_not_finite_number')
        if not isinstance(summary.get('run_id'), str) or not summary.get('run_id', '').strip() or not isinstance(score, dict) or score.get('run_id') != summary.get('run_id'):
            errors.append('score_summary_run_id_mismatch')
        if summary.get('result') != 'completed' or summary.get('route_completed') is not True:
            errors.append('route_not_completed')
        if summary.get('landed_and_disarmed') is not True:
            errors.append('not_landed_and_disarmed')
    if isinstance(score, dict):
        interval = score.get('flight_interval_sim_s')
        interval_span = None
        if (isinstance(interval, list) and len(interval) == 2
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in interval)):
            interval_span = float(interval[1]) - float(interval[0])
        if interval_span is None:
            errors.append('score_flight_interval_missing_or_invalid')
        elif row['elapsed_sim_s'] is not None and abs(interval_span-row['elapsed_sim_s']) > 1e-6:
            errors.append('score_summary_elapsed_mismatch')
        if score.get('full_route_verified_by_geometry') is not True:
            errors.append('full_route_geometry_not_verified')
        checks = score.get('checks')
        checks = checks if isinstance(checks, dict) else {}
        row['score_checks'] = {name: checks.get(name) is True for name in REQUIRED_SCORE_CHECKS}
        if not all(row['score_checks'].values()):
            errors.append('required_score_checks_failed_or_missing')
    if isinstance(recording, dict):
        row['recorder_exited_cleanly'] = _clean_recorder_exit(recording)
        if not row['recorder_exited_cleanly']:
            errors.append('recorder_did_not_exit_cleanly')
        row['truth_used_for_navigation'] = recording.get('truth_used_for_navigation')
        if recording.get('truth_used_for_navigation') is not False:
            errors.append('truth_navigation_not_explicitly_false')
    if isinstance(metadata, dict):
        row['metadata_run_id'] = metadata.get('run_id')
        row['sensor_diagnostic'] = metadata.get('sensor_diagnostic')
        row['launch'] = metadata.get('launch')
        if row['run_id'] is None or metadata.get('run_id') != row['run_id']:
            errors.append('metadata_summary_run_id_mismatch')
        profile = _controller_profile(metadata)
        if profile is None:
            errors.append('controller_profile_missing_or_invalid')
        else:
            row['profile'] = profile
            row['profile_sha256'] = _canonical_sha256(profile)
        source_hashes = metadata.get('runtime_source_sha256')
        if (isinstance(source_hashes, dict) and source_hashes
                and all(isinstance(key, str) and key.strip()
                        and isinstance(value, str) and value.strip()
                        for key, value in source_hashes.items())):
            row['runtime_source_sha256'] = source_hashes
        else:
            errors.append('runtime_source_hashes_missing_or_invalid')

    if row['elapsed_sim_s'] is not None:
        if row['elapsed_sim_s'] <= 0:
            errors.append('elapsed_time_not_positive')
        elif row['elapsed_sim_s'] >= TIME_LIMIT_S:
            errors.append('elapsed_time_not_under_60_seconds')
    row['failures'] = sorted(set(errors))
    row['passed'] = not row['failures']
    return row


def build_report(run_directories: list[str | Path]) -> dict[str, Any]:
    """Return a complete report for every supplied directory, including repeats."""
    attempts = [audit_attempt(directory) for directory in run_directories]
    seen_directories: set[str] = set()
    seen_run_ids: set[str] = set()
    for attempt in attempts:
        canonical_directory = str(Path(attempt['run_dir']).expanduser().resolve())
        if canonical_directory in seen_directories:
            attempt['failures'].append('duplicate_run_directory')
        seen_directories.add(canonical_directory)
        run_id = attempt.get('run_id')
        if isinstance(run_id, str) and run_id.strip():
            run_id = run_id.strip()
            if run_id in seen_run_ids:
                attempt['failures'].append('duplicate_run_id')
            seen_run_ids.add(run_id)
        attempt['failures'] = sorted(set(attempt['failures']))
        attempt['passed'] = not attempt['failures']

    # Passing attempts must use the same source files as well as the same
    # controller configuration. Preserve every attempt and flag later changes.
    source_reference = None
    for attempt in attempts:
        source_hashes = attempt.get('runtime_source_sha256')
        if attempt['passed'] and source_hashes is not None:
            if source_reference is None:
                source_reference = source_hashes
            elif source_hashes != source_reference:
                attempt['failures'].append('runtime_source_hashes_mismatch')
                attempt['failures'].sort()
                attempt['passed'] = False
    profiles = [attempt['profile_sha256'] for attempt in attempts]
    same_profile = bool(attempts) and all(profiles) and len(set(profiles)) == 1
    source_sets = [attempt['runtime_source_sha256'] for attempt in attempts]
    same_runtime_source = (bool(attempts) and all(source_sets)
                           and all(source == source_sets[0] for source in source_sets[1:]))
    passed = sum(attempt['passed'] for attempt in attempts)
    times = [attempt['elapsed_sim_s'] for attempt in attempts if attempt['elapsed_sim_s'] is not None]
    all_pass = bool(attempts) and passed == len(attempts)
    achieved = len(attempts) >= REPEAT_COUNT and all_pass and same_profile and same_runtime_source
    return {
        'schema_version': 1,
        'attempts': attempts,
        'total': len(attempts),
        'passed_runs': passed,
        'times_sim_s': times,
        'worst_time_sim_s': max(times) if times else None,
        'same_controller_profile': bool(same_profile),
        'same_runtime_source': bool(same_runtime_source),
        'achieved': bool(achieved),
        'achievement_rule': (
            f'at least {REPEAT_COUNT} explicitly supplied runs, every attempt passes, '
            'all runs have the same logged waypoints, course limits, speed scale, '
            'and runtime source hashes, with unique directories and run IDs'
        ),
        'interpretation': (
            'Finite repeated simulator runs are evidence for these recorded attempts; '
            'they are not a physical-flight or reliability guarantee.'
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directories', nargs='+',
                        help='explicit run artifact directories; each supplied entry is retained')
    parser.add_argument('--output', type=Path,
                        help='write JSON to this path instead of stdout')
    args = parser.parse_args(argv)
    report = build_report(args.run_directories)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    else:
        sys.stdout.write(rendered)
    return 0 if report['achieved'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
