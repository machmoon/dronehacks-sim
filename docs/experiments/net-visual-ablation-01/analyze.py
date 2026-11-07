#!/usr/bin/env python3
"""Offline analysis for the preregistered passive net-visual ablation captures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path(__file__).with_name('protocol.json')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _audit_source_path(path: str) -> str:
    normalized = str(path).replace('\\', '/')
    marker = '/models/'
    if marker in normalized:
        return normalized[normalized.index(marker) + 1:]
    if normalized.startswith('/tmp/'):
        return 'container:' + normalized
    return Path(normalized).name


def _world_summary(world: dict | None) -> dict | None:
    if not isinstance(world, dict):
        return None
    return {key: world.get(key) for key in (
        'mode', 'seed', 'configured_renderer', 'depth_topic', 'noise_mean_m',
        'noise_stddev_m', 'configuration_tool_sha256', 'source_world_sha256', 'world_sha256')}


def _validate_generation_manifest(artifacts: Path, protocol: dict, repo_root: Path) -> dict:
    generation = protocol['generation']
    manifest_path = artifacts / 'manifest.json'
    output_paths = {name: artifacts / name for name in ('control.sdf', 'treatment.sdf')}
    checks = {}
    details = {'manifest_path': _display_path(manifest_path, repo_root), 'output_sha256': {}}
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError) as error:
        return {'passed': False, 'checks': {'manifest_readable': False},
                'details': {'error': str(error), **details}, 'input_sha256': {}}

    expected_inputs = generation['inputs']
    actual_inputs = manifest.get('inputs') or {}
    checks['manifest_course_model_sha256'] = (
        (actual_inputs.get('course_model') or {}).get('sha256')
        == expected_inputs['course_model']['sha256'])
    checks['manifest_course_model_path_matches_protocol'] = (
        (actual_inputs.get('course_model') or {}).get('path')
        == expected_inputs['course_model']['path'])
    checks['manifest_source_world_sha256'] = (
        (actual_inputs.get('source_world') or {}).get('sha256')
        == expected_inputs['source_world']['sha256'])
    checks['manifest_source_world_path_matches_protocol'] = (
        (actual_inputs.get('source_world') or {}).get('path')
        == expected_inputs['source_world']['path'])
    checks['manifest_generator_sha256'] = (
        (manifest.get('tool') or {}).get('sha256') == generation['tool']['sha256'])
    checks['manifest_generator_path_matches_protocol'] = (
        (manifest.get('tool') or {}).get('path') == generation['tool']['path'])
    generator_path = repo_root / generation['tool']['path']
    checks['current_generator_source_matches_protocol'] = (
        generator_path.is_file() and _sha256(generator_path) == generation['tool']['sha256'])
    current_source_checks = {}
    for key, entry in expected_inputs.items():
        path = repo_root / entry['path']
        current_source_checks[entry['path']] = {
            'expected_sha256': entry['sha256'],
            'actual_sha256': _sha256(path) if path.is_file() else None,
        }
    checks['current_source_inputs_match_protocol'] = all(
        row['actual_sha256'] == row['expected_sha256'] for row in current_source_checks.values())

    expected_ablation = generation['ablation']
    actual_ablation = manifest.get('ablation') or {}
    ablation_checks = {
        'all_course_collision_semantics_equal':
            actual_ablation.get('all_course_collision_semantics_equal') is True,
        'net_collision_semantics_equal': actual_ablation.get('net_collision_semantics_equal') is True,
        'all_other_world_semantics_equal_after_flattening':
            actual_ablation.get('all_other_world_semantics_equal_after_flattening') is True,
        'removed_visual_names_match': actual_ablation.get('removed_visual_names')
            == expected_ablation['removed_visual_names'],
        'retained_net_collision_names_match': actual_ablation.get('net_collision_names_retained')
            == expected_ablation['net_collision_names_retained'],
        'worlds_are_registered_scene_base_inputs':
            actual_ablation.get('worlds_are_base_inputs_for_existing_vision_rgbd_scene')
            == expected_ablation['worlds_are_base_inputs_for_existing_vision_rgbd_scene'],
        'canonical_collision_hashes_match_protocol': all(
            actual_ablation.get(key) == expected_ablation.get(key)
            for key in ('canonical_all_course_collisions_sha256', 'canonical_net_collision_sha256')),
    }
    checks['collision_and_ablation_semantics_match'] = all(ablation_checks.values())

    output_checks = {}
    for name, path in output_paths.items():
        expected = generation['outputs'][name]
        record = (manifest.get('outputs') or {}).get(name) or {}
        actual_sha = _sha256(path) if path.is_file() else None
        output_checks[name] = {
            'manifest_path_matches_protocol': record.get('path') == name,
            'manifest_sha256_matches_protocol': record.get('sha256') == expected['sha256'],
            'file_sha256_matches_protocol': actual_sha == expected['sha256'],
            'byte_count_matches_protocol': (path.stat().st_size == expected['bytes']) if path.is_file() else False,
        }
        details['output_sha256'][name] = actual_sha
    checks['generated_worlds_match_protocol_hashes'] = all(
        all(status.values()) for status in output_checks.values())
    details['ablation_checks'] = ablation_checks
    details['output_checks'] = output_checks
    details['current_source_inputs'] = current_source_checks
    details['manifest_sha256'] = _sha256(manifest_path)
    input_hashes = {
        _display_path(manifest_path, repo_root): details['manifest_sha256'],
        **{_display_path(path, repo_root): details['output_sha256'][name]
           for name, path in output_paths.items()},
        **{path: row['actual_sha256'] for path, row in current_source_checks.items()},
        _display_path(generator_path, repo_root):
            _sha256(generator_path) if generator_path.is_file() else None,
    }
    return {'passed': all(checks.values()), 'checks': checks, 'details': details,
            'input_sha256': input_hashes}


def _check(checks: dict, name: str, passed: bool, details=None) -> bool:
    checks[name] = {'passed': bool(passed), 'details': details}
    return bool(passed)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _depth_distribution(depth: np.ndarray) -> dict:
    finite = np.isfinite(depth)
    pos_inf = np.isposinf(depth)
    neg_inf = np.isneginf(depth)
    nan = np.isnan(depth)
    values = depth[finite]
    total = int(depth.size)
    return {
        'sample_values': total,
        'finite_count': int(finite.sum()),
        'finite_fraction': float(finite.mean()),
        'positive_infinity_count': int(pos_inf.sum()),
        'positive_infinity_fraction': float(pos_inf.mean()),
        'negative_infinity_count': int(neg_inf.sum()),
        'negative_infinity_fraction': float(neg_inf.mean()),
        'nan_count': int(nan.sum()),
        'nan_fraction': float(nan.mean()),
        'finite_outside_0_1_to_8m_count': int((finite & ((depth < .1) | (depth > 8.))).sum()),
        'finite_min_m': _finite_or_none(values.min()) if values.size else None,
        'finite_median_m': _finite_or_none(np.median(values)) if values.size else None,
        'finite_max_m': _finite_or_none(values.max()) if values.size else None,
    }


def depth_change_summary(control: np.ndarray, treatment: np.ndarray,
                         minimum_fraction: float) -> dict:
    """Apply the protocol pixel-median rule using the full image denominator."""
    control_median = np.median(control, axis=0)
    treatment_median = np.median(treatment, axis=0)
    control_valid = np.isfinite(control_median) & (control_median >= .1) & (control_median <= 8.)
    increased_finite = (np.isfinite(treatment_median)
                        & (treatment_median > control_median + .5))
    changed = control_valid & (np.isposinf(treatment_median) | increased_finite)
    total_pixels = int(control_median.size)
    changed_count = int(changed.sum())
    control_median_finite = np.isfinite(control_median)
    treatment_median_finite = np.isfinite(treatment_median)
    return {
        'passed': bool(total_pixels and changed_count / total_pixels >= minimum_fraction),
        'changed_pixels': changed_count,
        'total_pixels_denominator': total_pixels,
        'changed_pixel_fraction_of_full_image': changed_count / total_pixels if total_pixels else 0.,
        'minimum_fraction': minimum_fraction,
        'control_pixel_median_classes': {
            'finite': int(control_median_finite.sum()),
            'nan_invalid': int(np.isnan(control_median).sum()),
            'positive_infinity': int(np.isposinf(control_median).sum()),
            'negative_infinity': int(np.isneginf(control_median).sum()),
            'finite_in_valid_depth_range': int(control_valid.sum()),
        },
        'treatment_pixel_median_classes': {
            'finite': int(treatment_median_finite.sum()),
            'nan_invalid': int(np.isnan(treatment_median).sum()),
            'positive_infinity': int(np.isposinf(treatment_median).sum()),
            'negative_infinity': int(np.isneginf(treatment_median).sum()),
        },
        'control_raw_depth': _depth_distribution(control),
        'treatment_raw_depth': _depth_distribution(treatment),
        'definition': ('For each pixel take the median of the 32 depth samples. A pixel changes only when '
                       'the control median is finite in [0.1, 8] m and the treatment median is +Inf or '
                       'more than 0.5 m farther. Divide by every image pixel, including invalid control pixels.'),
    }


def _depth_pair_is_well_formed(control, treatment, expected_shape) -> bool:
    return (
        isinstance(control, np.ndarray)
        and isinstance(treatment, np.ndarray)
        and control.ndim == 3
        and treatment.ndim == 3
        and control.shape == tuple(expected_shape)
        and treatment.shape == tuple(expected_shape)
        and control.shape == treatment.shape
        and np.issubdtype(control.dtype, np.number)
        and np.issubdtype(treatment.dtype, np.number)
    )


def _capture_native_dtypes(stamps_ns, rgb, depth) -> dict:
    checks = {
        'stamp_ns_int64': isinstance(stamps_ns, np.ndarray) and stamps_ns.dtype == np.dtype(np.int64),
        'rgb_uint8': isinstance(rgb, np.ndarray) and rgb.dtype == np.dtype(np.uint8),
        'depth_z_m_float32': isinstance(depth, np.ndarray) and depth.dtype == np.dtype(np.float32),
    }
    return {'passed': all(checks.values()), 'checks': checks}


def quaternion_angle_deg(first, second) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = min(1., abs(float(np.dot(a, b))))
    return math.degrees(2 * math.acos(dot))


def pose_summary(positions, orientations) -> dict:
    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError('pose positions must be a nonempty N by 3 array')
    if orientations.shape != (len(positions), 4):
        raise ValueError('pose orientations must be an N by 4 array')
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise ValueError('pose samples must be finite')
    norms = np.linalg.norm(orientations, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError('pose quaternion has zero norm')
    orientations = orientations / norms[:, None]
    median_position = np.median(positions, axis=0)
    position_deviations = np.linalg.norm(positions - median_position, axis=1)
    reference = orientations[0]
    aligned = orientations.copy()
    aligned[np.einsum('ij,j->i', aligned, reference) < 0] *= -1
    median_orientation = np.median(aligned, axis=0)
    median_orientation /= np.linalg.norm(median_orientation)
    orientation_deviations = [quaternion_angle_deg(q, median_orientation) for q in aligned]
    return {
        'sample_count': len(positions),
        'median_position_enu_m': median_position.tolist(),
        'max_position_deviation_from_median_m': float(position_deviations.max()),
        'median_orientation_xyzw_sign_aligned_component_median': median_orientation.tolist(),
        'max_orientation_deviation_from_median_deg': float(max(orientation_deviations)),
    }


def compare_median_poses(control: dict, treatment: dict,
                         maximum_position_difference_m: float,
                         maximum_orientation_difference_deg: float) -> dict:
    position_delta = float(np.linalg.norm(np.asarray(control['median_position_enu_m'])
                                          - np.asarray(treatment['median_position_enu_m'])))
    orientation_delta = quaternion_angle_deg(
        control['median_orientation_xyzw_sign_aligned_component_median'],
        treatment['median_orientation_xyzw_sign_aligned_component_median'])
    passed = (position_delta <= maximum_position_difference_m
              and orientation_delta <= maximum_orientation_difference_deg)
    return {
        'median_position_difference_m': position_delta,
        'maximum_allowed_position_difference_m': maximum_position_difference_m,
        'median_orientation_difference_deg': orientation_delta,
        'maximum_allowed_orientation_difference_deg': maximum_orientation_difference_deg,
        'passed': passed,
    }


def align_capture_truth(truth_rows, stamps_ns, max_gap_s: float) -> dict:
    """Align every capture using the same truth interpolation contract as offline scoring."""
    from tools.score_markerfree import Truth
    stamps_s = np.asarray(stamps_ns, dtype=np.float64) * 1e-9
    truth = Truth(truth_rows, max_gap_s)
    aligned = [truth.at(float(stamp)) for stamp in stamps_s]
    bracketed = all(sample is not None for sample in aligned)
    parts, continuous = truth.interval(float(stamps_s[0]), float(stamps_s[-1]))
    if bracketed:
        summary = pose_summary([sample[0] for sample in aligned], [sample[1] for sample in aligned])
    else:
        summary = None
    lo = max(0, int(np.searchsorted(truth.stamps, stamps_s[0])) - 1)
    hi = min(len(truth.stamps), int(np.searchsorted(truth.stamps, stamps_s[-1], side='right')) + 1)
    covered_stamps = truth.stamps[lo:hi]
    max_span_gap = float(np.diff(covered_stamps).max()) if len(covered_stamps) > 1 else 0.
    return {
        'passed': bracketed and continuous,
        'all_capture_stamps_bracketed': bracketed,
        'bracketed_capture_count': sum(sample is not None for sample in aligned),
        'capture_count': len(stamps_s),
        'full_capture_span_continuous': continuous,
        'truth_contiguous_parts_in_capture_span': len(parts),
        'max_truth_gap_in_capture_span_s': max_span_gap,
        'max_truth_gap_allowed_s': max_gap_s,
        'pose': summary,
    }


def observer_status_checks(status: dict) -> dict:
    # A user stop can close queues cleanly before the preregistered 30 s window.
    # The collection protocol requires the duration-complete terminal state.
    terminal = status.get('state') == 'duration_complete'
    zero_overflow = (status.get('truth_queue_overflow') == 0
                     and status.get('contact_queue_overflow') == 0)
    no_pending = (status.get('truth_queue_pending') == 0
                  and status.get('contact_queue_pending') == 0)
    def count_matches(total, *parts):
        if not isinstance(total, int) or isinstance(total, bool):
            return False
        if any(not isinstance(part, int) or isinstance(part, bool) for part in parts):
            return False
        return total == sum(parts)

    truth_accounting = count_matches(
        status.get('truth_messages_received'), status.get('truth_queue_overflow'),
        status.get('truth_messages_dequeued'), status.get('truth_queue_pending'))
    contact_accounting = count_matches(
        status.get('contact_messages_queued'), status.get('contact_queue_overflow'),
        status.get('contact_messages_dequeued'), status.get('contact_queue_pending'))
    topics = status.get('contacts_expected_topics')
    topic_observation = status.get('contact_topic_observation')
    topics_valid = isinstance(topics, list) and isinstance(topic_observation, dict)
    topic_rows = [topic_observation.get(topic) for topic in topics] if topics_valid else []
    topics_valid = topics_valid and all(isinstance(row, dict) for row in topic_rows)
    contact_receive_accounting = (topics_valid and count_matches(
        status.get('contact_messages_received'),
        *(row.get('transport_messages_received') for row in topic_rows)))
    contact_pair_accounting = (topics_valid and count_matches(
        status.get('contact_messages_queued'),
        *(row.get('messages_with_contact_pairs') for row in topic_rows)))
    truth_dequeue_accounting = count_matches(
        status.get('truth_messages_dequeued'), status.get('truth_samples'),
        status.get('truth_messages_ignored'))
    contact_write_accounting = (status.get('contact_messages_dequeued') is not None
                                and status.get('contact_messages_with_pairs') is not None
                                and status.get('contact_messages_dequeued')
                                == status.get('contact_messages_with_pairs'))
    checks = {
        'closed_terminal_state': terminal,
        'callback_acceptance_closed': status.get('callback_acceptance_closed') is True,
        'recording_complete': status.get('recording_complete') is True,
        'evidence_loss_false': status.get('evidence_loss') is False,
        'no_pending_queue_entries': no_pending,
        'zero_truth_and_contact_queue_overflow': zero_overflow,
        'truth_received_queue_accounting': truth_accounting,
        'truth_dequeue_accounting': truth_dequeue_accounting,
        'contact_queue_accounting': contact_accounting,
        'contact_receive_per_topic_accounting': contact_receive_accounting,
        'contact_pairs_per_topic_accounting': contact_pair_accounting,
        'contact_write_accounting': contact_write_accounting,
        'truth_subscribed_and_samples_present': (status.get('truth_subscribed') is True
                                                  and status.get('truth_samples', 0) > 0),
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'truth_messages_received': status.get('truth_messages_received'),
        'truth_samples_written': status.get('truth_samples'),
        'truth_queue_overflow': status.get('truth_queue_overflow'),
        'truth_queue_overflow_timestamp_range_ns': status.get('truth_queue_overflow_timestamp_range_ns'),
        'contact_messages_received': status.get('contact_messages_received'),
        'contact_messages_queued': status.get('contact_messages_queued'),
        'contact_messages_written': status.get('contact_messages_with_pairs'),
        'contact_queue_overflow': status.get('contact_queue_overflow'),
        'contact_queue_overflow_timestamp_range_ns': status.get('contact_queue_overflow_timestamp_range_ns'),
        'contact_topic_observation': status.get('contact_topic_observation'),
    }


def _load_json(path: Path):
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value


def _load_capture(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        needed = {'stamp_ns', 'rgb', 'depth_z_m'}
        if not needed.issubset(archive.files):
            raise ValueError(f'capture archive missing arrays: {sorted(needed-set(archive.files))}')
        return {key: archive[key] for key in needed}


def _valid_pose_quaternion(q):
    if len(q) != 4 or not np.isfinite(q).all():
        return False
    return abs(float(np.linalg.norm(q)) - 1.) <= .001


def _condition_analysis(condition: str, directory: Path, protocol: dict,
                        expected_world: dict, max_capture_gap_s: float,
                        max_truth_gap_s: float, minimum_clock_s: float,
                        repo_root: Path) -> dict:
    result = {'condition': condition, 'checks': {}, 'paths': {}}
    checks = result['checks']
    runtime_path = directory / 'runtime.json'
    capture_path = directory / 'capture.npz'
    observer_path = directory / 'observer'
    truth_path = observer_path / 'truth.jsonl'
    observer_status_path = observer_path / 'status.json'
    result['paths'] = {
        'runtime': _display_path(runtime_path, repo_root),
        'capture': _display_path(capture_path, repo_root),
        'truth': _display_path(truth_path, repo_root),
        'observer_status': _display_path(observer_status_path, repo_root),
    }
    result['input_sha256'] = {
        result['paths']['runtime']: _sha256(runtime_path) if runtime_path.is_file() else None,
        result['paths']['capture']: _sha256(capture_path) if capture_path.is_file() else None,
        result['paths']['truth']: _sha256(truth_path) if truth_path.is_file() else None,
        result['paths']['observer_status']:
            _sha256(observer_status_path) if observer_status_path.is_file() else None,
    }

    runtime = None
    try:
        runtime = _load_json(runtime_path)
        actual_world = runtime.get('sensor_diagnostic')
        _check(checks, 'runtime_world_configuration_exact', actual_world == expected_world,
               {'expected': _world_summary(expected_world),
                'actual': _world_summary(actual_world),
                'world_path_expected': 'container:/tmp/dronehacks-px4.sdf'})
    except (OSError, ValueError) as error:
        _check(checks, 'runtime_world_configuration_exact', False, str(error))

    if runtime is not None:
        px4 = runtime.get('px4_runtime') or {}
        expected_binary = protocol['fixed_configuration']['px4_binary_sha256']
        build = px4.get('build_manifest') or {}
        binary_pass = (px4.get('px4_binary_sha256') == expected_binary
                       and px4.get('manifest_verified') is True
                       and build.get('px4_binary_sha256') == expected_binary)
        _check(checks, 'px4_binary_and_manifest_match', binary_pass,
               {'expected_sha256': expected_binary,
                'actual_sha256': px4.get('px4_binary_sha256'),
                'manifest_verified': px4.get('manifest_verified'),
                'manifest_binary_sha256': build.get('px4_binary_sha256')})
        source_actual = runtime.get('source_sha256') or {}
        expected_sources = {
            'vision/capture_rgbd.py': protocol['passive_capture']['capture_source_sha256'],
            'vision/flight_observe.py': protocol['fixed_configuration']['flight_source_sha256'][
                'vision/flight_observe.py'],
        }
        source_pass = all(source_actual.get(path) == digest
                          for path, digest in expected_sources.items())
        _check(checks, 'capture_and_observer_sources_match', source_pass,
               {'expected': expected_sources,
                'actual': {path: source_actual.get(path) for path in expected_sources}})

        audit = runtime.get('navigation_audit') or {}
        expected_world_hash = expected_world.get('world_sha256')
        world_sources = audit.get('world_and_model_sources') or []
        recorded_world = next((row for row in world_sources
                               if row.get('path') == expected_world.get('world_path')), None)
        audit_checks = {
            'truth_odometry_publisher_zero': audit.get('truth_topic_publishers') == 0,
            'artificial_landmark_models_zero': audit.get('artificial_landmark_models') == 0,
            'live_simulator_position_not_used': audit.get('live_simulator_position_used') is False,
            'audited_runtime_world_hash_matches': (recorded_world is not None
                                                    and recorded_world.get('sha256') == expected_world_hash),
        }
        _check(checks, 'navigation_audit_passes', all(audit_checks.values()),
               {'checks': audit_checks, 'truth_topic': audit.get('truth_topic'),
                'audited_world': ({'path': _audit_source_path(recorded_world['path']),
                                   'sha256': recorded_world.get('sha256')}
                                  if recorded_world else None)})
        result['navigation_audit'] = {
            'truth_topic': audit.get('truth_topic'),
            'truth_topic_publishers': audit.get('truth_topic_publishers'),
            'artificial_landmark_models': audit.get('artificial_landmark_models'),
            'live_simulator_position_used': audit.get('live_simulator_position_used'),
            'audited_sources': [{'path': _audit_source_path(row.get('path', '')),
                                 'sha256': row.get('sha256')}
                                for row in world_sources],
        }
        result['px4_runtime'] = {
            'px4_binary_sha256': px4.get('px4_binary_sha256'),
            'manifest_verified': px4.get('manifest_verified'),
            'build_manifest_sha256': px4.get('build_manifest_sha256'),
        }
        result['source_sha256'] = {path: source_actual.get(path) for path in expected_sources}
    else:
        _check(checks, 'px4_binary_and_manifest_match', False, 'runtime.json unavailable')
        _check(checks, 'capture_and_observer_sources_match', False, 'runtime.json unavailable')
        _check(checks, 'navigation_audit_passes', False, 'runtime.json unavailable')

    try:
        status = _load_json(observer_status_path)
        observer_result = observer_status_checks(status)
        _check(checks, 'observer_closed_complete_no_known_queue_loss', observer_result['passed'],
               observer_result)
        result['observer'] = observer_result
        truth_rows = []
        with truth_path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f'truth line {line_number} is not an object')
                    truth_rows.append(row)
        _check(checks, 'truth_samples_match_observer_status',
               len(truth_rows) == status.get('truth_samples'),
               {'file_rows': len(truth_rows), 'status_samples': status.get('truth_samples')})
    except (OSError, ValueError, json.JSONDecodeError) as error:
        truth_rows = None
        _check(checks, 'observer_closed_complete_no_known_queue_loss', False, str(error))
        _check(checks, 'truth_samples_match_observer_status', False, str(error))

    try:
        archive = _load_capture(capture_path)
        stamps_ns = np.asarray(archive['stamp_ns'])
        rgb = np.asarray(archive['rgb'])
        depth = np.asarray(archive['depth_z_m'])
        expected_frames = protocol['passive_capture']['per_condition_frames']
        expected_height, expected_width = protocol['passive_capture']['rgbd_shape']
        count_pass = stamps_ns.ndim == 1 and len(stamps_ns) == expected_frames
        native_dtypes = _capture_native_dtypes(stamps_ns, rgb, depth)
        stamp_type_pass = native_dtypes['checks']['stamp_ns_int64']
        rgb_dtype_pass = native_dtypes['checks']['rgb_uint8']
        depth_dtype_pass = native_dtypes['checks']['depth_z_m_float32']
        shape_pass = (rgb.shape == (expected_frames, expected_height, expected_width, 3)
                      and depth.shape == (expected_frames, expected_height, expected_width))
        dtype_pass = native_dtypes['passed']
        _check(checks, 'capture_count_shape_and_pairing', count_pass and dtype_pass and shape_pass,
               {'frames': int(len(stamps_ns)), 'expected_frames': expected_frames,
                'stamp_dtype': str(stamps_ns.dtype), 'rgb_shape': list(rgb.shape),
                'depth_shape': list(depth.shape),
                'rgb_dtype': str(rgb.dtype), 'depth_dtype': str(depth.dtype),
                'expected_dtypes': {'stamp_ns': 'int64', 'rgb': 'uint8', 'depth_z_m': 'float32'},
                'dtype_checks': {'stamp_ns_int64': stamp_type_pass,
                                 'rgb_uint8': rgb_dtype_pass,
                                 'depth_z_m_float32': depth_dtype_pass},
                'expected_rgb_shape': [expected_frames, expected_height, expected_width, 3],
                'expected_depth_shape': [expected_frames, expected_height, expected_width]})
        increasing = bool(stamp_type_pass and stamps_ns.ndim == 1 and len(stamps_ns) > 1
                          and np.all(np.diff(stamps_ns.astype(np.int64)) > 0))
        gaps_s = np.diff(stamps_ns.astype(np.float64)) * 1e-9 if increasing else np.array([])
        max_gap = float(gaps_s.max()) if gaps_s.size else None
        gap_pass = increasing and max_gap <= max_capture_gap_s + 1e-9
        _check(checks, 'capture_timestamps_increasing_and_gap_bounded', gap_pass,
               {'strictly_increasing': increasing, 'count': int(len(stamps_ns)),
                'max_gap_s': max_gap, 'maximum_allowed_s': max_capture_gap_s,
                'first_stamp_ns': int(stamps_ns[0]) if len(stamps_ns) else None,
                'last_stamp_ns': int(stamps_ns[-1]) if len(stamps_ns) else None})

        pre_capture = runtime.get('pre_capture_imu_sim_s') if runtime else None
        try:
            pre_capture = float(pre_capture)
            pre_capture_pass = (math.isfinite(pre_capture) and pre_capture >= minimum_clock_s
                                and len(stamps_ns) > 0
                                and pre_capture <= float(stamps_ns[0]) * 1e-9)
        except (TypeError, ValueError):
            pre_capture_pass = False
        _check(checks, 'clock_warmup_before_capture', pre_capture_pass,
               {'pre_capture_imu_sim_s': pre_capture,
                'minimum_sim_clock_s': minimum_clock_s,
                'first_capture_sim_s': float(stamps_ns[0]) * 1e-9 if len(stamps_ns) else None})
        result['capture'] = {
            'stamp_ns': stamps_ns.astype(np.int64).tolist(),
            'max_capture_gap_s': max_gap,
            'rgb_shape': list(rgb.shape), 'depth_shape': list(depth.shape),
        }
    except (OSError, ValueError, KeyError) as error:
        archive = None
        _check(checks, 'capture_count_shape_and_pairing', False, str(error))
        _check(checks, 'capture_timestamps_increasing_and_gap_bounded', False, str(error))
        _check(checks, 'clock_warmup_before_capture', False, str(error))

    if (archive is not None and truth_rows is not None
            and np.asarray(archive['stamp_ns']).ndim == 1
            and len(archive['stamp_ns']) > 0):
        try:
            alignment = align_capture_truth(truth_rows, archive['stamp_ns'], max_truth_gap_s)
            result['truth_alignment'] = alignment
            _check(checks, 'truth_brackets_every_capture', alignment['all_capture_stamps_bracketed'],
                   {'capture_count': alignment['capture_count'],
                    'truth_bracketed_count': alignment['bracketed_capture_count'],
                    'maximum_allowed_bracket_s': max_truth_gap_s})
            _check(checks, 'truth_full_capture_span_continuous', alignment['full_capture_span_continuous'],
                   {'truth_contiguous_parts_in_capture_span': alignment[
                       'truth_contiguous_parts_in_capture_span'],
                    'max_truth_gap_in_capture_span_s': alignment['max_truth_gap_in_capture_span_s'],
                    'maximum_allowed_gap_s': max_truth_gap_s,
                    'full_span_covered': alignment['full_capture_span_continuous'],
                    'truth_rows': len(truth_rows)})
            if alignment['pose'] is not None:
                pose = alignment['pose']
                result['pose'] = pose
                max_deviation = protocol['passive_capture']['requirements'][
                    'within_each_capture_max_position_deviation_from_median_m']
                deviation_pass = pose['max_position_deviation_from_median_m'] <= max_deviation
                _check(checks, 'pose_stability_within_condition', deviation_pass,
                       {'max_position_deviation_m': pose['max_position_deviation_from_median_m'],
                        'maximum_allowed_m': max_deviation})
        except (ImportError, ValueError, KeyError, TypeError, IndexError) as error:
            _check(checks, 'truth_brackets_every_capture', False, str(error))
            _check(checks, 'truth_full_capture_span_continuous', False, str(error))
    else:
        _check(checks, 'truth_brackets_every_capture', False, 'capture or truth input unavailable')
        _check(checks, 'truth_full_capture_span_continuous', False, 'capture or truth input unavailable')
        _check(checks, 'pose_stability_within_condition', False, 'capture or aligned truth unavailable')
    result['passed'] = bool(checks) and all(item['passed'] for item in checks.values())
    return result


def _camera_intrinsics_check(protocol: dict, conditions: dict, repo_root: Path) -> dict:
    fixed = protocol['fixed_configuration']['flight_source_sha256']
    source_paths = ('vision/rgbd_scene.py', 'vision/rgbd_odometry.py')
    source_checks = {}
    for relative in source_paths:
        path = repo_root / relative
        expected = fixed.get(relative)
        actual = _sha256(path) if path.is_file() else None
        source_checks[relative] = {'expected_sha256': expected, 'actual_sha256': actual,
                                   'matches': expected is not None and actual == expected}
    try:
        sys.path.insert(0, str(repo_root))
        from vision.rgbd_scene import RGBD_HEIGHT, RGBD_HFOV, RGBD_WIDTH
        shape = protocol['passive_capture']['rgbd_shape']
        shape_match = [RGBD_HEIGHT, RGBD_WIDTH] == shape
        focal = RGBD_WIDTH / (2 * math.tan(RGBD_HFOV / 2))
        intrinsics = [[focal, 0., RGBD_WIDTH/2], [0., focal, RGBD_HEIGHT/2], [0., 0., 1.]]
    except (ImportError, AttributeError, ValueError, TypeError) as error:
        shape_match, intrinsics, error_text = False, None, str(error)
    else:
        error_text = None
    worlds_match = all((conditions.get(name) or {}).get('checks', {}).get(
        'runtime_world_configuration_exact', {}).get('passed') is True for name in ('control', 'treatment'))
    passed = (all(x['matches'] for x in source_checks.values()) and shape_match and worlds_match)
    return {
        'passed': passed,
        'same_pair_intrinsics_and_shape': passed,
        'derived_camera_intrinsics_px': intrinsics,
        'rgbd_shape_height_width': protocol['passive_capture']['rgbd_shape'],
        'source_hash_checks': source_checks,
        'both_runtime_worlds_match_registered_hashes': worlds_match,
        'error': error_text,
        'basis': ('The captures do not embed separate K matrices. Pair equality is established from equal image shapes, '
                  'the registered RGBD scene/odometry source hashes, and exact registered runtime-world hashes '
                  'whose camera settings are part of those generated worlds.'),
    }


def analyze_pair(artifacts: Path, protocol: dict, repo_root: Path = ROOT) -> dict:
    repo_root_text = str(repo_root.resolve())
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    requirements = protocol['passive_capture']['requirements']
    worlds = protocol['expected_runtime_worlds']
    max_capture_gap = requirements['strictly_increasing_capture_stamps_max_gap_s']
    max_truth_gap = requirements['truth_brackets_all_capture_stamps_max_gap_s']
    conditions = {}
    for condition in ('control', 'treatment'):
        conditions[condition] = _condition_analysis(
            condition, artifacts / condition, protocol, worlds[condition],
            max_capture_gap, max_truth_gap,
            protocol['passive_capture']['minimum_sim_clock_before_capture_s'], repo_root)
    checks = {}
    registered_capture_paths = protocol['passive_capture']['capture_paths']
    roots = {str(Path(path).parent.parent) for path in registered_capture_paths.values()}
    expected_artifact_root = next(iter(roots)) if len(roots) == 1 else None
    try:
        artifact_root_matches = (expected_artifact_root is not None
                                 and artifacts.resolve() == (repo_root / expected_artifact_root).resolve())
    except OSError:
        artifact_root_matches = False
    _check(checks, 'capture_artifact_root_matches_protocol', artifact_root_matches,
           {'expected': expected_artifact_root,
            'actual': _display_path(artifacts, repo_root),
            'protocol_capture_paths': registered_capture_paths})
    generation = _validate_generation_manifest(artifacts, protocol, repo_root)
    _check(checks, 'generated_world_manifest_and_collision_semantics', generation['passed'],
           {'checks': generation['checks'], 'details': generation['details']})
    _check(checks, 'control_condition_passes_all_gates', conditions['control']['passed'])
    _check(checks, 'treatment_condition_passes_all_gates', conditions['treatment']['passed'])
    intrinsics = _camera_intrinsics_check(protocol, conditions, repo_root)
    _check(checks, 'same_intrinsics_and_capture_shape', intrinsics['passed'], intrinsics)

    def non_runtime_audit_sources(condition):
        audit = condition.get('navigation_audit') or {}
        rows = audit.get('audited_sources') or []
        return sorted((row.get('path'), row.get('sha256')) for row in rows
                      if row.get('path') != 'container:/tmp/dronehacks-px4.sdf')

    control_sources = non_runtime_audit_sources(conditions['control'])
    treatment_sources = non_runtime_audit_sources(conditions['treatment'])
    audit_sources_match = bool(control_sources and control_sources == treatment_sources)
    _check(checks, 'navigation_model_sources_match_between_conditions', audit_sources_match,
           {'control': control_sources, 'treatment': treatment_sources})

    depth = None
    capture_control = conditions['control'].get('_depth')
    capture_treatment = conditions['treatment'].get('_depth')
    if capture_control is None or capture_treatment is None:
        for condition in ('control', 'treatment'):
            try:
                archive = _load_capture(artifacts / condition / 'capture.npz')
                conditions[condition]['_depth'] = archive['depth_z_m']
            except (OSError, ValueError, KeyError):
                conditions[condition]['_depth'] = None
        capture_control = conditions['control']['_depth']
        capture_treatment = conditions['treatment']['_depth']
    expected_frames = protocol['passive_capture']['per_condition_frames']
    height, width = protocol['passive_capture']['rgbd_shape']
    expected_shape = (expected_frames, height, width)
    arrays_well_formed = _depth_pair_is_well_formed(
        capture_control, capture_treatment, expected_shape)
    if arrays_well_formed:
        threshold = requirements['changed_depth_pixel_fraction_of_full_image_min']
        depth = depth_change_summary(capture_control, capture_treatment, threshold)
        _check(checks, 'material_depth_change_gate', depth['passed'], depth)
    else:
        details = {
            'passed_shape': list(capture_control.shape) if isinstance(capture_control, np.ndarray) else None,
            'treatment_shape': list(capture_treatment.shape) if isinstance(capture_treatment, np.ndarray) else None,
            'expected_shape': list(expected_shape),
            'passed_dtype': str(capture_control.dtype) if isinstance(capture_control, np.ndarray) else None,
            'treatment_dtype': str(capture_treatment.dtype) if isinstance(capture_treatment, np.ndarray) else None,
        }
        _check(checks, 'material_depth_change_gate', False,
               {'error': 'Depth arrays unavailable or malformed; pair comparison skipped.', **details})

    pose_control = conditions['control'].get('pose')
    pose_treatment = conditions['treatment'].get('pose')
    if pose_control is not None and pose_treatment is not None:
        position_limit = requirements['between_condition_median_position_difference_m']
        orientation_limit = requirements['between_condition_median_orientation_difference_degrees']
        pose_pair = compare_median_poses(pose_control, pose_treatment,
                                         position_limit, orientation_limit)
        _check(checks, 'between_condition_median_pose_matches', pose_pair['passed'], pose_pair)
    else:
        pose_pair = None
        _check(checks, 'between_condition_median_pose_matches', False,
               'One or both conditions lack complete capture-aligned truth pose.')

    # Drop in-memory arrays from the public result; retain only distributions/statistics.
    for condition in conditions.values():
        condition.pop('_depth', None)
    passed = bool(checks) and all(item['passed'] for item in checks.values())
    return {
        'schema_version': 1,
        'status': 'pass' if passed else 'fail',
        'overall_pass': passed,
        'artifacts_root': _display_path(artifacts, repo_root),
        'protocol_path': _display_path(PROTOCOL_PATH, repo_root),
        'protocol_sha256': _sha256(PROTOCOL_PATH),
        'analysis_source_sha256': _sha256(Path(__file__)),
        'input_sha256': {
            **generation.get('input_sha256', {}),
            _display_path(PROTOCOL_PATH, repo_root): _sha256(PROTOCOL_PATH),
        },
        'checks': checks,
        'conditions': conditions,
        'pair': {
            'intrinsics_and_shape': intrinsics,
            'median_pose_match': pose_pair,
            'depth_change': depth,
        },
        'limits': {
            'capture_gap_max_s': max_capture_gap,
            'truth_bracket_gap_max_s': max_truth_gap,
            'position_deviation_from_median_max_m': requirements[
                'within_each_capture_max_position_deviation_from_median_m'],
            'between_condition_median_position_max_m': requirements[
                'between_condition_median_position_difference_m'],
            'between_condition_median_orientation_max_deg': requirements[
                'between_condition_median_orientation_difference_degrees'],
            'minimum_changed_depth_fraction': requirements[
                'changed_depth_pixel_fraction_of_full_image_min'],
        },
        'claim_limit': ('This reports passive rendered RGBD and pose agreement only. It is not flight evidence, '
                        'a realistic wire-net optical model, sensor reliability, or real-camera transfer.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True,
                        help='new paired artifact folder containing control/ and treatment/')
    parser.add_argument('--output', type=Path, required=True,
                        help='new JSON result path; existing files are never overwritten')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error(f'output already exists: {output}')
    if not args.artifacts.is_dir():
        parser.error(f'artifacts folder does not exist: {args.artifacts}')
    protocol = _load_json(PROTOCOL_PATH)
    result = analyze_pair(args.artifacts.resolve(), protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'status': result['status'],
                      'overall_pass': result['overall_pass']}))
    return 0 if result['overall_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
