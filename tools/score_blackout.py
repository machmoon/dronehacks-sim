"""Score an injected RGBD blackout offline from telemetry, observer data, and PX4 ULog."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from learning import sim
from mission import airframe
from tools import score_markerfree as route_score


ROOT = Path(__file__).resolve().parents[1]
INJECTION = 'Injected RGBD capture loss; no further measurements sent'
FAULT = 'rgbd_pose_unavailable:stale_capture'
POLICY_FAULT = 'stale_policy_action'
EXPECTED_FAULTS = (FAULT, POLICY_FAULT)
LAND_REQUEST = 'Requested PX4 Land after mission stop'
LAND_ACK = 'PX4 command 21: result 0'
CONFIRMED = 'PX4 confirmed touchdown and disarm after abort'
AUTO_LAND = 18  # PX4 v1.16 vehicle_status.nav_state
ARMED, DISARMED = 2, 1
MAX_PRE_INJECTION_AID_AGE_S = .6
MAX_LAND_STATE_AGE_S = 1.
ULOG_TOPICS = ('vehicle_visual_odometry', 'vehicle_status', 'vehicle_land_detected',
               'vehicle_local_position', 'estimator_status_flags')


def _load_ulog(data):
    from pyulog.core import ULog
    return ULog(io.BytesIO(data), list(ULOG_TOPICS))


def _ordered_events(events, expected_fault=FAULT):
    required = [route_score.ARMED_EVENT, INJECTION, f'Mission stopped: {expected_fault}',
                LAND_REQUEST, LAND_ACK, CONFIRMED]
    positions = []
    for message in required:
        matches = [index for index, event in enumerate(events) if event.get('message') == message]
        if len(matches) != 1:
            return False
        positions.append(matches[0])
    ack = events[positions[-2]]
    return positions == sorted(positions) and ack.get('command') == 21 and ack.get('result') == 0


def _transitions(data, key, start, end):
    times = data['timestamp'] / 1e6
    values = data[key]
    if not len(times):
        return []
    indices = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    start_index = np.searchsorted(times[indices], start, side='left')
    return [{'sim_time_s': float(times[i]), 'value': int(values[i])}
            for i in indices[max(0, start_index-1):] if times[i] <= end]


def _entered(transitions, target, start, end, previous=None):
    for before, after in zip(transitions, transitions[1:]):
        if (after['value'] == target and start <= after['sim_time_s'] <= end
                and (previous is None or before['value'] == previous)):
            return after['sim_time_s']
    return None


def _recent_state(data, key, stamp, value):
    if key not in data:
        return False
    times = data['timestamp']/1e6
    index = np.searchsorted(times, stamp, side='right') - 1
    return bool(index >= 0 and 0 <= stamp-times[index] <= MAX_LAND_STATE_AGE_S
                and data[key][index] == value)


def _controller_vision(samples, summary, injection, fault):
    sent = summary.get('vision_measurements_sent')
    prior = [row for row in samples if row['sim_time'] < injection]
    after = [row for row in samples if row['sim_time'] >= injection]
    if type(sent) is not int or sent <= 0 or not prior or not after:
        return sent, False
    latest = max(prior, key=lambda row: row['sim_time'])
    first = min(after, key=lambda row: row['sim_time'])
    pre_count = latest.get('vision_measurements_sent')
    if type(pre_count) is not int or first.get('vision_stamp_sim_s') is None:
        return sent, False
    capture = route_score._time(first['vision_stamp_sim_s'])
    # A final pre-injection send can fall between controller telemetry samples.
    stable = (injection-MAX_PRE_INJECTION_AID_AGE_S <= capture < injection
              and latest.get('vision_valid') is True and 0 < pre_count <= sent
              and all(row['sim_time'] <= fault and row.get('vision_measurements_sent') == sent
                      and row.get('vision_stamp_sim_s') == capture for row in after))
    return sent, stable


def _px4_evidence(ulog, injection, fault, confirmed):
    topics = {item.name: item.data for item in ulog.data_list if item.multi_id == 0}
    visual = [(int(timestamp), int(capture)) for item in ulog.data_list
              if item.name == 'vehicle_visual_odometry'
              for timestamp, capture in zip(item.data['timestamp'], item.data['timestamp_sample'])]
    last_receipt, last_capture = (tuple(value/1e6 for value in max(visual)) if visual else (None, None))
    start, end = ulog.start_timestamp/1e6, ulog.last_timestamp/1e6
    after_injection = (sum(injection <= timestamp/1e6 for timestamp, _ in visual)
                       if injection is not None else None)
    # Dropout marks carry duration only. Pyulog attaches the last parsed timestamp.
    dropouts = [{'preceding_message_sim_s': row.timestamp/1e6, 'duration_ms': row.duration}
                for row in ulog.dropouts]
    overlaps = [row for row in dropouts if injection is not None and confirmed is not None
                and row['preceding_message_sim_s'] <= confirmed
                and row['preceding_message_sim_s'] + row['duration_ms']/1000 >= injection]
    coverage = (injection is not None and confirmed is not None and start <= injection < confirmed <= end
                and not overlaps and not ulog.file_corruption)
    credible_pre_aid = (injection is not None and last_receipt is not None
                        and injection-MAX_PRE_INJECTION_AID_AGE_S <= last_receipt < injection)
    log_evidence = {
        'coverage_sim_s': [start, end],
        'file_corruption_detected': ulog.file_corruption,
        'dropout_records': dropouts,
        'dropout_records_overlapping_blackout': overlaps,
        'spans_blackout_without_reported_dropouts_or_corruption': coverage,
        'vehicle_visual_odometry_record_count': len(visual),
        'last_visual_odometry_receipt_sim_s': last_receipt,
        'last_visual_odometry_capture_sim_s': last_capture,
        'logged_visual_odometry_records_after_injection': after_injection,
        'px4_git_revision': ulog.msg_info_dict.get('ver_sw'),
    }
    transition_fields = {
        'navigation_state': ('vehicle_status', 'nav_state'),
        'arming_state': ('vehicle_status', 'arming_state'),
        'failsafe': ('vehicle_status', 'failsafe'),
        'land_detected': ('vehicle_land_detected', 'landed'),
        'vertical_position_valid': ('vehicle_local_position', 'z_valid'),
        'external_vision_height_aid': ('estimator_status_flags', 'cs_ev_hgt'),
        'barometer_height_aid': ('estimator_status_flags', 'cs_baro_hgt'),
    }
    transitions = {name: _transitions(topics[topic], field, injection, confirmed)
                   for name, (topic, field) in transition_fields.items()
                   if injection is not None and confirmed is not None and field in topics.get(topic, {})}
    auto_land = disarmed = None
    landed = False
    if fault is not None and confirmed is not None:
        auto_land = _entered(transitions.get('navigation_state', []), AUTO_LAND, fault, confirmed)
        disarmed = _entered(transitions.get('arming_state', []), DISARMED, fault, confirmed, previous=ARMED)
        landed = (_recent_state(topics.get('vehicle_land_detected', {}), 'landed', confirmed, True)
                  and _recent_state(topics.get('vehicle_status', {}), 'arming_state', confirmed, DISARMED))
    return log_evidence, transitions, auto_land, disarmed, landed, coverage and credible_pre_aid


def _truth_pose(truth, stamp):
    sample = truth.at(stamp) if stamp is not None else None
    return sample[0] if sample is not None else None


def _displacement(start, end):
    return float(np.linalg.norm(end[:2]-start[:2])) if start is not None and end is not None else None


def score_blackout(run_directory, ulog_path, expected_fault=FAULT):
    if expected_fault not in EXPECTED_FAULTS:
        raise ValueError(f'expected_fault must be one of {EXPECTED_FAULTS!r}')
    directory, ulog_path = Path(run_directory).resolve(), Path(ulog_path).resolve()
    hashes = {}
    telemetry = route_score._load(directory/'telemetry.jsonl', hashes, lines=True)
    metadata = next(row for row in telemetry if row.get('type') == 'metadata')
    summary = next((row for row in telemetry if row.get('type') == 'summary'), {})
    if summary and metadata['run_id'] != summary['run_id']:
        raise ValueError('telemetry metadata and summary run IDs disagree')
    events = [row for row in telemetry if row.get('type') == 'event']
    samples = [row for row in telemetry if row.get('type') == 'sample']
    policy_actions = [row for row in telemetry if row.get('type') == 'policy_action']

    def stamp(message):
        value = route_score._event(events, message).get('sim_time')
        return route_score._time(value) if value is not None else None

    armed, injection, confirmed = map(stamp, (route_score.ARMED_EVENT, INJECTION, CONFIRMED))
    fault = armed + route_score._time(summary['elapsed_sim_s']) if armed is not None and summary else None
    timed = all(value is not None for value in (armed, injection, fault, confirmed))
    ordered_times = timed and armed < injection < fault < confirmed
    spans = []
    transit = None
    if ordered_times:
        hover_stamps = [row['sim_time'] for row in samples if row.get('phase') == 'HOVER_CHECK']
        hover_verified = stamp(route_score.HOVER_EVENT)
        if hover_verified is not None:
            transit = min(hover_stamps + [hover_verified])
            if armed <= transit < fault:
                spans = [('takeoff', armed, transit, False), ('transit', transit, fault, True),
                         ('abort_landing', fault, confirmed, False)]

    observer = directory/'observer'
    truth = route_score.Truth(route_score._load(observer/'truth.jsonl', hashes, lines=True),
                              route_score.DEFAULT_MAX_GAP_S)
    contacts_path, status_path = observer/'contacts.jsonl', observer/'status.json'
    contacts = route_score._load(contacts_path, hashes, lines=True) if contacts_path.is_file() else []
    status = route_score._load(status_path, hashes) if status_path.is_file() else {}
    course = sim.Course()
    envelope = airframe.load_x500_envelope(margin_m=route_score.X500_ENVELOPE_MARGIN_M)
    geometry = route_score._multipart_scene(truth, spans, course.scene(np.random.default_rng(0), 0.), envelope)
    contact_score = route_score._contacts(contacts, status, armed, confirmed, transit, fault) if spans else None
    counts = contact_score['raw_pair_observations'] if contact_score else None
    floor_contacts = [row for row in contacts if ordered_times and fault <= row['sim_time_s'] <= confirmed
                      and any('ground_plane' in name or 'floor_20_x_20_ft' in name
                              for pair in row['pairs'] for name in pair)]
    first_floor = min(floor_contacts, key=lambda row: row['sim_time_s'], default=None)
    first_floor_stamp = first_floor['sim_time_s'] if first_floor is not None else None
    launch, inject_pose, fault_pose, floor_pose, final_pose = (
        _truth_pose(truth, time) for time in (armed, injection, fault, first_floor_stamp, confirmed))

    ulog_data = ulog_path.read_bytes()
    hashes[str(ulog_path)] = hashlib.sha256(ulog_data).hexdigest()
    log, transitions, auto_land, disarmed, landed, valid_aid_log = _px4_evidence(
        _load_ulog(ulog_data), injection, fault, confirmed)
    count, unchanged = (_controller_vision(samples, summary, injection, fault)
                        if injection is not None and fault is not None else (None, False))
    touchdown = (auto_land is not None and first_floor_stamp is not None and auto_land <= first_floor_stamp
                 and all(pose is not None for pose in (launch, fault_pose, floor_pose, final_pose))
                 and fault_pose[2] - launch[2] > route_score.TOUCHDOWN_HEIGHT_TOLERANCE_M
                 and abs(final_pose[2]-launch[2]) <= route_score.TOUCHDOWN_HEIGHT_TOLERANCE_M)
    checks = {
        ('intentional_stale_capture_fault' if expected_fault == FAULT else 'expected_fault_observed'):
            summary.get('result') == 'failed' and summary.get('fault') == expected_fault,
        'injection_fault_land_ack_disarm_events_ordered': ordered_times and _ordered_events(events, expected_fault),
        'camera_aid_stopped_through_abort': unchanged and valid_aid_log,
        'px4_auto_land_mode_observed': auto_land is not None,
        'px4_ground_and_disarm_confirmed_after_fault': summary.get('landed_after_fault') is True
        and landed and disarmed is not None and auto_land is not None and auto_land < disarmed,
        'independent_touchdown_observed': bool(touchdown),
        'pre_fault_and_abort_multipart_geometry_certified_clear': geometry['certified_clear'],
        'no_unexpected_contact_in_received_messages': counts is not None
        and not (counts['obstacle'] or counts['transit_ground']),
    }
    if expected_fault == POLICY_FAULT:
        checks['policy_task_and_pre_injection_action_present'] = (
            metadata.get('task') == 'policy'
            and armed is not None
            and injection is not None
            and any(row.get('sim_time') is not None and armed <= row['sim_time'] < injection
                    for row in policy_actions)
        )

    model = airframe.X500_SDF
    runner_path = ROOT/'mission/markerfree_run.py'
    checked_runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    recorded_runner_hash = metadata.get('runtime_source_sha256', {}).get('mission/markerfree_run.py')
    for path in [Path(__file__), Path(route_score.__file__), Path(sim.__file__), Path(airframe.__file__),
                 runner_path, model, model.parents[1]/'flight_x500/model.sdf',
                 course.route_path, course.groups_path, *(
                     course.mesh_dir/group['mesh'] for group in course.groups['groups'] if group['collidable'])]:
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'run_id': metadata['run_id'],
        'mission_result': summary.get('result', 'incomplete'),
        'mission_fault': summary.get('fault'),
        'expected_fault': expected_fault,
        'fault_test_passed': all(checks.values()),
        'checks': checks,
        'timeline_sim_s': {
            'controller_armed_offboard': armed, 'injected_camera_aiding_stop': injection,
            'expected_fault': fault,
            **({'stale_capture_fault': fault} if expected_fault == FAULT else {}),
            'px4_entered_auto_land': auto_land,
            'independent_first_floor_contact': first_floor_stamp, 'px4_disarmed': disarmed,
            'controller_confirmed_ground_and_disarm': confirmed,
        },
        'vision_aiding': {
            'controller_total_vision_measurements_sent': count,
            'controller_send_count_unchanged_in_recorded_samples_after_injection': unchanged,
            'px4_ulog': log,
            'source_guard': {
                'recorded_runner_source_sha256': recorded_runner_hash,
                'checked_runner_source_sha256': checked_runner_hash,
                'recorded_hash_matches_checked_source': (recorded_runner_hash == checked_runner_hash
                                                           if recorded_runner_hash is not None else None),
                'interpretation': ('Telemetry records runtime_source_sha256 when available. A missing hash '
                                   'means runner-source identity cannot be established for that historical run; '
                                   'the scorer hashes the currently checked source but does not retroactively '
                                   'attribute it to the run.'),
            },
            'interpretation': ('The ULog count measures logged updates. Logger sampling and any unobserved '
                               'records limit absence claims, including when no dropout mark is present.'),
        },
        'independent_truth': {
            'at_injection_enu_m': inject_pose.tolist() if inject_pose is not None else None,
            'at_fault_enu_m': fault_pose.tolist() if fault_pose is not None else None,
            'at_first_floor_contact_enu_m': floor_pose.tolist() if floor_pose is not None else None,
            'at_confirmed_disarm_enu_m': final_pose.tolist() if final_pose is not None else None,
            'horizontal_displacement_injection_to_disarm_m': _displacement(inject_pose, final_pose),
            'horizontal_displacement_fault_to_disarm_m': _displacement(fault_pose, final_pose),
        },
        'cad_proxy': geometry,
        'contacts_through_disarm': {**contact_score, 'first_abort_floor_contact': first_floor} if contact_score else None,
        'px4_transitions': transitions,
        'provenance': {
            'sha256_by_file': hashes,
            'max_truth_interpolation_gap_s': route_score.DEFAULT_MAX_GAP_S,
            'fault_timestamp_source': 'Controller arming simulation time plus summary elapsed simulation time.',
        },
        'scope': ('Injected RGBD aid loss followed by floor landing and disarm. The multipart proxy checks '
                  'obstacles throughout and permits floor contact during takeoff and abort landing. '
                  'Contact observations are limited to received sensor messages.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--ulog', type=Path, required=True, help='closed PX4 ULog from this run')
    parser.add_argument('--expected-fault', choices=EXPECTED_FAULTS, default=FAULT,
                        help='exact fault expected after injection; defaults to legacy stale-capture behavior')
    parser.add_argument('--output', type=Path, help='new report path; defaults to RUN_DIRECTORY/fault-test.json')
    args = parser.parse_args()
    try:
        result = score_blackout(args.run_directory, args.ulog, expected_fault=args.expected_fault)
    except ModuleNotFoundError as error:
        if error.name == 'pyulog':
            parser.error('install pyulog in this Python environment to read PX4 ULogs')
        raise
    with (args.output or args.run_directory/'fault-test.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False, sort_keys=True)
        output.write('\n')
    return 0 if result['fault_test_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
