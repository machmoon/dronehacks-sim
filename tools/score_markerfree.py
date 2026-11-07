"""Score a recorded marker-free SITL run offline from independent Gazebo evidence."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from learning.sim import Course
from mission import airframe
from mission.crossing import evaluate_window_crossing as evaluate_sphere_crossing
from mission.px4_frames import camera_pose_to_px4

LEGACY_SPHERE_RADIUS_M = .36
X500_ENVELOPE_MARGIN_M = .015
DEFAULT_MAX_GAP_S = .12
TOUCHDOWN_HEIGHT_TOLERANCE_M = .12
ARMED_EVENT = 'PX4 armed in Offboard with marker-free RGBD aiding'
HOVER_EVENT = 'Sustained takeoff hover verified'
TOUCHDOWN_EVENT = 'PX4 confirmed touchdown and disarm inside the modeled pad'


def _load(path, provenance, *, lines=False):
    data = path.read_bytes()
    provenance[str(path)] = hashlib.sha256(data).hexdigest()
    if lines:
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    return json.loads(data)


def _vector(value, size):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f'expected {size} finite coordinates')
    return vector


def _time(value):
    stamp = float(value)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError('invalid simulation timestamp')
    return stamp


class Truth:
    def __init__(self, rows, max_gap_s):
        self.stamps = np.asarray([_time(row['sim_time_s']) for row in rows])
        self.positions = np.asarray([_vector(row['base_link_position_enu_m'], 3) for row in rows])
        self.orientations = np.asarray([_vector(row['base_link_orientation_xyzw'], 4) for row in rows])
        self.max_gap_s = max_gap_s
        if not len(rows) or np.any(np.diff(self.stamps) <= 0):
            raise ValueError('truth recording must contain strictly increasing timestamps')
        norms = np.linalg.norm(self.orientations, axis=1)
        if np.any(np.abs(norms - 1) > .001):
            raise ValueError('truth recording contains a non-unit orientation')
        self.orientations /= norms[:, None]

    def at(self, stamp):
        """Interpolate only inside a fresh truth bracket on the same clock."""
        i = int(np.searchsorted(self.stamps, stamp))
        # Observer nanoseconds and PX4 milliseconds serialize through floats.
        for index in (i, i-1):
            if 0 <= index < len(self.stamps) and abs(self.stamps[index]-stamp) <= 1e-9:
                return self.positions[index], self.orientations[index]
        if not 0 < i < len(self.stamps):
            return None
        t0, t1 = self.stamps[i-1:i+1]
        if t1-t0 > self.max_gap_s+1e-9:
            return None
        weight = (stamp-t0)/(t1-t0)
        position = (1-weight)*self.positions[i-1] + weight*self.positions[i]
        a, b = self.orientations[i-1:i+1]
        dot = float(np.dot(a, b))
        if dot < 0:
            b, dot = -b, -dot
        if dot > .9995:
            orientation = (1-weight)*a + weight*b
        else:
            angle = math.acos(min(1., dot))
            orientation = (math.sin((1-weight)*angle)*a + math.sin(weight*angle)*b)/math.sin(angle)
        return position, orientation/np.linalg.norm(orientation)

    def interval(self, start, end):
        interior = (self.stamps > start) & (self.stamps < end)
        points = list(zip(self.stamps[interior], self.positions[interior]))
        begin, finish = self.at(start), self.at(end)
        if begin is not None:
            points.insert(0, (start, begin[0]))
        if finish is not None and end != start:
            points.append((end, finish[0]))
        parts = []
        for stamp, position in points:
            if not parts or stamp-parts[-1][-1][0] > self.max_gap_s+1e-9:
                parts.append([])
            parts[-1].append((stamp, position))
        return parts, bool(begin is not None and finish is not None and len(parts) == 1)


def _position_stats(errors):
    if not errors:
        return None
    errors = np.array(errors)
    distances = np.linalg.norm(errors, axis=1)
    return {
        'sample_count': len(errors),
        'bias_enu_m': errors.mean(axis=0).tolist(),
        'rmse_enu_m': np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        'first_error_enu_m': errors[0].tolist(),
        'last_error_enu_m': errors[-1].tolist(),
        'rmse_m': float(np.sqrt(np.mean(distances**2))),
        'median_m': float(np.median(distances)),
        'p95_m': float(np.percentile(distances, 95)),
        'max_m': float(distances.max()),
    }


def _position_alignment(samples, truth, position_key, stamp_key, interval, *, approximate=False):
    errors, by_phase, used = [], {}, set()
    missing_bracket, approximate_count = 0, 0
    for row in samples:
        if row.get(position_key) is None:
            continue
        value = row.get(stamp_key)
        use_loop_time = approximate and value is None
        if use_loop_time:
            value = row.get('sim_time')
        if value is None:
            continue
        stamp = _time(value)
        if not interval[0] <= stamp <= interval[1] or stamp in used:
            continue
        used.add(stamp)
        actual = truth.at(stamp)
        if actual is None:
            missing_bracket += 1
            continue
        approximate_count += use_loop_time
        error = _vector(row[position_key], 3) - actual[0]
        errors.append(error)
        by_phase.setdefault(row.get('phase', 'unknown'), []).append(error)
    return {
        'timestamp_field': stamp_key,
        'approximate_loop_timestamp_samples': approximate_count,
        'unmatched_timestamp_count': missing_bracket,
        'stats': _position_stats(errors),
        'by_phase': {phase: _position_stats(values) for phase, values in by_phase.items()},
    }


def _rpy_quaternion(rpy):
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy/2), np.sin(rpy/2)
    return np.array((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def _attitude_alignment(samples, truth, interval):
    angles, used = [], set()
    approximate_count, missing_bracket = 0, 0
    for row in samples:
        if row.get('attitude_rpy_frd') is None:
            continue
        approximate = row.get('attitude_stamp_sim_s') is None
        stamp = _time(row['sim_time'] if approximate else row['attitude_stamp_sim_s'])
        if not interval[0] <= stamp <= interval[1] or stamp in used:
            continue
        used.add(stamp)
        actual = truth.at(stamp)
        if actual is None:
            missing_bracket += 1
            continue
        approximate_count += approximate
        quat = actual[1]
        _, actual_rpy = camera_pose_to_px4(actual[0], np.r_[quat[3], quat[:3]])
        q_est = _rpy_quaternion(_vector(row['attitude_rpy_frd'], 3))
        q_truth = _rpy_quaternion(np.array(actual_rpy))
        angle = 2*math.degrees(math.acos(min(1., abs(float(q_est @ q_truth)))))
        angles.append(angle)
    return {
        'sample_count': len(angles),
        'approximate_loop_timestamp_samples': approximate_count,
        'unmatched_timestamp_count': missing_bracket,
        'rmse_deg': float(np.sqrt(np.mean(np.square(angles)))) if angles else None,
        'p95_deg': float(np.percentile(angles, 95)) if angles else None,
        'max_deg': max(angles, default=None),
    }


def _window_summary(scores):
    valid = [score for score in scores if score['full_envelope_crossed']]
    finite = [score for score in scores if score['min_aperture_clearance_m'] is not None]
    if valid:
        best = min(valid, key=lambda s: s['min_aperture_clearance_m'])
    elif finite:
        best = max(finite, key=lambda s: s['min_aperture_clearance_m'])
    else:
        best = scores[0]
    return {**best, **{key: sum(score[key] for score in scores) for key in
                       ('valid_forward_crossings', 'forward_attempts', 'valid_reverse_crossings')}}


def _window(parts, window):
    if not window:
        return None
    return _window_summary([
        evaluate_sphere_crossing([p for _, p in part], window, LEGACY_SPHERE_RADIUS_M)
        for part in parts or [[]]
    ])


def _pose_samples(truth, part):
    return [p for _, p in part], [truth.at(stamp)[1] for stamp, _ in part]


def _multipart_window(truth, parts, window, envelope):
    if not window:
        return None
    return _window_summary([
        airframe.evaluate_window_crossing(*_pose_samples(truth, part), window, envelope)
        for part in parts or [[]]
    ])


def _multipart_scene(truth, spans, scene, envelope):
    phases = {}
    for phase, start, end, check_ground in spans:
        if end <= start:
            continue
        parts, complete = truth.interval(start, end)
        checked = 0
        first_unresolved = None
        all_clear = bool(parts)
        for part in parts:
            result = airframe.check_scene_path(*_pose_samples(truth, part), scene, envelope,
                                               check_ground=check_ground)
            checked += result['checked_subsegments']
            all_clear &= result['certified_clear']
            hit = result['first_unresolved']
            if hit is not None and first_unresolved is None:
                segment = hit['segment']
                first_unresolved = {**hit, 'segment_sim_s': [part[segment][0],
                                                            part[min(segment+1, len(part)-1)][0]]}
        phases[phase] = {
            'interval_sim_s': [start, end],
            'ground_checked': check_ground,
            'truth_complete': complete,
            'certified_clear': complete and all_clear,
            'checked_subsegments': checked,
            'first_unresolved': first_unresolved,
        }
    return {
        'certified_clear': bool(phases) and all(result['certified_clear'] for result in phases.values()),
        'phases': phases,
        'margin_m': envelope.margin_m,
        'component_count': len(envelope.components),
        'model': ('SDF body/gear boxes and swept rotor cylinders against the course CAD proxy boxes '
                  'and polygonal tunnel shell. Translation is linear and attitude follows the shortest arc.'),
        'interpretation': 'Unresolved component separation can be a conservative false positive.',
    }


def _proxy_collisions(parts, transit_start, landing_start, course):
    # Move both flight path and boxes above Scene.hits' built-in ground plane.
    # The translation preserves exact sphere/box sweeps. Score the real floor
    # separately so expected ground contact does not hide nearby obstacles.
    lowest = min((float(p[2]) for part in parts for _, p in part), default=0.)
    shift = np.array((0., 0., LEGACY_SPHERE_RADIUS_M + 1. - min(0., lowest)))
    base = course.scene(np.random.default_rng(0), 0.)
    scene = replace(base, centers=base.centers + shift, radius=LEGACY_SPHERE_RADIUS_M)
    first_obstacle = None
    hit_count, checked = 0, 0
    sampled_min = None
    first_floor = None
    for part in parts:
        shifted = np.array([p for _, p in part]) + shift
        q = np.abs(scene.local(shifted)) - scene.half
        signed = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
        clearance = float(signed.min()) - LEGACY_SPHERE_RADIUS_M
        sampled_min = clearance if sampled_min is None else min(clearance, sampled_min)
        for (ta, a), (tb, b) in zip(part, part[1:]):
            checked += 1
            if scene.hits(a+shift, b+shift):
                hit_count += 1
                if first_obstacle is None:
                    first_obstacle = [ta, tb]
            low, high = max(ta, transit_start), min(tb, landing_start)
            if first_floor is None and high > low:
                za, zb = (a[2]+(t-ta)/(tb-ta)*(b[2]-a[2]) for t in (low, high))
                if min(za, zb) <= LEGACY_SPHERE_RADIUS_M:
                    first_floor = [low, high]
    return {
        'sphere_radius_m': LEGACY_SPHERE_RADIUS_M,
        'checked_segments': checked,
        'hit_segment_count': hit_count,
        'first_hit_segment_sim_s': first_obstacle,
        'sampled_min_obstacle_clearance_m': sampled_min,
        'first_transit_floor_overlap_segment_sim_s': first_floor,
        'model': 'Continuous sphere sweeps over learning.sim CAD box and polygonal tunnel proxies, linear truth segments.',
        'ground_treatment': 'Floor scored separately from the start of hover verification to landing handoff.',
    }


def _truth_waypoints(parts, metadata):
    plan = metadata['waypoints']
    default = metadata['course_limits']['arrival_radius']
    goals = [_vector(item['position'], 3) for item in plan]
    radii = []
    for index, item in enumerate(plan):
        error = f'waypoint {index} requires 0 < arrival_radius <= verification_radius <= 0.2 m'
        try:
            arrival = float(item.get('arrival_radius', default))
            verification = float(item.get('verification_radius', arrival))
        except (TypeError, ValueError) as exc:
            raise ValueError(error) from exc
        if not 0 < arrival <= verification <= .2:
            raise ValueError(error)
        radii.append((arrival, verification))
    reached = []
    index = 0
    for part in parts:
        for (ta, a), (tb, b) in zip(part, part[1:]):
            delta = b-a
            length2 = float(delta @ delta)
            fraction = 0.
            while index < len(goals):
                offset = a-goals[index]
                projection = float(offset @ delta)
                arrival, verification = radii[index]
                c = float(offset @ offset) - verification**2
                if length2*fraction**2 + 2*projection*fraction + c > 1e-12:
                    discriminant = projection**2 - length2*c
                    if length2 < 1e-15 or discriminant < 0:
                        break
                    entry = (-projection-math.sqrt(discriminant))/length2
                    if not fraction <= entry <= 1:
                        break
                    fraction = entry
                visit = {'index': index, 'phase': plan[index]['phase'],
                         'arrival_radius_m': arrival, 'sim_time_s': ta+fraction*(tb-ta)}
                if 'verification_radius' in plan[index]:
                    visit['verification_radius_m'] = verification
                reached.append(visit)
                index += 1
    return {'reached': index, 'required': len(plan), 'arrivals': reached}


def _contacts(rows, status, start, end, transit_start, landing_start):
    observed_topics = status.get('contacts_topics_with_messages', [])
    expected_topics = status.get('contacts_expected_topics', [])
    counts = {'obstacle': 0, 'transit_ground': 0, 'expected_ground': 0, 'self': 0}
    first_obstacle = None
    for row in rows:
        stamp = _time(row['sim_time_s'])
        if not start <= stamp <= end:
            continue
        for pair in row['pairs']:
            first, second = pair
            if 'course_quad' in first and 'course_quad' in second:
                counts['self'] += 1
            elif any('ground_plane' in name or 'floor_20_x_20_ft' in name for name in pair):
                kind = 'transit_ground' if transit_start <= stamp < landing_start else 'expected_ground'
                counts[kind] += 1
            else:
                counts['obstacle'] += 1
                if first_obstacle is None or stamp < first_obstacle['sim_time_s']:
                    first_obstacle = {'sim_time_s': stamp, 'pair': pair, 'topic': row.get('topic')}
    return {
        'status': ('all_expected_topics_observed' if expected_topics and set(expected_topics) <= set(observed_topics)
                   else 'partial_topic_coverage' if observed_topics or rows else 'unavailable'),
        'topics_with_messages': observed_topics,
        'topics_expected': expected_topics,
        'topic_discovery_and_observations': status.get('contact_topic_observation'),
        'raw_pair_observations': counts if rows or observed_topics else None,
        'first_obstacle_observation': first_obstacle,
        'interpretation': ('Sensor observations only; counts can repeat. Publisher advertisement is not a '
                           'health check, and silence cannot establish zero physical contact.'),
    }


def _event(events, message):
    return next((row for row in events if row.get('message') == message), {})


def _abort_landing(summary, events, truth, contacts, status, fault_time, scene, envelope):
    """Keep post-fault clearance separate from the scored mission interval."""
    if summary.get('result') != 'failed' or fault_time is None:
        return None
    confirmed = _event(events, 'PX4 confirmed touchdown and disarm after abort').get('sim_time')
    confirmed = _time(confirmed) if confirmed is not None else None
    valid = (summary.get('landed_after_fault') is True and confirmed is not None
             and confirmed > fault_time)
    result = {'fault_sim_s': fault_time, 'confirmed_disarm_sim_s': confirmed,
              'controller_confirmed_ground_and_disarm': valid,
              'cad_proxy': None, 'contacts': None, 'certified_clear_through_disarm': False,
              'scope': 'Clearance covers the post-fault interval through controller-confirmed disarm. '
                       'It compares the margin-inflated multipart envelope with course CAD obstacle proxies. '
                       'Floor contact is permitted. Physical contact evidence is reported separately.'}
    if valid:
        geometry = _multipart_scene(truth, [('abort_landing', fault_time, confirmed, False)], scene, envelope)
        observed = _contacts(contacts, status, fault_time, confirmed, fault_time, fault_time)
        result.update(cad_proxy=geometry, contacts=observed,
                      certified_clear_through_disarm=geometry['certified_clear'])
    return result


def score_run(directory, max_truth_gap_s=DEFAULT_MAX_GAP_S):
    if not math.isfinite(max_truth_gap_s) or max_truth_gap_s <= 0:
        raise ValueError('max truth gap must be positive and finite')
    directory = Path(directory).resolve()
    source_hashes = {}
    telemetry = _load(directory/'telemetry.jsonl', source_hashes, lines=True)
    metadata = next(row for row in telemetry if row.get('type') == 'metadata')
    summary = next((row for row in telemetry if row.get('type') == 'summary'), {})
    if summary and metadata['run_id'] != summary['run_id']:
        raise ValueError('telemetry metadata and summary run IDs disagree')
    events = [row for row in telemetry if row.get('type') == 'event']
    samples = [row for row in telemetry if row.get('type') == 'sample']
    observer = directory/'observer'
    truth = Truth(_load(observer/'truth.jsonl', source_hashes, lines=True), max_truth_gap_s)
    contacts_path, status_path = observer/'contacts.jsonl', observer/'status.json'
    contacts = _load(contacts_path, source_hashes, lines=True) if contacts_path.is_file() else []
    status = _load(status_path, source_hashes) if status_path.is_file() else {}

    armed = _event(events, ARMED_EVENT)
    if armed.get('sim_time') is not None:
        start = _time(armed['sim_time'])
        end = start + _time(summary['elapsed_sim_s']) if summary else max(
            (_time(row['sim_time']) for row in samples), default=start)
        flight_samples = [row for row in samples if start <= _time(row['sim_time']) <= end]
        parts, complete_truth = truth.interval(start, end)
    else:
        start = end = None
        flight_samples, parts, complete_truth = [], [], False
    # The hover-settle check is already airborne. Start checking the floor
    # at its first recorded sample, with the completion event as a fallback.
    transit_start = min(
        _time(_event(events, HOVER_EVENT).get('sim_time', end or 0.)),
        min((_time(row['sim_time']) for row in flight_samples if row.get('phase') == 'HOVER_CHECK'),
            default=end or 0.),
    )
    landing_start = (start + _time(summary['route_elapsed_sim_s'])
                     if start is not None and summary.get('route_elapsed_sim_s') is not None else end or 0.)
    spans = []
    if start is not None:
        if not start <= transit_start <= landing_start <= end:
            raise ValueError('takeoff, transit, and landing timestamps are inconsistent')
        spans = [('takeoff', start, transit_start, False),
                 ('transit', transit_start, landing_start, True),
                 ('landing', landing_start, end, False)]

    course = Course()
    envelope = airframe.load_x500_envelope(margin_m=X500_ENVELOPE_MARGIN_M)
    model = airframe.X500_SDF
    for asset in [model, model.parents[1]/'flight_x500/model.sdf', Path(airframe.__file__).resolve(),
                  course.route_path, course.groups_path, *(
            course.mesh_dir / group['mesh'] for group in course.groups['groups'] if group['collidable'])]:
        source_hashes[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
    scene = course.scene(np.random.default_rng(0), 0.)
    proxy = _multipart_scene(truth, spans, scene, envelope)
    abort = _abort_landing(summary, events, truth, contacts, status, end, scene, envelope)
    crossing = _multipart_window(truth, parts, metadata.get('window'), envelope)
    truth_route = _truth_waypoints(parts, metadata)
    contact = _contacts(contacts, status, start or 0., end or 0., transit_start, landing_start)
    interval = (start or 0., end or 0.)
    position = _position_alignment(flight_samples, truth, 'position', 'position_stamp_sim_s', interval,
                                   approximate=True)
    camera = _position_alignment(flight_samples, truth, 'vision_position_enu', 'vision_stamp_sim_s', interval)
    attitude = _attitude_alignment(flight_samples, truth, interval)

    pad = metadata['landing']
    pad_center = _vector(pad['position'], 3)
    pad_radius = float(summary.get('pad_radius_m', .762))
    disarmed = summary.get('landed_and_disarmed') is True and bool(_event(events, TOUCHDOWN_EVENT))
    at_start = truth.at(start) if start is not None else None
    at_finish = truth.at(end) if end is not None and disarmed else None
    landing_position = at_finish[0] if at_finish is not None else None
    # A bound on the entire body radius also bounds the horizontal footprint
    # at any touchdown attitude, including the diagonal swept rotors.
    pad_footprint_radius = max(part.max_body_radius for part in envelope.components) + envelope.margin_m
    pad_margin, legacy_pad_margin, height_delta, landing_error = None, None, None, None
    if landing_position is not None:
        pad_distance = float(np.linalg.norm(landing_position[:2]-pad_center[:2]))
        pad_margin = pad_radius - pad_footprint_radius - pad_distance
        legacy_pad_margin = pad_radius - LEGACY_SPHERE_RADIUS_M - pad_distance
        if at_start is not None:
            height_delta = float(landing_position[2] - pad_center[2] - at_start[0][2])
        if summary.get('landing_position_enu') is not None:
            landing_error = (_vector(summary['landing_position_enu'], 3)-landing_position).tolist()
    landing = {
        'px4_confirmed_ground_and_disarm': disarmed,
        'truth_time_sim_s': end if disarmed else None,
        'truth_base_link_enu_m': landing_position.tolist() if landing_position is not None else None,
        'pad_center_enu_m': pad_center.tolist(),
        'pad_radius_m': pad_radius,
        'horizontal_footprint_radius_bound_m': pad_footprint_radius,
        'horizontal_envelope_clearance_m': pad_margin,
        'ground_height_error_vs_launch_m': height_delta,
        'px4_landing_error_enu_m': landing_error,
        'truth_envelope_inside_pad': pad_margin is not None and pad_margin >= 0,
        'truth_height_consistent_with_touchdown': height_delta is not None and
        abs(height_delta) <= TOUCHDOWN_HEIGHT_TOLERANCE_M,
    }
    controller_complete = (summary.get('result') == 'completed' and summary.get('route_completed') is True
                           and summary.get('waypoints_reached') == truth_route['required'])
    contact_counts = contact['raw_pair_observations']
    checks = {
        'controller_reports_all_waypoints': controller_complete,
        'truth_complete_during_flight': complete_truth,
        'truth_reaches_all_waypoints_in_order': truth_route['reached'] == truth_route['required'],
        'truth_crosses_window_with_full_envelope': bool(crossing and crossing['full_envelope_crossed']),
        'truth_multipart_path_certified_clear': complete_truth and proxy['certified_clear'],
        'truth_transit_floor_checked': 'transit' in proxy['phases'],
        'px4_ground_and_disarm_confirmed': disarmed,
        'truth_touchdown_inside_pad': landing['truth_envelope_inside_pad'] and landing['truth_height_consistent_with_touchdown'],
    }
    observed_contact = (bool(contact_counts['obstacle'] or contact_counts['transit_ground'])
                        if contact_counts is not None else None)
    contact['unexpected_contact_observed'] = observed_contact
    ages = [float(row['vision_age_sim_s']) for row in flight_samples
            if row.get('vision_valid') is True and row.get('vision_age_sim_s') is not None]
    return {
        'run_id': metadata['run_id'],
        'task': metadata['task'],
        'mission_result': summary.get('result', 'incomplete'),
        'mission_fault': summary.get('fault'),
        'full_route_verified_by_geometry': (metadata['task'] in ('course', 'policy') and all(checks.values())
                                            and observed_contact is not True),
        'checks': checks,
        'flight_interval_sim_s': [start, end],
        'max_truth_interpolation_gap_s': max_truth_gap_s,
        'truth_sample_count': len(truth.stamps),
        'truth_flight_points': sum(map(len, parts)),
        'truth_contiguous_chunks': len(parts),
        'route': truth_route,
        'window': crossing,
        'cad_proxy': proxy,
        'legacy_sphere': {
            'window': _window(parts, metadata.get('window')),
            'cad_proxy': _proxy_collisions(parts, transit_start, landing_start, course),
            'landing_horizontal_clearance_m': legacy_pad_margin,
        },
        'landing': landing,
        'abort_landing': abort,
        'contacts': contact,
        'px4_position_error': position,
        'px4_attitude_error': attitude,
        'camera_position_error': camera,
        'camera_capture_age_sim_s': {'median': float(np.median(ages)), 'p95': float(np.percentile(ages, 95))}
        if ages else None,
        'provenance': {
            'sha256_by_file': source_hashes,
            'navigation_source_audit': _event(events, 'Navigation source audit passed') or None,
            'truth_source': 'Independent Gazebo dynamic_pose/info observer, world ENU base_link.',
            'estimate_source': 'PX4 LOCAL_POSITION_NED converted to ENU in controller telemetry; camera ENU when logged.',
            'error_sign': 'Estimate minus truth; no fitted frame transform or time shift.',
            'geometry_scope': ('Simulated path, multipart x500 collision envelope, and course CAD proxies. '
                               'Legacy sphere diagnostics do not determine the score. '
                               'Geometry certification does not establish physical contacts or official scoring.'),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, help='new score JSON file; default stdout')
    parser.add_argument('--max-truth-gap', type=float, default=DEFAULT_MAX_GAP_S)
    args = parser.parse_args()
    result = json.dumps(score_run(args.run_directory, args.max_truth_gap), indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
