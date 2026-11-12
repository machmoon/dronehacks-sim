#!/usr/bin/env python3
"""Summarize flight sections from telemetry timestamps in simulation time.

Route intervals are bounded by the verified-hover event and route-complete
event (success) or the fault time (failure). Abort landing is reported
separately and is never counted as route time.
"""
import argparse
import json
import math
from pathlib import Path


def _rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _event_time(rows, phrase):
    for row in rows:
        if row.get('type') == 'event' and phrase in str(row.get('message', '')).lower():
            return _finite(row.get('sim_time'))
    return None


def _route_segments(samples, route_start, route_end, max_gap):
    if route_start is None or route_end is None or route_end <= route_start:
        return [], 0.0
    samples = sorted((r for r in samples if _finite(r.get('sim_time')) is not None
                      and float(r['sim_time']) >= route_start),
                     key=lambda r: float(r['sim_time']))
    intervals = []
    covered = 0.0

    def append_interval(start, end, row=None):
        nonlocal covered
        start, end = max(start, route_start), min(end, route_end)
        if end <= start:
            return
        if row is None:
            label = ('UNOBSERVED', None)
        else:
            label = (str(row.get('phase') or 'phase unavailable'), row.get('waypoint_index'))
            covered += end - start
        if intervals and intervals[-1]['phase'] == label[0] and intervals[-1]['waypoint_index'] == label[1] and abs(intervals[-1]['end_sim_s'] - start) < 1e-8:
            intervals[-1]['end_sim_s'] = end
            intervals[-1]['duration_s'] = end - intervals[-1]['start_sim_s']
        else:
            intervals.append({'phase': label[0], 'waypoint_index': label[1],
                              'start_sim_s': start, 'end_sim_s': end,
                              'duration_s': end - start})

    if not samples:
        append_interval(route_start, route_end)
        return intervals, covered

    first_time = float(samples[0]['sim_time'])
    if first_time > route_start:
        append_interval(route_start, min(first_time, route_end))
    for index, row in enumerate(samples):
        start = float(row['sim_time'])
        if start >= route_end:
            break
        if index + 1 < len(samples):
            next_time = float(samples[index + 1]['sim_time'])
            if next_time <= start:
                continue
        else:
            next_time = route_end
        end = min(next_time, route_end)
        if end - start > max_gap:
            append_interval(start, end)
        else:
            append_interval(start, end, row)
    return intervals, covered


def analyze_run(archive, max_sample_gap_s=0.5):
    archive = Path(archive)
    rows = _rows(archive / 'telemetry.jsonl')
    samples = [r for r in rows if r.get('type') == 'sample']
    summary = next((r for r in reversed(rows) if r.get('type') == 'summary'), {})
    metadata = next((r for r in rows if r.get('type') == 'metadata'), {})
    score_path = archive / 'score.json'
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    run_id = summary.get('run_id', metadata.get('run_id'))
    score_run_id = score.get('run_id')
    if score and (not score_run_id or score_run_id != run_id):
        raise ValueError(f'score run_id {score_run_id!r} does not match telemetry run_id {run_id!r}: {score_path}')
    scored_route = score.get('route', {}) if score else {}

    arm_time = _event_time(rows, 'px4 armed in offboard')
    if arm_time is None:
        arm_sample_times = [_finite(r.get('sim_time')) for r in samples if r.get('px4_armed')]
        arm_sample_times = [t for t in arm_sample_times if t is not None]
        arm_time = min(arm_sample_times) if arm_sample_times else None
    hover_time = _event_time(rows, 'sustained takeoff hover verified')
    result = str(summary.get('result', 'unknown')).lower()
    task = str(summary.get('task', metadata.get('task', 'unknown'))).lower()
    policy_actions = int(summary.get('policy_actions_received') or 0)
    waypoint_reached = summary.get('waypoints_reached')
    waypoint_count = summary.get('waypoint_count', metadata.get('waypoint_count'))
    armed = arm_time is not None
    if task == 'policy':
        run_kind = 'learned policy'
    elif policy_actions == 0:
        run_kind = 'waypoint baseline'
    else:
        run_kind = f'{task} (policy actions observed)'

    summary_elapsed = _finite(summary.get('elapsed_sim_s'))
    if not armed:
        flight_status = 'prearm_failed' if result == 'failed' else 'not_armed'
        flight_duration = None
        fault_time = None
        landing_time = None
    else:
        flight_status = 'completed' if result == 'completed' else ('failed_after_arm' if result == 'failed' else result)
        flight_duration = summary_elapsed
        fault_time = arm_time + summary_elapsed if result == 'failed' and summary_elapsed is not None else None
        landing_time = arm_time + summary_elapsed if result == 'completed' and summary_elapsed is not None else None

    route_completed = bool(summary.get('route_completed'))
    route_elapsed_from_arm = _finite(summary.get('route_elapsed_sim_s')) if route_completed else None
    if route_completed and armed and route_elapsed_from_arm is not None:
        route_end = arm_time + route_elapsed_from_arm
        route_end_kind = 'route_complete'
    elif armed and hover_time is not None and fault_time is not None:
        route_end = fault_time
        route_end_kind = 'censored_at_fault'
    else:
        route_end = None
        route_end_kind = 'no_route_interval'
    route_start = hover_time if armed else None

    segments, sample_covered = _route_segments(samples, route_start, route_end, max_sample_gap_s)
    route_duration = route_end - route_start if route_start is not None and route_end is not None else None
    route_unobserved = max(0.0, route_duration - sample_covered) if route_duration is not None else None

    touchdown_event = next((r for r in rows if r.get('type') == 'event' and
                            'confirmed touchdown and disarm' in str(r.get('message', '')).lower()), None)
    touchdown_message = str(touchdown_event.get('message')) if touchdown_event else None
    touchdown_time = _finite(touchdown_event.get('sim_time')) if touchdown_event else None
    abort_landing_duration = (touchdown_time - fault_time if fault_time is not None and touchdown_time is not None
                              and touchdown_time >= fault_time else None)
    route_complete_to_landing = (landing_time - route_end if landing_time is not None and route_end is not None else None)
    end_time = landing_time if landing_time is not None else fault_time
    return {
        'run_id': run_id,
        'archive': str(archive),
        'task': task,
        'run_kind': run_kind,
        'result': result,
        'flight_status': flight_status,
        'end_semantics': 'completed_landing' if landing_time is not None else ('fault_censored' if fault_time is not None else 'no_armed_flight'),
        'fault': summary.get('fault'),
        'armed': armed,
        'arm_sim_time_s': arm_time,
        'mission_flight_duration_s': flight_duration,
        'arm_to_flight_end_s': flight_duration,
        'flight_end_sim_time_s': end_time,
        'reported_summary_elapsed_sim_s': summary_elapsed,
        'prearm_elapsed_is_flight_time': False,
        'hover_verified_sim_time_s': hover_time,
        'arm_to_hover_s': hover_time - arm_time if armed and hover_time is not None else None,
        'route_start_sim_time_s': route_start,
        'route_end_sim_time_s': route_end,
        'route_end_semantics': route_end_kind,
        'route_duration_s': route_duration,
        'arm_to_route_complete_s': route_elapsed_from_arm,
        'route_complete_to_landing_s': route_complete_to_landing,
        'abort_landing_confirmed_by_event': bool(touchdown_message and 'after abort' in touchdown_message.lower()),
        'abort_landing_confirmation_message': touchdown_message if fault_time is not None else None,
        'abort_landing_disarm_sim_time_s': touchdown_time if fault_time is not None and touchdown_message else None,
        'abort_landing_elapsed_s': abort_landing_duration,
        'abort_landing_timing_note': ('Measured from fault time to the timestamped touchdown/disarm event; '
                                      'excluded from route time.' if abort_landing_duration is not None else
                                      ('Landing is confirmed after the fault but has no usable simulation timestamp; '
                                       'it is excluded from route time.' if fault_time is not None and touchdown_message else None)),
        'policy_actions_received': policy_actions,
        'controller_waypoints_reached': waypoint_reached,
        'waypoint_count': waypoint_count,
        'truth_waypoints_reached': scored_route.get('reached'),
        'truth_waypoint_count': scored_route.get('required'),
        'truth_waypoint_source': 'score.json independent truth observer' if scored_route else None,
        'route_sample_coverage_s': sample_covered,
        'route_unobserved_s': route_unobserved,
        'route_segments': segments,
        'timing_basis': 'simulation timestamps only; contiguous sample labels are held to the next sample; gaps over max_sample_gap_s are unobserved; route ends at success event or is censored at fault',
        'max_sample_gap_s': max_sample_gap_s,
    }


def _fmt(value):
    return '—' if value is None else f'{float(value):.3f}'


def render_table(results):
    lines = ['run        kind             status            arm->hover  route(s)  route end        route->land  arm->end  policy actions  truth waypoints']
    for result in results:
        wp = '—' if result['truth_waypoints_reached'] is None else f"{result['truth_waypoints_reached']}/{result['truth_waypoint_count']}"
        lines.append(f"{Path(result['archive']).name:<10} {result['run_kind']:<16} {result['flight_status']:<17} "
                     f"{_fmt(result['arm_to_hover_s']):>10} {_fmt(result['route_duration_s']):>9} "
                     f"{result['route_end_semantics']:<16} {_fmt(result['route_complete_to_landing_s']):>11} "
                     f"{_fmt(result['arm_to_flight_end_s']):>8} "
                     f"{result['policy_actions_received']:>14} {wp:>14}")
        if result['flight_status'] == 'prearm_failed':
            lines.append('  No armed flight; preflight wall time is not represented as simulation flight time.')
        elif result['route_end_semantics'] == 'censored_at_fault':
            lines.append(f"  Route duration is censored at fault ({_fmt(result['route_duration_s'])} s after hover); abort landing is excluded.")
            if result['abort_landing_elapsed_s'] is not None:
                lines.append(f"  Touchdown/disarm was confirmed {_fmt(result['abort_landing_elapsed_s'])} s after the fault (reported separately).")
        for segment in result['route_segments']:
            waypoint = '—' if segment['waypoint_index'] is None else str(segment['waypoint_index'])
            lines.append(f"  {segment['start_sim_s']:8.3f}–{segment['end_sim_s']:8.3f}  "
                         f"{segment['duration_s']:7.3f} s  phase={segment['phase']}  waypoint={waypoint}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, help='write JSON report to this path')
    parser.add_argument('--format', choices=('table', 'json'), default='table')
    parser.add_argument('--max-sample-gap-s', type=float, default=0.5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_sample_gap_s) or args.max_sample_gap_s <= 0:
        parser.error('--max-sample-gap-s must be finite and positive')
    results = [analyze_run(path, args.max_sample_gap_s) for path in args.archives]
    report = {'schema_version': 1, 'clock': 'simulation time', 'runs': results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2) if args.format == 'json' else render_table(results))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
