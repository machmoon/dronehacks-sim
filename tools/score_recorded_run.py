"""Accept a completed SITL recording only with full-flight geometry evidence.

Run with the same Python environment as evaluate_trajectory_geometry.py.
This is an offline simulation score, not a real-aircraft safety guarantee.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def score(summary, geometry, flight_start, *, baseline=59.0, margin=.04,
          max_gap=.12):
    end = flight_start + summary.get('elapsed_sim_s', 0)
    coverage = geometry['recorded_time_range_sim_s']
    obstacle = geometry['obstacle_clearance']
    clearance = obstacle['minimum_signed_m']
    if clearance is None:
        clearance = obstacle['search_distance_m']
    # The evaluator bounds each interpolated segment using its endpoint
    # clearances and length, rather than treating samples as continuous proof.
    lower_bound = geometry['linear_path_clearance_lower_bound_m']
    crossing = geometry['window_crossing']
    checks = {
        'mission_completed': summary.get('result') == 'completed',
        'all_waypoints': (summary.get('waypoint_count', 0) > 0 and
                          summary.get('waypoints_reached') == summary.get('waypoint_count')),
        'landed_and_disarmed': summary.get('landed_and_disarmed') is True,
        'full_flight_recorded': coverage[0] <= flight_start and coverage[1] >= end,
        'bounded_recording_gaps': geometry['maximum_recorded_gap_s'] <= max_gap,
        'no_envelope_overlap': obstacle['negative_clearance_samples'] == 0,
        'clearance_margin': lower_bound >= margin,
        'window_envelope_crossed': crossing.get('full_envelope_crossed') is True,
        'window_margin': (crossing.get('min_aperture_clearance_m') or 0) >= margin,
        'faster_than_baseline': 0 < summary.get('elapsed_sim_s', 0) < baseline,
    }
    return {'run_id': summary.get('run_id'), 'accepted': all(checks.values()), 'checks': checks,
            'elapsed_sim_s': summary.get('elapsed_sim_s'), 'baseline_sim_s': baseline,
            'required_clearance_m': margin, 'sampled_clearance_m': clearance,
            'interpolated_path_clearance_lower_bound_m': lower_bound,
            'window_clearance_m': crossing.get('min_aperture_clearance_m'),
            'flight_time_range_sim_s': [flight_start, end],
            'scope': 'Recorded simulation, static CAD and assumed 0.36 m sphere; '
                     'linear interpolation between truth observations. No hardware validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--course', type=Path, required=True)
    parser.add_argument('--margin', type=float, default=.04)
    parser.add_argument('--baseline', type=float, default=59)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.margin, args.baseline)):
        parser.error('margin and baseline must be positive finite values')
    rows = [json.loads(line) for line in (args.run/'telemetry.jsonl').read_text().splitlines()]
    starts = [row['sim_time'] for row in rows if row.get('type') == 'event' and
              row.get('message') == 'PX4 armed in Offboard using camera-aided EKF position']
    summary = json.loads((args.run/'telemetry.summary.json').read_text())
    if not starts:
        result = {'run_id': summary.get('run_id'), 'accepted': False,
                  'checks': {'flight_started': False}, 'fault': summary.get('fault'),
                  'scope': 'No flight-start event; no speed or full-flight clearance claim.'}
        (args.run/'score.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        return 1
    if len(starts) != 1:
        raise ValueError('exactly one flight-start event is required')
    geometry_path = args.run/'geometry.json'
    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_trajectory_geometry.py')),
                    '--trajectory', str(args.run/'truth.jsonl'), '--course', str(args.course),
                    '--output', str(geometry_path), '--radius', '.36',
                    '--sample-period', '.005',
                    '--source-label', 'Gazebo truth, offline evaluation only'], check=True,
                   stdout=subprocess.DEVNULL)
    result = score(summary, json.loads(geometry_path.read_text()), starts[0],
                   baseline=args.baseline, margin=args.margin)
    (args.run/'score.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
