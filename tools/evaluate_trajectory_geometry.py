#!/usr/bin/env python3
"""Offline window and static-CAD sphere-clearance diagnostics for JSONL paths.

This loads no flight stack and runs no simulator. Positions between recorded
samples are linearly interpolated before querying the supplied course mesh.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def read_path(path):
    rows = []
    for line in Path(path).open():
        row = json.loads(line)
        position = row.get('position')
        timestamp = row.get('sim_time')
        if (isinstance(position, list) and len(position) == 3
                and isinstance(timestamp, (int, float))):
            rows.append({'sim_time': float(timestamp), 'position': position,
                         'phase': row.get('phase'),
                         'waypoint_index': row.get('waypoint_index')})
    rows.sort(key=lambda row: row['sim_time'])
    if len(rows) < 2:
        raise ValueError('trajectory must contain at least two timestamped positions')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--course', type=Path, default=Path('mission/course.json'))
    parser.add_argument('--cad', type=Path, default=Path('assets/course-v3.zip'))
    parser.add_argument('--geometry-module', type=Path,
                        default=Path('../Drone008_Dronehacks/tools/course_demo'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-label', default='recorded positions; truth/estimate status unspecified')
    parser.add_argument('--radius', type=float, default=.36)
    parser.add_argument('--sample-period', type=float, default=1/48)
    parser.add_argument('--search-distance', type=float, default=.65)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in
               (args.radius, args.sample_period, args.search_distance)):
        parser.error('radius, sample period and search distance must be positive and finite')

    root = Path.cwd()
    geometry_module = (root / args.geometry_module).resolve()
    sys.path.insert(0, str(geometry_module))
    sys.path.insert(0, str((root / 'mission').resolve()))
    try:
        import pybullet as bullet
        from geometry import CourseGeometry
        from crossing import evaluate_window_crossing
    except ImportError as error:
        raise SystemExit('Requires pybullet and the existing course geometry/crossing modules: '
                         f'{error}')

    rows = read_path(args.trajectory)
    course = json.loads((root / args.course).read_text())
    crossing = evaluate_window_crossing([row['position'] for row in rows],
                                        course['window'], args.radius)
    client = bullet.connect(bullet.DIRECT)
    try:
        geometry = CourseGeometry(str((root / args.cad).resolve()), client,
                                  radius=args.radius)
        expanded = []
        for start, end in zip(rows, rows[1:]):
            dt = end['sim_time'] - start['sim_time']
            count = max(1, math.ceil(dt / args.sample_period))
            for index in range(count):
                fraction = index / count
                position = [start['position'][axis] + fraction *
                            (end['position'][axis] - start['position'][axis])
                            for axis in range(3)]
                expanded.append((start['sim_time'] + fraction * dt, position, start))
        expanded.append((rows[-1]['sim_time'], rows[-1]['position'], rows[-1]))

        minimum, nearest, at = math.inf, None, None
        linear_path_lower_bound = math.inf
        previous_probe = None
        negative = below_2cm = 0
        for timestamp, position, source_row in expanded:
            clearance, obstacle = geometry.clearance(position, args.search_distance)
            if previous_probe is not None:
                previous_position, previous_clearance = previous_probe
                # Every point on this line segment is at most half its length
                # from an endpoint. Static obstacle distance is 1-Lipschitz.
                linear_path_lower_bound = min(
                    linear_path_lower_bound, min(clearance, previous_clearance) -
                    math.dist(position, previous_position)/2)
            previous_probe = position, clearance
            negative += clearance < 0
            below_2cm += clearance < .02
            if clearance < minimum:
                minimum, nearest = clearance, obstacle
                at = {'sim_time_s': timestamp, 'position_m': position,
                      'phase': source_row['phase'],
                      'waypoint_index': source_row['waypoint_index']}
    finally:
        bullet.disconnect(client)

    result = {
        'offline_only': True,
        'trajectory': str(args.trajectory),
        'trajectory_source': args.source_label,
        'cad_asset': str(args.cad),
        'geometry_evaluator': 'Drone008_Dronehacks/tools/course_demo/geometry.py CourseGeometry',
        'sphere_radius_m': args.radius,
        'input_positions': len(rows),
        'recorded_time_range_sim_s': [rows[0]['sim_time'], rows[-1]['sim_time']],
        'maximum_recorded_gap_s': max(b['sim_time']-a['sim_time'] for a, b in zip(rows, rows[1:])),
        'interpolated_positions_checked': len(expanded),
        'maximum_sample_interval_s': args.sample_period,
        'maximum_interpolated_step_m': max(
            math.dist(a[1], b[1]) for a, b in zip(expanded, expanded[1:])),
        'linear_path_clearance_lower_bound_m': linear_path_lower_bound,
        'obstacle_clearance': {
            'minimum_signed_m': minimum if minimum < args.search_distance else None,
            'nearest_obstacle': nearest if minimum < args.search_distance
            else f'beyond search distance; clearance >= {args.search_distance} m',
            'minimum_at': at,
            'negative_clearance_samples': negative,
            'samples_below_0_02_m': below_2cm,
            'search_distance_m': args.search_distance,
        },
        'window_crossing': crossing,
        'assumptions_and_limits': [
            'Assumed spherical envelope; this is not the airframe swept volume.',
            'Only the recorded time range is checked; missing takeoff/landing data is not evidence of clearance.',
            'Static imported CAD mesh; not an official surveyed course or collision judge.',
            'Linear interpolation and sampled closest-point queries do not prove continuous-path clearance.',
            'Negative clearance is overlap with the geometry proxy, not proof of physical contact.',
            'Floor/pad surfaces are excluded from the static obstacle clearance probe.',
            'Boulder collision geometry uses the existing evaluator convex-hull approximation.',
        ],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
