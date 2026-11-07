"""Offline camera/estimator accuracy report; never publishes navigation data.

Uses an independent Gazebo observer only after the recorded mission ends. It
does not certify obstacle clearance, collisions, hardware accuracy or SLAM
success: the localizer and airframe are carried through without relabeling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def report(directory):
    directory = Path(directory)
    paths = [directory/'telemetry.jsonl', directory/'telemetry.summary.json',
             directory/'observer/truth.jsonl']
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    summary = json.loads(paths[1].read_text())
    truth = [json.loads(line) for line in paths[2].read_text().splitlines()]
    by_time = {r['sim_time_s']: r['base_link_position_enu_m'] for r in truth}
    stamps = np.array(sorted(by_time))
    xyz = np.array([by_time[t] for t in stamps])
    if len(stamps) < 2:
        raise ValueError('at least two independent observer samples required')
    output = dict(airframe=summary['vehicle_model'], localizer=summary.get('localizer', 'survey'),
                  navigation_source=summary['navigation_source'], result=summary['result'],
                  fault=summary.get('fault'), elapsed_sim_s=summary['elapsed_sim_s'],
                  landed_and_disarmed=summary['landed_and_disarmed'],
                  landed_after_fault=summary.get('landed_after_fault', False),
                  waypoints_reached=summary['waypoints_reached'], waypoint_count=summary['waypoint_count'],
                  hover_settled_sim_s=summary.get('hover_settled_sim_s'),
                  truth_used_for_control=summary['simulator_position_used_for_control'],
                  truth_samples=len(truth), clearance_or_collision_certified=False,
                  evidence_sha256={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in paths})
    for name, stamp_key, position_key in [('camera', 'vision_stamp_sim_s', 'vision_position_enu'),
                                          ('px4', 'position_stamp_sim_s', 'position')]:
        errors, armed_errors, used = [], [], set()
        for row in rows:
            t, p = row.get(stamp_key), row.get(position_key)
            if t is None or p is None or t in used or (name == 'camera' and not row.get('vision_valid')):
                continue
            i = np.searchsorted(stamps, t, side='right')
            if i == 0 or i == len(stamps) or stamps[i]-stamps[i-1] > .1:
                continue
            f = (t-stamps[i-1])/(stamps[i]-stamps[i-1])
            target = (1-f)*xyz[i-1]+f*xyz[i]
            errors.append(float(np.linalg.norm(np.asarray(p)-target)))
            if row.get('px4_armed'):
                armed_errors.append(errors[-1])
            used.add(t)
        output[name] = dict(aligned_samples=len(errors),
                            position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                            position_p95_m=float(np.percentile(errors,95)) if errors else None,
                            position_max_m=max(errors) if errors else None,
                            armed_samples=len(armed_errors),
                            armed_position_rmse_m=float(np.sqrt(np.mean(np.square(armed_errors)))) if armed_errors else None,
                            armed_position_p95_m=float(np.percentile(armed_errors,95)) if armed_errors else None,
                            timestamp_alignment='Linear interpolation at measurement time; gaps >100 ms rejected')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
