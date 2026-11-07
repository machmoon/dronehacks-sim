"""Compare section speed caps on the same bounded, delayed waypoint plant."""
import argparse
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import time

import numpy as np

from learning import sim
from mission import airframe, core, heading
from tools import check_classical_transfer as transfer


ROOT = Path(__file__).resolve().parents[1]
DELAYS_S = (.12, .20, .24)
RESPONSES_S = (.50, .66, .80)
GAINS_XY = (1., 1.08)
NOMINAL = (.20, .66, 1.)
LOOKAHEAD_M = 1.8
YAW_RESPONSE_S = .35
SECTIONS = {
    'maze': {'MAZE': .55},
    'window': {'WINDOW_ALIGN': .23, 'WINDOW_CROSS': .32},
    'straights': {'FIRST_CORRIDOR': .54, 'TUNNEL': .93, 'WINDOW_APPROACH': .42,
                  'LANDING_APPROACH': .71},
}


def candidate_plan(plan, sections):
    candidate = deepcopy(plan)
    caps = {phase: speed for section in sections for phase, speed in SECTIONS[section].items()}
    for waypoint in candidate['waypoints']:
        if waypoint['phase'] in caps:
            waypoint['speed'] = caps[waypoint['phase']]
    return candidate


def passed(result, waypoint_count):
    return (result['completed'] and result['fault'] is None
            and result['waypoints_reached'] == result['truth_waypoints_reached'] == waypoint_count
            and result['min_center_boundary_gap_m'] is not None and result['min_center_boundary_gap_m'] >= 0.
            and result['geometry']['certified_clear'] and result['window']['full_envelope_crossed'])


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def compare(plan, course_path, output, deadline):
    cases = [NOMINAL] + [case for case in product(DELAYS_S, RESPONSES_S, GAINS_XY) if case != NOMINAL]
    waypoint_count = len(plan['waypoints'])-1
    sections = list(SECTIONS)
    candidates = {'baseline': plan}
    for size in range(1, len(sections)+1):
        for selection in combinations(sections, size):
            candidates['+'.join(selection)] = candidate_plan(plan, selection)
    course = sim.Course()
    scene = course.scene(np.random.default_rng(transfer.GEOMETRY_SEED), 0.)
    envelope = airframe.load_x500_envelope(margin_m=transfer.ENVELOPE_MARGIN_M)
    paths = [course_path, Path(__file__), Path(transfer.__file__), Path(sim.__file__), Path(core.__file__),
             Path(heading.__file__), Path(airframe.__file__), airframe.X500_SDF, course.groups_path,
             course.route_path, *(course.mesh_dir/group['mesh'] for group in course.groups['groups']
                                  if group['collidable'])]
    report = {
        'profile': 'classical-waypoint-x500-post-slew-delay',
        'source_sha256_by_path': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'seed': transfer.GEOMETRY_SEED, 'geometry_difficulty': 0.,
        'limits': plan['limits'], 'bounds': plan['bounds'], 'footprint_radius_m': plan['footprint_radius'],
        'component_margin_m': transfer.ENVELOPE_MARGIN_M,
        'matrix_axes': {'command_delay_s': DELAYS_S, 'velocity_response_s': RESPONSES_S, 'horizontal_velocity_gain': GAINS_XY},
        'nominal': {'command_delay_s': NOMINAL[0], 'velocity_response_s': NOMINAL[1], 'horizontal_velocity_gain': NOMINAL[2]},
        'integration_step_s': transfer.DT, 'yaw_response_s': YAW_RESPONSE_S, 'heading_lookahead_m': LOOKAHEAD_M,
        'waypoint_count_after_hover': waypoint_count,
        'section_cap_options_mps': SECTIONS,
        'scope': ('The shared PD controller and its existing slew issue a velocity setpoint. A FIFO delays it '
                  'before an approximate velocity plant with horizontal gain. Identical simulated state feedback, '
                  'CAD proxies, and multipart checks apply to every case. Elapsed times start at settled hover and '
                  'end at the landing handoff. This finite nominal-geometry matrix gives no guarantee for PX4, '
                  'camera observability, real hardware, or independent touchdown.'),
        'selection_rule': ('Require all baseline stress cases to pass. Rank faster candidates by nominal elapsed '
                           'time, then select the first candidate that passes all 18 stress cases.'),
        'candidates': {name: {'cases': [], 'matrix_passed': None} for name in candidates},
        'selected': None, 'complete': False,
    }

    def run(name, case):
        if time.monotonic() >= deadline:
            return None
        delay, response, gain = case
        result = transfer.simulate(candidates[name], response, YAW_RESPONSE_S, LOOKAHEAD_M,
                                   command_delay_s=delay, velocity_gain_xy=gain, scene=scene, envelope=envelope)
        result['passed'] = passed(result, waypoint_count)
        report['candidates'][name]['cases'].append(result)
        print(f"{name} delay={delay:g} response={response:g} gain={gain:g} "
              f"passed={result['passed']} elapsed={result['elapsed_sim_s']:.2f}s", flush=True)
        write_report(output, report)
        return result

    def matrix(name, pending):
        for case in pending:
            if run(name, case) is None:
                return None
        accepted = all(result['passed'] for result in report['candidates'][name]['cases'])
        report['candidates'][name]['matrix_passed'] = accepted
        return accepted

    write_report(output, report)
    baseline_passed = matrix('baseline', cases)
    if baseline_passed is None:
        return report, None
    baseline = report['candidates']['baseline']['cases'][0]
    faster = []
    for name in candidates:
        if name == 'baseline':
            continue
        trial = run(name, NOMINAL)
        if trial is None:
            return report, None
        if trial['passed'] and trial['elapsed_sim_s'] < baseline['elapsed_sim_s']:
            faster.append((trial['elapsed_sim_s'], name))
    selected = None
    if baseline_passed:
        for elapsed, name in sorted(faster):
            qualification = matrix(name, cases[1:])
            if qualification is None:
                return report, None
            if qualification:
                selected = candidates[name]
                baseline_by_case = report['candidates']['baseline']['cases']
                chosen_by_case = report['candidates'][name]['cases']
                report['selected'] = {
                    'name': name, 'nominal_sim_s': elapsed, 'baseline_nominal_sim_s': baseline['elapsed_sim_s'],
                    'nominal_savings_s': round(baseline['elapsed_sim_s']-elapsed, 6),
                    'same_case_savings_range_s': [round(extreme(before['elapsed_sim_s']-after['elapsed_sim_s']
                                                               for before, after in zip(baseline_by_case, chosen_by_case)), 6)
                                                  for extreme in (min, max)],
                    'minimum_boundary_gap_m': min(case['min_center_boundary_gap_m'] for case in chosen_by_case),
                    'minimum_window_clearance_m': min(case['window']['min_aperture_clearance_m'] for case in chosen_by_case),
                }
                break
    report['complete'] = True
    report['outcome'] = ('qualified_candidate' if selected else
                         'baseline_failed_stress' if not baseline_passed else 'no_faster_qualified_candidate')
    return report, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', type=Path, default=ROOT/'mission/course-window-x500.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tuned-course', type=Path, help='new route path; written only after matrix qualification')
    parser.add_argument('--max-wall-seconds', type=float, default=270.)
    args = parser.parse_args()
    if not 0. < args.max_wall_seconds <= 300.:
        parser.error('--max-wall-seconds must be in (0, 300]')
    if args.output.exists() or (args.tuned_course is not None and args.tuned_course.exists()):
        parser.error('output paths must be new')
    if args.tuned_course is not None and args.output.resolve() == args.tuned_course.resolve():
        parser.error('report and tuned route must use different paths')
    plan = json.loads(args.course.read_text())
    started = time.monotonic()
    report, selected = compare(plan, args.course, args.output, started+args.max_wall_seconds)
    report['wall_seconds'] = round(time.monotonic()-started, 3)
    if selected is not None and args.tuned_course is not None:
        route = json.dumps(selected, indent=2, allow_nan=False)+'\n'
        with args.tuned_course.open('x') as output:
            output.write(route)
        report['selected']['route_path'] = str(args.tuned_course.resolve())
        report['selected']['route_sha256'] = hashlib.sha256(route.encode()).hexdigest()
    write_report(args.output, report)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
