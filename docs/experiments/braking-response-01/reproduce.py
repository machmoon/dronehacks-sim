#!/usr/bin/env python3
"""Reproduce the descriptive setpoint-to-PX4-velocity fit for policy runs 05/07/08/09.

No dependencies beyond Python's standard library. Commands are time-stamped at the
telemetry sample's `sim_time`; the PX4 velocity observation uses its own
`position_stamp_sim_s`. For each output sample, the delayed input is the latest
recorded command whose telemetry timestamp is <= output_stamp - delay. This is a
closed-loop trace fit, not an airframe identification or safety proof.
"""
from __future__ import annotations
import bisect
import hashlib
import json
import math
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'summary.json'
RUNS = ('policy-05', 'policy-07', 'policy-08', 'policy-09')
ROUTE_PHASES = {
    'FIRST_CORRIDOR', 'MAZE', 'GATE_A', 'TUNNEL_ALIGN', 'TUNNEL', 'CHICANE',
    'WINDOW_APPROACH', 'WINDOW_ALIGN', 'WINDOW_CROSS',
}
MAX_OUTPUT_GAP_S = 0.2
TAUS = [n / 100.0 for n in range(8, 161, 4)]
DELAYS = [n / 100.0 for n in range(0, 51, 2)]


def read_run(name):
    path = ROOT / 'artifacts' / 'markerfree' / name / 'telemetry.jsonl'
    records = [json.loads(line) for line in path.open()]
    samples = [r for r in records if r.get('type') == 'sample']
    # Input times are the telemetry log times of the already-sent MAVLink
    # velocity setpoint. No future record can satisfy a delayed lookup.
    commands = []
    for r in samples:
        t = r.get('sim_time')
        u = r.get('command_velocity')
        if isinstance(t, (int, float)) and isinstance(u, list) and len(u) >= 2:
            commands.append((float(t), tuple(float(v) for v in u[:2])))
    commands.sort(key=lambda item: item[0])
    cmd_times = [x[0] for x in commands]

    outputs = []
    segment = 0
    in_selected_run = False
    for r in samples:
        selected = (r.get('phase') in ROUTE_PHASES and bool(r.get('policy_actions_received', 0))
                    and isinstance(r.get('position_stamp_sim_s'), (int, float))
                    and isinstance(r.get('velocity'), list) and len(r['velocity']) >= 2)
        if not selected:
            in_selected_run = False
            continue
        if not in_selected_run:
            segment += 1
            in_selected_run = True
        t = float(r['position_stamp_sim_s'])
        y = tuple(float(v) for v in r['velocity'][:2])
        outputs.append((t, y, r, segment))
    return path, records, commands, cmd_times, outputs


def delayed_command(commands, cmd_times, when):
    # bisect_right(...)-1 chooses the newest command at or before `when`.
    # Commands with timestamps after `when` are never used.
    idx = bisect.bisect_right(cmd_times, when) - 1
    return commands[idx][1] if idx >= 0 else None


def integrate_interval(y0, t0, t1, tau, delay, commands, cmd_times):
    """Exact first-order response over ZOH input changes shifted by delay."""
    idx = bisect.bisect_right(cmd_times, t0 - delay) - 1
    if idx < 0:
        return None, 0
    pred = y0
    cursor = t0
    held = commands[idx][1]
    boundaries = 0
    next_idx = idx + 1
    # A command logged at c becomes an effective input change at c + delay.
    # Stop at t1: later command records cannot affect this output interval.
    while next_idx < len(commands):
        boundary = cmd_times[next_idx] + delay
        if boundary > t1:
            break
        if boundary > cursor:
            dt = boundary - cursor
            a = math.exp(-dt / tau)
            pred = (a * pred[0] + (1.0 - a) * held[0],
                    a * pred[1] + (1.0 - a) * held[1])
            cursor = boundary
        held = commands[next_idx][1]
        boundaries += 1
        next_idx += 1
    if t1 > cursor:
        dt = t1 - cursor
        a = math.exp(-dt / tau)
        pred = (a * pred[0] + (1.0 - a) * held[0],
                a * pred[1] + (1.0 - a) * held[1])
    return pred, boundaries


def self_check():
    commands = [(-1.0, (0.0, 0.0)), (0.5, (1.0, 1.0)), (1.2, (9.0, 9.0))]
    times = [item[0] for item in commands]
    no_delay, n0 = integrate_interval((0.0, 0.0), 0.0, 1.0, 1.0, 0.0, commands, times)
    delayed, nd = integrate_interval((0.0, 0.0), 0.0, 1.0, 1.0, 0.2, commands, times)
    expected_no_delay = 1.0 - math.exp(-0.5)
    expected_delayed = 1.0 - math.exp(-0.3)
    assert math.isclose(no_delay[0], expected_no_delay, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(delayed[0], expected_delayed, rel_tol=0.0, abs_tol=1e-12)
    assert n0 == 1 and nd == 1
    return {'passed': True,
            'u0_zero_delay': {'expected': expected_no_delay, 'actual': no_delay[0]},
            'u0_delay_0p2_s': {'expected': expected_delayed, 'actual': delayed[0]},
            'future_command_at_1p2_s_ignored': True}


def fit(outputs, commands, cmd_times, tau, delay):
    errors = []
    segment_starts = 0
    boundary_crossings = 0
    pred = None
    prev_t = None
    prev_segment = None
    for t1, y1, _record, segment in outputs:
        if (prev_t is None or segment != prev_segment or t1 <= prev_t
                or t1 - prev_t > MAX_OUTPUT_GAP_S):
            # Initialize each contiguous selected segment from its first actual
            # PX4 velocity sample; never bridge an excluded phase or large gap.
            pred = y1
            prev_t, prev_segment = t1, segment
            segment_starts += 1
            continue
        pred_next, crossings = integrate_interval(
            pred, prev_t, t1, tau, delay, commands, cmd_times)
        if pred_next is None:
            # No command history existed at interval start. Reset at this
            # observed velocity instead of filling the missing input with a
            # later command.
            pred = y1
            prev_t, prev_segment = t1, segment
            segment_starts += 1
            continue
        pred = pred_next
        boundary_crossings += crossings
        errors.extend((pred[0] - y1[0], pred[1] - y1[1]))
        prev_t, prev_segment = t1, segment
    if not errors:
        return None, 0, segment_starts, boundary_crossings
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return rmse, len(errors) // 2, segment_starts, boundary_crossings


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values) - 1) * q)]


def run_analysis(name):
    path, records, commands, cmd_times, outputs = read_run(name)
    best = None
    for tau in TAUS:
        for delay in DELAYS:
            rmse, intervals, resets, crossings = fit(outputs, commands, cmd_times, tau, delay)
            if rmse is not None and (best is None or rmse < best[0]):
                best = (rmse, tau, delay, intervals, resets, crossings)
    fixed = {}
    for key, tau, delay in (('configured_0p8_0p4', .8, .4), ('tau_0p8_zero_delay', .8, 0.0)):
        rmse, intervals, resets, crossings = fit(outputs, commands, cmd_times, tau, delay)
        fixed[key] = {'tau_s': tau, 'delay_s': delay, 'rmse_mps': rmse,
                      'intervals': intervals, 'segment_resets': resets,
                      'integrated_command_boundaries': crossings}
    metadata = next(r for r in records if r.get('type') == 'metadata')
    recorded_sources = metadata.get('runtime_source_sha256', {})
    actions = [r for r in records if r.get('type') == 'policy_action']
    def values(field):
        return [float(r[field]) for r in actions
                if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
    skew = [float(r['sim_time']) - float(r['position_stamp_sim_s'])
            for r in records if r.get('type') == 'sample'
            and isinstance(r.get('sim_time'), (int, float))
            and isinstance(r.get('position_stamp_sim_s'), (int, float))]
    return {
        'input': {'path': str(path.relative_to(ROOT)),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'recorded_runtime_source_sha256': {key: recorded_sources.get(key)
                                            for key in ('mission/markerfree_run.py',
                                                        'mission/boundary_brake.py')},
        'fit': {'selected_output_samples': len(outputs),
                'selected_command_samples': len(commands),
                'best_grid': {'tau_s': best[1], 'delay_s': best[2], 'rmse_mps': best[0],
                              'intervals': best[3], 'segment_resets': best[4],
                              'integrated_command_boundaries': best[5]},
                **fixed,
                'command_log_time_minus_px4_position_stamp_s': {
                    'median_abs_s': quantile(sorted(abs(x) for x in skew), .5),
                    'p95_abs_s': quantile(sorted(abs(x) for x in skew), .95),
                    'max_abs_s': max((abs(x) for x in skew), default=None)},
                'max_output_gap_s': MAX_OUTPUT_GAP_S,
                'grid': {'tau_min_s': TAUS[0], 'tau_max_s': TAUS[-1], 'tau_step_s': .04,
                         'delay_min_s': DELAYS[0], 'delay_max_s': DELAYS[-1], 'delay_step_s': .02}},
        'telemetry_timing': {
            'capture_age_sim_s': {'p50': quantile(values('capture_age_sim_s'), .5),
                                  'p95': quantile(values('capture_age_sim_s'), .95),
                                  'max': max(values('capture_age_sim_s'), default=None)},
            'capture_interval_sim_s': {'p50': quantile(values('capture_interval_sim_s'), .5),
                                       'p95': quantile(values('capture_interval_sim_s'), .95),
                                       'max': max(values('capture_interval_sim_s'), default=None)},
            'policy_latency_wall_s': {'p50': quantile(values('latency_wall_s'), .5),
                                      'p95': quantile(values('latency_wall_s'), .95),
                                      'max': max(values('latency_wall_s'), default=None)},
        },
    }


def main():
    result = {
        'schema_version': 3,
        'python': platform.python_version(),
        'method': {
            'output_clock': 'PX4 telemetry position_stamp_sim_s for velocity samples',
            'input_clock': 'telemetry sim_time when command_velocity (already-sent ENU setpoint) was logged; actual MAVLink transmission timestamp is not recorded',
            'input_selection': 'exact zero-order hold integration across command boundaries shifted by candidate_delay; starting input is latest logged command <= segment_start - delay; no command effective after the output time is used',
            'state_alignment': 'recursive first-order state is initialized from the first PX4 velocity sample of each contiguous selected segment; excluded phases, nonmonotonic stamps, and gaps >0.2s start a new segment and are not bridged',
            'selection': 'policy-action route phases only; takeoff/hover/wait/landing excluded; XY ENU only',
            'model': 'exact piecewise integration of first-order velocity response over delayed zero-order-held command changes',
            'self_check': self_check(),
            'fit_scope': 'descriptive closed-loop telemetry fit; PX4 velocity-control loop, policy feedback, estimator and setpoint slew are included; not a physical plant identification or worst-case stopping proof',
        },
        'runs': {name: run_analysis(name) for name in RUNS},
    }
    result['analysis_script'] = {'path': str(Path(__file__).relative_to(ROOT)),
                                'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for relative in ('mission/markerfree_run.py', 'mission/boundary_brake.py'):
        source = ROOT / relative
        result.setdefault('current_source_sha256', {})[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(OUT)
    for name, run in result['runs'].items():
        fit = run['fit']
        print(name, 'best', fit['best_grid'], 'configured', fit['configured_0p8_0p4'],
              'capture-age-p95', run['telemetry_timing']['capture_age_sim_s']['p95'])


if __name__ == '__main__':
    main()
