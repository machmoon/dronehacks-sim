"""Probe physical velocity-command delay after slew in the frozen x500 transfer plant."""
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools.check_policy_latency import isolated_transfer, load_snapshot, reason, write_report


def delayed_flight_type(module):
    class DelayedFlight(module.TransferFlight):
        def __init__(self, course, seed, response_s, delay_s, gain):
            super().__init__(course, seed=seed, difficulty=0., sensor_stress=1., response=response_s)
            self.gain = gain
            self.issued = deque(np.zeros(3) for _ in range(round(delay_s/module.SUBSTEP_S)))

        def step(self, action):
            if not self.issued and self.gain == 1.:
                return super().step(action)
            advance = module.plant_substep

            def delayed(motion, target, cap, response_s, wind, yaw_target):
                sim = module.sim
                change = sim.clip_norm(target-motion.command, module.COMMAND_SLEW*module.SUBSTEP_S)
                issued = sim.clip_norm(motion.command+change, cap)
                self.issued.append(issued)
                applied = self.issued.popleft()*self.gain
                # Pass the already-slewed delayed command through the original
                # dynamics, bypassing a second slew. Keep the issued command
                # as the memory for the next upstream slew calculation.
                advanced = advance(replace(motion, command=applied), applied, math.inf,
                                   response_s, wind, yaw_target)
                return replace(advanced, command=issued)

            module.plant_substep = delayed
            try:
                return super().step(action)
            finally:
                module.plant_substep = advance

    return DelayedFlight


def run_case(flight, module, model, deadline):
    started = time.monotonic()
    with torch.inference_mode():
        while not flight.done and time.monotonic() < deadline:
            observation = flight.observe(render=model is not None)
            if model is None:
                action = module.teacher(flight)
            else:
                depth = torch.from_numpy(observation.depth)[None, None]
                state = torch.from_numpy(observation.state)[None]
                action = model(depth, state)[0].numpy()
            flight.step(action)
    timed_out = not flight.done
    result = flight.result()
    return dict(**result, reason=reason(result, timed_out), verified_waypoints_reached=flight.truth_index,
                last_position_enu=flight.p.tolist(), last_velocity_enu=flight.v.tolist(),
                truth_min_center_boundary_gap_m=float(flight.course.bounds_gap(np.asarray(flight.path)).min()),
                wall_seconds=round(time.monotonic()-started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2100000704)
    parser.add_argument('--responses', type=float, nargs='+', default=[.66, .50, .80])
    parser.add_argument('--delays', type=float, nargs='+', default=[.12, .20, .24])
    parser.add_argument('--gain', type=float, default=1.05, help='extra case at .20 s delay and .66 s response')
    parser.add_argument('--skip-teacher', action='store_true')
    parser.add_argument('--max-wall-seconds', type=float, default=165.)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 175.:
        parser.error('--max-wall-seconds must be in (0, 175]')
    if not math.isfinite(args.gain) or not 1. <= args.gain <= 1.2:
        parser.error('--gain must be in [1, 1.2]')
    if args.output.exists():
        parser.error('--output must be a new path')
    deadline = time.monotonic()+args.max_wall_seconds
    torch.set_num_threads(1)
    module = isolated_transfer()
    if any(not module.RESPONSE_RANGE_S[0] <= response <= module.RESPONSE_RANGE_S[1] for response in args.responses):
        parser.error('--responses must be inside the frozen training range')
    if any(not 0. < delay <= 1. or not math.isclose(delay/module.SUBSTEP_S, round(delay/module.SUBSTEP_S), abs_tol=1e-8)
           for delay in args.delays):
        parser.error('--delays must be positive whole 40 ms steps up to 1 s')
    course = module.TransferCourse()
    flight_type = delayed_flight_type(module)
    snapshots = [load_snapshot(path, course) for path in args.snapshot]
    policy_cases = [(0., .66, 1.), *((delay, response, 1.) for response in args.responses for delay in args.delays)]
    if args.gain != 1.:
        policy_cases.insert(1, (.20, .66, args.gain))
    cases = [(model, checkpoint['steps'], delay, response, gain)
             for delay, response, gain in policy_cases for model, checkpoint in snapshots]
    if not args.skip_teacher:
        cases[2*len(snapshots):2*len(snapshots)] = [(None, None, 0., .66, 1.), (None, None, .20, .66, 1.)]

    report = {
        'checkpoints': [checkpoint for _, checkpoint in snapshots],
        'transfer_profile': course.profile_metadata(),
        'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': args.seed, 'geometry_difficulty': 0., 'sensor_stress': 1.,
        'command_delay_model': ('FIFO of world-frame velocity setpoints after the ordinary slew and waypoint speed '
                                'cap. Queue persists through waypoint handoffs. Optional gain multiplies the delayed '
                                'setpoint before the ordinary acceleration and attitude response.'),
        'evaluation': ('Frozen TransferFlight bounds, independent simulated waypoint radii, continuous multipart '
                       'component checks and full multipart window crossing.'),
        'limitations': ('Nominal seed and analytic depth. Actor timing is synchronous at 160 ms, and no inference or '
                        'transport delay is injected. The stock teacher predicts its immediate-command plant. '
                        'A fixed lag and gain stress test does not identify general PX4 dynamics.'),
        'max_wall_seconds': args.max_wall_seconds, 'planned_cases': len(cases), 'complete': False, 'episodes': [],
    }
    write_report(args.output, report)
    for model, step, delay, response, gain in cases:
        if time.monotonic() >= deadline:
            return 1
        flight = flight_type(course, args.seed, response, delay, gain)
        result = run_case(flight, module, model, deadline)
        row = dict(actor='policy' if model is not None else 'teacher', checkpoint_steps=step,
                   command_delay_s=delay, response_s=response, velocity_gain=gain, **result)
        report['episodes'].append(row)
        write_report(args.output, report)
        print(json.dumps(row, allow_nan=False), flush=True)
        if row['reason'] == 'probe_wall_timeout':
            return 1
    report['complete'] = True
    write_report(args.output, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
