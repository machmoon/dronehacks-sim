"""Bounded offline action-delay probe for a frozen x500 policy checkpoint."""
import argparse
from collections import deque
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

STARTED = time.monotonic()

import numpy as np
import torch

from learning import transfer
from learning.policy import DepthPolicy


def isolated_transfer():
    # Only this private module's execution step is wrapped. The trainer's
    # import and the teacher's candidate plant stay unchanged.
    spec = importlib.util.spec_from_file_location('learning._latency_probe_transfer', transfer.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flight_type(module, waypoint_handoff):
    class DelayedFlight(module.TransferFlight):
        def __init__(self, course, delay_s):
            super().__init__(course, seed=0, difficulty=0., response=.66)
            self.delay_steps = round(delay_s/module.SUBSTEP_S)
            self.targets = deque(np.zeros(3) for _ in range(self.delay_steps))
            self.segment = self.index
            self.queue_resets = 0

        def step(self, action):
            if self.index != self.segment:
                self.segment = self.index
                if waypoint_handoff == 'cancel':
                    self.targets.clear()
                    self.targets.extend(np.zeros(3) for _ in range(self.delay_steps))
                    self.queue_resets += 1
            if not self.delay_steps:
                return super().step(action)
            advance = module.plant_substep

            def delayed(motion, target, cap, response_s, wind, yaw_target):
                self.targets.append(target)
                return advance(motion, self.targets.popleft(), cap, response_s, wind, yaw_target)

            module.plant_substep = delayed
            try:
                return super().step(action)
            finally:
                module.plant_substep = advance

    return DelayedFlight


def load_snapshot(path, course):
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    weights = torch.load(io.BytesIO(contents), map_location='cpu', weights_only=True)
    profile = weights.get('transfer_profile', {})
    expected = course.profile_metadata()
    if (weights.get('format') != 1 or weights.get('architecture') != 'DepthPolicy'
            or weights.get('selection_split') != 'validation' or weights.get('profile') != 'x500-transfer'
            or profile != expected or weights.get('route_sha256') != expected['route_sha256']
            or weights.get('heading_sha256') != hashlib.sha256(
                (course.route_path.parent/'heading.py').read_bytes()).hexdigest()):
        raise ValueError('checkpoint and local x500 transfer profile differ')
    model = DepthPolicy().eval()
    model.load_state_dict(weights['model'], strict=True)
    return model, dict(sha256=digest, steps=weights['steps'], samples=weights['samples'],
                       path=str(path.resolve()), profile=profile['name'])


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def reason(result, timed_out):
    if timed_out:
        return 'probe_wall_timeout'
    if result['success']:
        return 'route_complete'
    if result['collision']:
        return 'uncertified_airframe_overlap'
    if result['bound_violation']:
        return 'mission_boundary'
    if result['estimated_waypoints_reached'] < 14:
        return 'episode_step_limit'
    if result['route_progress'] < 1.:
        return 'missed_verification_waypoint'
    return 'aperture_clearance'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=165.)
    parser.add_argument('--waypoint-handoff', choices=('fifo', 'cancel'), default='fifo')
    parser.add_argument('--delays', type=float, nargs='+', default=[0., .12, .20])
    parser.add_argument('--actors', choices=('policy', 'teacher'), nargs='+', default=['policy', 'teacher'])
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 175.:
        parser.error('--max-wall-seconds must be in (0, 175]')
    deadline = STARTED+args.max_wall_seconds
    torch.set_num_threads(1)
    module = isolated_transfer()
    if any(not math.isfinite(delay) or not 0. <= delay <= 1.
           or not math.isclose(delay/module.SUBSTEP_S, round(delay/module.SUBSTEP_S), abs_tol=1e-8)
           for delay in args.delays):
        parser.error('--delays must be between 0 and 1 s in whole 40 ms plant steps')
    course = module.TransferCourse()
    model, checkpoint = load_snapshot(args.snapshot, course)
    delayed_flight = flight_type(module, args.waypoint_handoff)
    handoff = ('At each waypoint change, discard pending actions and request zero until a fresh delayed action arrives. '
               'The sent velocity retains its normal slew.' if args.waypoint_handoff == 'cancel'
               else 'Queued actions persist across waypoints in this isolated delay stress.')
    report = dict(checkpoint=checkpoint, seed=0, geometry_difficulty=0., plant_response_s=.66,
                  actor_step_s=module.sim.DT, plant_substep_s=module.SUBSTEP_S,
                  waypoint_handoff=args.waypoint_handoff,
                  delay_description='FIFO of world-frame actions; fresh yaw and unchanged actor observations. '
                                    'Teacher predicts its original immediate-actuation plant. '+handoff,
                  limitations='Fixed delays and ideal actor cadence; no actual camera, estimator, PX4, or response jitter.',
                  max_wall_seconds=args.max_wall_seconds, complete=False, episodes=[])
    write_report(args.output, report)
    for delay in args.delays:
        for actor in args.actors:
            if time.monotonic() >= deadline:
                return 1
            start = time.monotonic()
            flight = delayed_flight(course, delay)
            timed_out = False
            while not flight.done:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                obs = flight.observe(render=actor == 'policy')
                if actor == 'policy':
                    with torch.inference_mode():
                        action = model(torch.from_numpy(obs.depth)[None, None], torch.from_numpy(obs.state)[None])[0].numpy()
                else:
                    action = module.teacher(flight)
                flight.step(action)
            result = flight.result()
            row = dict(actor=actor, delay_s=delay, **result, reason=reason(result, timed_out),
                       verified_waypoints_reached=flight.truth_index, last_position_enu=flight.p.tolist(),
                       waypoint_queue_resets=flight.queue_resets, wall_seconds=round(time.monotonic()-start, 3))
            report['episodes'].append(row)
            write_report(args.output, report)
            print(json.dumps(row, allow_nan=False), flush=True)
            if timed_out:
                return 1
    report['complete'] = True
    report['wall_seconds'] = round(time.monotonic()-STARTED, 3)
    write_report(args.output, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
