"""Read-only analytic teacher experiment for the stock x500 transfer gap.

Run as python -m tools.check_transfer_teacher. This uses the current analytic
course, changes only local process state, and never starts PX4 or training.
"""
import argparse
from dataclasses import replace
import json

import numpy as np

from learning import sim


def advance_with_slew(position, velocity, command, target, response, wind, cap,
                      interval=sim.DT, trace=None):
    """Integrate a held target through the setpoint ramp and plant response."""
    dt = interval / 4
    for _ in range(4):
        command = sim.clip_norm(command + sim.clip_norm(target-command, .8*dt), cap)
        acceleration = sim.clip_norm((command-velocity)/response, sim.ACCEL) + wind
        next_velocity = velocity + dt*acceleration
        position = position + (velocity+next_velocity)*(dt/2)
        velocity = next_velocity
        if trace is not None:
            trace.append(position)
    return position, velocity, acceleration, command


def with_advance(advance, function, *args):
    original = sim.advance
    sim.advance = advance
    try:
        return function(*args)
    finally:
        sim.advance = original


class TransferFlight(sim.Flight):
    def __init__(self, course, seed, difficulty, response):
        super().__init__(course, seed, difficulty)
        self.response = response
        self.command = np.zeros(3)

    def step(self, action):
        cap = self.course.speeds[self.index]
        radius = sim.GOAL_RADIUS
        last_position = self.p
        trace = []

        def advance(position, velocity, target, response, wind):
            p, v, a, self.command = advance_with_slew(
                position, velocity, self.command, target, response, wind, cap, trace=trace)
            return p, v, a

        sim.GOAL_RADIUS = self.course.arrival_radii[self.index]
        try:
            with_advance(advance, super().step, action)
        finally:
            sim.GOAL_RADIUS = radius
        self.path[-1:] = trace
        self.min_clearance = min(self.min_clearance, float(self.scene.clearance(trace).min()))
        self.collision = any(self.scene.hits(a, b) for a, b in zip([last_position]+trace, trace))


def slew_aware_teacher(flight, horizon_s=.8, reserve_m=0.):
    command = flight.command
    cap = flight.course.speeds[flight.index]

    def advance(position, velocity, target, response, wind):
        nonlocal command
        p, v, a, command = advance_with_slew(position, velocity, command, target, response, wind, cap,
                                             interval=horizon_s/5)
        return p, v, a

    scene = flight.scene
    if reserve_m:
        flight.scene = replace(scene, radius=scene.radius+reserve_m)
    try:
        return with_advance(advance, sim.teacher, flight)
    finally:
        flight.scene = scene


class TransferCourse(sim.Course):
    def __init__(self, heading_lookahead, max_geometry_difficulty=1.):
        super().__init__(camera_forward_m=.34, heading_lookahead_m=heading_lookahead)
        self.radius = .36
        self.arrival_radii = [.08]*len(self.goals)
        self.max_geometry_difficulty = max_geometry_difficulty

    def scene(self, rng, difficulty):
        return super().scene(rng, min(difficulty, self.max_geometry_difficulty))

    def set_window_exit(self, point, arrival_radius, post_exit):
        index = self.phases.index('WINDOW_CROSS')
        center = np.asarray(self.window['center'])
        self.goals[index] = center + self.axes @ point
        self.arrival_radii[index] = arrival_radius
        if post_exit is not None:
            self.goals = np.insert(self.goals, index+1, center + self.axes @ post_exit, axis=0)
            self.speeds = np.insert(self.speeds, index+1, .30)
            self.phases.insert(index+1, 'WINDOW_EXIT')
            self.arrival_radii.insert(index+1, .08)


def run_case(course, seed, difficulty, response, teacher):
    flight = TransferFlight(course, seed, difficulty, response)
    minima = {}
    while not flight.done:
        index = flight.index
        flight.observe(render=False)
        flight.step(teacher(flight))
        phase = f'{index+1}:{course.phases[index]}'
        clearance = float(flight.scene.clearance(flight.path[-4:]).min())
        minima[phase] = min(minima.get(phase, float('inf')), clearance)
    return dict(**flight.result(), seed=seed, difficulty=difficulty, response_s=response,
                last_position_enu=flight.p.tolist(), completed_route_goals=flight.index,
                min_clearance_by_segment_m=minima)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response', type=float, nargs='+', default=[.50, .66, .80])
    parser.add_argument('--seed', type=int, default=2100000688)
    parser.add_argument('--difficulty', type=float, default=0.)
    parser.add_argument('--heading-lookahead', type=float, default=1.2)
    parser.add_argument('--teacher', choices=('original', 'slew'), default='slew')
    parser.add_argument('--window-exit-local', type=float, nargs=3,
                        metavar=('NORMAL', 'HORIZONTAL', 'UP'))
    parser.add_argument('--post-exit-local', type=float, nargs=3,
                        metavar=('NORMAL', 'HORIZONTAL', 'UP'))
    parser.add_argument('--exit-arrival-radius', type=float, default=.08)
    parser.add_argument('--speed-scale', type=float, default=1.)
    parser.add_argument('--maze-speed-scale', type=float, default=1.)
    parser.add_argument('--max-geometry-difficulty', type=float, default=1.)
    parser.add_argument('--teacher-horizon', type=float, default=.8)
    parser.add_argument('--teacher-reserve', type=float, default=0.)
    args = parser.parse_args()
    course = TransferCourse(args.heading_lookahead, args.max_geometry_difficulty)
    if args.window_exit_local is not None:
        course.set_window_exit(args.window_exit_local, args.exit_arrival_radius, args.post_exit_local)
    course.speeds *= args.speed_scale
    for index, phase in enumerate(course.phases):
        if phase in ('FIRST_CORRIDOR', 'MAZE', 'GATE_A'):
            course.speeds[index] *= args.maze_speed_scale

    def teacher(flight):
        if args.teacher == 'original':
            return sim.teacher(flight)
        return slew_aware_teacher(flight, args.teacher_horizon, args.teacher_reserve)

    for response in args.response:
        report = run_case(course, args.seed, args.difficulty, response, teacher)
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
