"""Bounded offline check of waypoint tracking under lagged velocity/yaw response.

This is a diagnostic plant approximation, not PX4 or hardware validation.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path

import numpy as np

from learning.sim import Course, clip_norm
from mission.airframe import (check_scene_path, evaluate_window_crossing,
                              load_x500_envelope, quaternion_xyzw_from_rpy)
from mission.core import CourseController, wrap
from mission.heading import route_yaw


DT = .04
GEOMETRY_SEED = 0
ENVELOPE_MARGIN_M = .015


def simulate(plan, response_s, yaw_response_s=.35, lookahead_m=1.8, *, command_delay_s=0.,
             velocity_gain_xy=1., scene=None, envelope=None):
    delay_steps = round(command_delay_s/DT)
    if delay_steps < 0 or not math.isclose(command_delay_s, delay_steps*DT, abs_tol=1e-9):
        raise ValueError('command delay must be a nonnegative multiple of the plant step')
    if any(not math.isfinite(x) or x <= 0 for x in (response_s, yaw_response_s, velocity_gain_xy)):
        raise ValueError('response times and horizontal velocity gain must be positive and finite')
    controller = CourseController(dict(plan, waypoints=plan['waypoints'][1:]))
    position = np.array(plan['waypoints'][0]['position'], float)
    velocity = np.zeros(3)
    yaw = command_yaw = math.pi/2
    dt, time_s = DT, 0.
    issued = deque(np.zeros(3) for _ in range(delay_steps))
    gain = np.array([velocity_gain_xy, velocity_gain_xy, 1.])
    positions, attitudes, phases = [position.copy()], [quaternion_xyzw_from_rpy(0, 0, yaw)], []
    waypoints = plan['waypoints'][1:]
    waypoint_positions = np.array([item['position'] for item in waypoints])
    verification_radii = [item.get('verification_radius', item.get('arrival_radius', controller.arrival_radius))
                          for item in waypoints]
    verified_arrivals, controller_arrivals = [], []
    done, fault = False, None
    while time_s < 240:
        truth_index = len(verified_arrivals)
        if truth_index < len(waypoints):
            distance = np.linalg.norm(position-waypoint_positions[truth_index])
            if distance <= verification_radii[truth_index]:
                verified_arrivals.append(round(time_s, 6))
        previous_index = controller.waypoint_index
        command = controller.update(position, velocity, time_s)
        if controller.waypoint_index > previous_index:
            controller_arrivals.append(round(time_s, 6))
        if command.fault or command.done:
            done, fault = command.done, command.fault
            break
        index = controller.waypoint_index
        following = controller.waypoints[index+1]['position'] if index+1 < len(controller.waypoints) else None
        desired_yaw = route_yaw(position, command.target, following, command.phase, lookahead_m)
        if desired_yaw is not None:
            step = controller.max_yaw_rate*dt
            command_yaw = wrap(command_yaw + np.clip(wrap(desired_yaw-command_yaw), -step, step))
        yaw = wrap(yaw + np.clip(wrap(command_yaw-yaw)/yaw_response_s, -1.6, 1.6)*dt)
        # The real controller already slews its output. Delay that velocity
        # setpoint before the plant without applying another slew or cap.
        issued.append(np.asarray(command.velocity))
        applied = issued.popleft()*gain
        acceleration = clip_norm((applied-velocity)/response_s, .8)
        next_velocity = velocity + dt*acceleration
        position = position + dt*(velocity+next_velocity)/2
        velocity = next_velocity
        # Approximate the thrust tilt needed for this horizontal acceleration.
        forward = math.cos(yaw)*acceleration[0] + math.sin(yaw)*acceleration[1]
        left = -math.sin(yaw)*acceleration[0] + math.cos(yaw)*acceleration[1]
        roll, pitch = -math.atan2(left, 9.81), math.atan2(forward, 9.81)
        positions.append(position.copy())
        attitudes.append(quaternion_xyzw_from_rpy(roll, pitch, yaw))
        phases.append(command.phase)
        time_s += dt
    if scene is None:
        scene = Course().scene(np.random.default_rng(GEOMETRY_SEED), 0.)
    if envelope is None:
        envelope = load_x500_envelope(margin_m=ENVELOPE_MARGIN_M)
    geometry = check_scene_path(positions, attitudes, scene, envelope)
    if geometry['first_unresolved']:
        index = geometry['first_unresolved']['segment']
        geometry['first_unresolved'].update(phase=phases[index], position_enu=positions[index].tolist())
    window = evaluate_window_crossing(positions, attitudes, plan['window'], envelope)
    center_gap = None
    if controller.bounds:
        radius = controller.footprint_radius
        lower = np.asarray(controller.bound_min) + [radius, radius, -.05]
        upper = np.asarray(controller.bound_max) - radius
        points = np.asarray(positions)
        center_gap = float(np.minimum(points-lower, upper-points).min())
    return dict(response_s=response_s, yaw_response_s=yaw_response_s, heading_lookahead_m=lookahead_m,
                command_delay_s=command_delay_s, velocity_gain_xy=velocity_gain_xy,
                completed=done, fault=fault, elapsed_sim_s=round(time_s, 6),
                waypoints_reached=controller.waypoint_index, truth_waypoints_reached=len(verified_arrivals),
                truth_waypoint_arrivals_sim_s=verified_arrivals, controller_waypoint_arrivals_sim_s=controller_arrivals,
                min_center_boundary_gap_m=center_gap,
                geometry=geometry, window=window,
                limitation='Approximate velocity and yaw dynamics with exact simulated state feedback. '
                           'The path starts at takeoff hover and ends at landing handoff.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', type=Path, default=Path('mission/course-window-x500.json'))
    parser.add_argument('--response', type=float, nargs='+', default=[.5, .66, .8])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in args.response):
        parser.error('response times must be positive and finite')
    plan = json.loads(args.course.read_text())
    results = [simulate(plan, value) for value in args.response]
    text = json.dumps(results, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
