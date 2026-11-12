"""Analytic transfer profile for the centered x500 Gazebo course."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from mission.airframe import (X500_SDF, check_scene_path, evaluate_window_crossing,
                              face_clearance_at_poses, load_x500_envelope, quaternion_xyzw_from_rpy)
from mission.heading import route_yaw

from . import sim


PROFILE = 'x500-transfer-v1'
ROUTE = sim.ROOT / 'mission/course-window-x500.json'
CAMERA_FORWARD_M = .34
DEFAULT_LOOKAHEAD_M = 1.8
RESPONSE_RANGE_S = (.50, .80)
COMMAND_SLEW = .8
SUBSTEPS = 4
SUBSTEP_S = sim.DT / SUBSTEPS
YAW_RATE = 1.6
GEOMETRY_DIFFICULTY_MAX = .25
ENVELOPE_MARGIN_M = .015
INSTRUCTOR_STEPS = 7
INSTRUCTOR_RESERVE_M = .045


class InfeasibleScenario(ValueError):
    def __init__(self, seed, geometry_difficulty, detail):
        self.seed, self.geometry_difficulty, self.detail = seed, geometry_difficulty, detail
        super().__init__(f'uncertified reference route for seed={seed}, geometry={geometry_difficulty}: {detail}')


@dataclass(frozen=True)
class Motion:
    position: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    yaw: float
    roll: float | np.ndarray = 0.
    pitch: float | np.ndarray = 0.


def plant_substep(motion, target_enu, cap, response_s, wind, yaw_target):
    """Hold the action for 40 ms, applying the same setpoint slew in both rollouts."""
    change = sim.clip_norm(target_enu-motion.command, COMMAND_SLEW*SUBSTEP_S)
    command = sim.clip_norm(motion.command+change, cap)
    acceleration = sim.clip_norm((command-motion.velocity)/response_s, sim.ACCEL)+wind
    velocity = motion.velocity+SUBSTEP_S*acceleration
    position = motion.position+(motion.velocity+velocity)*(SUBSTEP_S/2)
    heading_error = math.remainder(yaw_target-motion.yaw, 2*math.pi)
    yaw = motion.yaw+float(np.clip(heading_error, -YAW_RATE*SUBSTEP_S, YAW_RATE*SUBSTEP_S))
    local_accel = acceleration @ sim.yaw_rotation(yaw)
    return Motion(position, velocity, command, yaw, -local_accel[..., 1]/9.81, local_accel[..., 0]/9.81)


def rotation_matrices(roll, pitch, yaw):
    """Batched version of sim.camera_rotation for candidate clearance."""
    roll, pitch, yaw = np.broadcast_arrays(roll, pitch, yaw)
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rows = (cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
            sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
            -sp, cp*sr, cp*cr)
    return np.stack(rows, axis=-1).reshape(*roll.shape, 3, 3)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TransferCourse(sim.Course):
    def __init__(self, *, route=ROUTE, heading_lookahead_m=DEFAULT_LOOKAHEAD_M, speed_scale=1.):
        super().__init__(camera_forward_m=CAMERA_FORWARD_M, heading_lookahead_m=heading_lookahead_m)
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 2:
            raise ValueError('speed_scale must be finite and in (0, 2]')
        self.route_path = Path(route).resolve()
        route_bytes = self.route_path.read_bytes()
        plan = json.loads(route_bytes)
        if plan['window'] != self.window:
            raise ValueError('transfer route must use the existing surveyed window geometry')
        self.route_sha256 = hashlib.sha256(route_bytes).hexdigest()
        self.speed_scale = speed_scale
        self.envelope = load_x500_envelope(margin_m=ENVELOPE_MARGIN_M)
        self.radius = float(plan['footprint_radius'])  # course boundary only
        self.max_body_radius = max(part.max_body_radius for part in self.envelope.components)
        launch, *waypoints = plan['waypoints']
        self.start = np.array(launch['position'], dtype=float)
        self.goals = np.array([w['position'] for w in waypoints], dtype=float)
        self.phases = [w['phase'] for w in waypoints]
        self.limits = plan['limits']
        self.speeds = np.minimum(self.limits['max_speed'], [speed_scale*w['speed'] for w in waypoints])
        max_arrival = self.limits['arrival_radius']
        self.estimated_arrivals = np.array([w.get('arrival_radius', max_arrival) for w in waypoints])
        self.truth_arrivals = np.array([w.get('verification_radius', arrival)
                                        for w, arrival in zip(waypoints, self.estimated_arrivals)])
        if not np.all((0 < self.estimated_arrivals) & (self.estimated_arrivals <= self.truth_arrivals)
                      & (self.truth_arrivals <= max_arrival)):
            raise ValueError('route radii must satisfy 0 < arrival_radius <= verification_radius <= limits.arrival_radius')
        self.lower = np.asarray(plan['bounds']['min']) + np.array([self.radius, self.radius, -.05])
        self.upper = np.asarray(plan['bounds']['max']) - self.radius
        self._geometry_cache = OrderedDict()
        self._reference = self._reference_path()
        if np.any(self.bounds_gap(self._reference[0]) < 0.):
            raise ValueError('reference route violates the mission boundary')
        self._profile = {
            'name': PROFILE, 'route_sha256': self.route_sha256, 'x500_sdf_sha256': _sha(X500_SDF),
            'transfer_source_sha256': _sha(Path(__file__)),
            'airframe_source_sha256': _sha(sim.ROOT/'mission/airframe.py'),
            'camera_forward_m': CAMERA_FORWARD_M, 'heading_lookahead_m': heading_lookahead_m,
            'speed_scale': speed_scale, 'envelope_margin_m': ENVELOPE_MARGIN_M,
            'envelope_model': 'SDF body and gear boxes, swept rotors; no sensor housing collider',
            'collision_result': 'Continuous conservative component sweeps; unresolved overlap counts as failure',
            'course_boundary_margin_m': self.radius,
            'estimated_waypoint_arrival_radii_m': self.estimated_arrivals.tolist(),
            'truth_waypoint_arrival_radii_m': self.truth_arrivals.tolist(),
            'velocity_response_s': list(RESPONSE_RANGE_S), 'plant_acceleration_mps2': sim.ACCEL,
            'velocity_setpoint_slew_mps2': COMMAND_SLEW, 'yaw_rate_rad_s': YAW_RATE,
            'actor_step_s': sim.DT, 'plant_substep_s': SUBSTEP_S, 'substeps_per_action': SUBSTEPS,
            'max_geometry_difficulty': GEOMETRY_DIFFICULTY_MAX,
            'instructor_horizon_s': INSTRUCTOR_STEPS*sim.DT,
            'instructor_reserve_m': INSTRUCTOR_RESERVE_M,
        }

    def profile_metadata(self):
        return dict(self._profile)

    def make_flight(self, seed, difficulty, sensor_stress=1., *, response=None):
        return TransferFlight(self, seed, difficulty, sensor_stress, response=response)

    def instructor(self, flight):
        return teacher(flight)

    def scene(self, rng, difficulty):
        return super().scene(rng, min(difficulty, GEOMETRY_DIFFICULTY_MAX))

    def bounds_gap(self, positions):
        return np.minimum(positions-self.lower, self.upper-positions).min(axis=-1)

    def _reference_path(self):
        positions, headings = [], []
        yaw = math.pi/2
        for index, (start, target) in enumerate(zip(np.vstack([self.start, self.goals[:-1]]), self.goals)):
            following = self.goals[index+1] if index+1 < len(self.goals) else None
            count = math.ceil(np.linalg.norm(target-start)/.035)+1
            for point in np.linspace(start, target, count):
                desired = route_yaw(point, target, following, self.phases[index], self.heading_lookahead_m)
                if desired is not None:
                    yaw = desired
                positions.append(point)
                headings.append(yaw)
        positions = np.asarray(positions)
        q = np.array([quaternion_xyzw_from_rpy(0., 0., yaw) for yaw in headings])
        rotations = rotation_matrices(0., 0., headings)
        return positions, q, rotations, _sweep_motion(positions, q, self.max_body_radius)

    def certify_scenario(self, scene, seed, difficulty):
        key = seed, min(difficulty, GEOMETRY_DIFFICULTY_MAX)
        if key not in self._geometry_cache:
            p, q, rotation, sweep = self._reference
            gap = face_clearance_at_poses(p, rotation, scene, self.envelope)
            result = None
            for index in _uncertain(gap, sweep):
                proof = check_scene_path(p[index:index+2], q[index:index+2], scene, self.envelope)
                if not proof['certified_clear']:
                    result = {**proof['first_unresolved'], 'segment': int(index)}
                    break
            if result is None:
                crossing = evaluate_window_crossing(p, q, scene.window, self.envelope)
                if not crossing['full_envelope_crossed']:
                    result = {'window': crossing['direction'], 'margin_m': crossing['min_aperture_clearance_m']}
            self._geometry_cache[key] = result
            if len(self._geometry_cache) > 128:
                self._geometry_cache.popitem(last=False)
        result = self._geometry_cache[key]
        if result is not None:
            raise InfeasibleScenario(*key, result)


def _sweep_motion(positions, q, max_radius):
    """Maximum distance to the nearest endpoint pose along each interpolation."""
    travel = np.linalg.norm(np.diff(positions, axis=0), axis=1)/2
    half_angle = np.arccos(np.abs(np.sum(q[1:]*q[:-1], axis=1)).clip(0., 1.))
    return travel + 2*max_radius*np.sin(half_angle/2)


def _uncertain(gaps, sweep):
    return np.flatnonzero(np.minimum(gaps[:-1], gaps[1:]) <= sweep+1e-9)


class TransferFlight(sim.Flight):
    def __init__(self, course, seed, difficulty, sensor_stress=1., *, response=None):
        if (not math.isfinite(difficulty) or difficulty < 0
                or not math.isfinite(sensor_stress) or sensor_stress < 0):
            raise ValueError('difficulty and sensor stress must be finite and nonnegative')
        super().__init__(course, seed, difficulty, sensor_stress)
        self.response = self.rng.uniform(*RESPONSE_RANGE_S) if response is None else float(response)
        if not RESPONSE_RANGE_S[0] <= self.response <= RESPONSE_RANGE_S[1]:
            raise ValueError('response is outside the transfer profile range')
        course.certify_scenario(self.scene, seed, difficulty)
        self.command = np.zeros(3)
        self.bound_violation = False
        self.first_unresolved = None
        self._reset_trace()
        if self._last_gap <= 0. or course.bounds_gap(self.p+self.bias) < 0.:
            raise InfeasibleScenario(seed, min(difficulty, GEOMETRY_DIFFICULTY_MAX), 'uncertified spawn pose')

    @property
    def done(self):
        return self.bound_violation or super().done

    def _reset_trace(self):
        self.path = [self.p.copy()]
        self.orientations = [quaternion_xyzw_from_rpy(self.roll, self.pitch, self.yaw)]
        self._last_gap = float(face_clearance_at_poses(self.p, sim.camera_rotation(self.yaw, self.roll, self.pitch),
                                                     self.scene, self.course.envelope))
        self.min_clearance = self._last_gap

    def start_at_window_approach(self):
        if self.steps:
            raise ValueError('window-only collection must start before the first action')
        self.index = self.truth_index = self.course.phases.index('WINDOW_APPROACH')
        self.p = self.course.goals[self.index-1].copy()
        delta = self.course.goals[self.index]-self.p
        self.yaw = math.atan2(delta[1], delta[0])
        self.yaw_command = self.yaw
        self._reset_trace()

    def motion(self):
        return Motion(self.p, self.v, self.command, self.yaw, self.roll, self.pitch)

    def step(self, action):
        cap = self.course.speeds[self.index]
        target_enu = self.rotation @ sim.clip_norm(np.asarray(action, dtype=float), cap)
        trace = []
        motion = self.motion()
        for _ in range(SUBSTEPS):
            motion = plant_substep(motion, target_enu, cap, self.response, self.wind, self.yaw_command)
            trace.append(motion)
        self.p, self.v, self.command = motion.position, motion.velocity, motion.command
        self.yaw, self.roll, self.pitch = motion.yaw, motion.roll, motion.pitch
        self.bias = np.clip(.997*self.bias + self.rng.normal(0., .002, 3)*self.difficulty, -.07, .07)
        self.steps += 1

        points = np.array([m.position for m in trace])
        roll, pitch, yaw = (np.array([getattr(m, key) for m in trace]) for key in ('roll', 'pitch', 'yaw'))
        q = np.array([quaternion_xyzw_from_rpy(r, p, y) for r, p, y in zip(roll, pitch, yaw)])
        gap = face_clearance_at_poses(points, rotation_matrices(roll, pitch, yaw), self.scene, self.course.envelope)
        path, attitude = np.vstack([self.path[-1], points]), np.vstack([self.orientations[-1], q])
        sweep = _sweep_motion(path, attitude, self.course.max_body_radius)
        for index in _uncertain(np.r_[self._last_gap, gap], sweep):
            proof = check_scene_path(path[index:index+2], attitude[index:index+2], self.scene, self.course.envelope)
            if not proof['certified_clear']:
                self.first_unresolved = proof['first_unresolved']
                self.collision = True
                break
        self.path.extend(points)
        self.orientations.extend(q)
        self._last_gap = float(gap[-1])
        self.min_clearance = min(self.min_clearance, float(gap.min()))
        estimated = points+self.bias
        self.bound_violation = bool(np.any(self.course.bounds_gap(estimated) < 0.))
        if np.any(np.linalg.norm(self.course.goals[self.index]-estimated, axis=1)
                  <= self.course.estimated_arrivals[self.index]):
            self.index += 1
        if self.truth_index < len(self.course.goals):
            error = np.linalg.norm(self.course.goals[self.truth_index]-points, axis=1)
            if np.any(error <= self.course.truth_arrivals[self.truth_index]):
                self.truth_index += 1

    def result(self):
        window = evaluate_window_crossing(self.path, self.orientations, self.scene.window, self.course.envelope)
        count = len(self.course.goals)
        return dict(success=not self.collision and not self.bound_violation and self.index == self.truth_index == count
                            and window['full_envelope_crossed'],
                    collision=self.collision, first_uncertified_contact=self.first_unresolved,
                    bound_violation=self.bound_violation, window_crossed=window['full_envelope_crossed'],
                    window_margin_m=window['min_aperture_clearance_m'], route_progress=self.truth_index/count,
                    estimated_waypoints_reached=self.index, sim_seconds=round(self.steps*sim.DT, 3),
                    sampled_min_clearance_m=self.min_clearance,
                    phase_at_end=self.course.phases[min(self.index, count-1)])


_SIDE, _UP = np.meshgrid((-.75, 0., .75), (-.75, 0., .75))
_SIDE, _UP = _SIDE.reshape(-1, 1), _UP.reshape(-1, 1)


def teacher(flight):
    """Choose a bounded target using the transfer plant and component clearance."""
    course, index = flight.course, flight.index
    target = course.goals[index]-.5*flight.bias
    if course.phases[index] == 'WINDOW_CROSS':
        center = np.asarray(flight.scene.window['center'])
        along = float((flight.p-center) @ course.axes[:, 0])
        target += np.clip((along+.50)/.15, 0., 1.)*(center-course.window['center'])
    goal = target-flight.p
    distance, cap = np.linalg.norm(goal), course.speeds[index]
    # A damped candidate gives the longer-lag teacher a way to settle at turns.
    nominal = sim.clip_norm(1.5*goal-flight.v-flight.response*flight.wind,
                            min(cap, math.sqrt(2*sim.ACCEL*distance)))
    heading = goal[:2]/max(.01, np.linalg.norm(goal[:2]))
    side = np.array([-heading[1], heading[0], 0.])
    offsets = _SIDE*side+_UP*sim.Z
    candidates = np.vstack([nominal, np.zeros(3), .5*nominal+.8*cap*offsets, .9*nominal+.5*cap*offsets])
    candidates = sim.clip_norm(candidates, cap)
    motion = flight.motion()
    probes = []
    for _ in range(INSTRUCTOR_STEPS):
        for _ in range(SUBSTEPS):
            motion = plant_substep(motion, candidates, cap, flight.response, flight.wind, flight.yaw_command)
        probes.append(motion)
    positions = np.stack([m.position for m in probes])
    roll, pitch = (np.stack([getattr(m, key) for m in probes]) for key in ('roll', 'pitch'))
    rotations = rotation_matrices(roll, pitch, np.array([m.yaw for m in probes])[:, None])
    clearance = face_clearance_at_poses(positions, rotations, flight.scene, course.envelope).min(axis=0)
    bounds = course.bounds_gap(positions+flight.bias).min(axis=0)
    gap = np.minimum(clearance, bounds)
    safety = 10.*np.maximum(INSTRUCTOR_RESERVE_M-gap, 0.)**2 + 24.*np.maximum(-gap, 0.)
    progress = np.linalg.norm(target-motion.position, axis=1)
    deviation = np.linalg.norm(nominal-candidates, axis=1)
    settle = .14*np.linalg.norm(motion.velocity, axis=1)*max(0., 1.-distance/.5)
    return candidates[np.argmin(progress+.23*deviation+settle+safety)] @ flight.rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2100000688)
    parser.add_argument('--difficulty', type=float, default=0.)
    parser.add_argument('--sensor-stress', type=float, default=1.)
    parser.add_argument('--response', type=float, nargs='+', default=[.50, .66, .80])
    parser.add_argument('--speed-scale', type=float, default=1.)
    args = parser.parse_args()
    course = TransferCourse(speed_scale=args.speed_scale)
    for response in args.response:
        start = time.perf_counter()
        try:
            flight = course.make_flight(args.seed, args.difficulty, args.sensor_stress, response=response)
        except InfeasibleScenario as error:
            print(json.dumps({'seed': error.seed, 'uncertified': error.detail}), flush=True)
            return 1
        while not flight.done:
            flight.observe(render=False)
            flight.step(course.instructor(flight))
        print(json.dumps({**flight.result(), 'seed': args.seed, 'response_s': response,
                          'wall_seconds': time.perf_counter()-start}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
