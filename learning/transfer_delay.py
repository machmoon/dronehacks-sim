"""Versioned x500 transfer profile with delayed post-slew velocity commands."""
from collections import deque
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np

from mission.airframe import (check_scene_path, face_clearance_at_poses,
                              quaternion_xyzw_from_rpy)
from . import sim, transfer


TRAIN_PROFILE = 'x500-transfer-delay'
PROFILE = 'x500-transfer-v2'
DELAY_STEPS = (3, 4, 5, 6)
GAIN_XY = (1., 1.08)


def delayed_substep(motion, target, cap, response, wind, yaw_target, issued, gain_xy):
    """Slew the outgoing command, then use the next queued command at the plant."""
    slew = transfer.COMMAND_SLEW*transfer.SUBSTEP_S
    command = sim.clip_norm(motion.command + sim.clip_norm(target-motion.command, slew), cap)
    issued.append(command)
    applied = issued.popleft()*np.array([gain_xy, gain_xy, 1.])
    # Start the base plant with its command already applied so it does not
    # slew a second time. Return the outgoing command as the next slew state.
    advanced = transfer.plant_substep(replace(motion, command=applied), applied, math.inf,
                                      response, wind, yaw_target)
    return replace(advanced, command=command)


class DelayCourse(transfer.TransferCourse):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._profile.update(
            name=PROFILE,
            transfer_delay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            command_delay_s=[round(steps*transfer.SUBSTEP_S, 2) for steps in (DELAY_STEPS[0], DELAY_STEPS[-1])],
            command_delay_quantum_s=transfer.SUBSTEP_S,
            command_delay_location='post_slew',
            velocity_gain_xy=list(GAIN_XY),
        )

    def make_flight(self, seed, difficulty, sensor_stress=1., *, response=None, delay_s=None, gain_xy=None):
        return DelayFlight(self, seed, difficulty, sensor_stress, response=response,
                           delay_s=delay_s, gain_xy=gain_xy)

    def instructor(self, flight):
        return delay_teacher(flight)


class DelayFlight(transfer.TransferFlight):
    def __init__(self, course, seed, difficulty, sensor_stress=1., *, response=None, delay_s=None, gain_xy=None):
        super().__init__(course, seed, difficulty, sensor_stress, response=response)
        dynamics_rng = np.random.default_rng([seed, 3])
        if delay_s is None:
            steps = int(dynamics_rng.choice(DELAY_STEPS))
        else:
            if not math.isfinite(delay_s):
                raise ValueError('command delay must be finite')
            steps = round(delay_s/transfer.SUBSTEP_S)
            if steps not in (0, *DELAY_STEPS) or not math.isclose(delay_s, steps*transfer.SUBSTEP_S, abs_tol=1e-9):
                raise ValueError('command delay must be zero or 0.12–0.24 s in 0.04 s steps')
        self.gain_xy = float(dynamics_rng.uniform(*GAIN_XY) if gain_xy is None else gain_xy)
        if not GAIN_XY[0] <= self.gain_xy <= GAIN_XY[1]:
            raise ValueError('horizontal velocity gain is outside the transfer profile range')
        self.command_delay_s = round(steps*transfer.SUBSTEP_S, 2)
        self.issued = deque(np.zeros(3) for _ in range(steps))

    def step(self, action):
        if not self.issued and self.gain_xy == 1.:
            return super().step(action)
        cap = self.course.speeds[self.index]
        target = self.rotation @ sim.clip_norm(np.asarray(action, dtype=float), cap)
        motion = self.motion()
        trace = []
        for _ in range(transfer.SUBSTEPS):
            motion = delayed_substep(motion, target, cap, self.response, self.wind,
                                     self.yaw_command, self.issued, self.gain_xy)
            trace.append(motion)
        self.p, self.v, self.command = motion.position, motion.velocity, motion.command
        self.yaw, self.roll, self.pitch = motion.yaw, motion.roll, motion.pitch
        self.bias = np.clip(.997*self.bias + self.rng.normal(0., .002, 3)*self.difficulty, -.07, .07)
        self.steps += 1

        # Apply the original transfer certification at each substep. The v1
        # implementation stays byte-for-byte compatible with old checkpoints.
        points = np.array([m.position for m in trace])
        roll, pitch, yaw = (np.array([getattr(m, key) for m in trace]) for key in ('roll', 'pitch', 'yaw'))
        q = np.array([quaternion_xyzw_from_rpy(r, p, y) for r, p, y in zip(roll, pitch, yaw)])
        rotations = transfer.rotation_matrices(roll, pitch, yaw)
        gap = face_clearance_at_poses(points, rotations, self.scene, self.course.envelope)
        path, attitudes = np.vstack([self.path[-1], points]), np.vstack([self.orientations[-1], q])
        sweep = transfer._sweep_motion(path, attitudes, self.course.max_body_radius)
        for index in transfer._uncertain(np.r_[self._last_gap, gap], sweep):
            proof = check_scene_path(path[index:index+2], attitudes[index:index+2], self.scene, self.course.envelope)
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
        return dict(**super().result(), command_delay_s=self.command_delay_s,
                    velocity_gain_xy=self.gain_xy, velocity_response_s=self.response)


def preview_commands(flight, candidates, cap):
    """Predict batched candidates from a private copy of the actual issued FIFO."""
    motion = flight.motion()
    motion = replace(motion, position=np.broadcast_to(motion.position, candidates.shape),
                     velocity=np.broadcast_to(motion.velocity, candidates.shape))
    issued = deque(flight.issued)
    probes = []
    for _ in range(transfer.INSTRUCTOR_STEPS):
        for _ in range(transfer.SUBSTEPS):
            motion = delayed_substep(motion, candidates, cap, flight.response, flight.wind,
                                     flight.yaw_command, issued, flight.gain_xy)
        probes.append(motion)
    return probes


def delay_teacher(flight):
    if not flight.issued and flight.gain_xy == 1.:
        return transfer.teacher(flight)
    course, index = flight.course, flight.index
    target = course.goals[index]-.5*flight.bias
    if course.phases[index] == 'WINDOW_CROSS':
        center = np.asarray(flight.scene.window['center'])
        along = float((flight.p-center) @ course.axes[:, 0])
        target += np.clip((along+.50)/.15, 0., 1.)*(center-course.window['center'])
    goal = target-flight.p
    distance, cap = np.linalg.norm(goal), course.speeds[index]
    nominal = sim.clip_norm(1.5*goal-flight.v-flight.response*flight.wind,
                            min(cap, math.sqrt(2*sim.ACCEL*distance)))
    heading = goal[:2]/max(.01, np.linalg.norm(goal[:2]))
    side = np.array([-heading[1], heading[0], 0.])
    offsets = transfer._SIDE*side+transfer._UP*sim.Z
    candidates = sim.clip_norm(np.vstack([nominal, np.zeros(3), .5*nominal+.8*cap*offsets,
                                          .9*nominal+.5*cap*offsets]), cap)
    probes = preview_commands(flight, candidates, cap)
    positions = np.stack([m.position for m in probes])
    roll, pitch = (np.stack([getattr(m, key) for m in probes]) for key in ('roll', 'pitch'))
    rotations = transfer.rotation_matrices(roll, pitch, np.array([m.yaw for m in probes])[:, None])
    clearance = face_clearance_at_poses(positions, rotations, flight.scene, course.envelope).min(axis=0)
    bounds = course.bounds_gap(positions+flight.bias).min(axis=0)
    gap = np.minimum(clearance, bounds)
    safety = 10.*np.maximum(transfer.INSTRUCTOR_RESERVE_M-gap, 0.)**2 + 24.*np.maximum(-gap, 0.)
    progress = np.linalg.norm(target-probes[-1].position, axis=1)
    deviation = np.linalg.norm(nominal-candidates, axis=1)
    settle = .14*np.linalg.norm(probes[-1].velocity, axis=1)*max(0., 1.-distance/.5)
    return candidates[np.argmin(progress+.23*deviation+settle+safety)] @ flight.rotation
