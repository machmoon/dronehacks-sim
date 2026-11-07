"""ROS-independent, feedback-driven course controller.

The controller consumes and emits course-frame quantities. It does not model the
vehicle, arm it, enter Offboard, or land it. ``done`` means landing *handoff* only.
The physics demonstrator and the ROS adapter intentionally use this same class.
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]
ZERO = (0.0, 0.0, 0.0)


def vector(values):
    result = tuple(float(v) for v in values)
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise ValueError('expected three finite coordinates')
    return result


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def limited(v, maximum):
    length = norm(v)
    return tuple(x * min(1.0, maximum / length) for x in v) if length else ZERO


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Command:
    velocity: Vector3
    yaw_rate: float
    phase: str
    done: bool
    fault: Optional[str]
    target: Vector3


class CourseController:
    """Sequential waypoint follower with speed/acceleration/freshness limits.

    Waypoints are only advanced within ``arrival_radius``; a time schedule never
    advances the aircraft past an obstacle. Validate the swept vehicle envelope
    along the path separately. This follower cannot infer unobserved obstacles.
    """

    def __init__(self, config, speed_scale=1.0):
        self.config = config
        self.start = vector(config['start'])
        if not math.isfinite(speed_scale) or not 0.0 < speed_scale <= 2.0:
            raise ValueError('speed_scale must be finite and in (0, 2]')
        limits = config.get('limits', {})
        self.max_speed = float(limits.get('max_speed', 1.0))
        self.max_accel = float(limits.get('max_accel', 0.8))
        self.arrival_radius = float(limits.get('arrival_radius', 0.12))
        self.odom_timeout = float(limits.get('odom_timeout', 0.25))
        self.state_timeout = float(limits.get('state_timeout', 0.6))
        self.max_yaw_rate = float(limits.get('max_yaw_rate', 0.7))
        self.jump_margin = float(limits.get('jump_margin', 0.35))
        self.position_gain = float(limits.get('position_gain', 2.0))
        self.velocity_damping = float(limits.get('velocity_damping', 0.0))
        if not math.isfinite(self.velocity_damping) or self.velocity_damping < 0:
            raise ValueError('velocity_damping must be finite and nonnegative')
        for name in ('max_speed', 'max_accel', 'arrival_radius', 'odom_timeout',
                     'state_timeout', 'max_yaw_rate', 'jump_margin', 'position_gain'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive and finite')
        if self.arrival_radius > 0.2:
            raise ValueError('arrival_radius > 0.2 m requires a new clearance review')
        self.speed_scale = float(speed_scale)
        self.bounds = config.get('bounds')
        self.footprint_radius = float(config.get('footprint_radius', 0.0))
        if not math.isfinite(self.footprint_radius) or self.footprint_radius < 0:
            raise ValueError('footprint_radius must be finite and nonnegative')
        if self.bounds:
            self.bound_min = vector(self.bounds['min'])
            self.bound_max = vector(self.bounds['max'])
            if any(lo >= hi for lo, hi in zip(self.bound_min, self.bound_max)):
                raise ValueError('invalid course bounds')
        self.waypoints = []
        for item in config['waypoints']:
            speed = float(item.get('speed', self.max_speed))
            arrival = float(item.get('arrival_radius', self.arrival_radius))
            if not math.isfinite(arrival) or not 0 < arrival <= self.arrival_radius:
                raise ValueError('waypoint arrival_radius must be positive and no larger than the global radius')
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError('waypoint speeds must be positive and finite')
            self.waypoints.append({'position': vector(item['position']),
                                   'phase': str(item.get('phase', 'COURSE')),
                                   'arrival_radius': arrival,
                                   'speed': min(self.max_speed, speed * speed_scale)})
        if not self.waypoints:
            raise ValueError('at least one waypoint is required')
        landing = config.get('landing', {})
        if landing:
            p = vector(landing['position'])
            self.handoff_target = (p[0], p[1], max(
                self.start[2] + 0.3, p[2] + float(landing.get('approach_height', 0.7))))
        else:
            self.handoff_target = self.waypoints[-1]['position']
            if self.handoff_target[2] < self.start[2] + 0.3:
                raise ValueError('terminal target must remain above the interface floor')
        if norm(sub(self.waypoints[-1]['position'], self.handoff_target)) > 1e-6:
            self.waypoints.append({'position': self.handoff_target,
                                   'phase': 'LANDING_APPROACH', 'speed': min(0.3, self.max_speed),
                                   'arrival_radius': self.arrival_radius})
        self.waypoint_index = 0
        self.started = False
        self.done = False
        self.fault = None
        self.last_time = None
        self.last_position = None
        self.last_command = ZERO

    @property
    def progress(self):
        return self.waypoint_index / len(self.waypoints)

    def _stop(self, reason, target):
        self.fault = self.fault or reason
        self.last_command = ZERO
        return Command(ZERO, 0.0, 'FAULT', False, self.fault, target)

    def update(self, position, velocity, now, offboard=True, odom_age=0.0,
               state_age=0.0, *, yaw=None):
        """Return a bounded velocity request, or a latched zero-command fault.

        ``yaw`` is optional course-frame heading. Omitting it holds heading by
        returning zero yaw rate; the ROS adapter always supplies measured yaw.
        Zero on fault is an intervention request, not a claim that a lost pose
        can support a safe hover. Restart only after diagnosis and pilot reset.
        """
        target = (self.handoff_target if self.done else
                  self.waypoints[self.waypoint_index]['position'])
        if self.fault:
            return self._stop(self.fault, target)
        try:
            position, velocity = vector(position), vector(velocity)
        except (TypeError, ValueError):
            return self._stop('invalid_state', target)
        if not all(math.isfinite(v) for v in (now, odom_age, state_age)):
            return self._stop('invalid_timestamp', target)
        if self.bounds:
            outside_xy = any(position[i] < self.bound_min[i]+self.footprint_radius or
                             position[i] > self.bound_max[i]-self.footprint_radius
                             for i in (0, 1))
            outside_z = (position[2] < self.bound_min[2]-0.05 or
                         position[2] > self.bound_max[2]-self.footprint_radius)
            if outside_xy or outside_z:
                return self._stop('course_boundary_breach', target)
        if odom_age < -0.05 or state_age < -0.05:
            return self._stop('future_timestamp', target)
        if self.started and not offboard:
            return self._stop('offboard_lost', target)
        if odom_age > self.odom_timeout or state_age > self.state_timeout:
            if self.started:
                return self._stop('stale_odometry' if odom_age > self.odom_timeout
                                  else 'stale_flight_state', target)
            return Command(ZERO, 0.0, 'WAIT_FOR_FRESH_STATE', False, None, target)
        if not offboard:
            self.last_time = now
            self.last_position = position
            self.last_command = ZERO
            return Command(ZERO, 0.0, 'WAIT_FOR_OFFBOARD', False, None, target)
        dt = 1.0 / 30.0 if self.last_time is None else now - self.last_time
        if dt < 0 or (self.started and dt > self.state_timeout):
            return self._stop('control_clock_discontinuity', target)
        dt = min(max(dt, 1e-4), 0.1)
        if self.last_position is not None and self.started:
            # Allow modest tracking overshoot, but never let a corrupt velocity
            # in the same packet justify an arbitrarily large position jump.
            motion_speed = min(2.0 * self.max_speed, max(self.max_speed, norm(velocity)))
            allowed_move = self.jump_margin + motion_speed * dt
            if norm(sub(position, self.last_position)) > allowed_move:
                return self._stop('odometry_jump', target)
        self.started = True
        self.last_time = now
        self.last_position = position
        if not self.done and norm(sub(target, position)) <= self.waypoints[self.waypoint_index]['arrival_radius']:
            # At most one stage per observation prevents jumping over co-located
            # checkpoints and makes progress inspectable in logs.
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                self.done = True
                target = self.handoff_target
            else:
                target = self.waypoints[self.waypoint_index]['position']
        delta = sub(target, position)
        distance = norm(delta)
        speed = min(0.3, self.max_speed) if self.done else self.waypoints[self.waypoint_index]['speed']
        # Position feedback approaches each waypoint conservatively. The
        # acceleration envelope starts braking before reaching a tight turn.
        requested_speed = min(speed, self.position_gain * distance,
                              math.sqrt(2.0 * self.max_accel * distance))
        desired = tuple(x / distance * requested_speed for x in delta) if distance else ZERO
        if self.velocity_damping:
            # Damp measured motion near turns. Saturating after the feedback
            # term preserves cruise speed far from a waypoint while braking
            # before position-only tracking would reverse its command.
            desired = limited(tuple(self.position_gain*x-self.velocity_damping*v
                                    for x, v in zip(delta, velocity)),
                              min(speed, math.sqrt(2.0*self.max_accel*distance)))
        change = limited(sub(desired, self.last_command), self.max_accel * dt)
        command = limited(tuple(a + b for a, b in zip(self.last_command, change)), self.max_speed)
        self.last_command = command
        yaw_rate = 0.0
        if yaw is not None:
            if not math.isfinite(yaw):
                return self._stop('invalid_yaw', target)
            if math.hypot(delta[0], delta[1]) > 0.05 and not self.done:
                error = wrap(math.atan2(delta[1], delta[0]) - yaw)
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, 1.5 * error))
        phase = 'LANDING_HANDOFF' if self.done else self.waypoints[self.waypoint_index]['phase']
        return Command(command, yaw_rate, phase, self.done, None, target)
