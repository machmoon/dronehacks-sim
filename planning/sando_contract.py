"""Validate SANDO trajectory samples for offline/shadow evaluation only.

No claim that this velocity-tracking approximation preserves SANDO's safety
proof. Overspeed proposals are rejected, not silently clipped into a different
trajectory. Goal.power is never an arm/disarm command.
"""
import math

from mission.frames import rotate_body_velocity


def vector(value):
    result = tuple(float(x) for x in value)
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError('expected finite xyz')
    return result


def state_from_odometry(position, quaternion_xyzw, velocity_body):
    p, v = vector(position), vector(velocity_body)
    q = tuple(float(x) for x in quaternion_xyzw)
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError('invalid quaternion')
    length = math.sqrt(sum(x*x for x in q))
    if length < 1e-9:
        raise ValueError('zero quaternion')
    q = tuple(x/length for x in q)
    return {'position': p, 'quaternion_xyzw': q,
            'velocity_world': rotate_body_velocity(q, v)}


def evaluate_goal(goal, position, now, *, expected_frame='map', max_age=.15,
                  max_speed=1.0, max_position_error=.4, position_gain=1.0):
    """Consume the actual upstream Goal fields p/v/a/j/yaw/dyaw/header."""
    if not all(math.isfinite(x) and x > 0 for x in
               (max_age, max_speed, max_position_error, position_gain)):
        raise ValueError('invalid evaluator limits')
    if not math.isfinite(now) or now < 0:
        raise ValueError('invalid evaluator clock')
    try:
        stamp = float(goal['stamp'])
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError('invalid timestamp')
        if goal['frame'] != expected_frame:
            raise ValueError('frame_mismatch')
        if stamp > now+.05:
            raise ValueError('future_goal')
        if now-stamp > max_age:
            raise ValueError('stale_goal')
        if goal.get('mode_xy', 0) != 0 or goal.get('mode_z', 0) != 0:
            raise ValueError('unsupported_tracking_mode')
        p, v = vector(goal['p']), vector(goal['v'])
        vector(goal['a']); vector(goal['j'])
        if not all(math.isfinite(float(goal[x])) for x in ('yaw', 'dyaw')):
            raise ValueError('invalid_yaw')
        error = tuple(a-b for a, b in zip(p, vector(position)))
        if math.sqrt(sum(x*x for x in error)) > max_position_error:
            raise ValueError('tracking_error_exceeded')
        candidate = tuple(a+position_gain*b for a, b in zip(v, error))
        speed = math.sqrt(sum(x*x for x in candidate))
        if speed > max_speed:
            raise ValueError('speed_limit_exceeded')
    except (KeyError, TypeError, ValueError) as error:
        return {'accepted': False, 'reason': str(error), 'candidate_velocity': None}
    return {'accepted': True, 'reason': None, 'candidate_velocity': candidate,
            'tracking_error_m': math.sqrt(sum(x*x for x in error)),
            'age_s': max(0, now-stamp), 'flight_authorized': False}
