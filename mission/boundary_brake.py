"""Predictive planar braking for learned ENU velocity setpoints."""

from dataclasses import dataclass
import math

from .core import Vector3, limited, sub, vector


@dataclass(frozen=True)
class BrakingStep:
    sent_velocity: Vector3
    unfiltered_velocity: Vector3
    faces: tuple
    stop_budget_deficits_m: tuple


class BoundaryBrake:
    """Limit outward setpoints using measured velocity and a stopping budget.

    For a first-order velocity response and a setpoint ramp to zero, the
    stopping distance is tau*v + lead*u + u**2/(2*slew) in the outward
    direction. The lead term covers actuator delay, estimate age, and control
    cadence. The decision budgets for the larger of the previous and proposed
    setpoints, with an extra position reserve. This model is a guard for the
    surveyed planar bounds; the controller still checks the original hard bounds.
    """

    def __init__(self, lower, upper, footprint_m, *, reserve_m=.04, response_s=.80, lead_s=.40):
        lower, upper = vector(lower), vector(upper)
        if (not all(math.isfinite(v) and v >= 0 for v in (footprint_m, reserve_m, lead_s))
                or not math.isfinite(response_s) or response_s <= 0
                or any(hi-lo <= 2*(footprint_m+reserve_m) for lo, hi in zip(lower[:2], upper[:2]))):
            raise ValueError('invalid planar braking bounds or response')
        self.lower = tuple(x+footprint_m for x in lower[:2])
        self.upper = tuple(x-footprint_m for x in upper[:2])
        self.reserve_m = reserve_m
        self.response_s = response_s
        self.lead_s = lead_s

    def _outward_budget(self, gap, outward_velocity, previous, axis_slew):
        distance = gap-self.reserve_m-self.response_s*max(outward_velocity, 0.)
        outward = max(previous, 0.)
        after_lead = distance-self.lead_s*outward
        # Both bounds must hold: the delay budget uses max(previous, next).
        # Solve lead*next + next**2/(2*slew) <= distance for next >= 0.
        rate = axis_slew*self.lead_s
        limit = min(math.sqrt(2*axis_slew*max(0., after_lead)),
                    math.sqrt(rate**2+2*axis_slew*max(0., distance))-rate)
        # Positive means the prior sent setpoint exceeds the stopping model's
        # budget. It can occur even if the current requested action is inward.
        return limit, max(0., outward**2/(2*axis_slew)-after_lead)

    def step(self, position, velocity, desired, previous, *, slew, dt, cap):
        position, velocity, desired, previous = map(vector, (position, velocity, desired, previous))
        if not (math.isfinite(slew) and slew > 0 and math.isfinite(dt) and dt >= 0
                and math.isfinite(cap) and cap > 0):
            raise ValueError('invalid setpoint slew, timestep, or speed cap')
        max_change = slew*dt
        change = limited(sub(desired, previous), max_change)
        normal = limited(tuple(v+d for v, d in zip(previous, change)), cap)
        # Reserve slew for two simultaneous planar corrections near a corner.
        axis_slew = slew/math.sqrt(2.)
        lower, upper, faces, deficits = [], [], [], []
        for i, axis in enumerate(('x', 'y')):
            backward, back_deficit = self._outward_budget(position[i]-self.lower[i], -velocity[i],
                                                            -previous[i], axis_slew)
            forward, front_deficit = self._outward_budget(self.upper[i]-position[i], velocity[i],
                                                           previous[i], axis_slew)
            lower.append(-backward)
            upper.append(forward)
            if normal[i] < -backward:
                faces.append(axis+'-')
            if normal[i] > forward:
                faces.append(axis+'+')
            if back_deficit > 0.:
                deficits.append((axis+'-', back_deficit))
            if front_deficit > 0.:
                deficits.append((axis+'+', front_deficit))
        faces, deficits = tuple(faces), tuple(deficits)
        if not faces:
            return BrakingStep(normal, normal, faces, deficits)

        def project(v):
            return tuple(max(lo, min(hi, x)) for lo, hi, x in zip(lower, upper, v))+(v[2],)

        if all(lower[i] <= previous[i] <= upper[i] for i in (0, 1)):
            # Projection onto an axis-aligned box cannot increase the change
            # from a previous setpoint already inside that box.
            return BrakingStep(project(normal), normal, faces, deficits)
        # If the current setpoint is already outside the stopping budget,
        # prioritize recovery over changes to tangential or vertical commands.
        correction = limited(sub(project(previous), previous), max_change)
        sent = limited(tuple(v+d for v, d in zip(previous, correction)), cap)
        return BrakingStep(sent, normal, faces, deficits)
