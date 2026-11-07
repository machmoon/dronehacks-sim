"""SITL finish verification. This does not expose a hardware landing interface."""
import math


class LandingMonitor:
    """Require fresh PX4 ground/disarm reports and the envelope inside the pad.

    A land request or a small height estimate is not proof of landing. The
    radius is a simulator envelope assumption, not a measured Skybeta size.
    """

    def __init__(self, center, pad_radius, aircraft_radius, timeout=30.0):
        self.center = tuple(float(x) for x in center)
        self.pad_radius = float(pad_radius)
        self.aircraft_radius = float(aircraft_radius)
        self.timeout = float(timeout)
        if (len(self.center) != 3 or
                not all(math.isfinite(x) for x in (*self.center, self.pad_radius,
                                                   self.aircraft_radius, self.timeout)) or
                self.aircraft_radius <= 0 or self.pad_radius <= self.aircraft_radius or
                self.timeout <= 0):
            raise ValueError('invalid landing geometry or timeout')
        self.started = None

    def start(self, now):
        if not math.isfinite(now):
            raise ValueError('invalid landing clock')
        if self.started is None:
            self.started = now

    def check(self, now, position, *, armed, on_ground, telemetry_fresh):
        if self.started is None:
            return 'waiting', None
        if not math.isfinite(now) or now < self.started:
            return 'failed', 'landing_clock_reversed'
        if not telemetry_fresh:
            return 'failed', 'landing_telemetry_lost'
        if (len(position) != 3 or not all(math.isfinite(x) for x in position) or
                type(armed) is not bool or type(on_ground) is not bool):
            return 'failed', 'invalid_landing_state'
        if not armed:
            if not on_ground:
                return 'failed', 'disarmed_without_ground_confirmation'
            error = math.hypot(position[0]-self.center[0], position[1]-self.center[1])
            if error + self.aircraft_radius > self.pad_radius:
                return 'failed', 'landed_outside_pad'
            return 'completed', None
        if now-self.started > self.timeout:
            return 'failed', 'landing_timeout'
        return 'landing', None
