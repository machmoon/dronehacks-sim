"""Short-window visual velocity from timestamped camera poses, without truth."""
from collections import deque
import math


class CameraOdometry:
    def __init__(self, window_s=.2):
        self.window_s = window_s
        self.samples = deque()

    def update(self, observation):
        stamp = observation['sim_usec'] / 1e6
        position = observation['position_enu']
        if (not observation.get('valid') or len(position) != 3 or
                not all(math.isfinite(x) for x in (stamp, *position))):
            self.samples.clear()
            return None
        if self.samples:
            dt = stamp - self.samples[-1]['sim_usec']/1e6
            if dt <= 0 or dt > .1:
                self.samples.clear()
        self.samples.append(observation)
        while stamp-self.samples[0]['sim_usec']/1e6 > self.window_s + 1e-6:
            self.samples.popleft()
        if len(self.samples) < 5:
            return None
        times = [r['sim_usec']/1e6 for r in self.samples]
        if times[-1]-times[0] < .12:
            return None
        mean_time = sum(times)/len(times)
        centered = [t-mean_time for t in times]
        denominator = sum(t*t for t in centered)
        means = [sum(r['position_enu'][axis] for r in self.samples)/len(times)
                 for axis in range(3)]
        velocity = [sum(t*(r['position_enu'][axis]-means[axis])
                        for t, r in zip(centered, self.samples))/denominator
                    for axis in range(3)]
        residual = math.sqrt(sum((r['position_enu'][axis]-means[axis]-velocity[axis]*t)**2
                                 for t, r in zip(centered, self.samples)
                                 for axis in range(3))/len(times))
        if math.hypot(*velocity) > 3.0 or residual > .025:
            self.samples.clear()
            return None
        # Velocity belongs at the middle of the fitting window, not at the
        # newest capture time. Send that capture's pose with the same stamp.
        result = dict(self.samples[len(self.samples)//2])
        result['velocity_enu'] = velocity
        result['velocity_fit_residual_m'] = residual
        result['velocity_std_mps'] = max(.05, residual/math.sqrt(denominator))
        return result
