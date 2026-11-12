"""Align PX4 estimates with camera capture time without extrapolating."""
from collections import deque
from dataclasses import dataclass
import math

from .core import wrap
from .px4_frames import enu_to_ned


HISTORY_USEC = 1_000_000
MAX_BRACKET_USEC = 60_000


@dataclass(frozen=True)
class CaptureState:
    position_enu: tuple
    velocity_enu: tuple
    attitude_rpy_frd: tuple


def _record(samples, stamp_usec, values):
    if samples and stamp_usec <= samples[-1][0]:
        return
    samples.append((stamp_usec, values if all(map(math.isfinite, values)) else None))
    while samples[0][0] < stamp_usec-HISTORY_USEC:
        samples.popleft()


def _sample(samples, stamp_usec, angles=False):
    later = None
    for sample_usec, values in reversed(samples):
        if sample_usec == stamp_usec:
            return values
        if sample_usec < stamp_usec:
            if later is None or later[0]-sample_usec > MAX_BRACKET_USEC:
                return None
            later_usec, later_values = later
            if values is None or later_values is None:
                return None
            fraction = (stamp_usec-sample_usec)/(later_usec-sample_usec)
            if angles:
                return tuple(wrap(a + fraction*wrap(b-a)) for a, b in zip(values, later_values))
            return tuple(a + fraction*(b-a) for a, b in zip(values, later_values))
        later = sample_usec, values
    return None


class PolicyStateHistory:
    def __init__(self):
        self._local = deque(maxlen=64)
        self._attitude = deque(maxlen=64)

    def record(self, position, attitude):
        if position is not None:
            _record(self._local, position.time_boot_ms*1000,
                    (position.x, position.y, position.z, position.vx, position.vy, position.vz))
        if attitude is not None:
            _record(self._attitude, attitude.time_boot_ms*1000,
                    (attitude.roll, attitude.pitch, attitude.yaw))

    def at(self, capture_usec):
        local = _sample(self._local, capture_usec)
        attitude = _sample(self._attitude, capture_usec, angles=True)
        if local is None or attitude is None:
            return None
        return CaptureState(enu_to_ned(local[:3]), enu_to_ned(local[3:]), attitude)
