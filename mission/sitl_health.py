"""Read-only PX4 SITL battery and power publication watchdog."""
from dataclasses import dataclass, replace
import math
import os
import re
import subprocess
from threading import Event, Lock, Thread
import time
from typing import Optional


POLL_WALL_S = .7
LISTENER_TIMEOUT_S = 1.5
MAX_READ_AGE_WALL_S = 4.
MIN_AGE_SIM_S, MAX_AGE_SIM_S = -.2, .8
_NUMBER = r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
_FIELD = re.compile(r'[ \t]*([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*(.*)')
_TIMESTAMP = re.compile(rf'([0-9]{{1,20}})(?:[ \t]+\({_NUMBER} seconds ago\))?')


class HealthReadError(ValueError):
    pass


class HealthTransportError(RuntimeError):
    pass


def _fields(text, topic, required):
    if not isinstance(text, str):
        raise HealthReadError(f'{topic}_invalid_output')
    values = {}
    for line in text.splitlines():
        match = _FIELD.fullmatch(line)
        if match and match[1] in required:
            key, value = match.groups()
            if key in values:
                raise HealthReadError(f'{topic}_duplicate_{key}')
            values[key] = value.strip()
    for key in required:
        if not values.get(key):
            raise HealthReadError(f'{topic}_missing_{key}')
    return values


def _timestamp(values, topic):
    match = _TIMESTAMP.fullmatch(values['timestamp'])
    if match is None or not 0 < int(match[1]) < 2**64:
        raise HealthReadError(f'{topic}_invalid_timestamp')
    return int(match[1])


def _finite_number(values, topic, key):
    value = values[key]
    if not re.fullmatch(_NUMBER, value):
        raise HealthReadError(f'{topic}_invalid_{key}')
    number = float(value)
    if not math.isfinite(number):
        raise HealthReadError(f'{topic}_invalid_{key}')
    return number


def _expect(values, topic, expected):
    for key, value in expected.items():
        if values[key].lower() != value:
            raise HealthReadError(f'{topic}_unexpected_{key}')


def parse_battery_status(text):
    """Return the healthy simulated battery publication timestamp and remaining fraction."""
    topic = 'battery_status'
    expected = dict(connected='true', source='0', id='1', warning='0',
                    faults='0', is_powering_off='false')
    values = _fields(text, topic, ('timestamp', *expected, 'remaining'))
    stamp = _timestamp(values, topic)
    _expect(values, topic, expected)
    remaining = _finite_number(values, topic, 'remaining')
    if remaining <= .15:
        raise HealthReadError('battery_status_low_remaining')
    if remaining > 1.:
        raise HealthReadError('battery_status_invalid_remaining')
    return stamp, remaining


def parse_system_power(text):
    """Return the healthy system-power publication timestamp and 5 V rail voltage."""
    topic = 'system_power'
    expected = dict(brick_valid='1', periph_5v_oc='0', hipower_5v_oc='0')
    values = _fields(text, topic, ('timestamp', *expected, 'voltage5v_v'))
    stamp = _timestamp(values, topic)
    _expect(values, topic, expected)
    voltage = _finite_number(values, topic, 'voltage5v_v')
    if not 4.5 <= voltage <= 5.5:
        raise HealthReadError('system_power_unsafe_voltage5v_v')
    return stamp, voltage


@dataclass(frozen=True)
class _Read:
    timestamp_usec: int
    value: float
    read_wall_s: float


@dataclass(frozen=True)
class _Snapshot:
    battery: Optional[_Read] = None
    power: Optional[_Read] = None
    consecutive: int = 0
    reason: Optional[str] = None
    read_error: Optional[str] = None
    consecutive_transport_errors: int = 0


class SitlHealthMonitor:
    """Poll uORB on a daemon thread; flight control only reads immutable snapshots."""

    def __init__(self, *, start=True, runner=None, clock=time.monotonic):
        if os.environ.get('DRONEHACKS_SITL') != '1':
            raise RuntimeError('SITL health monitoring requires DRONEHACKS_SITL=1')
        self._runner = subprocess.run if runner is None else runner
        self._clock = clock
        self._snapshot = _Snapshot()
        self._lock = Lock()
        self._poll_lock = Lock()
        self._stop = Event()
        self._thread = None
        if start:
            self._thread = Thread(target=self._run, daemon=True, name='px4-sitl-health')
            self._thread.start()

    def _read(self, topic, parse):
        try:
            result = self._runner(['px4-listener', topic, '-n', '1'],
                                  capture_output=True, text=True, check=True, timeout=LISTENER_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise HealthTransportError(f'{topic}_listener_timeout') from error
        except (subprocess.SubprocessError, OSError) as error:
            raise HealthTransportError(f'{topic}_listener_error:{type(error).__name__}') from error
        except UnicodeError as error:
            raise HealthReadError(f'{topic}_invalid_output') from error
        stamp, value = parse(result.stdout)
        return _Read(stamp, value, self._clock())

    def poll(self):
        """Read both publications once. Use status() in the control loop."""
        with self._poll_lock:
            if self._stop.is_set():
                return
            try:
                battery = self._read('battery_status', parse_battery_status)
                if self._stop.is_set():
                    return
                power = self._read('system_power', parse_system_power)
            except HealthTransportError as error:
                # Keep the last validated pair and its original ages. The
                # caller still checks both freshness limits on every status().
                with self._lock:
                    previous = self._snapshot
                    self._snapshot = replace(previous, read_error=str(error),
                                             consecutive_transport_errors=previous.consecutive_transport_errors+1)
                return
            except HealthReadError as error:
                with self._lock:
                    self._snapshot = replace(self._snapshot, consecutive=0, reason=str(error),
                                             read_error=None, consecutive_transport_errors=0)
                return
            with self._lock:
                previous = self._snapshot
                reason = None
                consecutive = 1
                if previous.battery is not None:
                    battery_delta = battery.timestamp_usec - previous.battery.timestamp_usec
                    power_delta = power.timestamp_usec - previous.power.timestamp_usec
                    if battery_delta < 0:
                        reason = 'battery_status_timestamp_regressed'
                    elif power_delta < 0:
                        reason = 'system_power_timestamp_regressed'
                    consecutive = (0 if reason else
                                   previous.consecutive + int(battery_delta > 0 and power_delta > 0))
                    # Reading an existing publication again cannot make the
                    # underlying sample newer on either freshness clock.
                    if battery_delta == 0:
                        battery = previous.battery
                    if power_delta == 0:
                        power = previous.power
                self._snapshot = _Snapshot(battery, power, consecutive, reason)

    def _run(self):
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(POLL_WALL_S)

    def status(self, sim_s, now):
        """Return a fresh JSON-compatible snapshot without running a command."""
        with self._lock:
            snapshot = self._snapshot
        reason = snapshot.reason
        if self._stop.is_set():
            reason = 'monitor_closed'
        elif reason is None and snapshot.consecutive < 2:
            reason = 'awaiting_advancing_battery_and_power_samples'

        try:
            clocks_valid = math.isfinite(sim_s) and math.isfinite(now) and sim_s >= 0
        except TypeError:
            clocks_valid = False
        if reason is None and not clocks_valid:
            reason = 'invalid_health_clock'
        result = {'consecutive_samples': snapshot.consecutive, 'read_error': snapshot.read_error,
                  'consecutive_transport_errors': snapshot.consecutive_transport_errors}
        for topic, sample in (('battery_status', snapshot.battery), ('system_power', snapshot.power)):
            sim_age = sim_s-sample.timestamp_usec*1e-6 if sample is not None and clocks_valid else None
            # The polling thread can finish just after the caller captured now.
            wall_age = max(0., now-sample.read_wall_s) if sample is not None and clocks_valid else None
            if reason is None:
                if not MIN_AGE_SIM_S-1e-9 <= sim_age <= MAX_AGE_SIM_S+1e-9:
                    reason = f'{topic}_stale_sim_timestamp'
                elif wall_age >= MAX_READ_AGE_WALL_S:
                    reason = f'{topic}_stale_wall_read'
            result[f'{topic}_timestamp_usec'] = sample.timestamp_usec if sample is not None else None
            result[f'{topic}_age_sim_s'] = sim_age
            result[f'{topic}_read_age_wall_s'] = wall_age
        result.update(healthy=reason is None, reason=reason,
                      battery_remaining=snapshot.battery.value if snapshot.battery is not None else None,
                      voltage5v_v=snapshot.power.value if snapshot.power is not None else None)
        return result

    def close(self):
        """Stop polling. Any in-progress listener is independently time-bounded."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=.05)
