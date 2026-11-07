import os
import subprocess
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mission.sitl_health import (HealthReadError, SitlHealthMonitor,
                                 parse_battery_status, parse_system_power)


def battery(stamp, remaining='.87'):
    return f'''
TOPIC: battery_status
 battery_status
    timestamp: {stamp} (0.004000 seconds ago)
    timestamp_sample: 4
    connected: True
    source: 0
    id: 1
    warning: 0
    faults: 0
    is_powering_off: False
    remaining: {remaining}
'''


def power(stamp, voltage='5.00000'):
    return f'''
TOPIC: system_power
 system_power
    timestamp: {stamp} (0.004000 seconds ago)
    voltage5v_v: {voltage}
    brick_valid: 1
    periph_5v_oc: 0
    hipower_5v_oc: 0
'''


class ParsingTests(unittest.TestCase):
    def test_parses_listener_records_by_exact_field_name(self):
        self.assertEqual(parse_battery_status(battery(132000)), (132000, .87))
        self.assertEqual(parse_battery_status(battery(132000, '1.0')), (132000, 1.))
        self.assertEqual(parse_system_power(power(140000)), (140000, 5.))
        self.assertEqual(parse_system_power(power(140000, '4.5')), (140000, 4.5))
        self.assertEqual(parse_system_power(power(140000, '5.5')), (140000, 5.5))

    def test_rejects_unhealthy_battery_or_malformed_output(self):
        for text in (battery(1, '.15'), battery(1, '1.01'), battery(1, 'nan'), battery(1, '1e1000'),
                     battery(1).replace('connected: True', 'connected: False'),
                     battery(1).replace('source: 0', 'source: 1'),
                     battery(1).replace('id: 1', 'id: 2'),
                     battery(1).replace('warning: 0', 'warning: 1'),
                     battery(1).replace('faults: 0', 'faults: 16'),
                     battery(1).replace('is_powering_off: False', 'is_powering_off: True'),
                     battery(1).replace('timestamp:', 'other_timestamp:'),
                     battery(1) + '    timestamp: 2\n',
                     battery(0), battery(2**64), 'Never published'):
            with self.subTest(text=text), self.assertRaises(HealthReadError):
                parse_battery_status(text)

    def test_rejects_invalid_rail_or_overcurrent(self):
        for text in (power(1, '4.49'), power(1, '5.51'), power(1, 'inf'),
                     power(1).replace('brick_valid: 1', 'brick_valid: 0'),
                     power(1).replace('periph_5v_oc: 0', 'periph_5v_oc: 1'),
                     power(1).replace('hipower_5v_oc: 0', 'hipower_5v_oc: 1')):
            with self.subTest(text=text), self.assertRaises(HealthReadError):
                parse_system_power(text)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'DRONEHACKS_SITL': '1'})
        env.start()
        self.addCleanup(env.stop)
        self.now = 10.
        self.calls = []
        self.responses = {}
        self.monitor = SitlHealthMonitor(start=False, runner=self.run_listener, clock=lambda: self.now)
        self.addCleanup(self.monitor.close)

    def run_listener(self, args, **kwargs):
        self.calls.append((args, kwargs))
        result = self.responses[args[1]]
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(stdout=result)

    def poll(self, battery_usec, power_usec):
        self.responses = {'battery_status': battery(battery_usec), 'system_power': power(power_usec)}
        self.monitor.poll()

    def test_live_constructor_requires_sitl_environment(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'DRONEHACKS_SITL'):
            SitlHealthMonitor(start=False)

    def test_readiness_requires_both_timestamps_to_advance(self):
        self.assertFalse(self.monitor.status(1., self.now)['healthy'])
        self.poll(1_000_000, 1_004_000)
        self.assertFalse(self.monitor.status(1.01, self.now)['healthy'])
        self.poll(1_200_000, 1_204_000)
        ready = self.monitor.status(1.21, self.now)
        self.assertTrue(ready['healthy'], ready)
        self.assertEqual(ready['consecutive_samples'], 2)
        self.assertEqual(ready['battery_status_timestamp_usec'], 1_200_000)
        self.assertEqual(ready['system_power_timestamp_usec'], 1_204_000)
        self.assertAlmostEqual(ready['battery_status_age_sim_s'], .01)
        self.assertAlmostEqual(ready['system_power_age_sim_s'], .006)
        self.assertEqual(ready['voltage5v_v'], 5.)
        for args, options in self.calls:
            self.assertIn(args, [['px4-listener', topic, '-n', '1']
                                for topic in ('battery_status', 'system_power')])
            self.assertEqual(options['timeout'], 1.5)
            self.assertTrue(options['check'])
            self.assertFalse(options.get('shell', False))

        self.poll(1_400_000, 1_204_000)
        mixed = self.monitor.status(1.41, self.now)
        self.assertTrue(mixed['healthy'])
        self.assertEqual(mixed['consecutive_samples'], 2)
        self.poll(1_399_000, 1_400_000)
        self.assertEqual(self.monitor.status(1.41, self.now)['reason'],
                         'battery_status_timestamp_regressed')
        self.poll(1_600_000, 1_604_000)
        self.assertFalse(self.monitor.status(1.61, self.now)['healthy'])
        self.poll(1_800_000, 1_804_000)
        self.assertTrue(self.monitor.status(1.81, self.now)['healthy'])
        self.poll(2_000_000, 1_803_000)
        self.assertEqual(self.monitor.status(2.01, self.now)['reason'],
                         'system_power_timestamp_regressed')

    def test_duplicate_before_readiness_does_not_count_as_a_new_pair(self):
        self.poll(1_000_000, 1_004_000)
        self.poll(1_000_000, 1_004_000)
        self.poll(1_100_000, 1_004_000)
        not_ready = self.monitor.status(1.11, self.now)
        self.assertFalse(not_ready['healthy'])
        self.assertEqual(not_ready['consecutive_samples'], 1)
        self.poll(1_200_000, 1_204_000)
        self.assertTrue(self.monitor.status(1.21, self.now)['healthy'])

    def test_fresh_duplicate_remains_ready_until_original_sample_expires(self):
        self.poll(126_600_000, 126_600_000)
        self.poll(126_800_000, 126_800_000)
        self.now += .7
        self.poll(126_800_000, 126_800_000)
        fresh = self.monitor.status(126.788, self.now)
        self.assertTrue(fresh['healthy'], fresh)
        self.assertEqual(fresh['consecutive_samples'], 2)
        self.assertAlmostEqual(fresh['battery_status_age_sim_s'], -.012)
        self.assertAlmostEqual(fresh['battery_status_read_age_wall_s'], .7)
        self.assertEqual(self.monitor.status(127.601, self.now)['reason'],
                         'battery_status_stale_sim_timestamp')
        self.now += 3.3
        self.poll(126_800_000, 126_800_000)
        self.assertEqual(self.monitor.status(126.788, self.now)['reason'],
                         'battery_status_stale_wall_read')

    def test_status_rejects_stale_and_future_timestamps_or_reads(self):
        self.poll(1_000_000, 1_000_000)
        self.poll(1_200_000, 1_200_000)
        self.assertTrue(self.monitor.status(1., self.now)['healthy'])
        self.assertTrue(self.monitor.status(2., self.now)['healthy'])
        self.assertEqual(self.monitor.status(.99, self.now)['reason'], 'battery_status_stale_sim_timestamp')
        self.assertEqual(self.monitor.status(2.01, self.now)['reason'], 'battery_status_stale_sim_timestamp')
        self.assertEqual(self.monitor.status(1.21, self.now+4.)['reason'], 'battery_status_stale_wall_read')
        invalid = self.monitor.status(float('nan'), self.now)
        self.assertEqual(invalid['reason'], 'invalid_health_clock')
        self.assertIsNone(invalid['battery_status_age_sim_s'])

    def test_transport_timeout_preserves_only_the_fresh_original_pair(self):
        self.poll(75_600_000, 75_600_000)
        self.poll(75_652_000, 75_652_000)
        self.now += 2.615
        self.responses['battery_status'] = subprocess.TimeoutExpired('px4-listener', 1.5)
        self.monitor.poll()
        status = self.monitor.status(75.844, self.now)
        self.assertTrue(status['healthy'], status)
        self.assertEqual(status['read_error'], 'battery_status_listener_timeout')
        self.assertEqual(status['consecutive_transport_errors'], 1)
        self.assertEqual(status['battery_status_timestamp_usec'], 75_652_000)
        self.assertAlmostEqual(status['battery_status_read_age_wall_s'], 2.615)
        self.assertEqual(self.monitor.status(76.453, self.now)['reason'],
                         'battery_status_stale_sim_timestamp')
        self.now += 1.385
        self.monitor.poll()
        expired = self.monitor.status(75.844, self.now)
        self.assertFalse(expired['healthy'])
        self.assertEqual(expired['reason'], 'battery_status_stale_wall_read')
        self.assertEqual(expired['consecutive_transport_errors'], 2)

    def test_transport_failure_before_first_sample_is_not_ready(self):
        self.responses['battery_status'] = subprocess.TimeoutExpired('px4-listener', 1.5)
        self.monitor.poll()
        status = self.monitor.status(1., self.now)
        self.assertFalse(status['healthy'])
        self.assertEqual(status['read_error'], 'battery_status_listener_timeout')
        self.assertIsNone(status['battery_status_timestamp_usec'])

    def test_bad_publication_drops_readiness_even_with_a_recent_good_sample(self):
        self.poll(1_000_000, 1_000_000)
        self.poll(1_200_000, 1_200_000)
        self.responses['battery_status'] = battery(1_220_000, '.08')
        self.monitor.poll()
        status = self.monitor.status(1.22, self.now)
        self.assertFalse(status['healthy'])
        self.assertEqual(status['reason'], 'battery_status_low_remaining')
        self.assertEqual(status['consecutive_samples'], 0)
        self.responses['battery_status'] = battery(1_220_000)
        self.responses['system_power'] = 'Never published'
        self.monitor.poll()
        self.assertEqual(self.monitor.status(1.22, self.now)['reason'], 'system_power_missing_timestamp')
        self.poll(1_400_000, 1_400_000)
        self.assertFalse(self.monitor.status(1.42, self.now)['healthy'])
        self.poll(1_600_000, 1_600_000)
        self.assertTrue(self.monitor.status(1.62, self.now)['healthy'])

    def test_close_returns_while_listener_is_in_progress(self):
        entered, release, finished = Event(), Event(), Event()
        calls = []

        def blocked_listener(args, **kwargs):
            calls.append(args[1])
            entered.set()
            release.wait(timeout=1.)
            finished.set()
            return SimpleNamespace(stdout=battery(132000))

        monitor = SitlHealthMonitor(runner=blocked_listener)
        try:
            self.assertTrue(entered.wait(timeout=.5))
            before = time.monotonic()
            monitor.close()
            self.assertLess(time.monotonic()-before, .3)
            self.assertFalse(finished.is_set())
            self.assertEqual(monitor.status(1., time.monotonic())['reason'], 'monitor_closed')
        finally:
            release.set()
            monitor.close()
        self.assertEqual(calls, ['battery_status'])


if __name__ == '__main__':
    unittest.main()
