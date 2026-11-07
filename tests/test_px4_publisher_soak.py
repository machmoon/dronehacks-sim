import unittest

from tools.soak_px4_publishers import assess, stamp


def sample(t):
    return dict(imu_before_usec=t, imu_after_usec=t+4000,
                battery_status_timestamp_usec=t, system_power_timestamp_usec=t,
                cpuload_timestamp_usec=t-400000)


class PublisherSoakTests(unittest.TestCase):
    def test_advancing_publishers_cover_interval(self):
        self.assertTrue(assess([sample(t) for t in (1000000, 2000000, 3000000)], 2)['passed'])

    def test_frozen_publisher_fails_with_advancing_imu(self):
        rows = [sample(t) for t in (1000000, 2000000, 3000000)]
        for row in rows:
            row['battery_status_timestamp_usec'] = 1000000
        result = assess(rows, 2)
        self.assertFalse(result['passed'])
        self.assertTrue(any('battery_status_stale' in error for error in result['errors']))

    def test_paused_clock_and_transport_errors_cannot_pass(self):
        self.assertFalse(assess([sample(1000000)]*3, 2)['passed'])
        result = assess([sample(1000000), {'error': 'timeout'}, sample(4000000)], 2)
        self.assertFalse(result['passed'])

    def test_duplicate_timestamp_fields_rejected(self):
        self.assertEqual(stamp('  timestamp: 4000000 (0.002 seconds ago)', 'cpuload'), 4000000)
        with self.assertRaises(ValueError):
            stamp('timestamp: 4000000\ntimestamp: 4000000', 'cpuload')
