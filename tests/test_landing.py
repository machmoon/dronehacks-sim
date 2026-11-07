import unittest

from mission.landing import LandingMonitor


class LandingTests(unittest.TestCase):
    def monitor(self):
        m = LandingMonitor((1, 2, 0), .762, .36, timeout=20)
        m.start(10)
        return m

    def test_hover_or_land_command_does_not_count_as_finish(self):
        result = self.monitor().check(11, (1, 2, .7), armed=True,
                                      on_ground=False, telemetry_fresh=True)
        self.assertEqual(result, ('landing', None))

    def test_touchdown_still_requires_disarm(self):
        self.assertEqual(self.monitor().check(11, (1, 2, .2), armed=True,
                         on_ground=True, telemetry_fresh=True)[0], 'landing')

    def test_confirmed_ground_and_disarm_inside_pad_complete(self):
        self.assertEqual(self.monitor().check(11, (1.1, 2, .2), armed=False,
                         on_ground=True, telemetry_fresh=True), ('completed', None))

    def test_center_inside_pad_is_insufficient_if_envelope_overhangs(self):
        self.assertEqual(self.monitor().check(11, (1.5, 2, .2), armed=False,
                         on_ground=True, telemetry_fresh=True)[1], 'landed_outside_pad')

    def test_disarm_without_ground_confirmation_fails(self):
        self.assertEqual(self.monitor().check(11, (1, 2, .7), armed=False,
                         on_ground=False, telemetry_fresh=True)[1],
                         'disarmed_without_ground_confirmation')

    def test_stale_ground_report_cannot_complete(self):
        self.assertEqual(self.monitor().check(11, (1, 2, .2), armed=False,
                         on_ground=True, telemetry_fresh=False)[1], 'landing_telemetry_lost')

    def test_timeout_and_clock_reversal_fail(self):
        for t, reason in ((31, 'landing_timeout'), (9, 'landing_clock_reversed')):
            self.assertEqual(self.monitor().check(t, (1, 2, .7), armed=True,
                             on_ground=False, telemetry_fresh=True)[1], reason)

    def test_invalid_geometry_rejected(self):
        for radius in (.2, float('nan')):
            with self.assertRaises(ValueError):
                LandingMonitor((0, 0, 0), radius, .36)
