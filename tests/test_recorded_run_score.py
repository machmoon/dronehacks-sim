import copy
import unittest

from tools.score_recorded_run import score


class RecordedRunScoreTests(unittest.TestCase):
    def setUp(self):
        self.summary = dict(result='completed', elapsed_sim_s=50,
                            waypoint_count=15, waypoints_reached=15,
                            landed_and_disarmed=True)
        self.geometry = dict(recorded_time_range_sim_s=[0, 56],
                             maximum_recorded_gap_s=.04, maximum_interpolated_step_m=.02,
                             linear_path_clearance_lower_bound_m=.05,
                             obstacle_clearance=dict(minimum_signed_m=.06,
                                                     negative_clearance_samples=0),
                             window_crossing=dict(full_envelope_crossed=True,
                                                  min_aperture_clearance_m=.06))

    def test_complete_clear_faster_flight_passes(self):
        self.assertTrue(score(self.summary, self.geometry, 5)['accepted'])

    def test_missing_takeoff_or_landing_or_long_gap_rejects(self):
        for key, value in [('recorded_time_range_sim_s', [6, 56]),
                           ('recorded_time_range_sim_s', [0, 54]),
                           ('maximum_recorded_gap_s', .5)]:
            g = copy.deepcopy(self.geometry)
            g[key] = value
            self.assertFalse(score(self.summary, g, 5)['accepted'])

    def test_sampling_margin_and_incomplete_run_reject(self):
        self.geometry['linear_path_clearance_lower_bound_m'] = .035
        self.assertFalse(score(self.summary, self.geometry, 5)['accepted'])
        self.geometry['linear_path_clearance_lower_bound_m'] = .05
        self.summary['landed_and_disarmed'] = False
        self.assertFalse(score(self.summary, self.geometry, 5)['accepted'])
