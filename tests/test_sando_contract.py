import math
import unittest

from planning.sando_contract import evaluate_goal, state_from_odometry


class SandoContractTests(unittest.TestCase):
    def goal(self, **changes):
        data = dict(stamp=10, frame='map', p=[0, 0, 1], v=[.2, 0, 0],
                    a=[0, 0, 0], j=[0, 0, 0], yaw=0, dyaw=0)
        data.update(changes)
        return data

    def test_body_velocity_rotates_into_planner_world_frame(self):
        state = state_from_odometry([1, 2, 3], [0, 0, math.sqrt(.5), math.sqrt(.5)], [1, 0, 0])
        self.assertAlmostEqual(state['velocity_world'][0], 0)
        self.assertAlmostEqual(state['velocity_world'][1], 1)

    def test_valid_sample_remains_shadow_only(self):
        r = evaluate_goal(self.goal(), [0, 0, 1], 10.05)
        self.assertTrue(r['accepted'])
        self.assertEqual(r['candidate_velocity'], (.2, 0, 0))
        self.assertFalse(r['flight_authorized'])

    def test_age_frame_and_tracking_errors_reject(self):
        for changes, reason in (({'stamp': 9}, 'stale_goal'),
                                ({'stamp': 11}, 'future_goal'),
                                ({'frame': 'odom'}, 'frame_mismatch'),
                                ({'p': [1, 0, 1]}, 'tracking_error_exceeded'),
                                ({'mode_xy': 2}, 'unsupported_tracking_mode')):
            r = evaluate_goal(self.goal(**changes), [0, 0, 1], 10)
            self.assertFalse(r['accepted'])
            self.assertEqual(r['reason'], reason)

    def test_vector_speed_limit_cannot_be_evaded_with_per_axis_limits(self):
        r = evaluate_goal(self.goal(v=[.8, .8, 0]), [0, 0, 1], 10)
        self.assertFalse(r['accepted'])
        self.assertEqual(r['reason'], 'speed_limit_exceeded')

    def test_nonfinite_and_zero_quaternion_rejected(self):
        self.assertFalse(evaluate_goal(self.goal(a=[float('nan'), 0, 0]), [0, 0, 1], 10)['accepted'])
        with self.assertRaises(ValueError):
            state_from_odometry([0, 0, 1], [0, 0, 0, 0], [0, 0, 0])
