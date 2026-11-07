"""Measured velocity should reduce corner overshoot with a lagging actuator."""
import unittest

from mission.core import CourseController


def replay(damping):
    config = {'start': [0, 0, 0],
              'waypoints': [{'position': [1, 0, 1], 'speed': .8},
                            {'position': [1, 1, 1], 'speed': .8}],
              'limits': {'arrival_radius': .08, 'velocity_damping': damping}}
    controller = CourseController(config)
    position, velocity, maximum_x = [0, 0, 1], [0, 0, 0], 0
    for step in range(2000):
        command = controller.update(position, velocity, step*.02)
        if command.fault:
            raise AssertionError(command.fault)
        # A first-order velocity response, deliberately independent of the
        # controller's command acceleration limit. This is not PX4 validation.
        velocity = [a+(.02/.3)*(b-a) for a, b in zip(velocity, command.velocity)]
        position = [a+.02*b for a, b in zip(position, velocity)]
        maximum_x = max(maximum_x, position[0])
        if command.done:
            return maximum_x-1, step*.02
    raise AssertionError('route did not finish')


class CornerBrakingTests(unittest.TestCase):
    def test_window_alignment_requires_its_tighter_tolerance(self):
        controller = CourseController({'start': [0, 0, 0],
            'waypoints': [{'position': [1, 0, 1], 'arrival_radius': .025},
                          {'position': [1, 1, 1]}], 'limits': {'arrival_radius': .08}})
        controller.update([.95, 0, 1], [0, 0, 0], 0)
        self.assertEqual(controller.waypoint_index, 0)
        controller.update([.98, 0, 1], [0, 0, 0], .02)
        self.assertEqual(controller.waypoint_index, 1)

    def test_feedback_brakes_before_lag_carries_vehicle_past_turn(self):
        baseline_error, baseline_time = replay(0)
        damped_error, damped_time = replay(.6)
        self.assertGreater(baseline_error, .15)
        self.assertLess(damped_error, .05)
        self.assertLess(damped_time, baseline_time+1)

    def test_invalid_damping_is_rejected(self):
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                replay(value)
