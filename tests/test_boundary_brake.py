from collections import deque
import math
import unittest

from mission.boundary_brake import BoundaryBrake
from mission.core import limited, norm, sub


class BoundaryBrakeTests(unittest.TestCase):
    def setUp(self):
        self.guard = BoundaryBrake((-3., -3., 0.), (3., 3.048, 2.), .401)

    def step(self, p, v, desired, previous, *, dt=.04):
        return self.guard.step(p, v, desired, previous, slew=.8, dt=dt, cap=.5)

    def test_returns_original_slew_and_speed_cap_away_from_bounds(self):
        result = self.step((0., 0., 1.), (0., 0., 0.), (.7, 0., 0.), (.48, 0., 0.))
        self.assertEqual(result.sent_velocity, (.5, 0., 0.))
        self.assertEqual(result.unfiltered_velocity, result.sent_velocity)
        self.assertEqual(result.faces, ())

    def test_uses_current_velocity_to_brake_before_north_limit(self):
        previous = (.04, .493, 0.)
        result = self.step((-.29, 2.10, .95), (.008, .531, 0.), (.02, .494, 0.), previous)
        self.assertEqual(result.faces, ('y+',))
        self.assertAlmostEqual(result.sent_velocity[1], previous[1]-.8*.04)
        self.assertEqual(result.sent_velocity[0], previous[0])
        self.assertLess(result.sent_velocity[1], result.unfiltered_velocity[1])

    def test_tangential_motion_survives_when_no_outward_command_is_needed(self):
        result = self.step((0., 2.62, .95), (0., .15, 0.), (.4, -.02, .08), (.1, 0., .03))
        self.assertEqual(result.faces, ())
        self.assertGreater(dict(result.stop_budget_deficits_m)['y+'], .10)
        self.assertGreater(result.sent_velocity[0], .1)
        self.assertLess(result.sent_velocity[1], 0.)
        self.assertGreater(result.sent_velocity[2], .03)

    def test_corner_recovery_respects_vector_slew(self):
        previous = (-.31, .32, .05)
        result = self.step((-2.56, 2.61, 1.), (-.3, .4, 0.), (-.2, .3, .15), previous)
        self.assertEqual(result.faces, ('x-', 'y+'))
        self.assertLessEqual(norm(sub(result.sent_velocity, previous)), .8*.04+1e-12)
        self.assertGreater(result.sent_velocity[0], previous[0])
        self.assertLess(result.sent_velocity[1], previous[1])
        self.assertEqual(result.sent_velocity[2], previous[2])

    def test_brakes_symmetrically_at_all_planar_faces(self):
        for i, face, edge, direction in ((0, 'x-', self.guard.lower[0], -1.),
                                         (0, 'x+', self.guard.upper[0], 1.),
                                         (1, 'y-', self.guard.lower[1], -1.),
                                         (1, 'y+', self.guard.upper[1], 1.)):
            with self.subTest(face=face):
                p, v, command = [0., 0., 1.], [0., 0., 0.], [0., 0., 0.]
                p[i] = edge-direction*.55
                v[i], command[i] = direction*.531, direction*.493
                result = self.step(p, v, command, command)
                self.assertEqual(result.faces, (face,))
                self.assertGreater(dict(result.stop_budget_deficits_m)[face], 0.)
                self.assertAlmostEqual(direction*result.sent_velocity[i], .493-.8*.04)

    def test_delayed_corner_stress_with_pose_bias_and_overspeed(self):
        dt, cap = .04, .495
        direction = (1/math.sqrt(2), -1/math.sqrt(2), 0.)
        desired = tuple(cap*u for u in direction)
        previous = desired
        pending = deque([previous]*6)
        p = (self.guard.upper[0]-.65, self.guard.lower[1]+.65, .95)
        v = tuple(.531*u for u in direction)
        activated = set()
        for _ in range(175):
            estimate = (p[0]-.02, p[1]+.02, p[2])
            result = self.guard.step(estimate, v, desired, previous, slew=.8, dt=dt, cap=cap)
            activated.update(result.faces)
            previous = result.sent_velocity
            pending.append(previous)
            applied = tuple(1.08*u for u in pending.popleft())
            acceleration = limited(tuple((u-speed)/.80 for u, speed in zip(applied, v)), .8)
            p = tuple(x+dt*speed+.5*dt**2*a for x, speed, a in zip(p, v, acceleration))
            v = tuple(speed+dt*a for speed, a in zip(v, acceleration))
            self.assertLess(p[0], self.guard.upper[0])
            self.assertGreater(p[1], self.guard.lower[1])
        self.assertEqual(activated, {'x+', 'y-'})

    def test_near_edge_allows_inward_escape_and_limits_outward_projection(self):
        result = self.step((0., 2.63, 1.), (0., .01, 0.), (.1, .1, 0.), (0., 0., 0.))
        self.assertEqual(result.faces, ('y+',))
        self.assertGreater(result.sent_velocity[0], 0.)
        self.assertEqual(result.sent_velocity[1], 0.)
        escape = self.step((0., 2.63, 1.), (0., .01, 0.), (.1, -.1, 0.), (0., 0., 0.))
        self.assertEqual(escape.faces, ())
        self.assertLess(escape.sent_velocity[1], 0.)

    def test_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError):
            self.step((0., float('nan'), 1.), (0., 0., 0.), (0., 0., 0.), (0., 0., 0.))


if __name__ == '__main__':
    unittest.main()
