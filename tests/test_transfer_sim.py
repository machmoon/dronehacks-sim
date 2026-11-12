from dataclasses import replace
import hashlib
import unittest

import numpy as np

from learning import sim
from learning.transfer import (GEOMETRY_DIFFICULTY_MAX, InfeasibleScenario, Motion, SUBSTEP_S,
                               TransferCourse, plant_substep)
from mission.airframe import X500_SDF


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = TransferCourse()

    def test_plant_slews_setpoint_before_applying_response(self):
        zero = np.zeros(3)
        motion = Motion(zero, zero, zero, 0.)
        fast = plant_substep(motion, np.array([.5, 0., 0.]), .5, .5, zero, 0.)
        slow = plant_substep(motion, np.array([.5, 0., 0.]), .5, .8, zero, 0.)
        self.assertAlmostEqual(fast.command[0], .8*SUBSTEP_S)
        self.assertAlmostEqual(fast.velocity[0], fast.command[0]/.5*SUBSTEP_S)
        self.assertLess(slow.velocity[0], fast.velocity[0])

    def test_rollout_executes_four_shared_plant_steps(self):
        flight = self.course.make_flight(101, 0., response=.66)
        obs = flight.observe()
        self.assertEqual(obs.depth.shape, (sim.HEIGHT, sim.WIDTH))
        self.assertEqual(obs.state.shape, (9,))
        action = self.course.instructor(flight)
        cap = self.course.speeds[0]
        self.assertLessEqual(np.linalg.norm(action), cap+1e-8)
        expected = flight.motion()
        for _ in range(4):
            expected = plant_substep(expected, flight.rotation@action, cap, flight.response,
                                     flight.wind, flight.yaw_command)
        flight.step(action)
        self.assertEqual(len(flight.path), 5)
        self.assertEqual(len(flight.orientations), 5)
        np.testing.assert_allclose(flight.p, expected.position)
        np.testing.assert_allclose(flight.command, expected.command)

    def test_geometry_cap_does_not_reduce_sensor_stress(self):
        low = self.course.scene(np.random.default_rng(7), GEOMETRY_DIFFICULTY_MAX)
        high = self.course.scene(np.random.default_rng(7), 1.)
        np.testing.assert_array_equal(low.centers, high.centers)
        self.assertEqual(low.window, high.window)
        flight = self.course.make_flight(7, 1., 1.8, response=.66)
        self.assertEqual((flight.difficulty, flight.sensor_stress), (1., 1.8))

    def test_uncertified_reference_reports_seed_and_collision_component(self):
        scene = self.course.scene(np.random.default_rng(4), 0.)
        obstacle = replace(scene, centers=np.vstack([scene.centers, self.course.start]),
                           axes=np.concatenate([scene.axes, np.eye(3)[None]], axis=0),
                           half=np.vstack([scene.half, np.ones(3)]))
        with self.assertRaises(InfeasibleScenario) as error:
            self.course.certify_scenario(obstacle, 100004, .9)
        self.assertEqual(error.exception.seed, 100004)
        self.assertEqual(error.exception.geometry_difficulty, GEOMETRY_DIFFICULTY_MAX)
        self.assertEqual(error.exception.detail['obstacle'], len(scene.centers))

    def test_route_uses_mission_acceptance_and_boundary(self):
        flight = self.course.make_flight(102, 0., response=.66)
        flight.p = self.course.goals[0] - [0., .081, 0.]
        flight._reset_trace()
        flight.step(np.zeros(3))
        self.assertEqual((flight.index, flight.truth_index), (0, 0))
        flight.p = self.course.goals[0] - [0., .079, 0.]
        flight._reset_trace()
        flight.step(np.zeros(3))
        self.assertEqual((flight.index, flight.truth_index), (0, 1))
        flight.p = self.course.goals[0] - [0., .059, 0.]
        flight._reset_trace()
        flight.step(np.zeros(3))
        self.assertEqual((flight.index, flight.truth_index), (1, 1))
        self.assertLess(self.course.bounds_gap(np.array([0., 2.65, .95])), 0.)

    def test_metadata_hashes_route_and_collision_model(self):
        profile = self.course.profile_metadata()
        self.assertEqual(profile['route_sha256'], hashlib.sha256(self.course.route_path.read_bytes()).hexdigest())
        self.assertEqual(profile['x500_sdf_sha256'], hashlib.sha256(X500_SDF.read_bytes()).hexdigest())
        self.assertEqual(profile['velocity_response_s'], [.5, .8])
        self.assertEqual(profile['envelope_margin_m'], .015)
        self.assertEqual(profile['estimated_waypoint_arrival_radii_m'], [.06]*len(self.course.goals))
        self.assertEqual(profile['truth_waypoint_arrival_radii_m'], [.08]*len(self.course.goals))


if __name__ == '__main__':
    unittest.main()
