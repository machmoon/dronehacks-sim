import unittest

import numpy as np

from learning.sim import Course, Flight, HEIGHT, WIDTH, Scene, teacher, yaw_rotation


class LearningSimulatorTests(unittest.TestCase):
    def test_sweep_crosses_thin_wall_even_when_endpoints_are_clear(self):
        scene = Scene(np.array([[0., 0., 1.]]), np.eye(3)[None],
                      np.array([[.01, .5, .5]]), {}, .2)
        a, b = np.array([-1., 0., 1.]), np.array([1., 0., 1.])
        self.assertGreater(scene.clearance(a), 0)
        self.assertGreater(scene.clearance(b), 0)
        self.assertTrue(scene.hits(a, b))
        # Inside the corner of an inflated AABB, but outside the rounded
        # swept sphere: the exact narrow phase must not report contact.
        a, b = a + [0, .66, .66], b + [0, .66, .66]
        self.assertFalse(scene.hits(a, b))

    def test_course_window_is_an_aperture_in_depth_and_collision(self):
        course = Course()
        scene = course.scene(np.random.default_rng(31), 0)
        c = np.array(scene.window["center"])
        n, h = course.axes[:, 0], course.axes[:, 1]
        a, b = c + .45 * n, c - .45 * n
        self.assertFalse(scene.hits(a, b))
        self.assertTrue(scene.hits(a + .27 * h, b + .27 * h))
        rotation = yaw_rotation(float(np.arctan2(-n[1], -n[0])))
        camera = scene.depth(a, rotation)
        off_center = scene.depth(a + .52 * h, rotation)
        self.assertEqual(camera.shape, (HEIGHT, WIDTH))
        self.assertGreater(camera[HEIGHT // 2, WIDTH // 2],
                           off_center[HEIGHT // 2, WIDTH // 2] + .35)

    def test_noisy_state_reproducible_without_rendering(self):
        course = Course()
        visible, blank = [Flight(course, 603, 1) for _ in range(2)]
        for _ in range(4):
            a, b = visible.observe(), blank.observe(render=False)
            np.testing.assert_array_equal(a.state, b.state)
            self.assertGreater(a.depth.std(), .10)
            self.assertFalse(b.depth.any())
            action = np.array([.2, 0., 0.])
            visible.step(action)
            blank.step(action)

    def test_expert_crosses_seeded_course_and_scorer_reaches_approach(self):
        flight = Flight(Course(), 89002, 1.)
        while not flight.done:
            flight.observe(render=False)
            flight.step(teacher(flight))
        result = flight.result()
        self.assertTrue(result["success"], result)
        self.assertTrue(result["window_crossed"])
        self.assertFalse(result["collision"])
        self.assertEqual(result["route_progress"], 1)
