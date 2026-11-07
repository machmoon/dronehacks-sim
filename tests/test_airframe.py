import math
from types import SimpleNamespace
import unittest

import numpy as np

from mission.airframe import (Component, Envelope, check_scene_path, evaluate_window_crossing,
                              face_clearance_at_poses,
                              load_x500_envelope, quaternion_xyzw_from_rpy)


IDENTITY = (0., 0., 0., 1.)
WINDOW = dict(center=(0., 0., 0.), normal_toward_entry=(1., 0., 0.),
              horizontal_axis=(0., 1., 0.), width=.9144, height=.9144, thickness=.1016)


def box(half, center=(0., 0., 0.)):
    return Component('box', center, np.eye(3), half)


class AirframeTests(unittest.TestCase):
    def test_x500_covers_body_legs_skids_and_all_rotors(self):
        envelope = load_x500_envelope()
        components = envelope.components
        self.assertEqual({part.name for part in components},
                         {f'base_link_collision_{i}' for i in range(5)}
                         | {f'rotor_{i}_collision' for i in range(4)})
        self.assertEqual(envelope.margin_m, .015)
        self.assertAlmostEqual(components[0].center_flu[2], .007)
        self.assertNotEqual(components[1].rotation_flu[1, 2], 0.)
        lowest = min(part.center_flu[2] - part.half_support(np.eye(3), (0., 0., 1.))
                     for part in components)
        self.assertAlmostEqual(lowest, -.227)

    def test_rotor_support_uses_full_spin_and_tilts_with_body(self):
        rotor = next(part for part in load_x500_envelope().components if part.swept_rotor)
        radius = math.hypot(.2792307692307692/2, .016923076923076923/2)
        np.testing.assert_allclose(rotor.half_support(np.eye(3), np.eye(3)),
                                   [radius, radius, .0008461538461538462/2])
        quarter_roll = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        self.assertAlmostEqual(rotor.half_support(quarter_roll, (0., 0., 1.)), radius)
        self.assertAlmostEqual(np.linalg.norm(rotor.center_flu[:2])+radius,
                               math.hypot(.174, .174)+radius)

    def test_center_over_wall_is_not_a_whole_airframe_crossing(self):
        envelope = load_x500_envelope()
        partial = evaluate_window_crossing([(1., 0., 0.), (-.20, 0., 0.)],
                                           [IDENTITY]*2, WINDOW, envelope)
        self.assertFalse(partial['full_envelope_crossed'])
        self.assertEqual(partial['direction'], 'incomplete')
        result = evaluate_window_crossing([(1., 0., 0.), (-1., 0., 0.)],
                                          [IDENTITY]*2, WINDOW, envelope)
        self.assertTrue(result['full_envelope_crossed'])
        self.assertEqual(result['valid_forward_crossings'], 1)

    def test_window_and_static_boxes_reject_touch_with_margin(self):
        envelope = Envelope((box((.1, .2, .1)),))
        window = {**WINDOW, 'width': .43}
        path = [(1., 0., 0.), (-1., 0., 0.)]
        result = evaluate_window_crossing(path, [IDENTITY]*2, window, envelope)
        self.assertFalse(result['full_envelope_crossed'])
        self.assertAlmostEqual(result['min_aperture_clearance_m'], 0.)
        obstacle = SimpleNamespace(centers=[[0., 0., 0.]], axes=[np.eye(3)], half=[[.2, .1, .1]])
        at_margin = [(0., .315, 0.)]
        self.assertFalse(check_scene_path(at_margin, [IDENTITY], obstacle, envelope,
                                          check_ground=False)['certified_clear'])
        self.assertTrue(check_scene_path([(0., .316, 0.)], [IDENTITY], obstacle, envelope,
                                         check_ground=False)['certified_clear'])

    def test_static_path_rejects_overlap_between_clear_endpoints(self):
        envelope = Envelope((box((.1, .1, .1)),))
        obstacle = SimpleNamespace(centers=[[0., 0., 0.]], axes=[np.eye(3)], half=[[.1, .1, .1]])
        result = check_scene_path([(1., 0., 0.), (-1., 0., 0.)], [IDENTITY]*2, obstacle,
                                  envelope, check_ground=False)
        self.assertFalse(result['certified_clear'])
        self.assertEqual(result['first_unresolved'], dict(segment=0, component='box', obstacle=0))

    def test_batched_candidate_clearance_distinguishes_separation_with_attitude(self):
        envelope = Envelope((box((.2, .1, .1)),))
        obstacle = SimpleNamespace(centers=[[0., 0., 0.]], axes=[np.eye(3)], half=[[.1, .1, .1]])
        quarter_yaw = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        positions = [(0., .25, 0.), (0., .25, 0.)]
        bounds = face_clearance_at_poses(positions, [np.eye(3), quarter_yaw], obstacle, envelope,
                                         check_ground=False)
        np.testing.assert_allclose(bounds, [.035, -.065], atol=2e-9)

    def test_window_checks_attitude_sweep_between_clear_endpoint_attitudes(self):
        envelope = Envelope((box((.4, .04, .03)),), margin_m=0.)
        window = {**WINDOW, 'width': .68}
        positions = [(1., 0., 0.), (.2, 0., 0.), (-.2, 0., 0.), (-1., 0., 0.)]
        a = quaternion_xyzw_from_rpy(0., 0., math.pi/4)
        b = quaternion_xyzw_from_rpy(0., 0., 3*math.pi/4)
        fixed = evaluate_window_crossing(positions, [a]*4, window, envelope)
        turn = evaluate_window_crossing(positions, [a, a, b, b], window, envelope)
        self.assertTrue(fixed['full_envelope_crossed'])
        self.assertFalse(turn['full_envelope_crossed'])
        self.assertLess(turn['min_aperture_clearance_m'], -.05)


if __name__ == '__main__':
    unittest.main()
