"""Offline scoring must not fabricate observations across recording gaps."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from learning.sim import Scene
from mission.airframe import load_x500_envelope, quaternion_xyzw_from_rpy
from tools.score_markerfree import (ARMED_EVENT, HOVER_EVENT, TOUCHDOWN_EVENT, Truth,
                                    _abort_landing, _contacts, _multipart_scene, _multipart_window,
                                    _position_alignment, _proxy_collisions, _truth_waypoints,
                                    _window, score_run)


WINDOW = dict(center=[10., 10., .8], normal_toward_entry=[1., 0., 0.],
              horizontal_axis=[0., 1., 0.], width=.9144, height=.9144, thickness=.1016)


def truth_row(stamp, position):
    return dict(sim_time_s=stamp, base_link_position_enu_m=list(position),
                base_link_orientation_xyzw=[0., 0., 0., 1.])


class ScoreTests(unittest.TestCase):
    def test_abort_disarm_does_not_hide_post_fault_obstacle_overlap(self):
        truth = Truth([truth_row(1., [-1., 0., .8]), truth_row(1.1, [1., 0., .227])], .12)
        obstacle = Scene(np.array([[0., 0., .6]]), np.eye(3)[None],
                         np.array([[.02, .5, .5]]), WINDOW, .28)
        events = [dict(message='PX4 confirmed touchdown and disarm after abort', sim_time=1.1)]
        summary = dict(result='failed', landed_after_fault=True)
        result = _abort_landing(summary, events, truth, [], {}, 1., obstacle, load_x500_envelope())
        self.assertTrue(result['controller_confirmed_ground_and_disarm'])
        self.assertFalse(result['certified_clear_through_disarm'])
        self.assertEqual(result['cad_proxy']['phases']['abort_landing']['first_unresolved']['obstacle'], 0)
        self.assertFalse(result['cad_proxy']['phases']['abort_landing']['ground_checked'])

    def test_abort_without_disarm_or_complete_truth_cannot_certify_clearance(self):
        truth = Truth([truth_row(1., [-1., 0., .8]), truth_row(1.5, [-1., 0., .227])], .12)
        far = Scene(np.array([[0., 10., .6]]), np.eye(3)[None],
                    np.array([[.02, .5, .5]]), WINDOW, .28)
        summary = dict(result='failed', landed_after_fault=True)
        args = (truth, [], {}, 1., far, load_x500_envelope())
        absent = _abort_landing(summary, [], *args)
        self.assertFalse(absent['controller_confirmed_ground_and_disarm'])
        self.assertFalse(absent['certified_clear_through_disarm'])
        self.assertIsNone(absent['cad_proxy'])
        events = [dict(message='PX4 confirmed touchdown and disarm after abort', sim_time=1.5)]
        no_confirmation_flag = _abort_landing(dict(result='failed'), events, *args)
        self.assertFalse(no_confirmation_flag['controller_confirmed_ground_and_disarm'])
        self.assertFalse(no_confirmation_flag['certified_clear_through_disarm'])
        incomplete = _abort_landing(summary, events, *args)
        self.assertTrue(incomplete['controller_confirmed_ground_and_disarm'])
        self.assertFalse(incomplete['certified_clear_through_disarm'])
        self.assertFalse(incomplete['cad_proxy']['phases']['abort_landing']['truth_complete'])

    def test_abort_cad_clearance_and_received_contacts_are_separate_evidence(self):
        truth = Truth([truth_row(1., [-1., 0., .8]), truth_row(1.1, [-1., 0., .227])], .12)
        far = Scene(np.array([[0., 10., .6]]), np.eye(3)[None],
                    np.array([[.02, .5, .5]]), WINDOW, .28)
        events = [dict(message='PX4 confirmed touchdown and disarm after abort', sim_time=1.1)]
        summary = dict(result='failed', landed_after_fault=True)
        contacts = [dict(sim_time_s=1.1, pairs=[['course_quad::base_link', 'other_obstacle']])]
        result = _abort_landing(summary, events, truth, contacts, {}, 1., far, load_x500_envelope())
        self.assertTrue(result['certified_clear_through_disarm'])
        self.assertEqual(result['contacts']['raw_pair_observations']['obstacle'], 1)

    def test_truth_waypoint_has_separate_verification_radius(self):
        parts = [[(1., np.array([-.1, .07, 1.])), (1.1, np.array([.1, .07, 1.]))]]
        waypoint = dict(position=[0., 0., 1.], phase='MAZE', arrival_radius=.06)
        metadata = dict(waypoints=[waypoint], course_limits={'arrival_radius': .08})
        self.assertEqual(_truth_waypoints(parts, metadata)['reached'], 0)
        waypoint['verification_radius'] = .08
        score = _truth_waypoints(parts, metadata)
        self.assertEqual(score['reached'], 1)
        self.assertEqual(score['arrivals'][0]['arrival_radius_m'], .06)
        self.assertEqual(score['arrivals'][0]['verification_radius_m'], .08)

    def test_truth_waypoint_rejects_invalid_radius_metadata(self):
        for arrival, verification in ((.06, .201), (.08, .06), (0., .08),
                                      (.06, float('nan')), (.06, None)):
            with self.subTest(arrival=arrival, verification=verification):
                metadata = dict(course_limits={'arrival_radius': .08}, waypoints=[
                    dict(position=[0., 0., 1.], phase='MAZE',
                         arrival_radius=arrival, verification_radius=verification)])
                with self.assertRaisesRegex(ValueError, 'verification_radius <= 0.2 m'):
                    _truth_waypoints([], metadata)

    def test_exact_timestamp_alignment_rejects_large_truth_gaps(self):
        truth = Truth([truth_row(1, [0, 0, 0]), truth_row(1.1, [1, 0, 0]),
                       truth_row(1.7, [7, 0, 0])], .12)
        rows = [
            dict(sim_time=1.1, position_stamp_sim_s=1.05, position=[.6, 0, 0], phase='MAZE'),
            dict(sim_time=1.4, position_stamp_sim_s=1.4, position=[4., 0, 0], phase='MAZE'),
        ]
        aligned = _position_alignment(rows, truth, 'position', 'position_stamp_sim_s',
                                      (1., 2.), approximate=True)
        self.assertEqual(aligned['unmatched_timestamp_count'], 1)
        self.assertEqual(aligned['stats']['sample_count'], 1)
        self.assertAlmostEqual(aligned['stats']['bias_enu_m'][0], .1)
        self.assertIsNone(truth.at(.99))
        self.assertIsNone(truth.at(1.4))

    def test_window_cannot_be_joined_across_unobserved_gap(self):
        truth = Truth([truth_row(1, [10.7, 10, .8]), truth_row(1.5, [9.3, 10, .8])], .12)
        parts, complete = truth.interval(1., 1.5)
        self.assertFalse(complete)
        self.assertFalse(_window(parts, WINDOW)['full_envelope_crossed'])
        envelope = load_x500_envelope()
        self.assertFalse(_multipart_window(truth, parts, WINDOW, envelope)['full_envelope_crossed'])

    def test_recorded_diagonal_rotor_attitude_can_fail_where_legacy_sphere_passes(self):
        rows = [truth_row(1., [10.7, 10.08, .8]), truth_row(1.1, [9.3, 10.08, .8])]
        diagonal = quaternion_xyzw_from_rpy(0., 0., math.pi/4).tolist()
        for row in rows:
            row['base_link_orientation_xyzw'] = diagonal
        truth = Truth(rows, .12)
        parts, _ = truth.interval(1., 1.1)
        self.assertTrue(_window(parts, WINDOW)['full_envelope_crossed'])
        crossing = _multipart_window(truth, parts, WINDOW, load_x500_envelope())
        self.assertFalse(crossing['full_envelope_crossed'])
        self.assertLess(crossing['min_aperture_clearance_m'], 0.)

    def test_multipart_scene_scopes_ground_but_keeps_obstacles_during_takeoff(self):
        truth = Truth([truth_row(1., [-1., 0., .22]), truth_row(1.1, [1., 0., .22])], .12)
        envelope = load_x500_envelope()
        obstacle = Scene(np.array([[0., 0., .22]]), np.eye(3)[None],
                         np.array([[.02, .5, .5]]), WINDOW, .28)
        far = Scene(np.array([[0., 10., .22]]), np.eye(3)[None],
                    np.array([[.02, .5, .5]]), WINDOW, .28)
        launch = [('takeoff', 1., 1.1, False)]
        transit = [('transit', 1., 1.1, True)]
        self.assertTrue(_multipart_scene(truth, launch, far, envelope)['certified_clear'])
        hit = _multipart_scene(truth, launch, obstacle, envelope)
        self.assertFalse(hit['certified_clear'])
        self.assertEqual(hit['phases']['takeoff']['first_unresolved']['obstacle'], 0)
        floor = _multipart_scene(truth, transit, far, envelope)
        self.assertFalse(floor['certified_clear'])
        self.assertEqual(floor['phases']['transit']['first_unresolved']['obstacle'], 'floor')

    def test_sphere_sweep_detects_thin_obstacle_even_with_clear_endpoints(self):
        class ThinWall:
            def scene(self, rng, difficulty):
                return Scene(np.array([[0., 0., .5]]), np.eye(3)[None],
                             np.array([[.02, .5, .5]]), WINDOW, .28)

        low = [(0., np.array([-1., 0., .2])), (.1, np.array([1., 0., .2]))]
        score = _proxy_collisions([low], .2, 1., ThinWall())
        self.assertEqual(score['hit_segment_count'], 1)
        self.assertGreater(score['sampled_min_obstacle_clearance_m'], 0.)
        self.assertIsNone(score['first_transit_floor_overlap_segment_sim_s'])
        transit = _proxy_collisions([low], 0., 1., ThinWall())
        self.assertIsNotNone(transit['first_transit_floor_overlap_segment_sim_s'])

    def test_contacts_distinguish_ground_and_external_obstacles(self):
        rows = [dict(sim_time_s=t, pairs=[['course_quad::base_link::base', name]])
                for t, name in [(1., 'ground_plane::ground::floor'),
                                (1.8, 'ground_plane::ground::floor'),
                                (1.9, 'course_v3::layout::barrier_1')]]
        result = _contacts(rows, {}, 1., 2., 1.5, 2.)
        self.assertEqual(result['raw_pair_observations'],
                         dict(obstacle=1, transit_ground=1, expected_ground=1, self=0))
        self.assertEqual(result['first_obstacle_observation']['sim_time_s'], 1.9)

    def test_full_geometry_score_and_independent_pad_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            observer = directory/'observer'
            observer.mkdir()
            rows = []
            for i in range(61):
                t = i/20
                x = float(np.interp(t, [0., 1., 2., 3.], [10.75, 10.75, 9.2, 9.2]))
                z = float(np.interp(t, [0., 1., 2., 3.], [.227, .8, .8, .227]))
                rows.append(truth_row(t, [x, 10., z]))
            self._write_lines(observer/'truth.jsonl', rows)
            metadata = dict(type='metadata', run_id='synthetic', task='course',
                            course_limits={'arrival_radius': .08}, window=WINDOW,
                            landing={'position': [9.2, 10., 0.]},
                            waypoints=[dict(position=p, phase=phase) for p, phase in [
                                ([10.75, 10., .8], 'TAKEOFF'), ([9.4, 10., .8], 'WINDOW_CROSS'),
                                ([9.2, 10., .8], 'LANDING_APPROACH')]])
            sample = dict(type='sample', sim_time=1.1, phase='WINDOW_CROSS',
                          position_stamp_sim_s=1., position=[10.76, 10., .8],
                          vision_stamp_sim_s=1., vision_position_enu=[10.73, 10., .8],
                          attitude_stamp_sim_s=1., attitude_rpy_frd=[0., 0., math.pi/2])
            summary = dict(type='summary', run_id='synthetic', elapsed_sim_s=3.,
                           route_elapsed_sim_s=2., route_completed=True, result='completed',
                           landed_and_disarmed=True, landing_position_enu=[9.2, 10., .227],
                           waypoints_reached=3)
            telemetry = [metadata, dict(type='event', message=ARMED_EVENT, sim_time=0.),
                         dict(type='sample', phase='HOVER_CHECK', sim_time=1.), sample,
                         dict(type='event', message=HOVER_EVENT, sim_time=1.2),
                         dict(type='event', message=TOUCHDOWN_EVENT), summary]
            self._write_lines(directory/'telemetry.jsonl', telemetry)

            result = score_run(directory)
            self.assertTrue(result['full_route_verified_by_geometry'], result['checks'])
            self.assertTrue(result['cad_proxy']['phases']['transit']['ground_checked'])
            self.assertEqual(result['cad_proxy']['phases']['transit']['interval_sim_s'], [1., 2.])
            self.assertFalse(result['cad_proxy']['phases']['takeoff']['ground_checked'])
            self.assertFalse(result['cad_proxy']['phases']['landing']['ground_checked'])
            self.assertGreaterEqual(result['landing']['horizontal_footprint_radius_bound_m'], .386+.015)
            self.assertIsNone(result['contacts']['unexpected_contact_observed'])
            self.assertAlmostEqual(result['px4_position_error']['stats']['rmse_m'], .01)
            self.assertAlmostEqual(result['camera_position_error']['stats']['bias_enu_m'][0], -.02)
            self.assertAlmostEqual(result['px4_attitude_error']['rmse_deg'], 0., places=5)
            self.assertIn(str((observer/'truth.jsonl').resolve()), result['provenance']['sha256_by_file'])

            metadata['landing']['position'][0] += .37
            self._write_lines(directory/'telemetry.jsonl', telemetry)
            off_pad = score_run(directory)
            self.assertFalse(off_pad['full_route_verified_by_geometry'])
            self.assertTrue(off_pad['landing']['px4_confirmed_ground_and_disarm'])
            self.assertFalse(off_pad['landing']['truth_envelope_inside_pad'])
            self.assertGreater(off_pad['legacy_sphere']['landing_horizontal_clearance_m'], 0.)

    @staticmethod
    def _write_lines(path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


if __name__ == '__main__':
    unittest.main()
