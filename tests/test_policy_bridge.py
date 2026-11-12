import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from learning.inference import (CAMERA_HEIGHT, CAMERA_WIDTH, VERSION, PolicyClient, PolicyError,
                                action_to_enu, checkpoint_camera_forward, policy_depth, policy_state,
                                read_json, response_name, write_json)
from learning.sim import HEIGHT, MAX_DEPTH, RAYS, WIDTH
from mission.markerfree_run import (FlightFault, LearnedRoute, NAVIGATION, observe_abort_landing,
                                   policy_camera_shift, prepare_course)
from mission.state_history import PolicyStateHistory


def px4_position(stamp, p=(0., 0., 0.), v=(0., 0., 0.)):
    return SimpleNamespace(time_boot_ms=stamp, x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2])


def px4_attitude(stamp, yaw=0., roll=0., pitch=0.):
    return SimpleNamespace(time_boot_ms=stamp, roll=roll, pitch=pitch, yaw=yaw)


class StateHistoryTests(unittest.TestCase):
    def test_interpolates_position_velocity_and_short_arc_attitude_at_capture(self):
        history = PolicyStateHistory()
        history.record(px4_position(1000, (10., 20., -1.), (1., 2., -.5)),
                       px4_attitude(1010, math.radians(179), .1, -.2))
        history.record(px4_position(1040, (14., 24., -5.), (3., 4., -1.5)),
                       px4_attitude(1050, math.radians(-179), .2, -.3))
        state = history.at(1_030_000)
        np.testing.assert_allclose(state.position_enu, (23., 13., 4.))
        np.testing.assert_allclose(state.velocity_enu, (3.5, 2.5, 1.25))
        self.assertAlmostEqual(state.attitude_rpy_frd[0], .15)
        self.assertAlmostEqual(state.attitude_rpy_frd[1], -.25)
        self.assertAlmostEqual(abs(state.attitude_rpy_frd[2]), math.pi)

    def test_requires_each_stream_to_bracket_without_extrapolation(self):
        history = PolicyStateHistory()
        history.record(px4_position(1000), px4_attitude(1000))
        history.record(px4_position(1040), None)
        self.assertIsNone(history.at(1_020_000))
        history.record(None, px4_attitude(1061))
        self.assertIsNone(history.at(1_020_000))  # Attitude gap exceeds 60 ms.
        self.assertIsNone(history.at(1_080_000))
        history.record(px4_position(2100), px4_attitude(2100))
        self.assertIsNone(history.at(1_000_000))

    def test_blackout_abort_keeps_clock_without_forwarding_vision(self):
        vision = SimpleNamespace(sim_usec_now=3_000_000, poll=Mock())
        link = SimpleNamespace(boot_s=3., landed_state_sim_s=3., armed=False, landed_state=1,
                               m=SimpleNamespace(MAV_LANDED_STATE_ON_GROUND=1),
                               drain=Mock(), gcs_heartbeat=Mock(), vision=Mock())
        self.assertTrue(observe_abort_landing(link, vision, Mock(), allow_vision_aid=False))
        self.assertEqual(link.sim_usec_now, 3_000_000)
        vision.poll.assert_not_called()
        link.vision.assert_not_called()


class ObservationTests(unittest.TestCase):
    def test_optical_plane_is_euclidean_ray_distance(self):
        z = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 1.6, np.float32)
        actual = policy_depth(z)
        expected = 1 - (1.6 / RAYS[:, 0] / MAX_DEPTH).reshape(HEIGHT, WIDTH)
        np.testing.assert_allclose(actual.depth, expected, atol=1e-7)
        self.assertEqual(actual.coverage, 1.)
        self.assertEqual(actual.depth.dtype, np.float32)

    def test_virtual_camera_sees_known_axial_obstacle_farther_back(self):
        z = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), np.inf, np.float32)
        z[100:156, 160:224] = .80
        result = policy_depth(z, .26)
        forward_cosine = RAYS[:, 0].reshape(HEIGHT, WIDTH)[7, 11]
        self.assertAlmostEqual(result.depth[7, 11], 1 - 1.06/forward_cosine/MAX_DEPTH, places=6)
        self.assertGreaterEqual(result.center_coverage, .90)
        self.assertGreaterEqual(result.coverage, .70)
        # Measured far rays remain far; unobserved rays cannot be clear.
        self.assertEqual(result.depth[4, 4], 0.)
        self.assertEqual(result.depth[0, 0], 1.)

    def test_virtual_camera_rejects_big_disocclusions(self):
        wall = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), .50, np.float32)
        with self.assertRaisesRegex(PolicyError, 'insufficient_policy_depth_coverage'):
            policy_depth(wall, .26)

    def test_rejects_missing_or_corrupt_depth(self):
        with self.assertRaisesRegex(PolicyError, 'shape_or_format'):
            policy_depth(None)
        z = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), np.inf, np.float32)
        with self.assertRaisesRegex(PolicyError, 'invalid_or_empty'):
            policy_depth(z)
        z[-1, -1] = np.nan
        with self.assertRaisesRegex(PolicyError, 'invalid_or_empty'):
            policy_depth(z)

    def test_localized_invalid_depth_semantics_are_pinned(self):
        plane = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 1.4, dtype=np.float32)
        row_slice, col_slice = slice(96, 160), slice(144, 240)
        policy_rows, policy_cols = slice(6, 10), slice(9, 15)
        plane_observation = policy_depth(plane).depth

        for invalid in (0., np.nan):
            with self.subTest(invalid=invalid):
                depth = plane.copy()
                depth[row_slice, col_slice] = invalid
                with self.assertRaisesRegex(PolicyError, 'invalid_or_empty_policy_depth'):
                    policy_depth(depth)

        for invalid, expected in ((np.inf, 0.), (-np.inf, 1.)):
            with self.subTest(invalid=invalid):
                depth = plane.copy()
                depth[row_slice, col_slice] = invalid
                observation = policy_depth(depth)
                expected_observation = plane_observation.copy()
                expected_observation[policy_rows, policy_cols] = expected
                np.testing.assert_array_equal(observation.depth, expected_observation)

    def test_raw_mode_requires_checkpoint_camera_to_match(self):
        legacy = checkpoint_camera_forward({})
        self.assertEqual(legacy, .08)
        self.assertAlmostEqual(policy_camera_shift(.34, legacy, 'virtual'), .26)
        with self.assertRaisesRegex(FlightFault, 'checkpoint_camera_mount_mismatch'):
            policy_camera_shift(.34, legacy, 'raw')
        spec = dict(height=16, width=24, horizontal_fov_degrees=90., max_range_m=4.,
                    encoding='1 - range / max_range', camera_forward_m=.34)
        trained = checkpoint_camera_forward(dict(camera_forward_m=.34, depth=spec, state_size=9))
        self.assertEqual(policy_camera_shift(.34, trained, 'raw'), 0.)
        with self.assertRaisesRegex(PolicyError, 'checkpoint_observation_contract_mismatch'):
            checkpoint_camera_forward(dict(camera_forward_m=.34, depth=spec | {'max_range_m': 8.}))

    def test_px4_attitude_and_north_heading_flu_round_trip(self):
        state, yaw = policy_state((1., 2., 3.), (-4., -5., -6.), .12, -.13, 0., .4)
        np.testing.assert_allclose(state, (2., -1., 3., -5., 4., -6., .60, .65, .4), atol=1e-7)
        self.assertAlmostEqual(yaw, math.pi/2)
        np.testing.assert_allclose(action_to_enu((.2, -.3, .1), yaw), (.3, .2, .1), atol=1e-12)


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.sha = 'a'*64
        self.client = PolicyClient(self.path, self.sha, 'current-run', timeout_s=1.)
        self.addCleanup(self.client.close)
        self.service = dict(version=VERSION, checkpoint_sha256=self.sha, session='current-session', camera_forward_m=.08)
        write_json(self.path/'service.json', self.service)
        self.depth = np.zeros((HEIGHT, WIDTH), np.float32)
        self.state = np.r_[np.zeros(8, np.float32), np.float32(.4)]

    def response(self, **overrides):
        request = read_json(self.path/'request.json')
        request.pop('depth')
        request.pop('state')
        write_json(self.path/response_name(request['reply_id']), request | {'velocity_flu': [.2, .1, 0.]} | overrides)

    def current_response_path(self):
        return self.path/response_name(read_json(self.path/'request.json')['reply_id'])

    def test_stale_run_session_checkpoint_sequence_and_capture_are_rejected(self):
        self.assertTrue(self.client.connect())
        self.client.request(self.depth, self.state, 1_000_000, 10.)
        for key, wrong_value in [('run_id', 'old-run'), ('session', 'old-service'),
                                 ('checkpoint_sha256', 'b'*64), ('seq', 0), ('capture_usec', 999_999)]:
            with self.subTest(key=key):
                self.response(**{key: wrong_value})
                self.assertIsNone(self.client.poll(10.05))
        self.response()
        reply = self.client.poll(10.2)
        self.assertEqual(reply.capture_usec, 1_000_000)
        self.assertAlmostEqual(reply.latency_s, .2)
        self.assertIsNone(self.client.poll(10.25))

    def test_matching_but_late_reply_fails(self):
        self.assertTrue(self.client.connect())
        self.client.request(self.depth, self.state, 1_000_000, 10.)
        self.response()
        with self.assertRaisesRegex(PolicyError, 'policy_response_timeout'):
            self.client.poll(11.01)

    def test_partial_response_retries_without_accepting_or_resetting_deadline(self):
        self.assertTrue(self.client.connect())
        self.client.request(self.depth, self.state, 1_000_000, 10.)
        self.current_response_path().write_text('{"version":2')
        with self.assertLogs('learning.inference', level='WARNING') as logs:
            self.assertIsNone(self.client.poll(10.05))
        self.assertIn('JSONDecodeError', logs.output[0])
        self.assertEqual(self.client.transport_read_errors, 1)
        self.assertIsNotNone(self.client.pending)
        self.response()
        self.assertEqual(self.client.poll(10.10).capture_usec, 1_000_000)

        self.client.request(self.depth, self.state, 1_200_000, 20.)
        self.current_response_path().write_text('[]')
        self.assertIsNone(self.client.poll(20.8))
        with self.assertRaisesRegex(PolicyError, 'policy_response_timeout.*expected object, got list'):
            self.client.poll(21.01)

    def test_response_filename_is_unique_after_cancellation(self):
        self.assertTrue(self.client.connect())
        self.client.request(self.depth, self.state, 1_000_000, 10.)
        first = self.current_response_path()
        self.response()
        self.client.cancel()
        self.client.request(self.depth, self.state, 1_200_000, 10.2)
        self.assertNotEqual(self.current_response_path(), first)
        self.assertIsNone(self.client.poll(10.25))
        self.response()
        self.assertEqual(self.client.poll(10.3).capture_usec, 1_200_000)

    def test_exchange_publish_fsyncs_contents_before_replacing_record(self):
        path = self.path/'response.json'
        with patch('learning.inference.os.fsync', side_effect=lambda _: self.assertFalse(path.exists())) as sync:
            write_json(path, {'ready': True})
        sync.assert_called_once()
        self.assertEqual(read_json(path), {'ready': True})

    def test_checkpoint_mismatch_blocks_startup(self):
        write_json(self.path/'service.json', self.service | {'checkpoint_sha256': 'b'*64})
        with self.assertRaisesRegex(PolicyError, 'policy_service_or_checkpoint_mismatch'):
            self.client.connect()

    def test_a_running_client_does_not_adopt_a_new_service_session(self):
        self.assertTrue(self.client.connect())
        write_json(self.path/'service.json', self.service | {'session': 'replacement', 'camera_forward_m': .34})
        self.assertTrue(self.client.connect())
        self.client.request(self.depth, self.state, 1_000_000, 10.)
        self.assertEqual(read_json(self.path/'request.json')['session'], 'current-session')
        self.assertEqual(self.client.camera_forward_m, .08)
        self.response(session='replacement')
        self.assertIsNone(self.client.poll(10.1))

    def test_bad_action_fails_closed(self):
        self.assertTrue(self.client.connect())
        self.client.request(self.depth, self.state, 1_000_000, 10.)
        self.response(velocity_flu=[.3, .3, .3])
        with self.assertRaisesRegex(PolicyError, 'invalid_policy_velocity'):
            self.client.poll(10.1)

    def test_reply_cannot_cross_waypoint_or_outlive_its_sim_capture(self):
        self.assertTrue(self.client.connect())
        controller = LearnedRoute(self.client)
        history = PolicyStateHistory()
        z = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 1., np.float32)

        def tick(segment, capture, sim_s):
            stamp = round(sim_s*1000)
            history.record(px4_position(stamp), px4_attitude(stamp))
            return controller.tick(segment, dict(sim_usec=capture, depth_z_m=z), history,
                                   (1., 1., 1.), .4, sim_s, 10.)

        self.assertIsNone(tick(0, 1_000_000, 1.)[1])
        self.response()
        self.assertIsNotNone(tick(0, 1_000_000, 1.05)[1])
        velocity, reply = tick(1, 1_200_000, 1.2)
        self.assertEqual(velocity, (0., 0., 0.))
        self.assertIsNone(reply)
        self.response()
        with self.assertRaisesRegex(PolicyError, 'stale_policy_response'):
            tick(1, 1_200_000, 1.7)

    def test_policy_waits_for_bracket_and_rotates_reply_with_capture_yaw(self):
        self.assertTrue(self.client.connect())
        log = Mock()
        controller = LearnedRoute(self.client, log=log)
        history = PolicyStateHistory()
        z = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 1., np.float32)
        frame = dict(sim_usec=1_000_000, depth_z_m=z)
        history.record(px4_position(980), px4_attitude(980, math.radians(179)))
        self.assertEqual(controller.tick(0, frame, history, (1., 1., 1.), .4, 1.02, 10.),
                         ((0., 0., 0.), None))
        self.assertIsNone(self.client.pending)
        history.record(px4_position(1020), px4_attitude(1020, math.radians(-179)))
        controller.tick(0, frame, history, (1., 1., 1.), .4, 1.04, 10.02)
        self.assertIsNotNone(self.client.pending)
        self.response()
        history.record(px4_position(1060), px4_attitude(1060, math.radians(-175)))
        velocity, reply = controller.tick(0, frame, history, (1., 1., 1.), .4, 1.07, 10.05)
        self.assertEqual(reply.capture_usec, 1_000_000)
        np.testing.assert_allclose(velocity, (.1, -.2, 0.), atol=1e-12)
        first = log.write.call_args.kwargs
        self.assertAlmostEqual(first['capture_age_sim_s'], .07)
        self.assertIsNone(first['capture_interval_sim_s'])
        history.record(px4_position(1200), px4_attitude(1200))
        next_frame = dict(sim_usec=1_200_000, depth_z_m=z)
        controller.tick(0, next_frame, history, (1., 1., 1.), .4, 1.21, 10.10)
        self.response()
        controller.tick(0, next_frame, history, (1., 1., 1.), .4, 1.24, 10.15)
        self.assertAlmostEqual(log.write.call_args.kwargs['capture_interval_sim_s'], .20)

    def test_aligned_but_old_camera_view_is_not_submitted(self):
        self.assertTrue(self.client.connect())
        controller = LearnedRoute(self.client)
        history = PolicyStateHistory()
        history.record(px4_position(1000), px4_attitude(1000))
        frame = dict(sim_usec=1_000_000, depth_z_m=np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), np.float32))
        controller.tick(0, frame, history, (1., 1., 1.), .4, 1.30, 10.)
        self.assertIsNone(self.client.pending)
        with self.assertRaisesRegex(PolicyError, 'no_fresh_policy_action'):
            controller.tick(0, frame, history, (1., 1., 1.), .4, 1.76, 10.1)

    def test_capture_schedule_averages_160_ms_on_quantized_20_hz_frames(self):
        self.assertTrue(self.client.connect())
        controller = LearnedRoute(self.client)
        history = PolicyStateHistory()
        depth = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), np.float32)
        captures = []
        for i in range(21):
            # The 4 ms Gazebo physics clock quantizes captures to 48/52 ms.
            capture = 1_000_000 + (i//2)*100_000 + (48_000 if i % 2 else 0)
            stamp = capture//1000
            history.record(px4_position(stamp), px4_attitude(stamp))
            frame = dict(sim_usec=capture, depth_z_m=depth)
            sim_s = capture/1e6 + .02
            controller.tick(0, frame, history, (1., 1., 1.), .4, sim_s, 10.+sim_s)
            if self.client.pending is not None:
                captures.append(capture)
                self.response()
                controller.tick(0, frame, history, (1., 1., 1.), .4, sim_s+.005, 10.01+sim_s)
        self.assertEqual([stamp-1_000_000 for stamp in captures],
                         [0, 200_000, 348_000, 500_000, 648_000, 800_000, 1_000_000])
        self.assertAlmostEqual(np.mean(np.diff(captures)[:5])/1e6, .16)

    def test_late_frame_rebases_deadline_and_preflight_clear_restarts_it(self):
        self.assertTrue(self.client.connect())
        controller = LearnedRoute(self.client)
        history = PolicyStateHistory()
        depth = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), np.float32)

        def request_at(capture, segment=0):
            stamp = capture//1000
            history.record(px4_position(stamp), px4_attitude(stamp))
            frame = dict(sim_usec=capture, depth_z_m=depth)
            sim_s = capture/1e6 + .02
            controller.tick(segment, frame, history, (1., 1., 1.), .4, sim_s, 10.+sim_s)
            if self.client.pending is None:
                return False
            self.response()
            controller.tick(segment, frame, history, (1., 1., 1.), .4, sim_s+.005, 10.01+sim_s)
            return True

        self.assertTrue(request_at(1_000_000, 'preflight'))
        controller.clear()
        self.assertTrue(request_at(1_050_000))
        self.assertTrue(request_at(1_300_000))
        self.assertEqual(controller.next_capture_due_usec, 1_460_000)
        self.assertFalse(request_at(1_350_000))
        self.assertFalse(request_at(1_400_000))
        self.assertTrue(request_at(1_500_000))


class PrerequisiteTests(unittest.TestCase):
    def test_course_requires_verified_hover_from_same_launch(self):
        launch = dict(position_enu=[-2.5146, .9144, .24], yaw_enu_deg=90.)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            proof = Path(tmp)/'hover.summary.json'
            args = SimpleNamespace(task='course', course=root/'mission/course-window-fast.json', hover_summary=proof)
            result = dict(task='hover', result='completed', landed_and_disarmed=True,
                          route_completed=True, navigation_source=NAVIGATION, hover_settled_sim_s=3.1,
                          simulator_position_used_for_control=False, launch=launch)
            write_json(proof, result | {'landed_and_disarmed': False})
            with self.assertRaisesRegex(ValueError, 'completed RGBD hover and landing'):
                prepare_course(args, launch)
            write_json(proof, result)
            course, takeoff = prepare_course(args, launch)
            self.assertEqual([w['phase'] for w in takeoff['waypoints']], ['TAKEOFF'])
            self.assertIn('WINDOW_CROSS', [w['phase'] for w in course['waypoints']])
            self.assertGreaterEqual(course['footprint_radius'], math.sqrt(2)*.174 + math.hypot(.27923077/2, .01692308/2) + .015)
            args.survey_normals = 'fals7'
            with self.assertRaisesRegex(ValueError, 'completed RGBD hover and landing'):
                prepare_course(args, launch)
            write_json(proof, result | {'survey_normals': {'method': 'fals7'}})
            prepare_course(args, launch)
            args.require_survey_startup = True
            with self.assertRaisesRegex(ValueError, 'completed RGBD hover and landing'):
                prepare_course(args, launch)
            write_json(proof, result | {'survey_normals': {'method': 'fals7'},
                                        'require_survey_startup': True})
            prepare_course(args, launch)


if __name__ == '__main__':
    unittest.main()
