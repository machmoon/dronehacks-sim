import math
import unittest
from unittest.mock import Mock
import numpy as np

from vision.slam_odometry import SLAMOdometry
from vision.rgbd_odometry import rotation_from_quaternion
from vision.rgbd_scene import RGBD_BODY_FROM_OPTICAL, RGBD_TRANSLATION


class SLAMBridgeTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((240, 320, 3), np.uint8)
        self.depth = np.ones((240, 320), np.float32)
        self.start = [-2., 1., .1]
        self.q = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        self.estimator = SLAMOdometry(self.start, self.q)

    def mock_pose(self, pose=None, **extra):
        result = dict(valid=True, reason=None, map_from_camera=(np.eye(4) if pose is None else pose).tolist(),
                      relocalized=False)
        result.update(extra)
        self.estimator.tracker.estimate = Mock(return_value=result)

    def test_launch_body_pose_removes_rotated_camera_lever_arm(self):
        self.mock_pose()
        r = self.estimator.estimate(self.rgb, self.depth, 1_000_000)
        self.assertTrue(r['valid'])
        np.testing.assert_allclose(r['position_enu'], self.start, atol=1e-12)
        np.testing.assert_allclose(r['orientation_enu_flu_wxyz'], self.q, atol=1e-12)
        self.assertFalse(r['static_survey_used'])

    def test_optical_motion_rotates_to_world_and_keeps_capture_time(self):
        self.mock_pose(); self.estimator.estimate(self.rgb, self.depth, 1_000_000)
        pose = np.eye(4); pose[0, 3] = .02
        self.mock_pose(pose)
        r = self.estimator.estimate(self.rgb, self.depth, 1_050_000, 1_100_000)
        expected = np.array(self.start)+rotation_from_quaternion(self.q) @ np.array(RGBD_BODY_FROM_OPTICAL) @ [.02, 0, 0]
        np.testing.assert_allclose(r['position_enu'], expected, atol=1e-12)
        self.assertEqual(r['sim_usec'], 1_050_000)

    def test_loss_latches_and_does_not_attempt_relocalization_in_flight(self):
        self.mock_pose(); self.estimator.estimate(self.rgb, self.depth, 950_000)
        self.mock_pose(valid=False, reason='insufficient_texture', map_from_camera=None)
        self.assertFalse(self.estimator.estimate(self.rgb, self.depth, 1_000_000)['valid'])
        self.mock_pose()
        self.assertFalse(self.estimator.estimate(self.rgb, self.depth, 1_050_000)['valid'])
        self.estimator.tracker.estimate.assert_not_called()

    def test_startup_black_frame_does_not_latch_before_map_initialization(self):
        self.mock_pose(valid=False, reason='insufficient_texture', map_from_camera=None)
        result = self.estimator.estimate(self.rgb, self.depth, 1_000_000)
        self.assertFalse(result['valid'])
        self.assertIn('waiting_for_observable_capture', result['reason'])
        self.mock_pose()
        self.assertTrue(self.estimator.estimate(self.rgb, self.depth, 1_050_000)['valid'])

    def test_relocalized_jump_is_never_sent_as_external_vision(self):
        self.mock_pose(relocalized=True)
        r = self.estimator.estimate(self.rgb, self.depth, 1_000_000)
        self.assertFalse(r['valid']); self.assertIn('relocalization', r['reason'])
        self.assertIsNone(r['position_enu'])

    def test_duplicate_stale_and_gap_timestamps_fail_closed(self):
        for stamp, now in [(1_000_000, None), (1_050_000, 1_400_001), (1_400_000, None)]:
            with self.subTest(stamp=stamp, now=now):
                self.setUp(); self.mock_pose()
                self.estimator.estimate(self.rgb, self.depth, 1_000_000)
                self.assertFalse(self.estimator.estimate(self.rgb, self.depth, stamp, now)['valid'])

    def test_pose_jump_rejected_even_if_tracker_says_valid(self):
        self.mock_pose(); self.estimator.estimate(self.rgb, self.depth, 1_000_000)
        pose = np.eye(4); pose[2, 3] = 2
        self.mock_pose(pose)
        self.assertIn('discontinuous', self.estimator.estimate(self.rgb, self.depth, 1_050_000)['reason'])

    def test_camera_rotation_corrects_lever_arm(self):
        self.mock_pose(); self.estimator.estimate(self.rgb, self.depth, 1_000_000)
        import cv2
        p = np.eye(4); p[:3,:3] = cv2.Rodrigues(np.array([0., .04, 0.]))[0]
        self.mock_pose(p)
        r = self.estimator.estimate(self.rgb, self.depth, 1_050_000)
        b = self.estimator.world_from_map @ p @ self.estimator.optical_from_body
        np.testing.assert_allclose(r['position_enu'], b[:3,3], atol=1e-12)

    def test_image_motion_reaches_mavlink_with_capture_time_and_ned_axes(self):
        from types import SimpleNamespace
        from tools.benchmark_sparse_slam import SyntheticRoom, trajectory
        from mission.markerfree_run import MarkerFreeLink
        from mission.px4_frames import camera_pose_to_px4
        import cv2
        cv2.setRNGSeed(42)
        room = SyntheticRoom()
        estimator = SLAMOdometry(self.start, self.q, intrinsics=room.k)
        messages = []
        link = MarkerFreeLink.__new__(MarkerFreeLink)
        link.link = SimpleNamespace(mav=SimpleNamespace(
            vision_position_estimate_send=lambda *args: messages.append(args)))
        for i in range(20):
            expected_optical = trajectory(i, 120)
            rgb, depth = room.render(expected_optical, .002)
            stamp = 1_000_000 + round(i*1e6/30)
            result = estimator.estimate(rgb, depth, stamp, stamp+20_000)
            self.assertTrue(result['valid'], result)
            link.vision(result)
            expected_body = estimator.world_from_map @ expected_optical @ estimator.optical_from_body
            self.assertLess(np.linalg.norm(np.asarray(result['position_enu'])-expected_body[:3,3]), .025)
            p, rpy = camera_pose_to_px4(result['position_enu'], result['orientation_enu_flu_wxyz'])
            np.testing.assert_allclose(messages[-1][1:7], [*p, *rpy])
            self.assertEqual(messages[-1][0], stamp)
            self.assertEqual(len(messages[-1][7]), 21)
        result = estimator.estimate(np.zeros_like(rgb), depth, stamp+33_333)
        self.assertFalse(result['valid'])
        with self.assertRaisesRegex(ValueError, 'invalid localization'):
            link.vision(result)
        self.assertEqual(len(messages), 20)

    def test_full_evidence_disk_does_not_block_abort_logging(self):
        from mission.markerfree_run import Log
        from unittest.mock import patch
        log = Log.__new__(Log)
        log.file = Mock()
        log.file.write.side_effect = OSError(28, 'No space left on device')
        log.write_error = None
        log.common = {}
        log.summary_path = 'never-written.json'
        log.write(type='sample')
        self.assertIn('No space left', log.write_error)
        # These calls must not throw before the runner can send Land.
        with patch('builtins.print', side_effect=OSError(28, 'stdout disk full')):
            log.event('Mission stopped: telemetry_logging_failed')
        with patch('mission.markerfree_run.write_json') as save:
            log.finish(result='failed')
            save.assert_not_called()
        self.assertEqual(log.file.write.call_count, 1)


if __name__ == '__main__':
    unittest.main()
