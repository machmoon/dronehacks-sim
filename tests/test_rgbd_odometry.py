"""Synthetic geometry checks; these do not establish course or hardware success."""
import json
import tempfile
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import cv2
    import numpy as np
    from vision.rgbd_odometry import (CapturePairs, Gates, RGBDOdometry, TiltHistory,
                                      TrackingFailure, Vision, estimate_motion,
                                      rotation_from_quaternion)
except ImportError:
    cv2 = None

from vision.rgbd_scene import generate, RGBD_HEIGHT, RGBD_TOPIC, RGBD_TRANSLATION, RGBD_WIDTH


class RGBDSceneTests(unittest.TestCase):
    def test_no_markers_and_forward_calibration(self):
        source = Path(__file__).resolve().parents[1] / 'worlds/course.sdf'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.sdf'
            generate(source, output)
            tree = ET.parse(output)
            self.assertIsNone(tree.find(".//model[@name='vision_landmarks']"))
            link = tree.find(".//model[@name='course_quad']/link[@name='down_camera_link']")
            self.assertEqual(link.find('pose').get('relative_to'), 'base_link')
            self.assertEqual(tuple(map(float, link.findtext('pose').split())), (*RGBD_TRANSLATION, 0., 0., 0.))
            sensor = link.find('sensor')
            self.assertEqual(sensor.get('type'), 'rgbd_camera')
            self.assertEqual(sensor.findtext('topic'), RGBD_TOPIC)
            self.assertEqual(int(sensor.findtext('camera/image/width')), RGBD_WIDTH)
            self.assertEqual(int(sensor.findtext('camera/image/height')), RGBD_HEIGHT)


@unittest.skipIf(cv2 is None, 'OpenCV/numpy required; run inside existing simulation image')
class MotionTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(7)
        self.rng = np.random.default_rng(11)
        self.k = np.array(((250., 0, 240), (0, 250., 180), (0, 0, 1.)))
        self.xyz = self.rng.uniform((-1., -.7, 1.2), (1., .7, 3.), (150, 3))
        self.rvec = np.array((.012, -.018, .01))
        self.tvec = np.array((.025, -.015, .02))
        self.uv = self.project(self.xyz)

    def project(self, xyz):
        return cv2.projectPoints(xyz, self.rvec, self.tvec, self.k, None)[0].reshape(-1, 2)

    def test_known_metric_motion_with_outliers(self):
        pixels = self.uv + self.rng.normal(0, .12, self.uv.shape)
        pixels[:25] = self.rng.uniform((10, 10), (470, 350), (25, 2))
        transform, stats = estimate_motion(self.xyz, pixels, self.k, .1)
        self.assertGreaterEqual(stats['inliers'], 120)
        np.testing.assert_allclose(transform[:3, 3], self.tvec, atol=.003)
        np.testing.assert_allclose(transform[:3, :3], cv2.Rodrigues(self.rvec)[0], atol=.002)
        # Inverse is necessary: pose moves oppositely to observed scene points.
        world_from_new = np.linalg.inv(transform)
        np.testing.assert_allclose(world_from_new @ transform, np.eye(4), atol=1e-10)

    def test_planar_floor_allowed(self):
        xyz = self.xyz.copy()
        xyz[:, 2] = 2.
        transform, _ = estimate_motion(xyz, self.project(xyz), self.k, .1)
        np.testing.assert_allclose(transform[:3, 3], self.tvec, atol=.003)

    def test_collinear_geometry_rejected(self):
        xyz = np.c_[np.linspace(-1, 1, 60), np.zeros(60), np.full(60, 2.)]
        with self.assertRaisesRegex(TrackingFailure, 'degenerate'):
            estimate_motion(xyz, self.project(xyz), self.k, .1)

    def test_majority_outliers_rejected(self):
        pixels = self.uv.copy()
        pixels[:100] = self.rng.uniform((0, 0), (480, 360), (100, 2))
        with self.assertRaises(TrackingFailure):
            estimate_motion(self.xyz, pixels, self.k, .1)

    def test_stale_or_impossible_motion_rejected(self):
        with self.assertRaisesRegex(TrackingFailure, 'capture_gap'):
            estimate_motion(self.xyz, self.uv, self.k, 1.)
        with self.assertRaisesRegex(TrackingFailure, 'translation|rotation'):
            estimate_motion(self.xyz, self.uv, self.k, .001)

    def test_nonfinite_rejected(self):
        self.xyz[0, 0] = np.nan
        with self.assertRaisesRegex(TrackingFailure, 'nonfinite'):
            estimate_motion(self.xyz, self.uv, self.k, .1)
        with self.assertRaises(ValueError):
            rotation_from_quaternion((np.nan, 0, 0, 0))
        with self.assertRaises(ValueError):
            rotation_from_quaternion((2, 0, 0, 0))

    def texture(self):
        rgb = self.rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)
        return cv2.GaussianBlur(rgb, (3, 3), 0), np.full((360, 480), 2., dtype=np.float32)

    def test_known_launch_and_stationary_frames(self):
        estimator = RGBDOdometry((1, 2, 3), intrinsics=self.k)
        rgb, depth = self.texture()
        first = estimator.estimate(rgb, depth, 1_000_000, 1_000_000)
        self.assertTrue(first['valid'], first)
        np.testing.assert_allclose(first['position_enu'], (1, 2, 3))
        second = estimator.estimate(rgb, depth, 1_050_000, 1_050_000)
        self.assertTrue(second['valid'], second)
        np.testing.assert_allclose(second['position_enu'], (1, 2, 3), atol=1e-5)

    def test_textured_plane_translation_body_frame(self):
        estimator = RGBDOdometry((1, 2, 3), intrinsics=self.k)
        rgb, depth = self.texture()
        self.assertTrue(estimator.estimate(rgb, depth, 1_000_000)['valid'])
        shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 3], [0, 1, 0]]), (480, 360))
        record = estimator.estimate(shifted, depth, 1_050_000)
        self.assertTrue(record['valid'], record)
        # +3 image pixels is +.024m scene optical X; camera optical X is body -Y.
        np.testing.assert_allclose(record['position_enu'], (1, 2.024, 3), atol=.002)

    def test_tracking_loss_latches_and_pose_is_absent(self):
        estimator = RGBDOdometry((1, 2, 3), intrinsics=self.k)
        rgb, depth = self.texture()
        estimator.estimate(rgb, depth, 1_000_000)
        invalid = estimator.estimate(np.zeros_like(rgb), depth, 1_050_000)
        self.assertFalse(invalid['valid'])
        self.assertIsNone(invalid['position_enu'])
        self.assertFalse(estimator.estimate(rgb, depth, 1_100_000)['valid'])
        estimator.reset((4, 5, 6))
        reset = estimator.estimate(rgb, depth, 1_200_000)
        self.assertTrue(reset['valid'])
        np.testing.assert_allclose(reset['position_enu'], (4, 5, 6))

    def test_inconsistent_current_depth_rejected(self):
        rgb, depth = self.texture()
        estimator = RGBDOdometry((0, 0, 0), intrinsics=self.k)
        self.assertTrue(estimator.estimate(rgb, depth, 1_000_000)['valid'])
        record = estimator.estimate(rgb, depth*.5, 1_050_000)
        self.assertFalse(record['valid'])
        self.assertIn('inconsistent_depth', record['reason'])
        self.assertIsNone(record['position_enu'])

    def test_stale_capture_and_reversed_time_latch(self):
        rgb, depth = self.texture()
        estimator = RGBDOdometry((0, 0, 0), intrinsics=self.k)
        self.assertFalse(estimator.estimate(rgb, depth, 1_000_000, 2_000_000)['valid'])
        estimator.reset((0, 0, 0))
        estimator.estimate(rgb, depth, 1_000_000)
        self.assertFalse(estimator.estimate(rgb, depth, 999_999)['valid'])

    def test_capture_gap_gate_accepts_normal_and_single_missed_frame(self):
        rgb, depth = self.texture()
        for interval_usec in (50_000, 100_000):
            with self.subTest(interval_usec=interval_usec):
                estimator = RGBDOdometry((0, 0, 0), intrinsics=self.k)
                self.assertTrue(estimator.estimate(rgb, depth, 1_000_000)['valid'])
                record = estimator.estimate(rgb, depth, 1_000_000+interval_usec)
                self.assertTrue(record['valid'], record)
                self.assertEqual(estimator.last_usec, 1_000_000+interval_usec)

    def test_capture_gap_rejects_before_fit_or_reference_update(self):
        rgb, depth = self.texture()
        estimator = RGBDOdometry((0, 0, 0), intrinsics=self.k)
        self.assertTrue(estimator.estimate(rgb, depth, 1_000_000)['valid'])
        prior = estimator.last_usec
        with patch.object(estimator, '_update_depth', side_effect=AssertionError('fit must not run')) as fit:
            record = estimator.estimate(rgb, depth, 1_120_001)
        self.assertFalse(record['valid'])
        self.assertIn('capture_gap', record['reason'])
        fit.assert_not_called()
        self.assertEqual(estimator.last_usec, prior)

    def test_capture_age_is_independent_of_inter_capture_gap(self):
        rgb, depth = self.texture()
        estimator = RGBDOdometry((0, 0, 0), intrinsics=self.k)
        self.assertTrue(estimator.estimate(rgb, depth, 1_000_000, 1_000_000)['valid'])
        # A fresh capture 50 ms after the previous one remains usable even
        # when processing is delayed by 200 ms.
        delayed = estimator.estimate(rgb, depth, 1_050_000, 1_250_000)
        self.assertTrue(delayed['valid'], delayed)
        self.assertEqual(estimator.last_usec, 1_050_000)

        late_estimator = RGBDOdometry((0, 0, 0), intrinsics=self.k)
        self.assertTrue(late_estimator.estimate(rgb, depth, 2_000_000, 2_000_000)['valid'])
        prior = late_estimator.last_usec
        with patch.object(late_estimator, '_update_depth', side_effect=AssertionError('fit must not run')) as fit:
            too_late = late_estimator.estimate(rgb, depth, 2_050_000, 2_350_001)
        self.assertFalse(too_late['valid'])
        self.assertIn('stale_capture', too_late['reason'])
        fit.assert_not_called()
        self.assertEqual(late_estimator.last_usec, prior)

    def test_exact_pairing_bounded_and_no_replay(self):
        pairs = CapturePairs(capacity=2)
        pairs.push('color', 1001, 'c1')
        pairs.push('depth', 1002, 'wrong')
        self.assertIsNone(pairs.take())
        pairs.push('depth', 1001, 'd1')
        self.assertIsNone(pairs.take(through_ns=1000))
        self.assertEqual(pairs.take(through_ns=1001), (1001, 'c1', 'd1'))
        pairs.push('color', 1001, 'replay')
        self.assertIsNone(pairs.take())
        for stamp in range(1003, 1010):
            pairs.push('color', stamp, stamp)
        self.assertEqual(len(pairs.color), 2)

    def test_tilt_requires_fresh_valid_brackets(self):
        history = TiltHistory()
        history.record(100_000, 0., 0.)
        history.record(140_000, 0., .02)
        history.record(140_000, 1., 1.)  # Re-reading the same MAVLink packet is harmless.
        self.assertIsNone(history.at(99_999))
        self.assertIsNone(history.at(140_001))
        prior = history.at(120_000)
        self.assertEqual(prior.capture_usec, 120_000)
        self.assertEqual(prior.bracket_usec, 40_000)
        self.assertGreater(prior.std_rad, .01)
        np.testing.assert_allclose(prior.up_optical, (0., -np.cos(.01), np.sin(.01)))
        self.assertEqual(history.at(140_000).std_rad, .01)
        history.record(200_000, np.nan, 0.)
        history.record(220_000, 0., 0.)
        self.assertIsNone(history.at(200_000))
        self.assertIsNone(history.at(210_000))
        history.record(300_000, 0., 0.)
        self.assertIsNone(history.at(260_000))

    def tilt_adapter(self):
        class Estimator:
            lost = None

            def estimate(self, rgb, z, stamp, now, *, downward_range, inertial_tilt):
                if inertial_tilt is not None:
                    return {'valid': True, 'sim_usec': stamp, 'reason': None}
                self.lost = 'unobservable_depth_geometry'
                return {'valid': False, 'sim_usec': stamp, 'reason': 'tracking_lost:' + self.lost}

        adapter = Vision.__new__(Vision)
        adapter.estimator = Estimator()
        adapter._lock, adapter._pairs = Lock(), CapturePairs()
        adapter._tilts = TiltHistory()
        adapter._tilt_wait_started = {}
        adapter._ranges = {}
        adapter._rejected_frame_dir = None
        adapter._rejected_frame = None
        adapter._formats = (1, 2)
        return adapter

    def push_pair(self, adapter, capture_usec):
        for kind, pixel_format, stride in (('color', 1, 3), ('depth', 2, 4)):
            adapter._pairs.push(kind, capture_usec*1000, SimpleNamespace(
                width=RGBD_WIDTH, height=RGBD_HEIGHT, pixel_format_type=pixel_format,
                step=RGBD_WIDTH*stride, data=bytes(RGBD_WIDTH*RGBD_HEIGHT*stride)))

    def test_early_camera_capture_waits_for_real_right_side_tilt_and_is_used_once(self):
        adapter = self.tilt_adapter()
        self.push_pair(adapter, 39_100_000)
        adapter.record_attitude(39_088_000, .01, .01)
        adapter._clock_usec = 39_116_000
        with patch('vision.rgbd_odometry.time.monotonic', return_value=10.):
            self.assertIsNone(adapter.poll())
            self.assertIsNone(adapter.estimator.lost)
            adapter.record_attitude(39_124_000, .02, .02)
            adapter._clock_usec = 39_126_000
            record = adapter.poll()
            self.assertTrue(record['valid'])
            self.assertEqual(record['sim_usec'], 39_100_000)
            self.assertEqual(record['px4_tilt_pairing_status'], 'capture_aligned')
            # Later attitude alone cannot produce another camera observation.
            adapter.record_attitude(39_156_000, .02, .02)
            adapter._clock_usec = 39_157_000
            self.assertIsNone(adapter.poll())

    def test_missing_tilt_releases_capture_at_sim_or_wall_deadline(self):
        for clock_usec, wall in ((160_000, 10.1), (116_000, 10.301)):
            with self.subTest(clock_usec=clock_usec, wall=wall):
                adapter = self.tilt_adapter()
                self.push_pair(adapter, 100_000)
                adapter.record_attitude(88_000, 0., 0.)
                adapter._clock_usec = 116_000
                with patch('vision.rgbd_odometry.time.monotonic', return_value=10.):
                    self.assertIsNone(adapter.poll())
                adapter._clock_usec = clock_usec
                with patch('vision.rgbd_odometry.time.monotonic', return_value=wall):
                    record = adapter.poll()
                    self.assertFalse(record['valid'])
                    self.assertEqual(record['sim_usec'], 100_000)
                    self.assertEqual(record['px4_tilt_pairing_status'], 'missing_after_deadline')
                    self.assertIn('px4_tilt_bracket_missing_after_deadline', record['reason'])
                    adapter.record_attitude(124_000, 0., 0.)
                    self.assertIsNone(adapter.poll())

    def test_newer_waiting_capture_cannot_starve_an_expired_older_pair(self):
        adapter = self.tilt_adapter()
        self.push_pair(adapter, 100_000)
        self.push_pair(adapter, 150_000)
        adapter._clock_usec = 167_000
        with patch('vision.rgbd_odometry.time.monotonic', return_value=10.):
            self.assertEqual(adapter.poll()['sim_usec'], 100_000)
            self.assertIsNone(adapter.poll())

    def test_opt_in_rejection_saves_first_raw_pair_only(self):
        class Estimator:
            lost = None
            last_usec = 50_000
            world_from_optical = np.eye(4)
            intrinsics = np.eye(3)

            def estimate(self, rgb, z, stamp, now, *, downward_range, inertial_tilt):
                self.lost = 'unobservable_depth_geometry'
                return dict(reason='tracking_lost:' + self.lost, valid=False,
                            survey_normals={'method': 'fals7', 'sample_stride_px': 8},
                            rejected_fits={'survey': {'depth_observability': .0035}})

        with tempfile.TemporaryDirectory() as directory:
            adapter = Vision.__new__(Vision)
            adapter.estimator = Estimator()
            adapter._rejected_frame_dir = Path(directory)
            adapter._rejected_frame = None
            adapter._tilts = None
            adapter._subscriptions = []
            adapter._lock, adapter._pairs = Lock(), CapturePairs()
            adapter._ranges = {100_000_000: .92, 150_000_000: .94}
            adapter._tof_origin = np.zeros(3)
            adapter._tof_direction = np.array([0., 1., 0.])
            adapter._formats = (1, 2)
            for stamp, fill in ((100_000_000, 17), (150_000_000, 91)):
                adapter._clock_usec = stamp//1000 + 25_000
                rgb = np.full((RGBD_HEIGHT, RGBD_WIDTH, 3), fill, dtype='uint8')
                depth = np.full((RGBD_HEIGHT, RGBD_WIDTH), .001*fill, dtype='<f4')
                for kind, data, pixel_format, stride in (('color', rgb, 1, 3), ('depth', depth, 2, 4)):
                    adapter._pairs.push(kind, stamp, SimpleNamespace(
                        width=RGBD_WIDTH, height=RGBD_HEIGHT, pixel_format_type=pixel_format,
                        step=RGBD_WIDTH*stride, data=data.tobytes()))
                self.assertFalse(adapter.poll()['valid'])
            adapter.close()
            output = Path(directory) / 'first-rejected.npz'
            self.assertLess(output.stat().st_size, 2_000_000)
            with np.load(output) as saved:
                metadata = json.loads(saved['metadata_json'].item())
                self.assertEqual(metadata['capture_stamp_ns'], 100_000_000)
                self.assertEqual(metadata['tof_stamp_ns'], 100_000_000)
                self.assertEqual(metadata['tof_range_m'], .92)
                self.assertEqual(metadata['survey_normals'], {'method': 'fals7', 'sample_stride_px': 8})
                self.assertEqual(metadata['rejected_fits']['survey']['depth_observability'], .0035)
                self.assertTrue(np.all(saved['rgb'] == 17))
                np.testing.assert_allclose(saved['depth_z_m'], .017)


if __name__ == '__main__':
    unittest.main()
