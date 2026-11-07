"""Synthetic geometry checks; these do not establish course or hardware success."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import cv2
    import numpy as np
    from vision.rgbd_odometry import (CapturePairs, Gates, RGBDOdometry,
                                      TrackingFailure, estimate_motion,
                                      rotation_from_quaternion)
except ImportError:
    cv2 = None

from vision.rgbd_scene import generate, RGBD_TOPIC


class RGBDSceneTests(unittest.TestCase):
    def test_no_markers_and_mount_preserved(self):
        source = Path(__file__).resolve().parents[1] / 'worlds/course.sdf'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.sdf'
            generate(source, output)
            tree = ET.parse(output)
            self.assertIsNone(tree.find(".//model[@name='vision_landmarks']"))
            link = tree.find(".//model[@name='course_quad']/link[@name='down_camera_link']")
            self.assertEqual(link.find('pose').get('relative_to'), 'base_link')
            self.assertEqual(link.find('pose').text.split()[:3], ['0.0', '0.0', '-0.075'])
            sensor = link.find('sensor')
            self.assertEqual(sensor.get('type'), 'rgbd_camera')
            self.assertEqual(sensor.findtext('topic'), RGBD_TOPIC)


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

    def test_exact_pairing_bounded_and_no_replay(self):
        pairs = CapturePairs(capacity=2)
        pairs.push('color', 1001, 'c1')
        pairs.push('depth', 1002, 'wrong')
        self.assertIsNone(pairs.take())
        pairs.push('depth', 1001, 'd1')
        self.assertEqual(pairs.take(), (1001, 'c1', 'd1'))
        pairs.push('color', 1001, 'replay')
        self.assertIsNone(pairs.take())
        for stamp in range(1003, 1010):
            pairs.push('color', stamp, stamp)
        self.assertEqual(len(pairs.color), 2)


if __name__ == '__main__':
    unittest.main()
