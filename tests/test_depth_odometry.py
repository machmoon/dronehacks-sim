"""Camera-only localization against analytic, textureless depth scenes."""
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    from vision.depth_odometry import (DepthFrame, DepthGates, DepthSurvey, DownwardRange, FalsSurveyNormals,
                                        InertialTilt, SurveySurface, TrackingFailure,
                                        register_depth, tilt_residual)
    from vision.rgbd_odometry import RGBDOdometry, calibration
    from vision.rgbd_scene import RGBD_BODY_FROM_OPTICAL, RGBD_TRANSLATION
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'OpenCV and NumPy required')
class DepthOdometryTests(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 384, 256
        self.k = calibration(self.width, self.height)
        self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.rays = np.stack(((u-self.k[0, 2])/self.k[0, 0],
                              (v-self.k[1, 2])/self.k[1, 1], np.ones_like(u)), axis=-1)
        self.world_from_camera = np.eye(4)
        self.world_from_camera[:3, :3] = RGBD_BODY_FROM_OPTICAL
        self.world_from_camera[:3, 3] = np.array(RGBD_TRANSLATION) + (0., 0., 1.)
        # A back wall, offset side wall, and floor. The RGB image is blank.
        self.planes = ((0, 3.0), (1, -.83), (2, 0.))

    def render(self, pose, planes=None):
        rays = self.rays @ pose[:3, :3].T
        position = pose[:3, 3]
        depth = np.full(rays.shape[:2], np.inf)
        for axis, value in (self.planes if planes is None else planes):
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (value-position[axis])/rays[:, :, axis]
            depth = np.minimum(depth, np.where(z > .04, z, np.inf))
        return depth.astype(np.float32)

    def moved(self):
        pose = self.world_from_camera.copy()
        pose[:3, :3] = cv2.Rodrigues(np.array([-.014, .023, .019]))[0] @ pose[:3, :3]
        pose[:3, 3] += [.047, -.031, .025]
        return pose

    def test_metric_motion_without_rgb_texture(self):
        estimator = RGBDOdometry((0, 0, 1), intrinsics=self.k)
        self.assertTrue(estimator.estimate(self.rgb, self.render(self.world_from_camera), 100_000)['valid'])
        pose = self.moved()
        record = estimator.estimate(self.rgb, self.render(pose), 150_000)
        self.assertTrue(record['valid'], record)
        self.assertEqual(record['source'], 'depth_odometry')
        rotation = pose[:3, :3] @ np.array(RGBD_BODY_FROM_OPTICAL).T
        np.testing.assert_allclose(record['position_enu'], pose[:3, 3]-rotation @ RGBD_TRANSLATION,
                                   atol=.004)
        np.testing.assert_allclose(estimator.world_from_optical, pose, atol=.004)
        self.assertTrue(all(0 < x < .1 for x in record['position_std_m']))

    def survey(self):
        points, normals = [], []
        a, b = np.meshgrid(np.arange(-4, 4, .035), np.arange(-3, 3, .035))
        for axis, value in self.planes[:2]:
            p = np.empty((a.size, 3))
            p[:, axis] = value
            p[:, [i for i in range(3) if i != axis]] = np.c_[a.ravel(), b.ravel()]
            points.extend(p)
            normals.extend(np.tile(np.eye(3)[axis], (len(p), 1)))
        return DepthSurvey(points, normals)

    def test_survey_corrects_an_offset_without_an_image_reference(self):
        survey = self.survey()
        pose = self.moved()
        frame = DepthFrame(self.render(pose), self.k, DepthGates())
        found, quality = register_depth(frame, survey, self.world_from_camera, survey=True)
        np.testing.assert_allclose(found, pose, atol=.005)
        self.assertGreater(quality['depth_overlap'], .9)

    def test_shallow_noisy_depth_recovers_survey_with_raw_centers(self):
        cv2.setRNGSeed(7)
        self.planes = ((0, 2.12), (1, -.65), (2, 0.))
        estimator = RGBDOdometry((0., 0., .23), intrinsics=self.k, survey=self.survey())
        pose = estimator.world_from_optical.copy()
        depth = self.render(pose)
        depth += np.random.default_rng(1337).normal(0., .01, depth.shape).astype(np.float32)
        before = depth.copy()
        with self.assertRaisesRegex(TrackingFailure, 'insufficient_depth_overlap'):
            register_depth(DepthFrame(depth, self.k, DepthGates()), estimator.survey, pose, survey=True)
        record = estimator.estimate(self.rgb, depth, 100_000)
        self.assertTrue(record['valid'], record)
        self.assertEqual(record['source'], 'depth_survey')
        self.assertEqual(record['depth_normal_radius_px'], 2)
        self.assertGreater(record['depth_overlap'], .55)
        np.testing.assert_allclose(record['position_enu'], (0., 0., .23), atol=.012)
        np.testing.assert_array_equal(depth, before)

    def test_wide_survey_normals_keep_holes_and_depth_steps_out_of_support(self):
        depth = np.full(self.rgb.shape[:2], .7, dtype=np.float32)
        depth[15:45, 192:] = .95
        depth[85:115, 192] = np.nan
        depth[155:185, 192] = .95
        wide = DepthFrame(depth, self.k, DepthGates(), normal_radius_px=2)
        u, v = np.array([191, 191, 191, 184]), np.array([30, 100, 170, 220])
        points, normals = wide._surface_at(u, v)
        np.testing.assert_array_equal(np.isfinite(normals).all(axis=1), [False, False, False, True])
        np.testing.assert_array_equal(points[:, 2], depth[v, u])

        projected = self.rays[1:3, 1] * .7
        projected[1] = self.rays[2, 2] * .7
        ids, _, _ = wide.correspond(projected, np.tile([0., 0., 1.], (2, 1)), .08)
        np.testing.assert_array_equal(ids, [1])
        for invalid in (0, 3, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'normal radius'):
                DepthFrame(depth, self.k, DepthGates(), normal_radius_px=invalid)

    def test_wide_source_cannot_make_an_exact_survey_plane_observable(self):
        plane = np.full(self.rgb.shape[:2], 2., dtype=np.float32)
        measurement = plane + np.random.default_rng(7).normal(0., .01, plane.shape).astype(np.float32)
        exact = DepthFrame(plane, self.k, DepthGates())
        surveyed = DepthSurvey(exact.sample_points, exact.sample_normals, floor_z=-10.)
        wide = DepthFrame(measurement, self.k, DepthGates(), normal_radius_px=2)
        with self.assertRaisesRegex(TrackingFailure, 'unobservable_depth_geometry'):
            register_depth(wide, surveyed, np.eye(4), survey=True)
        fals = DepthFrame(measurement, self.k, DepthGates(), stride=8, normal_radius_px=2,
                          survey_fals=FalsSurveyNormals(self.k))
        with self.assertRaisesRegex(TrackingFailure, 'unobservable_depth_geometry'):
            register_depth(fals, surveyed, np.eye(4), survey=True)

    def test_fals_keeps_raw_centers_and_rejects_steps_thin_strips_and_holes(self):
        fals = FalsSurveyNormals(self.k)
        u, v = np.array([191, 191, 191, 189, 189, 184]), np.array([30, 100, 170, 100, 130, 220])
        with patch.object(cv2.rgbd, 'RgbdNormals_create', wraps=cv2.rgbd.RgbdNormals_create) as create:
            for height in (.03, .06, .10, .25):
                with self.subTest(height=height):
                    depth = np.full(self.rgb.shape[:2], .7, dtype=np.float32)
                    depth[15:45, 192:] += height
                    depth[85:115, 192] = np.nan
                    depth[155:185, 192] += height
                    depth[120:145, 192:] = .95
                    frame = DepthFrame(depth, self.k, DepthGates(), stride=8, normal_radius_px=2,
                                       survey_fals=fals)
                    points, normals = frame._surface_at(u, v)
                    np.testing.assert_array_equal(points[:, 2], depth[v, u])
                    np.testing.assert_array_equal(np.isfinite(normals).all(axis=1),
                                                  [False, False, False, False, False, True])
            create.assert_called_once()

    def test_fals_recovers_maze_divider_support_without_accepting_lateral_alias(self):
        # This oblique view sees the end wall and a narrow strip of the maze divider.
        # Short-baseline source normals can discard the strip under range noise.
        roll, pitch, yaw = np.radians([-.407, .553, 69.])
        body_rotation = (cv2.Rodrigues(np.array([0., 0., yaw]))[0]
                         @ cv2.Rodrigues(np.array([0., pitch, 0.]))[0]
                         @ cv2.Rodrigues(np.array([roll, 0., 0.]))[0])
        body_position = np.array([-.29565, 1.54764, .95099])
        body_from_optical = np.asarray(RGBD_BODY_FROM_OPTICAL)
        rotation = body_rotation @ body_from_optical
        camera = body_position + body_rotation @ RGBD_TRANSLATION
        rays = self.rays @ rotation.T
        depth = np.full(self.rgb.shape[:2], np.inf)
        faces = ((1, 3.048, (-3., 3.047, 0.), (3., 3.049, 3.05)),
                 (0, .254, (.253, .508, 0.), (.255, 2.1209, 1.7272)),
                 (1, 2.1209, (.254, 2.1208, 0.), (.3556, 2.121, 1.7272)),
                 (2, 0., (-10., -10., -.0001), (10., 10., .0001)))
        for axis, coordinate, lower, upper in faces:
            distance = (coordinate-camera[axis])/rays[..., axis]
            points = camera + distance[..., None]*rays
            hit = (distance > .04) & np.all((points >= lower) & (points <= upper), axis=-1)
            depth = np.minimum(depth, np.where(hit, distance, np.inf))
        depth = (depth + np.random.default_rng(29).normal(0., .01, depth.shape)).astype(np.float32)
        survey = DepthSurvey.from_course(Path(__file__).resolve().parents[1]/'models/course_v3/model.sdf')
        initial = np.eye(4)
        initial[:3, :3], initial[:3, 3] = rotation, camera + [.015, -.025, .006]
        origin = body_from_optical.T @ (np.array([0., 0., -.075])-RGBD_TRANSLATION)
        ray = body_from_optical.T @ np.array([0., 0., -1.])
        distance = -(camera+rotation@origin)[2]/(rotation@ray)[2]
        tof = DownwardRange(distance, origin, ray)
        tilt = InertialTilt(100_000, rotation.T @ np.array([0., 0., 1.]), .01, 0)

        central = DepthFrame(depth, self.k, DepthGates(), normal_radius_px=2)
        wrong, _ = register_depth(central, survey, initial, survey=True, downward_range=tof, inertial_tilt=tilt)
        self.assertGreater(abs(wrong[0, 3]-camera[0]), .10)

        estimator = RGBDOdometry(body_position, intrinsics=self.k, survey=survey, survey_normals='fals7')
        estimator.world_from_optical = initial
        record = estimator.estimate(self.rgb, depth, 100_000, downward_range=tof, inertial_tilt=tilt)
        self.assertTrue(record['valid'], record)
        self.assertTrue(record['source'].startswith('depth_survey'))
        np.testing.assert_allclose(record['position_enu'], body_position, atol=.01)
        self.assertEqual(record['depth_normals']['method'], 'fals7')
        self.assertEqual(record['depth_normals']['sample_stride_px'], 8)
        self.assertEqual(record['depth_normals']['center_curvature_radii_px'], [1, 2])
        self.assertEqual(record['depth_normal_radius_px'], 3)

    def test_fals_selection_requires_contrib(self):
        with patch.object(cv2, 'rgbd', None):
            with self.assertRaisesRegex(RuntimeError, 'OpenCV contrib cv2.rgbd'):
                RGBDOdometry((0., 0., 1.), intrinsics=self.k, survey=self.survey(), survey_normals='fals7')
            RGBDOdometry((0., 0., 1.), intrinsics=self.k, survey=self.survey())

    def test_noisy_plane_cannot_gain_rank_from_wider_relative_normals(self):
        rng = np.random.default_rng(741)
        old, new = [(2.+rng.normal(0., .01, self.rgb.shape[:2])).astype(np.float32) for _ in range(2)]
        # This fixture would falsely pass if the survey radius leaked into
        # the relative target normals, which do not come from surveyed planes.
        wide_frames = [DepthFrame(image, self.k, DepthGates(), normal_radius_px=2) for image in (new, old)]
        _, wrong = register_depth(*wide_frames, np.eye(4))
        self.assertGreater(wrong['depth_observability'], .004)

        textured = rng.integers(0, 256, self.rgb.shape, dtype=np.uint8)
        survey = self.survey()
        for method, radius in (('central2', 2), ('fals7', 3)):
            with self.subTest(method=method):
                estimator = RGBDOdometry((0., 0., 1.), intrinsics=self.k, survey=survey, survey_normals=method)
                self.assertTrue(estimator.estimate(textured, old, 100_000)['valid'])
                self.assertEqual(estimator.previous[2].normal_radius_px, 1)
                with patch.object(estimator, '_rgb_motion', side_effect=TrackingFailure('insufficient_inliers')):
                    record = estimator.estimate(self.rgb, new, 150_000)
                self.assertFalse(record['valid'])
                self.assertEqual(record['rejected_fits']['survey']['depth_normal_radius_px'], radius)
                self.assertEqual(record['rejected_fits']['survey']['depth_normals']['method'], method)
                relative = record['rejected_fits']['frame_depth']
                self.assertEqual(relative['depth_normals']['method'], 'central1')
                self.assertEqual(relative['depth_normal_radius_px'], 1)
                self.assertEqual(relative['reason'], 'insufficient_depth_overlap')

    def test_tof_recovers_the_missing_vertical_axis_only(self):
        survey = self.survey()
        pose = self.moved()
        frame = DepthFrame(self.render(pose, self.planes[:2]), self.k, DepthGates())
        direction = np.asarray(RGBD_BODY_FROM_OPTICAL).T @ np.array([0., 0., -1.])
        expected = -pose[2, 3]/(pose[:3, :3] @ direction)[2]
        tof = DownwardRange(expected, np.zeros(3), direction)
        with self.assertRaisesRegex(TrackingFailure, 'unobservable'):
            register_depth(frame, survey, self.world_from_camera, survey=True)
        found, quality = register_depth(frame, survey, self.world_from_camera,
                                        survey=True, downward_range=tof)
        np.testing.assert_allclose(found, pose, atol=.005)
        self.assertLess(abs(quality['tof_range_residual_m']), .002)

        one_wall = DepthFrame(self.render(pose, self.planes[:1]), self.k, DepthGates())
        with self.assertRaisesRegex(TrackingFailure, 'unobservable'):
            register_depth(one_wall, survey, self.world_from_camera, survey=True,
                           downward_range=tof)

    def test_tof_uses_nearest_mesh_surface_before_ground(self):
        triangle = np.array([[[-1., -1., .2], [1., -1., .2], [0., 1., .6]]])
        surface = SurveySurface(triangle, floor_z=0.)
        ray = np.array([0., 0., -1.])
        distance, normal = surface.intersect(np.array([0., 0., 1.]), ray)
        self.assertAlmostEqual(distance, .6)
        np.testing.assert_allclose(normal, (0., -.2, 1.)/np.linalg.norm((0., -.2, 1.)))
        distance, normal = surface.intersect(np.array([2., 0., 1.]), ray)
        self.assertAlmostEqual(distance, 1.)
        np.testing.assert_allclose(normal, (0., 0., 1.))

    def test_tilt_jacobian_matches_the_scaled_rotation_step(self):
        rotation = cv2.Rodrigues(np.array([.012, -.028, .08]))[0]
        up = np.array([.03, -.01, 1.])
        up /= np.linalg.norm(up)
        _, jac, _ = tilt_residual(rotation, up)
        for column, direction in enumerate(np.eye(3)):
            step = 1e-6
            plus = cv2.Rodrigues(step*direction/2)[0] @ rotation
            minus = cv2.Rodrigues(-step*direction/2)[0] @ rotation
            numeric = (tilt_residual(plus, up)[0]-tilt_residual(minus, up)[0])/(2*step)
            np.testing.assert_allclose(jac[:, 3+column], numeric, atol=1e-8)
        np.testing.assert_array_equal(jac[:, :3], 0.)
        np.testing.assert_array_equal(jac[:, 5], 0.)

    def test_tilt_resolves_pitch_null_but_cannot_invent_translation(self):
        a = np.linspace(-1., 1., 80)
        points = np.vstack([np.c_[np.full_like(a, 2.), a, 0*a],
                            np.c_[a, np.ones_like(a), a**2],
                            np.c_[0*a, a, 0*a]])
        normals = np.repeat(np.eye(3), len(a), axis=0)
        initial = np.eye(4)
        initial[:3, :3] = cv2.Rodrigues(np.array([0., .02, 0.]))[0]
        tilt = InertialTilt(1_000_000, np.array([0., 0., 1.]), .01, 0)
        wrong_tilt = InertialTilt(1_000_000, np.array([1., 0., 0.]), .01, 0)
        overconfident_tilt = InertialTilt(1_000_000, np.array([0., 0., 1.]), .001, 0)

        def pair(drop):
            xyz, directions = points[drop:], normals[drop:]
            frame = SimpleNamespace(sample_points=xyz, sample_normals=directions)
            reference = SimpleNamespace(correspond=lambda p, n, d: (np.arange(len(xyz)), xyz, directions))
            return frame, reference

        with self.assertRaisesRegex(TrackingFailure, 'unobservable'):
            register_depth(*pair(0), initial, survey=True)
        found, diagnostics = register_depth(*pair(0), initial, survey=True, inertial_tilt=tilt)
        np.testing.assert_allclose(found, np.eye(4), atol=.0002)
        self.assertEqual(diagnostics['px4_tilt_prior_status'], 'applied')
        self.assertEqual(diagnostics['px4_tilt_capture_usec'], 1_000_000)
        self.assertGreaterEqual(diagnostics['rotation_std'][1], .074)
        with self.assertRaisesRegex(TrackingFailure, 'unobservable'):
            register_depth(*pair(len(a)), initial, survey=True, inertial_tilt=tilt)
        with self.assertRaisesRegex(TrackingFailure, 'unobservable') as failure:
            register_depth(*pair(0), initial, survey=True, inertial_tilt=wrong_tilt)
        self.assertEqual(failure.exception.details['px4_tilt_prior_status'], 'disagreement')
        with self.assertRaisesRegex(TrackingFailure, 'unobservable'):
            register_depth(*pair(0), initial, survey=True, inertial_tilt=overconfident_tilt)

    def test_tilt_cannot_invent_yaw_during_disagreement(self):
        a = np.linspace(-1., 1., 80)
        points = np.vstack([np.c_[np.full_like(a, 2.), 0*a, a],
                            np.c_[0*a, np.ones_like(a), a],
                            np.c_[a, a**2, 0*a]])
        normals = np.repeat(np.eye(3), len(a), axis=0)
        frame = SimpleNamespace(sample_points=points, sample_normals=normals)
        reference = SimpleNamespace(correspond=lambda p, n, d: (np.arange(len(points)), points, normals))
        tilt = InertialTilt(1_000_000, np.array([np.sin(.03), 0., np.cos(.03)]), .01, 0)
        with self.assertRaisesRegex(TrackingFailure, 'unobservable'):
            register_depth(frame, reference, np.eye(4), survey=True, inertial_tilt=tilt)

    def test_stale_tilt_is_ignored_at_odometry_boundary(self):
        pose = self.world_from_camera
        depth = self.render(pose)
        estimator = RGBDOdometry((0, 0, 1), intrinsics=self.k, survey=self.survey())
        stale = InertialTilt(50_000, pose[:3, :3].T @ np.array([0., 0., 1.]), .01, 0)
        record = estimator.estimate(self.rgb, depth, 100_000, inertial_tilt=stale)
        self.assertTrue(record['valid'], record)
        self.assertNotIn('px4_tilt_prior_status', record)

    def test_unobservable_planes_are_rejected_and_loss_latches_after_startup(self):
        wall = self.render(self.world_from_camera, ((0, 3.),))
        estimator = RGBDOdometry((0, 0, 1), intrinsics=self.k)
        before_start = estimator.estimate(self.rgb, wall, 100_000)
        self.assertFalse(before_start['valid'])
        self.assertIn('unobservable', before_start['reason'])
        self.assertIsNone(before_start['position_enu'])
        usable = self.render(self.world_from_camera)
        self.assertTrue(estimator.estimate(self.rgb, usable, 150_000)['valid'])
        failed = estimator.estimate(self.rgb, wall, 200_000)
        self.assertFalse(failed['valid'])
        self.assertTrue(failed['reason'].startswith('tracking_lost:'))
        rejected_fit = failed['rejected_fits']['frame_depth']
        self.assertLess(rejected_fit['depth_observability'], 1e-6)
        self.assertEqual(len(rejected_fit['reference_from_optical_3x4']), 3)
        self.assertFalse(estimator.estimate(self.rgb, usable, 250_000)['valid'])

    def test_clipped_depth_is_not_a_surface(self):
        for fill in (np.nan, np.inf, .02, 8.):
            frame = DepthFrame(np.full(self.rgb.shape[:2], fill), self.k, DepthGates())
            self.assertEqual(len(frame.sample_points), 0)


if __name__ == '__main__':
    unittest.main()
