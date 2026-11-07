import contextlib
import io
import json
from pathlib import Path
import signal
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from tools.check_oak_depth import Probe, calibration, image, main, record_bag


TOPICS = {key: '/camera/'+key for key in ('image', 'depth', 'camera_info', 'imu')}


def stamped(ns, frame='oak_left_optical_frame'):
    sec, nanosec = divmod(ns, 10**9)
    return NS(stamp=NS(sec=sec, nanosec=nanosec), frame_id=frame)


def mono(ns, frame='oak_left_optical_frame'):
    return NS(header=stamped(ns, frame), encoding='mono8', width=2, height=2,
              is_bigendian=0, step=2, data=bytes(4))


def mm(ns, frame='oak_left_optical_frame'):
    return NS(header=stamped(ns, frame), encoding='16UC1', width=2, height=2,
              is_bigendian=1, step=6, data=struct.pack('>HHxxHHxx', 0, 1000, 1500, 2000))


def info(ns):
    return NS(header=stamped(ns), width=2, height=2, distortion_model='plumb_bob',
              k=[1., 0., 1., 0., 1., 1., 0., 0., 1.],
              p=[1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0.],
              r=[1., 0., 0., 0., 1., 0., 0., 0., 1.], d=[0.]*5)


def imu(ns):
    return NS(header=stamped(ns, 'oak_imu_frame'), angular_velocity=NS(x=0., y=.1, z=0.),
              linear_acceleration=NS(x=0., y=0., z=9.81),
              angular_velocity_covariance=[0.]*9, linear_acceleration_covariance=[0.]*9)


class OakDepthTests(unittest.TestCase):
    def test_big_endian_millimetres_with_row_padding(self):
        fields, depth = image(mm(10**9), depth=True)
        self.assertEqual(fields['metres_per_unit'], .001)
        self.assertEqual(depth['valid_ratio'], .75)
        self.assertEqual(depth['min_m'], 1.)
        self.assertEqual(depth['max_m'], 2.)
        self.assertEqual(depth['central_roi']['pixel_bounds'], [0, 2, 0, 2])
        tile_pixels = sum(tile['total_pixels'] for tile in depth['tiles_3x3'])
        self.assertEqual(tile_pixels, 4)
        self.assertEqual(sum(tile['valid_pixels'] for tile in depth['tiles_3x3']), 3)

    def test_float_metres_exclude_zero_and_nonfinite(self):
        msg = NS(encoding='32FC1', width=3, height=2, step=12, is_bigendian=0,
                 data=struct.pack('<6f', 0., float('inf'), float('nan'), -1., .5, 2.))
        fields, depth = image(msg, depth=True)
        self.assertEqual(fields['metres_per_unit'], 1.)
        self.assertEqual(depth['valid_ratio'], 1/3)
        self.assertEqual(depth['min_m'], .5)
        self.assertEqual(depth['max_m'], 2.)
        self.assertEqual(depth['valid_pixels'], 2)
        self.assertEqual(depth['total_pixels'], 6)
        for field, invalid in (('step', 11), ('is_bigendian', 2), ('encoding', '16SC1')):
            broken = NS(**(vars(msg) | {field: invalid}))
            with self.assertRaises(ValueError):
                image(broken, depth=True)

    def test_localized_hole_is_visible_in_central_roi_and_tiles_without_changing_global_ratio(self):
        width, height = 9, 9
        values = [1000] * (width*height)
        for row in range(3, 6):
            for col in range(3, 6):
                values[row*width+col] = 0
        msg = NS(encoding='16UC1', width=width, height=height, step=width*2,
                 is_bigendian=0, data=struct.pack('<81H', *values))

        _, stats = image(msg, depth=True)

        self.assertEqual(stats['valid_ratio'], .8888888888888888)
        self.assertEqual(stats['central_roi'], dict(pixel_bounds=[2, 7, 2, 7],
                                                    valid_pixels=16, total_pixels=25,
                                                    valid_ratio=.64))
        center_tile = stats['tiles_3x3'][4]
        self.assertEqual((center_tile['row'], center_tile['col']), (1, 1))
        self.assertEqual(center_tile['pixel_bounds'], [3, 6, 3, 6])
        self.assertEqual(center_tile['valid_ratio'], 0.)

    def test_spatial_metrics_are_diagnostic_and_legacy_global_preflight_gate_is_unchanged(self):
        width = height = 9
        stamp = 10**9
        raw = [1000] * (width*height)
        for row in range(3, 6):
            for col in range(3, 6):
                raw[row*width+col] = 0
        depth = NS(header=stamped(stamp), encoding='16UC1', width=width, height=height,
                   is_bigendian=0, step=width*2, data=struct.pack('<81H', *raw))
        image_msg = NS(header=stamped(stamp), encoding='mono8', width=width, height=height,
                       is_bigendian=0, step=width, data=bytes(width*height))
        camera_info = info(stamp)
        camera_info.width = width
        camera_info.height = height
        camera_info.k[2] = camera_info.p[2] = width/2
        camera_info.k[5] = camera_info.p[6] = height/2
        probe = Probe(TOPICS, minimum_pairs=1)
        for role, msg in (('camera_info', camera_info), ('image', image_msg),
                          ('depth', depth), ('imu', imu(stamp))):
            probe.ingest(role, msg, 1.)

        report = probe.report(1.1)

        self.assertTrue(report['passed'], report['problems'])
        self.assertEqual(report['matched_pairs'], 1)
        self.assertAlmostEqual(report['latest_depth']['valid_ratio'], 8/9)
        self.assertEqual(report['latest_depth']['tiles_3x3'][4]['valid_ratio'], 0.)
        self.assertEqual(report['latest_depth']['min_m'], 1.)
        self.assertEqual(report['latest_depth']['max_m'], 1.)

    def test_odd_dimensions_tiles_are_disjoint_and_cover_every_pixel(self):
        width, height = 5, 7
        msg = NS(encoding='32FC1', width=width, height=height, step=width*4,
                 is_bigendian=0, data=struct.pack('<35f', *([1.25]*35)))

        _, stats = image(msg, depth=True)

        covered = []
        for tile in stats['tiles_3x3']:
            r0, r1, c0, c1 = tile['pixel_bounds']
            covered.extend((r, c) for r in range(r0, r1) for c in range(c0, c1))
            self.assertEqual(tile['valid_ratio'], 1.)
        self.assertEqual(len(covered), width*height)
        self.assertEqual(set(covered), {(r, c) for r in range(height) for c in range(width)})
        self.assertEqual(stats['central_roi']['pixel_bounds'], [1, 6, 1, 4])

    def test_camera_info_rejects_uncalibrated_or_nonfinite_values(self):
        for broken in (0., float('nan')):
            camera_info = info(10**9)
            camera_info.k[0] = broken
            with self.assertRaisesRegex(ValueError, 'CameraInfo'):
                calibration(camera_info)

    def test_accepts_aligned_pairs_with_nearby_imu_and_tof(self):
        probe = Probe({**TOPICS, 'tof': '/tof/range'}, minimum_pairs=2)
        probe.ingest('camera_info', info(10**9), 1.)
        for ns in (10**9, 1_040_000_000):
            probe.ingest('image', mono(ns), 1.)
            probe.ingest('depth', mm(ns+5_000_000), 1.)
            probe.ingest('imu', imu(ns+6_000_000), 1.)
            probe.ingest('tof', NS(header=stamped(ns+12_000_000, 'range'), range=.9,
                                   min_range=.1, max_range=4.), 1.)
        report = probe.report(1.1)
        self.assertTrue(report['passed'], report['problems'])
        self.assertEqual(report['synchronized_recent_pairs'], 2)
        self.assertEqual(report['max_pair_skew_ms'], 5.)
        self.assertFalse(probe.report(2.1)['passed'])

    def test_rejects_frame_mismatch_and_nonmonotonic_capture(self):
        probe = Probe(TOPICS, minimum_pairs=1)
        probe.ingest('camera_info', info(10**9), 1.)
        probe.ingest('image', mono(10**9), 1.)
        probe.ingest('depth', mm(10**9, 'different_optical_frame'), 1.)
        probe.ingest('depth', mm(999_999_999, 'different_optical_frame'), 1.)
        probe.ingest('imu', imu(10**9), 1.)
        report = probe.report(1.1)
        self.assertFalse(report['passed'])
        self.assertEqual(report['invalid_counts']['depth'], 1)
        self.assertTrue(any('disagree on frame' in problem for problem in report['problems']))

    def test_rejects_wrong_clock_even_when_image_depth_pair_matches(self):
        probe = Probe(TOPICS, minimum_pairs=1)
        for role, msg in [('camera_info', info(10**9)), ('image', mono(10**9)),
                          ('depth', mm(10**9)), ('imu', imu(2*10**9))]:
            probe.ingest(role, msg, 1.)
        report = probe.report(1.1)
        self.assertEqual(report['matched_pairs'], 1)
        self.assertEqual(report['synchronized_recent_pairs'], 0)
        self.assertFalse(report['passed'])

    def test_missing_depth_does_not_start_requested_bag(self):
        def stereo_imu_only(probe, seconds, latched):
            for role, msg in [('camera_info', info(10**9)), ('image', mono(10**9)), ('imu', imu(10**9))]:
                probe.ingest(role, msg, 1.)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'manifest.json'
            args = ['check_oak_depth', '--output', str(out), '--record-seconds', '1',
                    '--bag-output', str(Path(directory)/'capture')]
            for role, topic in TOPICS.items():
                args.extend(['--'+role.replace('_', '-')+'-topic', topic])
            with patch.object(sys, 'argv', args), patch('tools.check_oak_depth.observe', stereo_imu_only), \
                    patch('tools.check_oak_depth.record_bag') as bag, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
                bag.assert_not_called()
            report = json.loads(out.read_text())
        self.assertEqual(report['counts']['depth'], 0)
        self.assertIsNone(report['recording'])
        self.assertTrue(any('no valid depth messages' in problem for problem in report['problems']))

    def test_bag_handoff_interrupts_recorder_and_verifies_topic_counts(self):
        child = Mock(pid=321, returncode=0)
        child.poll.side_effect = [None, None, 0]
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory)/'capture'
            bag.mkdir()
            (bag/'metadata.yaml').write_text(json.dumps({'rosbag2_bagfile_information': {
                'topics_with_message_count': [{'topic_metadata': {'name': '/depth'}, 'message_count': 3}]
            }}))
            with patch.dict(sys.modules, {'yaml': NS(safe_load=json.loads)}), \
                    patch('tools.check_oak_depth.subprocess.Popen', return_value=child) as launch, \
                    patch('tools.check_oak_depth.os.killpg') as kill, \
                    patch('tools.check_oak_depth.shutil.disk_usage', return_value=NS(free=10**9)), \
                    patch('tools.check_oak_depth.time.monotonic', side_effect=[0., 1., 1.1]):
                result = record_bag(bag, ['/depth', '/tf_static'], 1., None)
        kill.assert_called_once_with(321, signal.SIGINT)
        child.wait.assert_called_once_with(timeout=8)
        self.assertEqual(launch.call_args.args[0], ['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(bag),
                                                   '/depth', '/tf_static'])
        self.assertEqual(result['topic_counts'], {'/depth': 3})
        self.assertEqual(result['missing_topics'], ['/tf_static'])
        self.assertFalse(result['complete'])

    def test_sigterm_during_recorder_start_flushes_child_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        child = Mock(pid=321, returncode=0)
        child.poll.side_effect = [None, None, 0]

        def start(*args, **kwargs):
            # Deliver the installed callback before Popen returns its handle.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return child

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'yaml': NS(safe_load=json.loads)}), \
                patch('tools.check_oak_depth.subprocess.Popen', side_effect=start), \
                patch('tools.check_oak_depth.os.killpg') as kill, \
                patch('tools.check_oak_depth.shutil.disk_usage', return_value=NS(free=10**9)):
            result = record_bag(Path(directory)/'capture', ['/depth'], 1., None)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        kill.assert_called_once_with(321, signal.SIGINT)
        child.wait.assert_called_once_with(timeout=8)
        self.assertEqual(result['stop_reason'], 'terminated')
        self.assertFalse(result['complete'])


if __name__ == '__main__':
    unittest.main()
