import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from vision.flight_observe import base_link_pose
from vision.flight_record import FrameArchive, RGBDPairQueue, ToFBuffer, decode_tof_message


def pose(position, quaternion=(0, 0, 0, 1)):
    return SimpleNamespace(
        position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
        orientation=SimpleNamespace(x=quaternion[0], y=quaternion[1],
                                    z=quaternion[2], w=quaternion[3]),
    )


class FlightRecordTests(unittest.TestCase):
    def test_rgbd_queue_pairs_only_matching_capture_stamps(self):
        queue = RGBDPairQueue(max_pending=2, max_ready=2)
        queue.push('color', 10, 'rgb10')
        queue.push('depth', 11, 'depth11')
        self.assertIsNone(queue.take())
        queue.push('depth', 10, 'depth10')
        self.assertEqual(queue.take(), (10, 'rgb10', 'depth10'))
        self.assertIsNone(queue.take())

    def test_rgbd_queue_and_archive_remain_bounded(self):
        queue = RGBDPairQueue(max_pending=2, max_ready=1)
        for stamp in (10, 20, 30):
            queue.push('color', stamp, f'c{stamp}')
        self.assertLessEqual(len(queue.color), 2)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'record'
            archive = FrameArchive(output, max_frames=8, max_bytes=10)
            self.assertTrue(archive.add(1_000_000_000, b'123', b'456'))
            self.assertFalse(archive.add(2_000_000_000, b'789', b'012'))
            self.assertEqual(archive.limit_reason, 'max_bytes')
            self.assertEqual(archive.frame_count, 1)
            self.assertEqual(archive.bytes_written, 6)
            archive.close(state='bounded_recording_complete')
            row = json.loads((output / 'frames.jsonl').read_text())
            self.assertEqual(row['capture_ns'], 1_000_000_000)
            self.assertEqual(row['depth_z_mm_u16'], 'frames/depth-000000.png')
            status = json.loads((output / 'status.json').read_text())
            self.assertFalse(status['includes_vehicle_pose'])
            self.assertNotIn('position', status)

    def test_truth_transform_applies_model_rotation_and_translation(self):
        half = 2**-0.5
        model = pose((2, 3, 4), (0, 0, half, half))
        link = pose((1, 0, 0))
        position, orientation = base_link_pose(model, link)
        for actual, expected in zip(position, (2, 4, 4)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(orientation, (0, 0, half, half)):
            self.assertAlmostEqual(actual, expected)

    def test_truth_transform_identity_preserves_local_offset(self):
        model = pose((-2.5, 0.9, -0.013))
        link = pose((0, 0, 0.24))
        position, orientation = base_link_pose(model, link)
        for actual, expected in zip(position, (-2.5, 0.9, 0.227)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(orientation, (0, 0, 0, 1))

    def test_tof_matches_nearest_sample_across_out_of_order_arrival(self):
        matcher = ToFBuffer()
        matcher.push(1_010_000_000, 1.01)
        matcher.push(996_000_000, .996)
        result = matcher.match(1_000_000_000)
        self.assertEqual(result['status'], 'paired')
        self.assertEqual(result['capture_ns'], 996_000_000)
        self.assertEqual(result['range_m'], .996)
        self.assertAlmostEqual(result['signed_age_ms'], -4)

    def test_tof_missing_and_bad_age_are_explicit_but_optional(self):
        matcher = ToFBuffer()
        missing = matcher.match(1_000_000_000)
        self.assertIsNone(missing['range_m'])
        self.assertEqual(missing['reason'], 'no_tof_sample')
        matcher.push(1_020_000_001, .8)
        stale = matcher.match(1_000_000_000)
        self.assertIsNone(stale['range_m'])
        self.assertIsNone(stale['capture_ns'])
        self.assertEqual(stale['reason'], 'source_bad_age')
        with tempfile.TemporaryDirectory() as temp:
            archive = FrameArchive(Path(temp) / 'record')
            self.assertTrue(archive.add(1_000_000_000, b'rgb', b'depth',
                                        {'tof': missing}))
            archive.close(state='duration_complete')
            row = json.loads((Path(temp) / 'record/frames.jsonl').read_text())
            self.assertEqual(row['diagnostics']['tof']['reason'], 'no_tof_sample')

    def test_tof_message_requires_single_float32_pixel_and_in_clip_range(self):
        message = SimpleNamespace(pixel_format_type=3, width=1, height=1, step=4,
                                  data=__import__('struct').pack('<f', 1.25))
        value, reason = decode_tof_message(message, float_format=3,
                                           min_depth_m=.025, max_depth_m=4.0)
        self.assertAlmostEqual(value, 1.25)
        self.assertIsNone(reason)
        message.width = 2
        value, reason = decode_tof_message(message, float_format=3,
                                           min_depth_m=.025, max_depth_m=4.0)
        self.assertIsNone(value)
        self.assertEqual(reason, 'unexpected_image_dimensions')


if __name__ == '__main__':
    unittest.main()
