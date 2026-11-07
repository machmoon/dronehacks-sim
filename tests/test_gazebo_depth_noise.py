import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from mission.navigation_audit import _read_model_sources
from tools.check_gazebo_depth_noise import (DEFAULT_PATCH, MODES, compare_plane,
                                            configure_world, sha256, verify)
from vision.rgbd_scene import generate


ROOT = Path(__file__).resolve().parents[1]
CAMERA = './world/model[@name="course_quad"]/link[@name="down_camera_link"]/sensor/camera'


class GazeboDepthNoiseTests(unittest.TestCase):
    def test_generated_world_contains_only_forward_camera_change_and_is_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            world = Path(directory)/'world.sdf'
            generate(ROOT/'worlds/course.sdf', world)
            original = world.read_bytes()
            control = configure_world(world, 'control')
            self.assertEqual(world.read_bytes(), original)
            noisy = configure_world(world, 'gaussian-10mm')
            patched = ET.fromstring(world.read_bytes())
            camera = patched.find(CAMERA)
            self.assertEqual(camera.findtext('noise/type'), 'gaussian')
            self.assertEqual(float(camera.findtext('noise/stddev')), .01)
            camera.remove(camera.find('noise'))
            ET.indent(patched, space='  ')
            self.assertEqual(ET.tostring(patched), ET.tostring(ET.fromstring(original)))
            hashes = _read_model_sources(world, (ROOT/'models', ROOT/'assets'))
            self.assertEqual(hashes[0]['sha256'], noisy['world_sha256'])
            self.assertEqual(noisy['source_world_sha256'], control['world_sha256'])
            self.assertNotEqual(noisy['world_sha256'], control['world_sha256'])
            with self.assertRaisesRegex(ValueError, 'unmodified forward camera'):
                configure_world(world, 'gaussian-10mm')

    def test_static_plane_verification_separates_noise_from_noop_and_global_motion(self):
        rng = np.random.default_rng(4417)
        plane = np.broadcast_to(np.full((32, 32), 2.), (24, 32, 32))
        self.assertTrue(compare_plane(plane, plane+rng.normal(0., .010, plane.shape))['passed'])
        self.assertFalse(compare_plane(plane, plane)['checks']['added_noise_in_range'])
        self.assertFalse(compare_plane(plane, plane+.03)['checks']['same_view'])
        frame_jitter = plane+rng.normal(0., .010, (len(plane), 1, 1))
        self.assertFalse(compare_plane(plane, frame_jitter)['checks']['not_global_frame_jitter'])

    def test_capture_verifier_requires_audit_to_hash_actual_world(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            world = folder/'world.sdf'
            rng = np.random.default_rng(4417)
            for mode, stddev in MODES.items():
                generate(ROOT/'worlds/course.sdf', world)
                config = configure_world(world, mode)
                (folder/f'{mode}.config.json').write_text(json.dumps(config))
                audit = dict(world_and_model_sources=[{'path': str(world.resolve()), 'sha256': sha256(world.read_bytes())}],
                             truth_topic_publishers=0, artificial_landmark_models=0, live_simulator_position_used=False)
                (folder/f'{mode}.audit.json').write_text(json.dumps(audit))
                frames = (2.+rng.normal(0., stddev, (12, 256, 384))).astype(np.float32)
                np.savez(folder/f'{mode}.npz', depth_z_m=frames, stamp_ns=np.arange(12)*50_000_000)
            self.assertTrue(verify(folder, DEFAULT_PATCH)['passed'])
            audit['world_and_model_sources'][0]['sha256'] = '0'*64
            (folder/'gaussian-10mm.audit.json').write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'provenance did not match'):
                verify(folder, DEFAULT_PATCH)


if __name__ == '__main__':
    unittest.main()
