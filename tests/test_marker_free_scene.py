import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from vision import scene
from vision.course import CAMERA_TOPIC


COURSE = Path(__file__).resolve().parents[1] / 'worlds' / 'course.sdf'


class MarkerFreeSceneTests(unittest.TestCase):
    def test_default_generation_keeps_course_and_camera_without_marker_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'course-px4.sdf'
            scene.generate(COURSE, output, 0.0)

            world = ET.parse(output).getroot().find('world')
            self.assertIn('model://course_v3', [include.findtext('uri')
                                               for include in world.findall('include')])
            self.assertEqual(
                world.findtext("model[@name='course_quad']/link/sensor[@name='down_camera']/topic"),
                CAMERA_TOPIC,
            )
            self.assertIsNone(world.find("model[@name='vision_landmarks']"))
            self.assertFalse(output.with_suffix('.assets').exists())

    def test_cli_marker_fixture_requires_explicit_opt_in(self):
        for flag, setting, expected in (
            ([], '', False), ([], '0', False), ([], 'true', False),
            (['--with-test-markers'], '0', True), ([], '1', True),
        ):
            with self.subTest(flag=flag, setting=setting), \
                    patch.dict(os.environ, {'SIM_TEST_MARKERS': setting}), \
                    patch.object(sys, 'argv', ['scene', '--output', 'test.sdf', *flag]), \
                    patch.object(scene, 'generate') as generate, \
                    patch.object(sys, 'stderr') as diagnostic:
                scene.main()

                generate.assert_called_once_with(
                    Path('worlds/course.sdf'), Path('test.sdf'), 0.0, with_test_markers=expected,
                )
                self.assertEqual(diagnostic.write.called, expected)


if __name__ == '__main__':
    unittest.main()
