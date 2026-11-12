import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


STARTUP = Path(__file__).resolve().parents[1] / 'runtime' / 'start.sh'


def gui_config_script():
    startup = STARTUP.read_text()
    match = re.search(r"python3 [^\n]*<<'PY'\n(.*?)\nPY\n",
                      startup, re.S)
    if not match:
        raise AssertionError('GUI config Python heredoc not found in runtime/start.sh')
    return match.group(1)


class GazeboGuiConfigTests(unittest.TestCase):
    def run_configurator(self, home, size, world=None):
        namespace = {'__name__': '__main__'}
        with patch.object(Path, 'home', return_value=home), \
                patch.object(sys, 'argv', ['-', size] + ([str(world)] if world else [])):
            exec(compile(gui_config_script(), str(STARTUP), 'exec'), namespace)

    def test_resizes_window_fragment_and_preserves_sibling_plugin_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / '.gz/sim/8/gui.config'
            config.parent.mkdir(parents=True)
            plugin = '<plugin name="view"><width>123</width></plugin>'
            config.write_text('<window><width>1280</width><height>800</height></window>\n'
                              + plugin + '\n')

            self.run_configurator(home, '800x600')

            updated = config.read_text()
            window_match = re.search(r'<window(?:\s[^>]*)?>.*?</window>', updated, re.S)
            self.assertIsNotNone(window_match)
            window = ET.fromstring(window_match.group())
            self.assertEqual(window.findtext('width'), '800')
            self.assertEqual(window.findtext('height'), '600')
            self.assertIn(plugin, updated)
            plugin_match = re.search(r'<plugin\b.*?</plugin>', updated, re.S)
            self.assertEqual(ET.fromstring(plugin_match.group()).findtext('width'), '123')

    def test_invalid_size_fails_before_overwriting_existing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / '.gz/sim/8/gui.config'
            config.parent.mkdir(parents=True)
            original = '<window><width>1280</width><height>800</height></window>\n'
            config.write_text(original)

            with self.assertRaisesRegex(SystemExit, 'SIM_DESKTOP_SIZE must be WIDTHxHEIGHT'):
                self.run_configurator(home, '80x600oops')

            self.assertEqual(config.read_text(), original)

    def test_explicit_config_keeps_world_camera_and_window_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / '.gz/sim/8/gui.config'
            config.parent.mkdir(parents=True)
            config.write_text('<window><width>1280</width><height>800</height></window>')
            world = home / 'course.sdf'
            world.write_text('<sdf><world><gui><plugin filename="MinimalScene">'
                             '<camera_pose>-6 -6 8 0 .724 .785</camera_pose>'
                             '</plugin></gui></world></sdf>')
            self.run_configurator(home, '800x600', world)
            fragments = config.with_name('dronehacks-gui.config').read_text()
            root = ET.fromstring('<config>' + fragments + '</config>')
            self.assertEqual(root.findtext('window/width'), '800')
            self.assertEqual(root.findtext('window/height'), '600')
            self.assertEqual(root.findtext('plugin/camera_pose'), '-6 -6 8 0 .724 .785')


if __name__ == '__main__':
    unittest.main()
