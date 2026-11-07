import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tools.make_net_visual_ablation import (NET_VISUAL_NAMES, canonical, generate_pair)
from vision.rgbd_scene import generate as generate_rgbd_world

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / 'worlds/course.sdf'
COURSE = ROOT / 'models/course_v3/model.sdf'


def write_tree(tree, path):
    element = tree.getroot() if isinstance(tree, ET.ElementTree) else tree
    path.write_bytes(ET.tostring(element, encoding='utf-8', xml_declaration=True))


class NetVisualAblationTests(unittest.TestCase):
    def test_pair_flattens_control_and_removes_only_five_net_visuals(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'pair'
            manifest = generate_pair(WORLD, COURSE, output)
            source_world = ET.parse(WORLD).getroot()
            source_model = ET.parse(COURSE).getroot().find('model')
            control = ET.parse(output / 'control.sdf').getroot()
            treatment = ET.parse(output / 'treatment.sdf').getroot()
            cw, tw = control.find('world'), treatment.find('world')
            control_model = cw.find("model[@name='course_v3']")
            treatment_model = tw.find("model[@name='course_v3']")
            self.assertIsNotNone(control_model)
            self.assertEqual(canonical(control_model), canonical(source_model))
            expected_control = copy.deepcopy(source_world)
            expected_world = expected_control.find('world')
            course_include = next(i for i in expected_world.findall('include')
                                  if i.findtext('uri') == 'model://course_v3')
            insert_at = list(expected_world).index(course_include)
            expected_world.remove(course_include)
            expected_world.insert(insert_at, copy.deepcopy(source_model))
            self.assertEqual(canonical(expected_control), canonical(control))

            source_vehicle_include = next(i for i in source_world.findall('./world/include')
                                          if i.findtext('uri') == 'model://course_x3')
            control_vehicle_include = next(i for i in cw.findall('include')
                                            if i.findtext('uri') == 'model://course_x3')
            treatment_vehicle_include = next(i for i in tw.findall('include')
                                              if i.findtext('uri') == 'model://course_x3')
            self.assertEqual(canonical(source_vehicle_include), canonical(control_vehicle_include))
            self.assertEqual(canonical(source_vehicle_include), canonical(treatment_vehicle_include))

            control_visual_names = [v.get('name') for v in control_model.findall('./link/visual')]
            treatment_visual_names = [v.get('name') for v in treatment_model.findall('./link/visual')]
            self.assertEqual(set(n for n in control_visual_names if n.startswith('net_')),
                             set(NET_VISUAL_NAMES))
            self.assertFalse(set(treatment_visual_names) & set(NET_VISUAL_NAMES))
            control_collisions = control_model.findall('./link/collision')
            treatment_collisions = treatment_model.findall('./link/collision')
            self.assertEqual([canonical(c) for c in control_collisions],
                             [canonical(c) for c in treatment_collisions])
            self.assertEqual({c.get('name') for c in treatment_collisions
                              if c.get('name', '').startswith('net_')}, set(NET_VISUAL_NAMES))

            control_without_net = copy.deepcopy(control)
            model = control_without_net.find("./world/model[@name='course_v3']")
            for visual in list(model.findall('./link/visual')):
                if visual.get('name') in NET_VISUAL_NAMES:
                    model.find('link').remove(visual)
            self.assertEqual(canonical(control_without_net), canonical(treatment))
            self.assertEqual(manifest['ablation']['all_course_collision_semantics_equal'], True)
            self.assertEqual(manifest['ablation']['net_collision_semantics_equal'], True)
            self.assertEqual(manifest['ablation']['removed_visual_names'], list(NET_VISUAL_NAMES))
            self.assertEqual(manifest['outputs']['control.sdf']['sha256'],
                             __import__('hashlib').sha256((output/'control.sdf').read_bytes()).hexdigest())

    def test_both_variants_feed_the_existing_rgbd_world_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pair = folder / 'pair'
            generate_pair(WORLD, COURSE, pair)
            runtime = {}
            for variant in ('control', 'treatment'):
                runtime[variant] = folder / f'{variant}-rgbd.sdf'
                generate_rgbd_world(pair/f'{variant}.sdf', runtime[variant])
            trees = {name: ET.parse(path).getroot() for name, path in runtime.items()}
            for root in trees.values():
                self.assertIsNotNone(root.find('.//sensor[@type="rgbd_camera"]'))
                self.assertIsNotNone(root.find('.//sensor[@name="down_tof"]'))
            control_model = trees['control'].find("./world/model[@name='course_v3']")
            treatment_model = trees['treatment'].find("./world/model[@name='course_v3']")
            self.assertEqual({v.get('name') for v in control_model.findall('./link/visual')
                              if v.get('name', '').startswith('net_')}, set(NET_VISUAL_NAMES))
            self.assertFalse({v.get('name') for v in treatment_model.findall('./link/visual')
                              if v.get('name', '').startswith('net_')})
            self.assertEqual([canonical(c) for c in control_model.findall('.//collision')],
                             [canonical(c) for c in treatment_model.findall('.//collision')])

    def test_rejects_unexpected_course_include_override(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tree = ET.parse(WORLD)
            include = next(i for i in tree.getroot().findall('./world/include')
                           if i.findtext('uri') == 'model://course_v3')
            ET.SubElement(include, 'pose').text = '0.1 0 0 0 0 0'
            changed = folder / 'world.sdf'
            write_tree(tree, changed)
            with self.assertRaisesRegex(ValueError, 'course_v3 include override'):
                generate_pair(changed, COURSE, folder/'out')
            self.assertFalse((folder/'out').exists())

    def test_rejects_unexpected_vehicle_include_override(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tree = ET.parse(WORLD)
            include = next(i for i in tree.getroot().findall('./world/include')
                           if i.findtext('uri') == 'model://course_x3')
            ET.SubElement(include, 'static').text = 'true'
            changed = folder / 'world.sdf'
            write_tree(tree, changed)
            with self.assertRaisesRegex(ValueError, 'course_quad include override'):
                generate_pair(changed, COURSE, folder/'out')

    def test_rejects_missing_net_visual(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tree = ET.parse(COURSE)
            link = tree.getroot().find('./model/link')
            link.remove(next(v for v in link.findall('visual') if v.get('name') == 'net_roof'))
            changed = folder / 'model.sdf'
            write_tree(tree, changed)
            with self.assertRaisesRegex(ValueError, 'five named net visuals'):
                generate_pair(WORLD, changed, folder/'out')

    def test_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing'
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, 'refusing to overwrite'):
                generate_pair(WORLD, COURSE, output)


if __name__ == '__main__':
    unittest.main()
