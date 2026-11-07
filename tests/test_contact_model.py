import hashlib
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / 'models/flight_x500_base/model.sdf'
X500_MODEL = ROOT / 'models/flight_x500/model.sdf'
STOCK_BASE_SHA256 = 'd97e53de8c50f167b0b3a4f86dc21413236387f7b72cb817549362b63534e83c'
STOCK_X500_SHA256 = '76d8668c9c1a10348f9b7011d892203f319e5051dc8c0eb06c62e52db8e6714e'


class ContactModelTests(unittest.TestCase):
    def test_contact_sensors_cover_only_existing_collisions(self):
        root = ET.parse(BASE_MODEL).getroot()
        model = root.find('model')
        self.assertEqual(model.get('name'), 'x500_base')
        expected = {
            'base_link': {f'base_link_collision_{i}' for i in range(5)},
            **{f'rotor_{i}': {f'rotor_{i}_collision'} for i in range(4)},
        }
        links = {link.get('name'): link for link in model.findall('link')}
        self.assertEqual(set(links), set(expected))
        for link_name, collisions in expected.items():
            link = links[link_name]
            self.assertEqual({collision.get('name') for collision in link.findall('collision')},
                             collisions)
            sensors = [sensor for sensor in link.findall('sensor')
                       if sensor.get('type') == 'contact']
            self.assertEqual(len(sensors), len(collisions))
            selected = set()
            for sensor in sensors:
                contact = sensor.find('contact')
                collision = contact.findtext('collision')
                selected.add(collision)
                self.assertIn(collision, collisions)
                self.assertEqual(contact.findtext('topic'),
                                 f'/model/course_quad/contact/{link_name}')
                self.assertEqual(sensor.findtext('always_on'), 'true')
                self.assertEqual(sensor.findtext('update_rate'), '50')
            self.assertEqual(selected, collisions)

    def test_removing_sensor_blocks_restores_exact_stock_physics_model(self):
        source = BASE_MODEL.read_text()
        sensor = re.compile(
            r' {0,8}<sensor name="flight_contact_[^"]+" type="contact">\n'
            r'        <always_on>true</always_on>\n'
            r'        <update_rate>50</update_rate>\n'
            r'        <contact>\n'
            r'          <collision>[^<]+</collision>\n'
            r'          <topic>/model/course_quad/contact/[^<]+</topic>\n'
            r'        </contact>\n'
            r'      </sensor>\n')
        restored = sensor.sub('', source)
        self.assertNotEqual(restored, source)
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(), STOCK_BASE_SHA256)

        x500 = X500_MODEL.read_text()
        restored_x500 = x500.replace('model://flight_x500_base', 'model://x500_base')
        self.assertEqual(hashlib.sha256(restored_x500.encode()).hexdigest(), STOCK_X500_SHA256)

    def test_mesh_uris_and_contact_world_system_are_retained(self):
        source = BASE_MODEL.read_text()
        self.assertIn('model://x500_base/meshes/', source)
        self.assertNotIn('model://flight_x500_base/meshes/', source)
        x500 = ET.parse(X500_MODEL).getroot().find('model')
        self.assertEqual(x500.findtext('include/uri'), 'model://flight_x500_base')
        world = (ROOT / 'worlds/course.sdf').read_text()
        self.assertIn('filename="gz-sim-contact-system"', world)


if __name__ == '__main__':
    unittest.main()
