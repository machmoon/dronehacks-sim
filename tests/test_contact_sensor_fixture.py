import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tools.check_contact_sensors import (
    BASE_MODEL,
    COURSE_WORLD,
    FLIGHT_MODEL,
    NEGATIVE_OFFSET_XY_M,
    INITIAL_CLEARANCE_M,
    PROBE_SIZE,
    TARGET_MODEL,
    generate,
    matching_expected_pairs,
    score_fixture,
    extract_contact_targets,
    validate_probe_geometry,
    _select_exposed_probe,
    _colliders_at_pose,
    _vertical_interval,
    _probe_samples,
    _support_face_points,
)


class ContactSensorFixtureTests(unittest.TestCase):
    def test_generator_emits_all_nine_source_mappings_and_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'fixtures'
            manifest = generate(output)
            generated_files = {path.name for path in output.glob('*.sdf')}
            generated_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in output.glob('*.sdf')}

        fixtures = manifest['fixtures']
        positive = [fixture for fixture in fixtures if fixture['kind'] == 'positive']
        negative = [fixture for fixture in fixtures if fixture['kind'] != 'positive']
        self.assertEqual(len(positive), 9)
        self.assertEqual(len(negative), 1)
        self.assertEqual(
            [(fixture['link_name'], fixture['collision_name'], fixture['topic'])
             for fixture in positive],
            [('base_link', f'base_link_collision_{i}', '/model/course_quad/contact/base_link')
             for i in range(5)] +
            [(f'rotor_{i}', f'rotor_{i}_collision', f'/model/course_quad/contact/rotor_{i}')
             for i in range(4)])

        for fixture in positive:
            self.assertEqual(fixture['collider_scoped_name'].split('::')[0], TARGET_MODEL)
            self.assertEqual(fixture['probe_scoped_name'].split('::')[0], fixture['probe_model_name'])
            self.assertAlmostEqual(fixture['initial_clearance_m'], INITIAL_CLEARANCE_M)
            self.assertAlmostEqual(
                fixture['target_surface_min_z_at_probe_m'] - fixture['probe_top_z_world_m'],
                INITIAL_CLEARANCE_M, places=6)
            self.assertTrue(fixture['geometry_validation']['valid'])
            self.assertIn(fixture['world_file'], generated_files)
            self.assertEqual(fixture['world_sha256'], generated_hashes[fixture['world_file']])

        displaced = negative[0]
        source = positive[0]
        self.assertEqual(displaced['fixture_id'], source['fixture_id'] + '__negative_displaced')
        self.assertEqual(displaced['source_fixture_id'], source['fixture_id'])
        self.assertEqual(displaced['collider_scoped_name'], source['collider_scoped_name'])
        self.assertEqual(displaced['topic'], source['topic'])
        self.assertEqual(displaced['pedestal_displacement_xy_m'], list(NEGATIVE_OFFSET_XY_M))
        self.assertNotEqual(displaced['probe_pose_world_m'][:2], source['probe_pose_world_m'][:2])
        reference = displaced['reference_pedestal']
        self.assertEqual(reference['displacement_xy_m'], [0., 0.])
        self.assertEqual(reference['pose_world_m'][:2], source['probe_pose_world_m'][:2])
        self.assertNotEqual(reference['scoped_name'], displaced['displaced_pedestal']['scoped_name'])

        for path in (FLIGHT_MODEL, BASE_MODEL, COURSE_WORLD):
            key = str(path.relative_to(FLIGHT_MODEL.parents[2]))
            self.assertEqual(manifest['source_sha256'][key], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_world_preserves_sensor_include_and_has_no_controller_or_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'fixtures'
            manifest = generate(output)
            fixture = manifest['fixtures'][0]
            root = ET.parse(output / fixture['world_file']).getroot()
            contents = (output / fixture['world_file']).read_text()

        world = root.find('world')
        self.assertEqual(world.findtext('gravity'), '0 0 -9.8')
        self.assertEqual(world.findtext('physics/max_step_size'), '0.004')
        plugins = world.findall('plugin')
        self.assertIn('gz-sim-contact-system', [plugin.get('filename') for plugin in plugins])
        self.assertIn('gz-sim-user-commands-system', [plugin.get('filename') for plugin in plugins])
        physics = next(plugin for plugin in plugins if plugin.get('filename') == 'gz-sim-physics-system')
        self.assertEqual(physics.findtext('engine/filename'), 'gz-physics-dartsim-plugin')
        include = world.find('include')
        self.assertEqual(include.findtext('uri'), 'model://flight_x500')
        self.assertEqual(include.findtext('name'), 'course_quad')
        self.assertIsNone(world.find("model[@name='contact_fixture_ground']"))
        for fixture in manifest['fixtures']:
            if fixture['kind'] != 'positive' or not fixture['link_name'].startswith('rotor_'):
                continue
            x, y, _ = fixture['probe_pose_world_m']
            self.assertGreater(abs(x), .20)
            self.assertGreater(abs(y), .15)
        self.assertNotIn('px4', contents.lower())
        self.assertNotIn('command/motor_speed', contents)
        self.assertEqual(manifest['px4_plugins_or_models'], [])
        self.assertFalse(manifest['motor_commands_sent'])

    def test_occluded_upright_leg_and_initial_overlap_are_rejected(self):
        targets = extract_contact_targets()
        leg = next(item for item in targets if item['collision_name'] == 'base_link_collision_1')
        x, y, _ = next(point for point in _support_face_points(leg, (0., 0., 0.))
                       if abs(point[0]) < 1e-8 and point[1] < -.13)
        colliders = _colliders_at_pose(targets, (0., 0., 0.))
        leg_box = next(box for box in colliders if box['collision_name'] == leg['collision_name'])
        surface_z = min(_vertical_interval(leg_box, sx, sy)[0]
                        for sx, sy in _probe_samples(x, y))
        bad = dict(leg)
        bad.update(model_pose_rpy_rad=[0., 0., 0.], probe_contact_xy_world_m=[x, y],
                   target_surface_min_z_at_probe_m=surface_z,
                   probe_pose_world_m=[x, y, surface_z-INITIAL_CLEARANCE_M-PROBE_SIZE[2]/2])
        report = validate_probe_geometry(targets, bad)
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'probe_occluded_by_source_collision')

        exposed = _select_exposed_probe(leg, targets)
        overlapping = dict(exposed)
        overlapping['probe_pose_world_m'] = list(exposed['probe_pose_world_m'])
        overlapping['probe_pose_world_m'][2] += INITIAL_CLEARANCE_M
        self.assertFalse(validate_probe_geometry(targets, overlapping)['valid'])

    def test_scorer_requires_expected_pair_and_exact_sensor_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(Path(tmp) / 'fixtures')
        fixture = manifest['fixtures'][0]
        collider, probe = fixture['collider_scoped_name'], fixture['probe_scoped_name']
        records = [
            {'topic': '/wrong/topic', 'pairs': [[collider, probe]]},
            {'topic': fixture['topic'], 'pairs': [['course_quad::base_link::other', probe]]},
            {'topic': fixture['topic'], 'pairs': [[f'world::{collider}', probe]]},
        ]
        self.assertEqual(matching_expected_pairs(fixture, records), 1)
        scored = score_fixture(fixture, records)
        self.assertEqual(scored['outcome'], 'positive_pair_observed')
        self.assertEqual(scored['expected_pair_observations'], 1)

    def test_negative_control_requires_exact_reference_pair_and_absent_displaced_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(Path(tmp) / 'fixtures')
        fixture = next(item for item in manifest['fixtures'] if item['kind'] != 'positive')
        collider = fixture['collider_scoped_name']
        displaced = fixture['displaced_pedestal']['scoped_name']
        reference = fixture['reference_pedestal']['scoped_name']
        self.assertEqual(score_fixture(fixture, [])['outcome'],
                         'negative_control_inconclusive_reference_pair_not_observed')
        ref_row = {'topic': fixture['topic'], 'pairs': [[collider, reference]]}
        result = score_fixture(fixture, [ref_row])
        self.assertEqual(result['outcome'], 'negative_control_passed_reference_only')
        both = {'topic': fixture['topic'], 'pairs': [[collider, reference], [collider, displaced]]}
        self.assertEqual(score_fixture(fixture, [both])['outcome'],
                         'negative_control_failed_displaced_pair_observed')
        self.assertEqual(score_fixture(fixture, [ref_row], {'contact_topic_observation': {fixture['topic']:
                         {'transport_messages_received': 0}}})['outcome'],
                         'negative_control_passed_reference_only')
        wrong = [{'topic': '/wrong/topic', 'pairs': [[fixture['collider_scoped_name'],
                                                       reference]]}]
        self.assertEqual(score_fixture(fixture, wrong)['reference_pair_observations'], 0)


if __name__ == '__main__':
    unittest.main()
