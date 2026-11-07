import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'docs/experiments/net-visual-ablation-01/analyze.py'
SPEC = importlib.util.spec_from_file_location('net_visual_ablation_analyze', SCRIPT)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


class NetVisualAblationAnalysisTests(unittest.TestCase):
    def test_depth_change_gate_counts_positive_infinity_over_full_image(self):
        control = np.ones((32, 2, 100), dtype=np.float32)
        treatment = control.copy()
        treatment[:, 0, :2] = np.inf
        result = analyze.depth_change_summary(control, treatment, .01)
        self.assertTrue(result['passed'])
        self.assertEqual(result['changed_pixels'], 2)
        self.assertEqual(result['total_pixels_denominator'], 200)
        self.assertAlmostEqual(result['changed_pixel_fraction_of_full_image'], .01)
        below = analyze.depth_change_summary(control, treatment[:, :, :], .0101)
        self.assertFalse(below['passed'])

    def test_depth_statistics_separate_finite_invalid_and_infinities(self):
        depth = np.ones((32, 1, 5), dtype=np.float32)
        depth[:, 0, 1] = np.nan
        depth[:, 0, 2] = np.inf
        depth[:, 0, 3] = -np.inf
        depth[:, 0, 4] = 9.0
        result = analyze.depth_change_summary(depth, depth, 0.)
        stats = result['control_raw_depth']
        self.assertEqual(stats['finite_count'], 2 * 32)
        self.assertEqual(stats['nan_count'], 32)
        self.assertEqual(stats['positive_infinity_count'], 32)
        self.assertEqual(stats['negative_infinity_count'], 32)
        self.assertEqual(stats['finite_outside_0_1_to_8m_count'], 32)

    def test_malformed_depth_pair_is_rejected_without_broadcasting(self):
        control = np.ones((32, 256, 384), dtype=np.float32)
        treatment = np.ones((32, 256, 383), dtype=np.float32)
        self.assertFalse(analyze._depth_pair_is_well_formed(
            control, treatment, (32, 256, 384)))
        self.assertFalse(analyze._depth_pair_is_well_formed(
            control[:, :, :0], control[:, :, :0], (32, 256, 384)))

    def test_capture_requires_native_archive_dtypes(self):
        stamps = np.arange(32, dtype=np.int64)
        rgb = np.zeros((32, 256, 384, 3), dtype=np.uint8)
        depth = np.ones((32, 256, 384), dtype=np.float32)
        self.assertTrue(analyze._capture_native_dtypes(stamps, rgb, depth)['passed'])
        self.assertFalse(analyze._capture_native_dtypes(stamps.astype(np.float64), rgb, depth)['passed'])
        self.assertFalse(analyze._capture_native_dtypes(stamps, rgb.astype(np.float32), depth)['passed'])
        self.assertFalse(analyze._capture_native_dtypes(stamps, rgb, depth.astype(np.float64))['passed'])

    def test_pose_mismatch_exceeds_registered_translation_and_orientation_limits(self):
        identity = np.tile([0., 0., 0., 1.], (32, 1))
        control = analyze.pose_summary(np.zeros((32, 3)), identity)
        angle = np.deg2rad(.2)
        rotated = np.tile([0., 0., np.sin(angle/2), np.cos(angle/2)], (32, 1))
        treatment = analyze.pose_summary(np.tile([.003, 0., 0.], (32, 1)), rotated)
        result = analyze.compare_median_poses(control, treatment, .002, .1)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['median_position_difference_m'], .003)
        self.assertAlmostEqual(result['median_orientation_difference_deg'], .2, places=5)

    def test_missing_truth_bracket_fails_capture_alignment(self):
        q = [0., 0., 0., 1.]
        rows = [
            {'sim_time_s': 0., 'base_link_position_enu_m': [0., 0., 0.],
             'base_link_orientation_xyzw': q},
            {'sim_time_s': 1., 'base_link_position_enu_m': [0., 0., 0.],
             'base_link_orientation_xyzw': q},
        ]
        result = analyze.align_capture_truth(rows, np.array([500_000_000]), .12)
        self.assertFalse(result['passed'])
        self.assertFalse(result['all_capture_stamps_bracketed'])
        self.assertFalse(result['full_capture_span_continuous'])
        self.assertIsNone(result['pose'])

    def test_observer_overflow_or_open_callback_cutoff_fails_acceptance(self):
        topics = ['/contact/a', '/contact/b']
        status = {
            'state': 'duration_complete', 'callback_acceptance_closed': True,
            'recording_complete': True, 'evidence_loss': False,
            'truth_subscribed': True, 'truth_samples': 2,
            'truth_messages_received': 2, 'truth_messages_dequeued': 2,
            'truth_messages_ignored': 0, 'truth_queue_overflow': 0,
            'truth_queue_pending': 0, 'contact_messages_queued': 0,
            'contact_queue_overflow': 0, 'contact_messages_dequeued': 0,
            'contact_queue_pending': 0, 'contact_messages_received': 0,
            'contact_messages_with_pairs': 0,
            'contacts_expected_topics': topics,
            'contact_topic_observation': {
                topic: {'transport_messages_received': 0, 'messages_with_contact_pairs': 0}
                for topic in topics
            },
        }
        self.assertTrue(analyze.observer_status_checks(status)['passed'])
        early_stop = dict(status, state='stopped')
        self.assertFalse(analyze.observer_status_checks(early_stop)['passed'])
        lost = dict(status)
        lost.update(recording_complete=False, evidence_loss=True,
                    truth_messages_received=3, truth_queue_overflow=1,
                    truth_queue_overflow_timestamp_range_ns=[10_000_000, 10_000_000])
        result = analyze.observer_status_checks(lost)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['zero_truth_and_contact_queue_overflow'])
        open_cutoff = dict(status, callback_acceptance_closed=False)
        self.assertFalse(analyze.observer_status_checks(open_cutoff)['passed'])

    def test_missing_observer_accounting_fields_fail_closed(self):
        result = analyze.observer_status_checks({'state': 'duration_complete',
                                                 'recording_complete': True})
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['callback_acceptance_closed'])
        self.assertFalse(result['checks']['truth_received_queue_accounting'])

    def test_generation_manifest_requires_protocol_collision_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo'
            artifacts = root / 'artifacts'
            root.mkdir()
            artifacts.mkdir()
            (root / 'worlds').mkdir()
            (root / 'models/course_v3').mkdir(parents=True)
            (root / 'tools').mkdir()
            generator = root / 'tools/make_net_visual_ablation.py'
            generator.write_bytes(b'test generator')
            source_world = root / 'worlds/course.sdf'
            course_model = root / 'models/course_v3/model.sdf'
            source_world.write_bytes(b'world')
            course_model.write_bytes(b'course')
            outputs = {}
            for name, content in (('control.sdf', b'control'), ('treatment.sdf', b'treatment')):
                path = artifacts / name
                path.write_bytes(content)
                outputs[name] = {'path': name, 'sha256': analyze._sha256(path), 'bytes': len(content)}
            ablation = {
                'all_course_collision_semantics_equal': True,
                'all_other_world_semantics_equal_after_flattening': True,
                'canonical_all_course_collisions_sha256': 'all-collisions',
                'canonical_net_collision_sha256': 'net-collisions',
                'net_collision_names_retained': ['net_a'],
                'net_collision_semantics_equal': True,
                'removed_visual_names': ['net_a'],
                'worlds_are_base_inputs_for_existing_vision_rgbd_scene': True,
            }
            generation = {
                'inputs': {
                    'course_model': {'path': 'models/course_v3/model.sdf',
                                     'sha256': analyze._sha256(course_model)},
                    'source_world': {'path': 'worlds/course.sdf',
                                     'sha256': analyze._sha256(source_world)},
                },
                'outputs': outputs,
                'tool': {'path': 'tools/make_net_visual_ablation.py',
                         'sha256': analyze._sha256(generator)},
                'ablation': ablation,
            }
            manifest = {
                'inputs': generation['inputs'], 'outputs': outputs,
                'tool': generation['tool'], 'ablation': ablation,
            }
            (artifacts / 'manifest.json').write_text(json.dumps(manifest))
            protocol = {'generation': generation}
            passed = analyze._validate_generation_manifest(artifacts, protocol, root)
            self.assertTrue(passed['passed'])
            changed = json.loads((artifacts / 'manifest.json').read_text())
            changed['ablation']['net_collision_semantics_equal'] = False
            (artifacts / 'manifest.json').write_text(json.dumps(changed))
            failed = analyze._validate_generation_manifest(artifacts, protocol, root)
            self.assertFalse(failed['passed'])
            self.assertFalse(failed['checks']['collision_and_ablation_semantics_match'])
            changed['inputs']['course_model']['path'] = 'models/other/model.sdf'
            (artifacts / 'manifest.json').write_text(json.dumps(changed))
            failed_path = analyze._validate_generation_manifest(artifacts, protocol, root)
            self.assertFalse(failed_path['checks']['manifest_course_model_path_matches_protocol'])


if __name__ == '__main__':
    unittest.main()
