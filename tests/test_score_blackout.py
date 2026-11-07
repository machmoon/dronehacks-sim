"""The blackout predicate requires controller, PX4, and independent observer evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import score_blackout as blackout
from tools import score_markerfree as route_score


def event(message, **fields):
    return {'type': 'event', 'message': message, **fields}


def topic(name, times, **fields):
    return SimpleNamespace(name=name, multi_id=0, data={
        'timestamp': np.array([round(time*1e6) for time in times]),
        **{key: np.array(values) for key, values in fields.items()},
    })


class BlackoutScoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name)
        (self.run/'observer').mkdir()
        self.ulog_path = self.run/'px4.ulg'
        self.ulog_path.write_bytes(b'synthetic PX4 log')
        self.telemetry = [
            {'type': 'metadata', 'run_id': 'recording'},
            event(route_score.ARMED_EVENT, sim_time=1.),
            event(route_score.HOVER_EVENT, sim_time=2.),
            {'type': 'sample', 'sim_time': 2.95, 'vision_stamp_sim_s': 2.92, 'vision_valid': True,
             'vision_measurements_sent': 40},
            event(blackout.INJECTION, sim_time=3.),
            {'type': 'sample', 'sim_time': 3.2, 'vision_stamp_sim_s': 2.92,
             'vision_measurements_sent': 40},
            event(f'Mission stopped: {blackout.FAULT}'),
            event(blackout.LAND_REQUEST),
            event(blackout.LAND_ACK, command=21, result=0),
            event(blackout.CONFIRMED, sim_time=6.),
            {'type': 'summary', 'run_id': 'recording', 'result': 'failed', 'fault': blackout.FAULT,
             'elapsed_sim_s': 2.5, 'landed_after_fault': True, 'vision_measurements_sent': 40},
        ]
        stamps = np.arange(20, 121)/20
        heights = np.interp(stamps, [1., 1.6, 4., 4.7, 6.], [.227, .9, .9, .227, .227])
        self.truth = [{'sim_time_s': t, 'base_link_position_enu_m': [-2.52, 1., z],
                       'base_link_orientation_xyzw': [0., 0., 0., 1.]} for t, z in zip(stamps, heights)]
        self.contacts = [{'sim_time_s': 4.7, 'topic': '/model/course_quad/contact/base_link',
                          'pairs': [['course_quad::base_link::collision', 'ground_plane::ground::floor']]}]
        self.visual = topic('vehicle_visual_odometry', [2.94], timestamp_sample=[2920000])
        self.vehicle = topic('vehicle_status', [2.8, 3.54, 5.9, 6.1],
                             nav_state=[14, 18, 14, 14], arming_state=[2, 2, 1, 1], failsafe=[0, 0, 0, 0])
        self.land = topic('vehicle_land_detected', [2.8, 5.8], landed=[0, 1])
        self.log = SimpleNamespace(start_timestamp=1000000, last_timestamp=6100000,
                                   file_corruption=False, msg_info_dict={}, dropouts=[])

    def score(self, expected_fault=blackout.FAULT):
        for name, rows in [('telemetry.jsonl', self.telemetry), ('observer/truth.jsonl', self.truth),
                           ('observer/contacts.jsonl', self.contacts)]:
            (self.run/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.log.data_list = [self.visual, self.vehicle, self.land]
        with patch.object(blackout, '_load_ulog', return_value=self.log):
            return blackout.score_blackout(self.run, self.ulog_path, expected_fault=expected_fault)

    def test_complete_fault_response_passes_separately_from_mission(self):
        report = self.score()
        self.assertTrue(report['fault_test_passed'], report['checks'])
        self.assertEqual(len(report['checks']), 8)
        self.assertEqual(report['mission_result'], 'failed')
        self.assertEqual(report['timeline_sim_s']['px4_disarmed'], 5.9)
        self.assertEqual(report['vision_aiding']['px4_ulog']['logged_visual_odometry_records_after_injection'], 0)

    def test_restored_vision_in_abort_fails_even_with_frozen_controller_counter(self):
        self.visual = topic('vehicle_visual_odometry', [2.94, 4.5], timestamp_sample=[2920000, 4490000])
        report = self.score()
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['camera_aid_stopped_through_abort'])
        self.assertEqual(report['vision_aiding']['px4_ulog']['logged_visual_odometry_records_after_injection'], 1)

    def test_missing_disarm_cannot_pass(self):
        self.vehicle.data['arming_state'][:] = 2
        report = self.score()
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['px4_ground_and_disarm_confirmed_after_fault'])
        self.telemetry = [row for row in self.telemetry if row.get('message') != blackout.CONFIRMED]
        report = self.score()
        self.assertFalse(report['fault_test_passed'])
        self.assertIsNone(report['timeline_sim_s']['controller_confirmed_ground_and_disarm'])

    def test_abort_truth_gap_cannot_certify_geometry(self):
        self.truth = [row for row in self.truth if not 3.8 <= row['sim_time_s'] <= 4.2]
        report = self.score()
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['pre_fault_and_abort_multipart_geometry_certified_clear'])
        self.assertFalse(report['cad_proxy']['phases']['abort_landing']['truth_complete'])

    def test_transit_floor_collision_cannot_pass(self):
        next(row for row in self.truth if row['sim_time_s'] == 2.5)['base_link_position_enu_m'][2] = .227
        report = self.score()
        self.assertFalse(report['fault_test_passed'])
        self.assertEqual(report['cad_proxy']['phases']['transit']['first_unresolved']['obstacle'], 'floor')

    def test_overlapping_dropout_or_short_ulog_cannot_verify_aid_loss(self):
        self.log.dropouts = [SimpleNamespace(timestamp=2975000, duration=75)]
        report = self.score()
        self.assertFalse(report['fault_test_passed'])
        self.assertEqual(len(report['vision_aiding']['px4_ulog']['dropout_records_overlapping_blackout']), 1)
        self.log.dropouts = []
        self.log.last_timestamp = 5950000
        self.assertFalse(self.score()['checks']['camera_aid_stopped_through_abort'])

    def prepare_policy_fault(self, actual_fault=blackout.POLICY_FAULT, include_action=True, task='policy',
                             action_time=2.8):
        self.telemetry[0]['task'] = task
        if include_action:
            self.telemetry.insert(3, {'type': 'policy_action', 'sim_time': action_time, 'capture_usec': 2760000,
                                      'capture_age_sim_s': .04, 'latency_wall_s': .015})
        else:
            self.telemetry = [row for row in self.telemetry if row.get('type') != 'policy_action']
        fault_event = next(row for row in self.telemetry
                           if row.get('message', '').startswith('Mission stopped:'))
        fault_event['message'] = f'Mission stopped: {actual_fault}'
        next(row for row in self.telemetry if row.get('type') == 'summary')['fault'] = actual_fault

    def test_expected_policy_fault_passes_with_preinjection_policy_action(self):
        self.prepare_policy_fault()
        report = self.score(expected_fault=blackout.POLICY_FAULT)
        self.assertTrue(report['fault_test_passed'], report['checks'])
        self.assertEqual(report['expected_fault'], blackout.POLICY_FAULT)
        self.assertTrue(report['checks']['expected_fault_observed'])
        self.assertTrue(report['checks']['policy_task_and_pre_injection_action_present'])
        self.assertEqual(report['timeline_sim_s']['expected_fault'], 3.5)
        self.assertNotIn('stale_capture_fault', report['timeline_sim_s'])

    def test_wrong_policy_fault_does_not_pass_by_observed_fault_inference(self):
        self.prepare_policy_fault(actual_fault=blackout.FAULT)
        report = self.score(expected_fault=blackout.POLICY_FAULT)
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['expected_fault_observed'])
        self.assertFalse(report['checks']['injection_fault_land_ack_disarm_events_ordered'])

    def test_policy_fault_requires_policy_task_and_prior_action(self):
        self.prepare_policy_fault(task='hover')
        report = self.score(expected_fault=blackout.POLICY_FAULT)
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['policy_task_and_pre_injection_action_present'])

    def test_policy_action_before_arming_does_not_satisfy_active_policy_evidence(self):
        self.prepare_policy_fault(action_time=.8)
        report = self.score(expected_fault=blackout.POLICY_FAULT)
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['policy_task_and_pre_injection_action_present'])

        self.prepare_policy_fault(include_action=False)
        report = self.score(expected_fault=blackout.POLICY_FAULT)
        self.assertFalse(report['fault_test_passed'])
        self.assertFalse(report['checks']['policy_task_and_pre_injection_action_present'])

    def test_expected_fault_rejects_unlisted_values(self):
        with self.assertRaises(ValueError):
            blackout.score_blackout(self.run, self.ulog_path, expected_fault='some_other_failure')


if __name__ == '__main__':
    unittest.main()
