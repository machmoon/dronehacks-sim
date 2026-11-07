"""Focused artifact-audit tests for the sprint repeatability report."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.report_sprint import REQUIRED_SCORE_CHECKS, audit_attempt, build_report


class SprintReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, name, *, elapsed=52.5, run_id=None, speed_scale=1.0,
                 failed_check=None, recording_overrides=None, summary_overrides=None,
                 score_overrides=None, metadata_overrides=None):
        directory = self.root / name
        directory.mkdir()
        run_id = run_id or name + '-id'
        summary = {
            'type': 'summary', 'run_id': run_id, 'result': 'completed',
            'route_completed': True, 'landed_and_disarmed': True,
            'elapsed_sim_s': elapsed,
        }
        summary.update(summary_overrides or {})
        score = {
            'run_id': run_id,
            'flight_interval_sim_s': [10.0, 10.0 + elapsed],
            'full_route_verified_by_geometry': True,
            'checks': {key: True for key in REQUIRED_SCORE_CHECKS},
        }
        if failed_check:
            score['checks'][failed_check] = False
        score.update(score_overrides or {})
        recording = {
            'mission_exit_code': 0, 'runner_error': None,
            'process_exit_codes': [0, 0, 0], 'truth_used_for_navigation': False,
        }
        recording.update(recording_overrides or {})
        metadata = {
            'type': 'metadata', 'run_id': run_id,
            'waypoints': [
                {'position': [0.0, 0.0, 0.8], 'phase': 'TAKEOFF',
                 'speed': 0.3, 'arrival_radius': 0.06, 'verification_radius': 0.08},
                {'position': [1.0, 0.0, 0.8], 'phase': 'WINDOW_CROSS',
                 'speed': 0.29, 'arrival_radius': 0.06, 'verification_radius': 0.08},
            ],
            'course_limits': {'max_speed': 1.0, 'max_accel': 0.8,
                              'arrival_radius': 0.08, 'position_gain': 1.5},
            'speed_scale': speed_scale,
            'runtime_source_sha256': {'mission/core.py': 'abc123'},
        }
        metadata.update(metadata_overrides or {})
        (directory / 'telemetry.summary.json').write_text(json.dumps(summary))
        (directory / 'score.json').write_text(json.dumps(score))
        (directory / 'recording.json').write_text(json.dumps(recording))
        (directory / 'telemetry.jsonl').write_text(json.dumps(metadata) + '\n')
        return directory

    def test_five_explicit_clean_repeats_achieve_with_retained_evidence(self):
        directories = [self.make_run(f'run-{i}', elapsed=50.0 + i / 10) for i in range(5)]
        report = build_report(directories)
        self.assertEqual(report['total'], 5)
        self.assertEqual(report['passed_runs'], 5)
        self.assertTrue(report['achieved'])
        self.assertTrue(report['same_controller_profile'])
        self.assertAlmostEqual(report['worst_time_sim_s'], 50.4)
        self.assertEqual(len(report['attempts']), 5)
        attempt = report['attempts'][0]
        self.assertEqual(attempt['runtime_source_sha256'], {'mission/core.py': 'abc123'})
        self.assertEqual(set(attempt['input_sha256']), {
            'telemetry.summary.json', 'telemetry.jsonl', 'score.json', 'recording.json'})
        self.assertEqual(len(attempt['profile_sha256']), 64)

    def test_failed_and_slow_attempts_are_not_dropped(self):
        good = self.make_run('good')
        slow = self.make_run('slow', elapsed=60.0)
        failed = self.make_run('bad', failed_check=REQUIRED_SCORE_CHECKS[2])
        report = build_report([good, slow, failed])
        self.assertEqual(report['total'], 3)
        self.assertEqual(report['passed_runs'], 1)
        self.assertEqual(report['times_sim_s'], [52.5, 60.0, 52.5])
        self.assertEqual(report['worst_time_sim_s'], 60.0)
        self.assertFalse(report['achieved'])
        self.assertIn('elapsed_time_not_under_60_seconds', report['attempts'][1]['failures'])
        self.assertIn('required_score_checks_failed_or_missing', report['attempts'][2]['failures'])

    def test_mismatched_ids_and_unproven_recorder_or_truth_fail(self):
        directory = self.make_run(
            'mismatch', score_overrides={'run_id': 'other'},
            recording_overrides={'process_exit_codes': [0, 1], 'truth_used_for_navigation': None})
        attempt = audit_attempt(directory)
        self.assertFalse(attempt['passed'])
        self.assertIn('score_summary_run_id_mismatch', attempt['failures'])
        self.assertIn('recorder_did_not_exit_cleanly', attempt['failures'])
        self.assertIn('truth_navigation_not_explicitly_false', attempt['failures'])

    def test_profile_mismatch_prevents_achievement_even_if_attempts_pass(self):
        directories = [self.make_run(f'profile-{i}', speed_scale=1.1 if i == 4 else 1.0)
                       for i in range(5)]
        report = build_report(directories)
        self.assertEqual(report['passed_runs'], 5)
        self.assertFalse(report['same_controller_profile'])
        self.assertFalse(report['achieved'])

    def test_duplicate_directories_or_run_ids_do_not_count_as_repeats(self):
        directory = self.make_run('same-run')
        duplicated_directory = build_report([directory] * 5)
        self.assertEqual(duplicated_directory['total'], 5)
        self.assertEqual(duplicated_directory['passed_runs'], 1)
        self.assertFalse(duplicated_directory['achieved'])
        self.assertIn('duplicate_run_directory', duplicated_directory['attempts'][1]['failures'])
        self.assertIn('duplicate_run_id', duplicated_directory['attempts'][1]['failures'])

        same_id = [self.make_run(f'different-dir-{i}', run_id='one-run-id') for i in range(5)]
        duplicated_id = build_report(same_id)
        self.assertEqual(duplicated_id['passed_runs'], 1)
        self.assertFalse(duplicated_id['achieved'])
        self.assertIn('duplicate_run_id', duplicated_id['attempts'][1]['failures'])

    def test_elapsed_must_match_score_interval_and_source_hashes_must_match(self):
        inconsistent = self.make_run('time-mismatch', score_overrides={
            'flight_interval_sim_s': [1.0, 54.0],
        })
        attempt = audit_attempt(inconsistent)
        self.assertIn('score_summary_elapsed_mismatch', attempt['failures'])

        dirs = []
        for i in range(5):
            hashes = {'mission/core.py': 'changed' if i == 4 else 'same'}
            dirs.append(self.make_run(f'source-{i}', metadata_overrides={
                'runtime_source_sha256': hashes,
            }))
        report = build_report(dirs)
        self.assertEqual(report['passed_runs'], 4)
        self.assertFalse(report['same_runtime_source'])
        self.assertFalse(report['achieved'])
        self.assertIn('runtime_source_hashes_mismatch', report['attempts'][4]['failures'])

    def test_missing_source_hashes_and_malformed_json_roots_are_failures_not_crashes(self):
        missing_hashes = self.make_run('no-hashes', metadata_overrides={
            'runtime_source_sha256': {},
        })
        self.assertIn('runtime_source_hashes_missing_or_invalid',
                      audit_attempt(missing_hashes)['failures'])
        malformed = self.make_run('bad-root')
        (malformed / 'score.json').write_text('[]')
        (malformed / 'recording.json').write_text('null')
        report = build_report([malformed])
        self.assertEqual(report['total'], 1)
        self.assertFalse(report['attempts'][0]['passed'])
        self.assertIn('invalid_score', report['attempts'][0]['failures'])
        self.assertIn('invalid_recording', report['attempts'][0]['failures'])

    def test_missing_or_malformed_directory_is_retained_as_failure(self):
        malformed = self.root / 'malformed'
        malformed.mkdir()
        report = build_report([malformed, self.root / 'absent'])
        self.assertEqual(report['total'], 2)
        self.assertEqual(report['passed_runs'], 0)
        self.assertFalse(report['achieved'])
        self.assertTrue(report['attempts'][0]['failures'])
        self.assertIn('missing_telemetry.summary.json', report['attempts'][1]['failures'])


if __name__ == '__main__':
    unittest.main()
