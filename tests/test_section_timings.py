import json
import tempfile
import unittest
from pathlib import Path

from tools.section_timings import analyze_run, main, render_table


def _archive(tmp_path, rows, score=None):
    (tmp_path / 'telemetry.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    if score is not None:
        (tmp_path / 'score.json').write_text(json.dumps(score))
    return tmp_path


class SectionTimingTests(unittest.TestCase):
  def setUp(self):
    self._temp = tempfile.TemporaryDirectory()
    self.tmp_path = Path(self._temp.name)

  def tearDown(self):
    self._temp.cleanup()

  def test_completed_route_uses_sim_time_through_landing(self):
    rows = [
        {'type': 'metadata', 'task': 'course', 'waypoint_count': 2},
        {'type': 'event', 'message': 'PX4 armed in Offboard', 'sim_time': 100.0},
        {'type': 'event', 'message': 'Sustained takeoff hover verified', 'sim_time': 110.0},
        *[{'type': 'sample', 'sim_time': t, 'phase': 'CORRIDOR', 'waypoint_index': 1}
          for t in (110, 120)],
        *[{'type': 'sample', 'sim_time': t, 'phase': 'WINDOW', 'waypoint_index': 2}
          for t in (130, 140)],
        {'type': 'event', 'message': 'Route complete; requesting PX4 AUTO_LAND',
         'route_elapsed_sim_s': 50.0},
        {'type': 'summary', 'task': 'course', 'result': 'completed', 'elapsed_sim_s': 60.0,
         'route_elapsed_sim_s': 50.0, 'route_completed': True, 'landed_and_disarmed': True,
         'waypoints_reached': 2, 'waypoint_count': 2, 'policy_actions_received': 0},
    ]
    result = analyze_run(_archive(self.tmp_path, rows), max_sample_gap_s=10)
    self.assertEqual(result['arm_to_hover_s'], 10)
    self.assertEqual(result['route_duration_s'], 40)
    self.assertEqual(result['arm_to_route_complete_s'], 50)
    self.assertEqual(result['route_complete_to_landing_s'], 10)
    self.assertEqual(result['arm_to_flight_end_s'], 60)
    self.assertEqual([s['phase'] for s in result['route_segments']], ['CORRIDOR', 'WINDOW'])
    self.assertEqual(sum(s['duration_s'] for s in result['route_segments']), 40)


  def test_failed_route_is_censored_at_fault_and_abort_landing_excluded(self):
    rows = [
        {'type': 'metadata', 'task': 'policy', 'waypoint_count': 15},
        {'type': 'event', 'message': 'PX4 armed in Offboard', 'sim_time': 20.0},
        {'type': 'event', 'message': 'Sustained takeoff hover verified', 'sim_time': 30.0},
        *[{'type': 'sample', 'sim_time': t, 'phase': 'MAZE', 'waypoint_index': 5,
           'policy_actions_received': 12} for t in (30, 32, 34, 36, 38)],
        {'type': 'event', 'message': 'Mission stopped: course_boundary_breach'},
        {'type': 'event', 'message': 'Observing PX4 abort landing', 'sim_time': 44.0,
         'phase': 'ABORT_LAND'},
        {'type': 'event', 'message': 'PX4 confirmed touchdown and disarm after abort', 'sim_time': 46.568},
        {'type': 'summary', 'run_id': 'matching-run', 'task': 'policy', 'result': 'failed', 'fault': 'course_boundary_breach',
         'elapsed_sim_s': 20.0, 'route_elapsed_sim_s': None, 'route_completed': False,
         'landed_and_disarmed': False, 'waypoints_reached': 5, 'waypoint_count': 15,
         'policy_actions_received': 12},
    ]
    score = {'run_id': 'matching-run', 'route': {'reached': 5, 'required': 15}}
    result = analyze_run(_archive(self.tmp_path, rows, score), max_sample_gap_s=3)
    self.assertEqual(result['flight_status'], 'failed_after_arm')
    self.assertEqual(result['arm_to_flight_end_s'], 20)
    self.assertEqual(result['flight_end_sim_time_s'], 40)
    self.assertEqual(result['route_end_semantics'], 'censored_at_fault')
    self.assertEqual(result['route_duration_s'], 10)
    self.assertIs(result['abort_landing_confirmed_by_event'], True)
    self.assertEqual(result['abort_landing_disarm_sim_time_s'], 46.568)
    self.assertAlmostEqual(result['abort_landing_elapsed_s'], 6.568)
    self.assertEqual(result['policy_actions_received'], 12)
    self.assertEqual(result['truth_waypoints_reached'], 5)
    self.assertTrue(all(segment['phase'] != 'ABORT_LAND' for segment in result['route_segments']))
    self.assertIn('abort landing is excluded', render_table([result]).lower())


  def test_long_final_sample_gap_is_emitted_once_as_unobserved(self):
    rows = [
        {'type': 'metadata', 'task': 'course'},
        {'type': 'event', 'message': 'PX4 armed in Offboard', 'sim_time': 0.0},
        {'type': 'event', 'message': 'Sustained takeoff hover verified', 'sim_time': 10.0},
        {'type': 'sample', 'sim_time': 10.0, 'phase': 'ROUTE', 'waypoint_index': 1},
        {'type': 'sample', 'sim_time': 11.0, 'phase': 'ROUTE', 'waypoint_index': 1},
        {'type': 'summary', 'task': 'course', 'result': 'failed', 'fault': 'test_fault',
         'elapsed_sim_s': 5.0, 'route_completed': False, 'policy_actions_received': 0},
    ]
    # Fault is at t=5, before this synthetic route interval can be valid.
    # Instead, set the arm epoch so the fault lands at 13 and leaves a long tail gap.
    rows[1]['sim_time'] = 8.0
    rows[-1]['elapsed_sim_s'] = 5.0
    result = analyze_run(_archive(self.tmp_path, rows), max_sample_gap_s=1.5)
    self.assertEqual(result['route_duration_s'], 3)
    self.assertAlmostEqual(sum(s['duration_s'] for s in result['route_segments']), 3)
    self.assertEqual(result['route_segments'][-1]['phase'], 'UNOBSERVED')
    self.assertAlmostEqual(result['route_unobserved_s'], 2)


  def test_stale_score_run_id_is_rejected(self):
    rows = [
        {'type': 'metadata', 'task': 'course', 'run_id': 'actual-run'},
        {'type': 'summary', 'run_id': 'actual-run', 'task': 'course', 'result': 'failed',
         'elapsed_sim_s': 0.0, 'route_completed': False},
    ]
    with self.assertRaisesRegex(ValueError, 'does not match telemetry run_id'):
        analyze_run(_archive(self.tmp_path, rows, {'run_id': 'old-run', 'route': {'reached': 3}}))


  def test_nonfinite_sample_gap_is_rejected(self):
    with self.assertRaises(SystemExit):
        main(['artifacts/markerfree/course-08', '--max-sample-gap-s', 'nan'])


  def test_prearm_failure_has_no_flight_or_route_duration(self):
    rows = [
        {'type': 'metadata', 'task': 'policy', 'waypoint_count': 15},
        {'type': 'event', 'message': 'Mission stopped: invalid_policy_exchange_record'},
        {'type': 'summary', 'task': 'policy', 'result': 'failed',
         'fault': 'invalid_policy_exchange_record:response.json', 'elapsed_sim_s': 0.0,
         'route_completed': False, 'policy_actions_received': 0, 'waypoints_reached': 0,
         'waypoint_count': 15},
    ]
    result = analyze_run(_archive(self.tmp_path, rows))
    self.assertEqual(result['flight_status'], 'prearm_failed')
    self.assertIs(result['armed'], False)
    self.assertIsNone(result['mission_flight_duration_s'])
    self.assertIsNone(result['route_duration_s'])
    self.assertEqual(result['route_segments'], [])
    self.assertIs(result['prearm_elapsed_is_flight_time'], False)


if __name__ == '__main__':
    unittest.main()
