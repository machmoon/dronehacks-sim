import unittest

from tools.render_flight_video import _events, _mission_label, _phase_at


class FlightVideoTimelineTests(unittest.TestCase):
    def test_phase_uses_nearest_prior_event_even_when_rows_unsorted(self):
        rows = [
            {'type': 'sample', 'sim_time': 2.0, 'phase': 'MAZE'},
            {'type': 'event', 'sim_time': 1.0, 'message': 'armed'},
            {'type': 'event', 'sim_time': 2.5, 'message': 'Observing PX4 abort landing'},
        ]
        self.assertEqual(_phase_at(rows, 1.5), 'mission')
        self.assertEqual(_phase_at(rows, 2.4), 'MAZE')
        self.assertEqual(_phase_at(rows, 2.5), 'ABORT_LAND')

    def test_event_timeline_inserts_gap_without_reusing_previous_frame(self):
        frames = [
            {'capture_sim_s': 1.0, 'rgb': 'a.jpg'},
            {'capture_sim_s': 2.0, 'rgb': 'b.jpg'},
        ]
        events = _events(frames, [], .4)
        self.assertEqual([(e['t'], e['gap']) for e in events],
                         [(1.0, False), (1.4, True), (2.0, False)])
        self.assertNotIn('frame', events[1])
        self.assertAlmostEqual(sum(e['duration'] for e in events), 1.4)

    def test_labels_distinguish_baseline_policy_and_hover(self):
        self.assertEqual(_mission_label([
            {'type': 'metadata', 'task': 'course'},
            {'type': 'summary', 'task': 'course', 'result': 'failed'},
        ]), ('TASK BASELINE', 'FAILED'))
        self.assertEqual(_mission_label([
            {'type': 'metadata', 'task': 'policy'},
            {'type': 'summary', 'task': 'policy', 'result': 'failed',
             'policy_checkpoint_sha256': 'abc', 'policy_actions_received': 0},
        ]), ('POLICY NOT STARTED', 'FAILED'))
        self.assertEqual(_mission_label([
            {'type': 'metadata', 'task': 'hover'},
            {'type': 'summary', 'task': 'hover', 'result': 'completed'},
        ]), ('HOVER CHECK', 'COMPLETED'))


if __name__ == '__main__':
    unittest.main()
