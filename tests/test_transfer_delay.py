import hashlib
from pathlib import Path
import unittest

import numpy as np

from learning import transfer, transfer_delay


class DelayTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = transfer_delay.DelayCourse()

    def test_v2_contract_preserves_base_profile_hash_and_records_delay(self):
        profile = self.course.profile_metadata()
        base = transfer.TransferCourse().profile_metadata()
        self.assertEqual(profile['name'], 'x500-transfer-v2')
        self.assertEqual(profile['transfer_source_sha256'], base['transfer_source_sha256'])
        self.assertEqual(profile['transfer_delay_source_sha256'], hashlib.sha256(
            Path(transfer_delay.__file__).read_bytes()).hexdigest())
        self.assertEqual(profile['command_delay_s'], [.12, .24])
        self.assertEqual(profile['command_delay_quantum_s'], .04)
        self.assertEqual(profile['command_delay_location'], 'post_slew')
        self.assertEqual(profile['velocity_gain_xy'], [1., 1.08])

    def test_no_delay_unit_gain_matches_v1_step_and_teacher(self):
        old = transfer.TransferCourse().make_flight(42, 0., response=.66)
        new = self.course.make_flight(42, 0., response=.66, delay_s=0., gain_xy=1.)
        for _ in range(4):
            np.testing.assert_array_equal(old.observe(render=False).state, new.observe(render=False).state)
            action = old.course.instructor(old)
            np.testing.assert_array_equal(action, new.course.instructor(new))
            old.step(action)
            new.step(action)
        np.testing.assert_array_equal(old.path, new.path)
        np.testing.assert_array_equal(old.orientations, new.orientations)

    def test_gain_scales_delayed_horizontal_command_after_slew_only(self):
        flight = self.course.make_flight(42, 0., response=.50, delay_s=.12, gain_xy=1.08)
        flight.observe(render=False)
        target = np.array([.25, .0, .25])
        flight.step(flight.rotation.T@target)
        first_issued = target/np.linalg.norm(target)*(.8*.04)
        self.assertAlmostEqual(np.linalg.norm(flight.command), 4*.8*.04)
        np.testing.assert_allclose(flight.v, first_issued*np.array([1.08, 1.08, 1.])*.04/.50)

    def test_candidate_prediction_matches_next_real_step_and_keeps_pending_fifo(self):
        flight = self.course.make_flight(42, 0., response=.66, delay_s=.24, gain_xy=1.04)
        flight.observe(render=False)
        flight.step(np.array([.30, .0, .0]))
        # Crossing a waypoint keeps commands already issued to the plant.
        flight.index = 1
        flight.observe(render=False)
        queued = np.asarray(flight.issued).copy()
        candidates = np.array([[0., -.2, .0], [.2, .0, .0]])
        probes = transfer_delay.preview_commands(flight, candidates, self.course.speeds[flight.index])
        np.testing.assert_array_equal(flight.issued, queued)
        action = self.course.instructor(flight)
        self.assertLessEqual(np.linalg.norm(action), self.course.speeds[flight.index]+1e-12)
        np.testing.assert_array_equal(flight.issued, queued)
        flight.step(flight.rotation.T@candidates[0])
        np.testing.assert_allclose(flight.p, probes[0].position[0])
        np.testing.assert_allclose(flight.v, probes[0].velocity[0])
        np.testing.assert_allclose(flight.command, probes[0].command[0], atol=1e-14)

    def test_delay_and_gain_draws_are_repeatable_and_separate_from_base_response(self):
        first = self.course.make_flight(42, 0.)
        second = self.course.make_flight(42, 0.)
        base = transfer.TransferCourse().make_flight(42, 0.)
        self.assertEqual(first.command_delay_s, second.command_delay_s)
        self.assertEqual(first.gain_xy, second.gain_xy)
        self.assertEqual(first.response, base.response)
        self.assertIn(first.command_delay_s, (.12, .16, .20, .24))
        with self.assertRaises(ValueError):
            self.course.make_flight(42, 0., delay_s=.15)


if __name__ == '__main__':
    unittest.main()
