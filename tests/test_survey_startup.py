import unittest

from mission.survey_startup import SurveyStartup


def frame(t, *, valid=True, source="depth_survey"):
    return {"valid": valid, "source": source, "sim_usec": int(t * 1_000_000)}


class SurveyStartupTests(unittest.TestCase):
    def test_relative_or_unknown_backend_does_not_qualify(self):
        gate = SurveyStartup()
        self.assertFalse(gate.update(frame(0, source="relative")).get("ready"))
        status = gate.update(frame(.05, source="backend_unknown"))
        self.assertEqual(status["accepted_count"], 0)
        self.assertEqual(status["reason"], "unqualified_source")
        self.assertFalse(gate.ready(.05))

    def test_sustained_valid_stream_qualifies_after_three_seconds(self):
        gate = SurveyStartup()
        for i in range(61):
            gate.update(frame(i * .05, source="depth_survey_tof" if i % 2 else "depth_survey"))
        status = gate.status(3.0)
        self.assertTrue(status["ready"])
        self.assertEqual(status["reason"], "ready")
        self.assertEqual(status["accepted_count"], 61)
        self.assertAlmostEqual(status["span_s"], 3.0)
        self.assertAlmostEqual(status["last_capture_sim_s"], 3.0)

    def test_invalid_interleaved_frame_resets_span(self):
        gate = SurveyStartup()
        for i in range(30):
            gate.update(frame(i * .05))
        status = gate.update(frame(1.5, valid=False))
        self.assertEqual(status["accepted_count"], 0)
        self.assertEqual(status["span_s"], 0.0)
        gate.update(frame(1.55))
        self.assertFalse(gate.ready(1.55))
        self.assertEqual(gate.status()["accepted_count"], 1)

    def test_stale_capture_is_not_ready_and_reports_reason(self):
        gate = SurveyStartup()
        for i in range(61):
            gate.update(frame(i * .05))
        self.assertTrue(gate.ready(3.12))
        self.assertFalse(gate.ready(3.121))
        self.assertEqual(gate.status(3.121)["reason"], "capture_stale")

    def test_exact_span_and_gap_boundaries_qualify(self):
        gate = SurveyStartup(min_span_s=3.0, max_gap_s=.12)
        for i in range(26):
            gate.update({"valid": True, "source": "depth_survey", "sim_usec": i * 120_000})
        self.assertEqual(gate.status(3.0)["accepted_count"], 26)
        self.assertTrue(gate.ready(3.0))

    def test_tilt_survey_sources_are_allowed_but_unknown_suffix_is_not(self):
        gate = SurveyStartup()
        gate.update(frame(0, source="depth_survey_px4_tilt"))
        self.assertEqual(gate.status()["accepted_count"], 1)
        gate.update(frame(.05, source="depth_survey_tof_px4_tilt"))
        self.assertEqual(gate.status()["accepted_count"], 2)
        status = gate.update(frame(.10, source="depth_survey_other"))
        self.assertEqual(status["accepted_count"], 0)
        self.assertEqual(status["reason"], "unqualified_source")

    def test_nonfinite_configuration_and_query_are_rejected(self):
        for kwargs in ({"min_span_s": float("nan")},
                       {"min_span_s": float("inf")},
                       {"max_gap_s": float("nan")},
                       {"max_gap_s": float("-inf")}):
            with self.assertRaises(ValueError):
                SurveyStartup(**kwargs)
        gate = SurveyStartup()
        for i in range(61):
            gate.update(frame(i * .05))
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assertFalse(gate.ready(value))
            status = gate.status(value)
            self.assertFalse(status["ready"])
            self.assertEqual(status["reason"], "invalid_sim_time")

    def test_duplicate_and_reversed_timestamps_reset(self):
        for times in ((0.0, .05, .05), (0.0, .05, .04)):
            gate = SurveyStartup()
            for t in times:
                gate.update(frame(t))
            status = gate.status()
            self.assertEqual(status["accepted_count"], 0)
            self.assertEqual(status["reason"], "nonmonotonic_capture_timestamp")

    def test_gap_over_limit_resets_then_next_frame_starts_new_span(self):
        gate = SurveyStartup()
        gate.update(frame(0))
        status = gate.update(frame(.121))
        self.assertEqual(status["accepted_count"], 0)
        self.assertEqual(status["reason"], "capture_gap_exceeded")
        status = gate.update(frame(.171))
        self.assertEqual(status["accepted_count"], 1)
        self.assertAlmostEqual(status["last_capture_sim_s"], .171)


if __name__ == "__main__":
    unittest.main()
