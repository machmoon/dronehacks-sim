"""Optional qualification of a contiguous, current depth-survey pose stream.

This is a startup gate only. It does not estimate pose or provide a safety
guarantee; callers should feed only newly captured estimator records.
"""

import math


class SurveyStartup:
    """Require a recent 3 s span of accepted RGB-D survey captures."""

    ACCEPTED_SOURCES = frozenset((
        "depth_survey", "depth_survey_tof",
        "depth_survey_px4_tilt", "depth_survey_tof_px4_tilt",
    ))

    def __init__(self, min_span_s=3.0, max_gap_s=0.12):
        if (not math.isfinite(min_span_s) or not math.isfinite(max_gap_s)
                or min_span_s <= 0 or max_gap_s <= 0):
            raise ValueError("startup span and capture gap must be positive")
        self.min_span_s = float(min_span_s)
        self.max_gap_s = float(max_gap_s)
        self.reset("waiting_for_survey")

    def reset(self, reason="waiting_for_survey"):
        self.accepted_count = 0
        self.first_capture_sim_s = None
        self.last_capture_sim_s = None
        self._last_capture_usec = None
        self.reason = reason

    @staticmethod
    def _field(frame, name, default=None):
        if isinstance(frame, dict):
            return frame.get(name, default)
        return getattr(frame, name, default)

    def update(self, frame):
        """Consume one new estimator record and return its qualification status."""
        valid = self._field(frame, "valid", False)
        source = self._field(frame, "source")
        sim_usec = self._field(frame, "sim_usec")
        if valid is not True or source not in self.ACCEPTED_SOURCES:
            reason = "invalid_frame" if valid is not True else "unqualified_source"
            self.reset(reason)
            return self.status()
        if isinstance(sim_usec, bool) or not isinstance(sim_usec, int) or sim_usec < 0:
            self.reset("invalid_capture_timestamp")
            return self.status()

        capture_s = sim_usec * 1e-6
        if self._last_capture_usec is not None:
            gap_usec = sim_usec - self._last_capture_usec
            if gap_usec <= 0:
                self.reset("nonmonotonic_capture_timestamp")
                return self.status()
            if gap_usec > self.max_gap_s * 1_000_000 + 1e-6:
                self.reset("capture_gap_exceeded")
                return self.status()

        if self.accepted_count == 0:
            self.first_capture_sim_s = capture_s
        self.accepted_count += 1
        self.last_capture_sim_s = capture_s
        self._last_capture_usec = sim_usec
        self.reason = "awaiting_contiguous_span"
        return self.status()

    def ready(self, sim_s):
        """Whether the qualified span still ends at a sufficiently recent capture."""
        try:
            sim_s = float(sim_s)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(sim_s):
            return False
        if self.accepted_count <= 1 or self.first_capture_sim_s is None:
            return False
        if self.last_capture_sim_s is None:
            return False
        age = float(sim_s) - self.last_capture_sim_s
        if age < -1e-9 or age > self.max_gap_s + 1e-9:
            return False
        return self.last_capture_sim_s - self.first_capture_sim_s + 1e-9 >= self.min_span_s

    def status(self, sim_s=None):
        span = 0.0
        if self.first_capture_sim_s is not None and self.last_capture_sim_s is not None:
            span = self.last_capture_sim_s - self.first_capture_sim_s
        is_ready = False
        reason = self.reason
        if sim_s is not None:
            try:
                sim_s = float(sim_s)
            except (TypeError, ValueError):
                sim_s = math.nan
            if not math.isfinite(sim_s):
                return {
                    "accepted_count": self.accepted_count,
                    "span_s": span,
                    "last_capture_sim_s": self.last_capture_sim_s,
                    "ready": False,
                    "reason": "invalid_sim_time",
                }
        if sim_s is not None and self.accepted_count > 1 and self.last_capture_sim_s is not None:
            age = sim_s - self.last_capture_sim_s
            if age < -1e-9 or age > self.max_gap_s + 1e-9:
                reason = "capture_stale"
            elif span + 1e-9 >= self.min_span_s:
                is_ready = True
                reason = "ready"
            else:
                reason = "awaiting_contiguous_span"
        return {
            "accepted_count": self.accepted_count,
            "span_s": span,
            "last_capture_sim_s": self.last_capture_sim_s,
            "ready": is_ready,
            "reason": reason,
        }
