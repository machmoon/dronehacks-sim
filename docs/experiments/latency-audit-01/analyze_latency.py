#!/usr/bin/env python3
"""Rebuild the policy-05/07/08 timestamp and wall-latency summary."""
from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNS = ("policy-05", "policy-07", "policy-08", "policy-09")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[float], multiplier: float = 1000.0) -> dict:
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}

    def quantile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "p50": round(quantile(0.50) * multiplier, 3),
        "p95": round(quantile(0.95) * multiplier, 3),
        "max": round(values[-1] * multiplier, 3),
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    by_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    result = {
        "schema_version": 1,
        "runs": {},
        "per_run_compact": {},
        "pooled_phase_distributions": {},
        "definitions": {
            "sample_vision_age_sim_ms": "1000 * (sample.sim_time - sample.vision_stamp_sim_s); difference in the simulator timestamp domain.",
            "policy_capture_age_sim_ms": "1000 * policy_action.capture_age_sim_s; action acceptance sim_time minus originating capture timestamp, in the simulator timestamp domain.",
            "policy_latency_wall_ms": "1000 * policy_action.latency_wall_s; PolicyClient request submission to result acceptance, measured by the mission host monotonic clock. It includes local service/inference, exchange, and polling/scheduling time; it excludes image acquisition and any prior wait for an eligible frame.",
            "policy_capture_gap_sim_ms": "1000 * policy_action.capture_interval_sim_s; gap between capture timestamps attached to consecutive accepted actions, not direct request-send cadence.",
            "maximum_capture_gap_context": "For each run, select the last largest capture_interval_sim_s among adjacent accepted policy_action records (ties resolve to the later pair); retain both records, whether waypoint_index changed, their simulator-time acceptance interval, and sample rows within 250 ms before/after the pair. Samples contain only sim_time, phase, waypoint_index, policy_capture_usec, policy_velocity_flu, command_velocity, and policy_actions_received.",
            "action_phase_assignment": "Assign each policy_action to the phase on the latest sample whose sim_time is less than or equal to the action sim_time.",
            "archive_frame_gap_sim_ms": "Adjacent capture_sim_s gaps in camera/frames.jsonl. This is the throttled approximately 5 Hz archive recorder, not native camera cadence.",
            "setpoint_counts": "summary.policy_setpoints_sent counts route-loop setpoints emitted while policy_active and an action reply exists. route_elapsed_sim_s starts at arming, before takeoff/hover, so this count divided by route duration is not a setpoint frequency; only the aggregate count is reported.",
        },
        "limitations": [
            "Simulator timestamp differences are not a real-time Jetson benchmark and do not include camera exposure/readout, OAK transport, Jetson scheduling/inference, or deployment networking.",
            "policy_action logs accepted responses, not every request send; exact request cadence and unresolved requests are unavailable from these fields.",
            "camera/frames.jsonl is an approximately 5 Hz archive and must not be used to infer native approximately 20 Hz navigation cadence; see docs/native-capture-gap-evidence.json for separate native timestamp evidence.",
            "Summary setpoint count and route_elapsed_sim_s have different scopes; no setpoint rate is inferred.",
        ],
    }

    for run in RUNS:
        folder = ROOT / "artifacts" / "markerfree" / run
        telemetry_path = folder / "telemetry.jsonl"
        frames_path = folder / "camera" / "frames.jsonl"
        telemetry = read_jsonl(telemetry_path)
        frames = read_jsonl(frames_path)
        samples = sorted((row for row in telemetry if row.get("type") == "sample"),
                         key=lambda row: row["sim_time"])
        sample_times = [row["sim_time"] for row in samples]
        actions = [row for row in telemetry if row.get("type") == "policy_action"]
        summary = next(row for row in telemetry if row.get("type") == "summary")
        metadata = next(row for row in telemetry if row.get("type") == "metadata")
        per_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        run_action_values: dict[str, list[float]] = defaultdict(list)

        for sample in samples:
            stamp = sample.get("vision_stamp_sim_s")
            if stamp is not None:
                age = sample["sim_time"] - stamp
                per_phase[sample["phase"]]["sample_vision_age_sim_ms"].append(age)
                by_phase[sample["phase"]]["sample_vision_age_sim_ms"].append(age)

        for action in actions:
            index = bisect.bisect_right(sample_times, action["sim_time"]) - 1
            if index < 0:
                continue
            phase = samples[index]["phase"]
            for field, output in (
                ("capture_age_sim_s", "policy_capture_age_sim_ms"),
                ("latency_wall_s", "policy_latency_wall_ms"),
                ("capture_interval_sim_s", "policy_capture_gap_sim_ms"),
            ):
                value = action.get(field)
                if value is not None:
                    per_phase[phase][output].append(float(value))
                    by_phase[phase][output].append(float(value))
                    run_action_values[output].append(float(value))

        archive_times = [row["capture_sim_s"] for row in frames
                         if row.get("capture_sim_s") is not None]
        archive_gaps = [b - a for a, b in zip(archive_times, archive_times[1:])]
        adjacent_gaps = [
            (float(current["capture_interval_sim_s"]), index)
            for index, current in enumerate(actions)
            if index > 0 and current.get("capture_interval_sim_s") is not None
        ]
        max_gap_context = None
        if adjacent_gaps:
            max_gap_s, current_index = max(adjacent_gaps, key=lambda item: (item[0], item[1]))
            previous_action, current_action = actions[current_index - 1], actions[current_index]
            window_start = previous_action["sim_time"] - .25
            window_end = current_action["sim_time"] + .25
            nearby = [sample for sample in samples
                      if window_start <= sample["sim_time"] <= window_end]
            compact_sample_fields = (
                "sim_time", "phase", "waypoint_index", "policy_capture_usec",
                "policy_velocity_flu", "command_velocity", "policy_actions_received",
            )
            compact_action_fields = (
                "sim_time", "waypoint_index", "capture_usec", "capture_age_sim_s",
                "latency_wall_s", "capture_interval_sim_s",
            )
            max_gap_context = {
                "capture_gap_sim_ms": round(max_gap_s * 1000, 3),
                "previous_action": {key: previous_action.get(key) for key in compact_action_fields},
                "current_action": {key: current_action.get(key) for key in compact_action_fields},
                "waypoint_changed": previous_action.get("waypoint_index") != current_action.get("waypoint_index"),
                "acceptance_interval_sim_ms": round(
                    (current_action["sim_time"] - previous_action["sim_time"]) * 1000, 3
                ),
                "nearby_samples": [
                    {key: sample.get(key) for key in compact_sample_fields}
                    for sample in nearby
                ],
            }
        result["runs"][run] = {
            "source_sha256": {
                "telemetry.jsonl": sha256(telemetry_path),
                "camera/frames.jsonl": sha256(frames_path),
            },
            "counts": {
                "telemetry_samples": len(samples),
                "accepted_policy_actions": len(actions),
                "archive_frames": len(frames),
                "policy_setpoints_sent_aggregate_only": summary.get("policy_setpoints_sent"),
                "vision_measurements_sent": summary.get("vision_measurements_sent"),
            },
            "configuration": {
                "checkpoint_sha256": metadata.get("policy_checkpoint_sha256"),
                "policy_depth_mode": metadata.get("policy_depth_mode"),
                "speed_scale": metadata.get("speed_scale"),
                "heading_lookahead_m": metadata.get("heading_lookahead_m"),
                "survey_normals": metadata.get("survey_normals"),
                "policy_boundary_brake_enabled": metadata.get("policy_boundary_brake_enabled"),
                "inertial_tilt_prior_enabled": metadata.get("inertial_tilt_prior_enabled"),
                "sensor_diagnostic": {
                    key: metadata.get("sensor_diagnostic", {}).get(key)
                    for key in ("mode", "seed", "noise_mean_m", "noise_stddev_m", "source_world_sha256", "world_sha256")
                },
            },
            "result": {
                "completed": summary.get("result"),
                "fault": summary.get("fault"),
                "route_completed": summary.get("route_completed"),
                "landed_and_disarmed": summary.get("landed_and_disarmed"),
            },
            "simulated_durations_s": {
                "route_elapsed_sim_s": summary.get("route_elapsed_sim_s"),
                "elapsed_sim_s": summary.get("elapsed_sim_s"),
            },
            "phases": {
                phase: {name: distribution(values) for name, values in sorted(metrics.items())}
                for phase, metrics in sorted(per_phase.items())
            },
            "archive_frame_gap_sim_ms": distribution(archive_gaps),
            "maximum_capture_gap_context": max_gap_context,
        }
        result["per_run_compact"][run] = {
            "accepted_policy_actions": len(actions),
            "wall_latency_ms": distribution(run_action_values["policy_latency_wall_ms"]),
            "capture_age_sim_ms": distribution(run_action_values["policy_capture_age_sim_ms"]),
            "capture_gap_sim_ms": distribution(run_action_values["policy_capture_gap_sim_ms"]),
            "policy_setpoints_sent_aggregate_only": summary.get("policy_setpoints_sent"),
            "route_elapsed_sim_s": summary.get("route_elapsed_sim_s"),
        }

    result["pooled_phase_distributions"] = {
        phase: {name: distribution(values) for name, values in sorted(metrics.items())}
        for phase, metrics in sorted(by_phase.items())
    }
    native_evidence = ROOT / "docs" / "native-capture-gap-evidence.json"
    native_data = json.loads(native_evidence.read_text(encoding="utf-8"))
    result["native_capture_evidence"] = {
        "path": "docs/native-capture-gap-evidence.json",
        "sha256": sha256(native_evidence),
        "separate_evidence_summary": {
            run: data["positive_capture_gap_ms"]
            for run, data in native_data["runs"].items()
            if run.startswith(("hover-", "policy-"))
        },
        "interpretation": "Separate PX4 vehicle_visual_odometry timestamp_sample analysis in the cited evidence file (four closed logs, median positive gaps approximately 48-52 ms). These native timestamp results are not derived from the throttled archive frame log and are not an end-to-end inference benchmark.",
    }
    result["analysis_script_sha256"] = sha256(Path(__file__))
    output = Path(__file__).with_name("latency-summary.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
