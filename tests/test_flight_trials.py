import json
import tempfile
import unittest
from pathlib import Path

from tools.flight_trials import build_inventory, markdown, _classify, _jsonl


def write_run(root: Path, name: str, run_id: str, *, events=None, summary=None):
    archive = root / "artifacts/markerfree" / name
    archive.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "metadata", "run_id": run_id, "task": "policy", "waypoint_count": 15,
             "policy_checkpoint_sha256": "abc", "policy_boundary_brake_enabled": True}]
    rows.extend(events or [])
    (archive / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if summary is not None:
        (archive / "telemetry.summary.json").write_text(json.dumps({"run_id": run_id, **summary}))
    return archive


class FlightTrialsInventoryTests(unittest.TestCase):
    def test_only_unfinished_last_record_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'telemetry.jsonl'
            path.write_text('{"type":"metadata"}\n{"unfinished":')
            self.assertEqual(_jsonl(path), [{'type': 'metadata'}])
            path.write_text('{"type":"metadata"}\ncorrupt\n')
            with self.assertRaisesRegex(ValueError, 'Invalid telemetry record'):
                _jsonl(path)

    def test_source_filename_does_not_make_a_blackout_trial(self):
        metadata = {'runtime_source_sha256': {'mission/blackout.py': 'example'}}
        self.assertEqual(_classify('policy', metadata, []), 'full_route_attempts')
        self.assertEqual(_classify('course', metadata, [
            {'message': 'Injected RGBD capture loss'}]), 'blackout_diagnostics')

    def test_refresh_replaces_running_facts_when_summary_arrives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {"type": "event", "sim_time": 10.0, "message": "PX4 armed in Offboard with marker-free RGBD aiding"},
                {"type": "event", "sim_time": 18.0, "message": "Route complete; requesting PX4 AUTO_LAND"},
            ]
            archive = write_run(root, "policy-05", "run-5", events=events)
            first = build_inventory(root)["groups"]["full_route_attempts"][0]
            self.assertEqual(first["result"], "in_progress")
            self.assertFalse(first["landing"]["confirmed"])
            self.assertFalse(first["score"]["matched_run_id"])

            (archive / "telemetry.summary.json").write_text(json.dumps({
                "run_id": "run-5", "result": "completed", "fault": None,
                "elapsed_sim_s": 12.0, "route_elapsed_sim_s": 8.0,
                "route_completed": True, "landed_and_disarmed": True,
                "landed_after_fault": False, "waypoints_reached": 15,
                "waypoint_count": 15, "policy_actions_received": 9,
            }))
            refreshed = build_inventory(root)["groups"]["full_route_attempts"][0]
            self.assertEqual(refreshed["result"], "completed")
            self.assertIsNone(refreshed["fault"])
            self.assertEqual(refreshed["controller_waypoints_reached"], 15)
            self.assertTrue(refreshed["landing"]["confirmed"])
            self.assertEqual(refreshed["timestamps_sim_s"]["route_complete_event"], 18.0)
            self.assertIsNone(refreshed["timestamps_sim_s"]["touchdown_disarm_event"])
            self.assertFalse(refreshed["score"]["matched_run_id"])

    def test_current_alias_tracks_its_current_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "policy-04", "run-4")
            write_run(root, "policy-05", "run-5")
            current = root / "artifacts/markerfree/current"

            current.symlink_to("policy-04", target_is_directory=True)
            inventory = build_inventory(root)
            alias = inventory["inventory"]["duplicate_archive_aliases"][0]
            self.assertEqual(alias["canonical_archive"], "policy-04")
            self.assertIn("`current` is an identical alias of `policy-04`", markdown(inventory))

            current.unlink()
            current.symlink_to("policy-05", target_is_directory=True)
            inventory = build_inventory(root)
            alias = inventory["inventory"]["duplicate_archive_aliases"][0]
            self.assertEqual(alias["canonical_archive"], "policy-05")
            self.assertIn("`current` is an identical alias of `policy-05`", markdown(inventory))
            self.assertEqual(inventory["inventory"]["unique_run_ids"], 2)

    def test_export_alias_remains_named_when_current_sorts_after_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            target = write_run(export, "policy-01", "export-run")
            archives = root / "artifacts/markerfree"
            archives.mkdir(parents=True)
            for name in ("clean-export-policy-01", "current"):
                (archives / name).symlink_to(target, target_is_directory=True)
            inventory = build_inventory(root)
            item = inventory["groups"]["full_route_attempts"][0]
            self.assertEqual(item["archive"], "artifacts/markerfree/clean-export-policy-01")
            self.assertIn("`current` is an identical alias of `clean-export-policy-01`", markdown(inventory))

    def test_canonical_record_does_not_take_fields_from_different_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "a-run", "same-id", summary={"result": "failed", "fault": "preserved"})
            write_run(root, "z-copy", "same-id", summary={"result": "completed"})
            inventory = build_inventory(root)
            item = inventory["groups"]["full_route_attempts"][0]
            self.assertEqual(item["archive"], "artifacts/markerfree/a-run")
            self.assertEqual(item["result"], "failed")
            self.assertEqual(item["fault"], "preserved")

    def test_mismatched_summary_is_not_claimed_as_result_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = write_run(root, "policy-01", "this-run")
            (archive / "telemetry.summary.json").write_text(json.dumps({
                "run_id": "another-run", "result": "completed"}))
            item = build_inventory(root)["groups"]["full_route_attempts"][0]
            self.assertEqual(item["result"], "in_progress")
            self.assertFalse(item["reproducibility"]["summary_run_id_matched"])
            self.assertIsNone(item["reproducibility"]["summary_sha256"])

    def test_explicit_prearm_hover_refusal_has_no_landing_to_confirm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = write_run(root, "startup-hover", "refusal", summary={
                "result": "failed", "fault": "absolute_survey_startup_unqualified",
                "arm_request_sent": False})
            path = archive / "telemetry.jsonl"
            path.write_text(path.read_text().replace('"task": "policy"', '"task": "hover"'))
            item = build_inventory(root)["groups"]["diagnostic_hovers"][0]
            self.assertEqual(item["result"], "failed")
            self.assertFalse(item["landing"]["confirmed"])
            self.assertEqual(item["landing"]["status"], "no armed flight; touchdown not applicable")


if __name__ == "__main__":
    unittest.main()
