#!/usr/bin/env python3
"""Build a marker-free flight inventory from archived evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_DOC = ROOT / "docs/flight-trials.json"


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    out = []
    raw = path.read_bytes()
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines)-1 and not raw.endswith((b'\n', b'\r')):
                    continue  # only an unfinished final write can be skipped
                raise ValueError(f'Invalid telemetry record {path}:{index+1}')
    return out


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _score(path: Path, run_id: str) -> tuple[dict | None, Path | None]:
    # Prefer the canonical root score; a review score remains usable when that
    # is the only independently produced score for this run.
    for name in ("score.json", "score-review.json"):
        candidate = path / name
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("run_id") == run_id:
                return value, candidate
    return None, None


def _compact_score(value: dict | None, path: Path | None) -> dict:
    if value is None:
        return {"matched_run_id": False, "status": "no matching score.json or score-review.json; truth fields omitted"}
    route, window = value.get("route", {}), value.get("window", {})
    landing, abort = value.get("landing", {}), value.get("abort_landing", {})
    return {
        "matched_run_id": True,
        "score_file": path.name,
        "mission_result": value.get("mission_result"),
        "full_route_verified_by_geometry": value.get("full_route_verified_by_geometry"),
        "truth_waypoints_reached": route.get("reached"),
        "truth_waypoint_count": route.get("required"),
        "truth_reaches_all_waypoints_in_order": value.get("checks", {}).get("truth_reaches_all_waypoints_in_order"),
        "checks": value.get("checks", {}),
        "window_min_aperture_clearance_m": window.get("min_aperture_clearance_m"),
        "window_full_envelope_crossed": window.get("full_envelope_crossed"),
        "pad_horizontal_envelope_clearance_m": landing.get("horizontal_envelope_clearance_m"),
        "landing_truth_inside_pad": landing.get("truth_envelope_inside_pad"),
        "landing_confirmed_ground_disarm": landing.get("px4_confirmed_ground_and_disarm"),
        "abort_landing": {
            "fault_sim_s": abort.get("fault_sim_s"),
            "confirmed_disarm_sim_s": abort.get("confirmed_disarm_sim_s"),
            "scope": abort.get("scope"),
            "controller_confirmed_ground_and_disarm": abort.get("controller_confirmed_ground_and_disarm"),
            "certified_clear_through_disarm": abort.get("certified_clear_through_disarm"),
            "cad_proxy_clear": abort.get("cad_proxy", {}).get("certified_clear"),
            "contacts_status": abort.get("contacts", {}).get("status"),
            "contacts_unexpected_observed": abort.get("contacts", {}).get("unexpected_contact_observed"),
        } if abort else None,
    }


def _event(events: list[dict], needle: str) -> dict | None:
    return next((e for e in events if needle in str(e.get("message", ""))), None)


def _at(event: dict | None):
    if event is None:
        return None
    return event.get("sim_time", event.get("sim_time_s"))


def _classify(task: str, metadata: dict, events: list[dict]) -> str:
    if task == "hover":
        return "diagnostic_hovers"
    messages = " ".join(str(e.get("message", "")).lower() for e in events)
    if (task == 'blackout' or metadata.get('blackout_after') is not None
            or 'injected rgbd capture loss' in messages):
        return "blackout_diagnostics"
    return "full_route_attempts"


def _run_record(archive: Path, root: Path) -> dict | None:
    telemetry_path = archive / "telemetry.jsonl"
    rows = _jsonl(telemetry_path)
    metadata = next((r for r in rows if r.get("type") == "metadata" and r.get("run_id")), None)
    if metadata is None:
        return None
    run_id, task = metadata["run_id"], metadata.get("task", "unknown")
    events = [r for r in rows if r.get("type") == "event"]
    summary_path = archive / "telemetry.summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            candidate = json.loads(summary_path.read_text())
            if candidate.get("run_id") == run_id:
                summary = candidate
        except (OSError, json.JSONDecodeError):
            pass
    armed = _event(events, "PX4 armed in Offboard with marker-free RGBD aiding")
    hover = _event(events, "Sustained takeoff hover verified")
    route_done = _event(events, "Route complete; requesting PX4 AUTO_LAND")
    stopped = _event(events, "Mission stopped:")
    touchdown = _event(events, "PX4 confirmed touchdown and disarm")
    injection = next((e for e in events if "injected" in str(e.get("message", "")).lower()), None)
    fault = summary.get("fault")
    if fault is None and stopped:
        fault = str(stopped.get("message", "")).split("Mission stopped:", 1)[-1].strip() or None
    result = summary.get("result") or ("completed" if route_done and touchdown else ("failed" if stopped else "in_progress"))
    confirmed = bool(touchdown or summary.get("landed_and_disarmed") or summary.get("landed_after_fault"))
    landing_status = ("confirmed touchdown and disarm" if confirmed else
                      "no armed flight; touchdown not applicable" if not armed and
                      (task == "policy" or summary.get("arm_request_sent") is False) else
                      "not confirmed in archived telemetry/summary")
    cp = metadata.get("policy_checkpoint_sha256")
    brake_present = "policy_boundary_brake_enabled" in metadata
    score, score_path = _score(archive, run_id)
    source_map = metadata.get("runtime_source_sha256")
    aliases = []  # populated by build_inventory after grouping by run ID
    signals = [e for e in events if any(n in str(e.get("message", "")).lower() for n in
               ("navigation source audit passed", "armed in offboard", "hover verified", "injected", "mission stopped", "route complete", "confirmed touchdown", "disarmed"))]
    return {
        "archive": str(archive.relative_to(root)),
        "archive_aliases": aliases,
        "run_id": run_id,
        "category": _classify(task, metadata, events).removesuffix("s"),
        "task": task,
        "controller": "learned policy" if task == "policy" else ("waypoint course controller" if task == "course" else task),
        "result": result,
        "fault": fault,
        "policy_checkpoint_sha256": cp,
        "policy_boundary_brake_enabled": metadata.get("policy_boundary_brake_enabled"),
        "policy_boundary_brake_flag_recorded": brake_present,
        "policy_actions_received": summary.get("policy_actions_received"),
        "controller_waypoints_reached": summary.get("waypoints_reached"),
        "controller_waypoint_count": summary.get("waypoint_count", metadata.get("waypoint_count")),
        "landing": {
            "confirmed": confirmed,
            "status": landing_status,
            "landed_and_disarmed": bool(touchdown or summary.get("landed_and_disarmed")),
            "landed_after_fault": summary.get("landed_after_fault"),
            "confirmation_event": {"sim_time_s": _at(touchdown), "message": touchdown.get("message")} if touchdown else None,
        },
        "timestamps_sim_s": {
            "armed_event": _at(armed),
            "verified_hover_event": _at(hover),
            "route_complete_elapsed_from_arm": summary.get("route_elapsed_sim_s"),
            "summary_elapsed_from_arm": summary.get("elapsed_sim_s"),
            "injection_event": _at(injection),
            "mission_stopped_event": _at(stopped),
            "route_complete_event": _at(route_done),
            "touchdown_disarm_event": _at(touchdown),
        },
        "score": _compact_score(score, score_path),
        "reproducibility": {
            "telemetry_sha256": _sha(telemetry_path),
            "metadata_record_sha256": _canonical_hash(metadata),
            "runtime_source_hash_map_sha256": _canonical_hash(source_map) if source_map else None,
            "runtime_source_hash_count": len(source_map or {}),
            "summary_sha256": _sha(summary_path) if summary else None,
            "summary_run_id_matched": bool(summary),
            "score_sha256": _sha(score_path) if score_path else None,
        },
        "key_events": [{"sim_time_s": _at(e), "message": e.get("message")} for e in signals],
    }


def build_inventory(root: Path = ROOT) -> dict:
    archives_root = root / "artifacts/markerfree"
    paths = sorted(p.parent for p in archives_root.glob("*/telemetry.jsonl"))
    grouped: dict[str, list[Path]] = {}
    records: dict[str, dict] = {}
    for path in paths:
        record = _run_record(path, root)
        if record is None:
            continue
        grouped.setdefault(record["run_id"], []).append(path)
        records[record["run_id"]] = record

    duplicates = []
    for run_id, copies in grouped.items():
        # A stable run name remains canonical even when both it and `current`
        # are symlinks into an exported checkout outside this archive directory.
        current = next((p for p in copies if p.name == "current"), None)
        named = [p for p in copies if p.name != "current"]
        targets = [p for p in named if current and p.resolve() == current.resolve()]
        canonical = next(iter(targets or named or copies))
        item = _run_record(canonical, root)
        records[run_id] = item
        aliases = [p for p in copies if p != canonical]
        item["archive_aliases"] = [p.name for p in aliases]
        for alias in aliases:
            duplicates.append({
                "archive": alias.name,
                "canonical_archive": canonical.name,
                "same_run_id": run_id,
                "telemetry_byte_identical": (alias / "telemetry.jsonl").read_bytes() == (canonical / "telemetry.jsonl").read_bytes(),
            })

    grouped_records = {"full_route_attempts": [], "diagnostic_hovers": [], "blackout_diagnostics": []}
    for item in records.values():
        group = item["category"] + "s"
        grouped_records.setdefault(group, []).append(item)
    for collection in grouped_records.values():
        collection.sort(key=lambda item: (item["archive"], item["run_id"]))
    return {
        "schema": "dronehacks.markerfree-flight-trials.v1",
        "scope": "All marker-free telemetry archives with a flight run ID. Non-flight diagnostics without a run ID are excluded. Hover and injected blackout protocols are separated from full-route attempts. No pooled reliability rate is calculated across changing code/configuration or protocols.",
        "inventory": {
            "telemetry_archive_paths_seen": sum(len(paths) for paths in grouped.values()),
            "unique_run_ids": len(records),
            "duplicate_archive_aliases": sorted(duplicates, key=lambda d: d["archive"]),
        },
        "timing_semantics": "Event sim_time_s values are absolute simulation-clock timestamps when recorded. elapsed_sim_s and route_elapsed_sim_s come from the per-run summary and are relative to arming. A null event time means telemetry did not record a timestamp for that event.",
        "groups": grouped_records,
    }


def markdown(data: dict) -> str:
    inventory = data["inventory"]
    aliases = inventory["duplicate_archive_aliases"]
    alias_text = ("; ".join(f"`{a['archive']}` {'is an identical alias of' if a['telemetry_byte_identical'] else 'shares a run ID but differs from'} `{a['canonical_archive']}`" for a in aliases)
                  if aliases else "no duplicate archive aliases were found")
    out = ["# Marker-free flight trials", "", f"This inventory contains {inventory['unique_run_ids']} unique run IDs across {inventory['telemetry_archive_paths_seen']} flight telemetry archive paths; {alias_text}. No overall reliability percentage is inferred across changing code, configuration, and diagnostic protocols.", "", "## Full-route attempts", "", "| Archive | Controller / checkpoint | Outcome / fault | Controller waypoints | Independent truth | Landing |", "|---|---|---|---:|---:|---|"]
    for item in data["groups"].get("full_route_attempts", []):
        score = item["score"]
        truth = f"{score['truth_waypoints_reached']}/{score['truth_waypoint_count']}" if score.get("matched_run_id") and score.get("truth_waypoint_count") is not None else "not scored"
        if score.get("truth_reaches_all_waypoints_in_order"):
            truth += " ordered"
        cp = item.get("policy_checkpoint_sha256")
        result = item.get("fault") if item.get("result") == "failed" and item.get("fault") else item.get("result")
        landing = item["landing"]["status"]
        out.append(f"| {Path(item['archive']).name} | {item['controller']} / `{cp[:8] if cp else '—'}` | {result} | {item.get('controller_waypoints_reached','—')}/{item.get('controller_waypoint_count','—')} | {truth} | {landing} |")
    out += ["", "## Independent score details", "", "Only a score file whose run ID matches its telemetry archive is joined. Abort-landing clearance and contact coverage are reported separately from route completion.", ""]
    for item in data["groups"].get("full_route_attempts", []):
        s = item["score"]
        if not s.get("matched_run_id"):
            continue
        truth = f"{s.get('truth_waypoints_reached')}/{s.get('truth_waypoint_count')} ordered={s.get('truth_reaches_all_waypoints_in_order')}"
        abort = s.get("abort_landing")
        extra = f"; abort landing {abort.get('fault_sim_s')}→{abort.get('confirmed_disarm_sim_s')} s, CAD clear through disarm={abort.get('certified_clear_through_disarm')}, contact coverage={abort.get('contacts_status')}" if abort else ""
        out.append(f"- **{Path(item['archive']).name}:** truth {truth}{extra}.")
    out += ["", "## Diagnostic hovers", "", "| Archive | Outcome | Fault | Landing | Independent score |", "|---|---|---|---|---|"]
    for item in data["groups"].get("diagnostic_hovers", []):
        out.append(f"| {Path(item['archive']).name} | {item['result']} | {item.get('fault') or 'none'} | {item['landing']['status']} | {'matched' if item['score'].get('matched_run_id') else 'not scored'} |")
    out += ["", "## Blackout diagnostic", "", "| Archive | Outcome | Injection/fault | Controller waypoints | Independent truth | Landing |", "|---|---|---|---:|---:|---|"]
    for item in data["groups"].get("blackout_diagnostics", []):
        s = item["score"]
        truth = f"{s.get('truth_waypoints_reached')}/{s.get('truth_waypoint_count')}" if s.get("matched_run_id") and s.get("truth_waypoint_count") is not None else "not scored"
        out.append(f"| {Path(item['archive']).name} | {item['result']} | {item.get('fault') or 'none'} | {item.get('controller_waypoints_reached','—')}/{item.get('controller_waypoint_count','—')} | {truth} | {item['landing']['status']} |")
    out += ["", "## Interpretation and provenance", "", "The JSON preserves each run ID, checkpoint and braking-guard metadata, event and summary simulation times, matching independent-score checks, and SHA-256 fingerprints of source records. Missing scores remain unknown; the inventory does not infer truth waypoints from controller progress.", "", "All results are simulator evidence. Contact-topic coverage may be partial; absence of unexpected observed contacts is not a zero-physical-collision claim.", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write refreshed JSON and Markdown into this directory")
    parser.add_argument("--replace-docs", action="store_true", help="explicitly replace docs/flight-trials.json and .md")
    args = parser.parse_args()
    if bool(args.output) == args.replace_docs:
        parser.error("specify exactly one of --output DIR or --replace-docs")
    data = build_inventory(ROOT)
    out = (ROOT / args.output if args.output and not args.output.is_absolute() else args.output) if args.output else ROOT / "docs"
    if args.replace_docs and out != ROOT / "docs":
        parser.error("--replace-docs resolves only to the repository docs directory")
    if args.output and not out.exists():
        out.mkdir(parents=True)
    (out / "flight-trials.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "flight-trials.md").write_text(markdown(data))


if __name__ == "__main__":
    main()
