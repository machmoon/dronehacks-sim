#!/usr/bin/env python3
"""Tiny local viewer for the real Gazebo GUI and its telemetry JSONL."""
from __future__ import annotations

import json
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TELEMETRY = REPO / "artifacts" / "gazebo" / "telemetry.jsonl"
PX4_TELEMETRY = REPO / "artifacts" / "px4" / "telemetry.jsonl"
VISION = REPO / "artifacts" / "vision"
PREVIEW = REPO / "artifacts" / "preview"
LEARNING = REPO / "artifacts" / "learning"
MARKERFREE = REPO / 'artifacts' / 'markerfree' / 'current'
VNC_URL = os.environ.get('DRONEHACKS_VNC_URL', 'http://127.0.0.1:8082/vnc.html?autoconnect=1&resize=scale')
MAX_TAIL = 128 * 1024
MAX_HEAD = 32 * 1024
STALE_SECONDS = 3.0


def read_json(path):
    try:
        row = json.loads(path.read_text())
        return row if isinstance(row, dict) else None
    except (OSError, ValueError):
        return None


def training_state():
    current = read_json(LEARNING / 'current.json')
    if not current:
        return {'state': 'not_started'}
    directory = (LEARNING / current.get('run_directory', '')).resolve()
    if not directory.is_relative_to(LEARNING.resolve()):
        return {'state': 'invalid_run_path'}
    status = read_json(directory / 'status.json') or {'state': 'starting'}
    status['run_directory'] = str(directory)
    status['launcher'] = current
    status['status_age_s'] = (max(0, time.time() - (directory / 'status.json').stat().st_mtime)
                              if (directory / 'status.json').exists() else None)
    if status.get('state') in ('running', 'starting'):
        if current.get('finished_at'):
            status['state'] = 'stopped' if current.get('stop_reason') == 'user_stop' else 'failed'
            status['error'] = 'Supervisor exited before a final trainer status was saved.'
        elif status['status_age_s'] is not None and status['status_age_s'] > 60:
            status['state'] = 'stale'
            status['error'] = 'No recent trainer heartbeat; inspect the run log.'
    candidate = read_json(REPO / 'checkpoints' / 'window-depth-v2.manifest.json')
    if candidate and current.get('run_directory') == candidate.get('training_run'):
        status['checkpoint_candidate'] = {
            'file': candidate.get('checkpoint_file'),
            'sha256': candidate.get('packaged_checkpoint_sha256'),
            'selected_steps': candidate.get('selected_steps'),
            'profile': candidate.get('profile'),
            'flight_candidate': candidate.get('analytic_comparison', {}).get('flight_candidate'),
        }
    return status


def identify_flight_checkpoint(metadata):
    """Identify a policy checkpoint only by a manifest-matched telemetry hash."""
    if not metadata:
        return {'label': 'Awaiting flight metadata', 'sha256': None}
    if metadata.get('task') != 'policy':
        task = metadata.get('task')
        label = 'Waypoint baseline' if task == 'course' else 'No learned checkpoint in this run'
        return {'label': label, 'sha256': None}
    digest = metadata.get('policy_checkpoint_sha256')
    if not isinstance(digest, str):
        return {'label': 'Unidentified checkpoint', 'sha256': digest}
    for path, digest_key, label in (
        (REPO / 'checkpoints' / 'window-depth-v1.manifest.json', 'sha256', 'Packaged v1'),
        (REPO / 'checkpoints' / 'window-depth-v2.manifest.json', 'packaged_checkpoint_sha256', 'Packaged v2'),
    ):
        manifest = read_json(path)
        if manifest and digest == manifest.get(digest_key):
            return {
                'label': label,
                'sha256': digest,
                'profile': manifest.get('profile'),
            }
    return {'label': 'Unidentified checkpoint', 'sha256': digest}


def telemetry_path():
    preferred = os.environ.get("DRONEHACKS_TELEMETRY")
    if preferred:
        path = Path(preferred)
        return path if path.is_absolute() else REPO / path
    if (MARKERFREE / 'telemetry.jsonl').exists():
        return MARKERFREE / 'telemetry.jsonl'
    try:
        if PX4_TELEMETRY.exists() and (not TELEMETRY.exists() or
                                      PX4_TELEMETRY.stat().st_mtime > TELEMETRY.stat().st_mtime):
            return PX4_TELEMETRY
    except OSError:
        pass
    return TELEMETRY


def read_records(path):
    """Read bounded head and tail windows; ignore a partial leading tail line."""
    try:
        stat = path.stat()
        with path.open("rb") as f:
            head = f.read(MAX_HEAD)
            if stat.st_size <= MAX_TAIL:
                tail = head + f.read()
            else:
                f.seek(-MAX_TAIL, 2)
                f.readline()
                tail = f.read()
    except FileNotFoundError:
        return None, [], []

    def parse_window(raw):
        parsed = []
        for line in raw.splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    parsed.append(row)
            except (ValueError, UnicodeDecodeError):
                continue
        return parsed

    return stat, parse_window(head), parse_window(tail)


def read_summary_sidecar(path):
    """Course writer atomically refreshes this small file at mission finish."""
    path = path.with_name(path.stem + ".summary.json")
    try:
        row = json.loads(path.read_text())
        return row if isinstance(row, dict) and row.get("type") == "summary" else None
    except (OSError, ValueError):
        return None


def read_matched_score(path, metadata):
    """Read only the score beside the selected telemetry, for that exact run."""
    run_id = metadata.get('run_id') if metadata else None
    if not run_id:
        return None
    score = read_json(path.parent / 'score.json')
    return score if score and score.get('run_id') == run_id else None


def selected_camera_dir():
    return telemetry_path().parent / 'camera'


def state():
    path = telemetry_path()
    preview = read_json(PREVIEW / 'session.json')
    if preview and preview.get('started_at', 0) > (path.stat().st_mtime if path.exists() else 0):
        latest = read_json(PREVIEW / 'latest.json') or {}
        if latest.get('session_started_at') != preview.get('started_at'):
            latest = {}
        age = max(0, time.time() - latest.get('updated_at', 0)) if latest else None
        return {
            'mode': 'preview', 'status': 'live' if age is not None and age < 3 else 'stale',
            'metadata': preview, 'vision': latest, 'file_age_s': age,
            'camera_url': '/preview-camera.jpg' if latest else None,
            'labels': ['NO ARTIFICIAL TAGS', 'CAMERA PREVIEW', 'LOCALIZATION NOT CONNECTED'],
            'sample': None, 'summary': None, 'training': training_state(),
            'flight_checkpoint': {'label': 'No PX4 flight metadata', 'sha256': None},
        }
    stat, head, tail = read_records(path)
    sample = next((r for r in reversed(tail) if r.get("type") == "sample" and
                   isinstance(r.get("position"), list) and len(r["position"]) >= 3), None)
    now = time.time()
    age = max(0.0, now - stat.st_mtime) if stat else None
    status = "waiting" if sample is None else ("stale" if age is None or age > STALE_SECONDS else "live")
    metadata = next((r for r in head if r.get("type") == "metadata"), None)
    if metadata is None:
        metadata = next((r for r in tail if r.get("type") == "metadata"), None)
    summary = read_summary_sidecar(path)
    if summary is None:
        summary = next((r for r in reversed(tail) if r.get("type") == "summary"), None)
    if metadata and metadata.get("run_id") and summary:
        if summary.get("run_id") != metadata["run_id"]:
            summary = None
    events = [r for r in tail if r.get("type") == "event"][-8:]
    markerfree = bool(metadata and str(metadata.get('navigation_source', '')).startswith('RGBD'))
    camera_dir = path.parent / 'camera'
    if markerfree and summary:
        status = 'completed' if summary.get('result') == 'completed' else 'stopped'
    is_px4 = markerfree or bool(metadata and metadata.get("source") == "PX4 EKF2 + camera marker localization")
    vision = None
    vision_age = None
    vision_wall_age = None
    if is_px4 and not markerfree:
        try:
            vision = json.loads((VISION / "latest.json").read_text())
            vision_wall_age = max(0.0, now - (VISION / "latest.json").stat().st_mtime)
            if not isinstance(vision, dict):
                vision = None
            elif sample and vision.get("stamp_sim_s") is not None and sample.get("sim_time") is not None:
                vision_age = max(0.0, float(sample["sim_time"]) - float(vision["stamp_sim_s"]))
        except (OSError, ValueError, TypeError):
            vision = None
    return {
        "status": status, "file_age_s": round(age, 2) if age is not None else None,
        "sample": sample, "metadata": metadata, "summary": summary, "events": events,
        "flight_checkpoint": identify_flight_checkpoint(metadata) if markerfree else None,
        "score": read_matched_score(path, metadata) if markerfree else None,
        "camera": read_json(camera_dir/'latest.json') if markerfree else None,
        "source": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), "gazebo_url": VNC_URL,
        "mode": "markerfree" if markerfree else ("px4" if is_px4 else "gazebo"),
        "vision": vision, "vision_age_s": round(vision_age, 2) if vision_age is not None else None,
        "vision_wall_age_s": round(vision_wall_age, 2) if vision_wall_age is not None else None,
        "camera_url": ('/rgbd-camera.jpg' if markerfree and (camera_dir/'latest.jpg').is_file()
                       else '/camera.jpg' if is_px4 and not markerfree and (VISION / 'camera.jpg').is_file() else None),
        "depth_url": '/rgbd-depth.jpg' if markerfree and (camera_dir/'depth.jpg').is_file() else None,
        "labels": (["PX4 SITL", "RGBD + TOF", "NO ARTIFICIAL TAGS", "NO TRUTH FEEDBACK"] if markerfree else
                   ["PX4 SITL", "CAMERA POSITION", "MAPPED FIDUCIALS", "NOT HARDWARE VALIDATED"]
                   if is_px4 else ["REAL GAZEBO PHYSICS", "GROUND-TRUTH SIM STATE", "NOT PX4", "NOT A REAL DRONE"]),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if urlparse(self.path).path in ('/rgbd-camera.jpg', '/rgbd-depth.jpg'):
            image = selected_camera_dir()/('latest.jpg' if 'camera' in self.path else 'depth.jpg')
            if not image.is_file():
                self.send_error(404)
                return
            body = image.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if urlparse(self.path).path == '/api/training':
            body = json.dumps(training_state()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if urlparse(self.path).path == "/api/state":
            body = json.dumps(state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if urlparse(self.path).path in ("/camera.jpg", "/preview-camera.jpg"):
            image = (PREVIEW if urlparse(self.path).path == '/preview-camera.jpg' else VISION) / 'camera.jpg'
            if image.is_file():
                body = image.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
            return
        if urlparse(self.path).path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):
        print("[viewer] " + fmt % args)


if __name__ == "__main__":
    port = int(os.environ.get('DRONEHACKS_VIEWER_PORT', '8083'))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Marker-free PX4 viewer: http://127.0.0.1:{port}")
    server.serve_forever()
