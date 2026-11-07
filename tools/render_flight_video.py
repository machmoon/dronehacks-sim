#!/usr/bin/env python3
"""Render an honest, simulation-time RGB/depth replay from a flight archive.

Host mode uses the existing simulator container's OpenCV and the host ffmpeg.
Only the requested MP4 is retained; temporary composite frames are removed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

MAX_OUTPUT_BYTES = 20 * 1024 * 1024
DEFAULT_CONTAINER = 'dronehacks-markerfree-sim-1'
DEFAULT_CONTEXT = 'colima-dronehacks-sim'


def _load_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mission_label(rows):
    metadata = next((r for r in rows if r.get('type') == 'metadata'), {})
    summary = next((r for r in reversed(rows) if r.get('type') == 'summary'), {})
    actions = int(summary.get('policy_actions_received') or 0)
    is_hover = metadata.get('task') == 'hover' or summary.get('task') == 'hover'
    policy_attempt = (metadata.get('task') == 'policy' or summary.get('task') == 'policy'
                      or summary.get('policy_checkpoint_sha256')
                      or summary.get('policy_service_session'))
    if is_hover:
        mode = 'HOVER CHECK'
    elif policy_attempt and actions:
        mode = 'LEARNED POLICY'
    elif policy_attempt:
        mode = 'POLICY NOT STARTED'
    else:
        mode = 'TASK BASELINE'
    result = str(summary.get('result', 'unknown')).upper()
    if result not in {'FAILED', 'COMPLETED', 'UNKNOWN'}:
        result = 'UNKNOWN'
    return mode, result


def _phase_at(rows, stamp):
    eligible = [(float(row['sim_time']), row) for row in rows
                if row.get('sim_time') is not None and float(row['sim_time']) <= stamp]
    recent = max(eligible, key=lambda x: x[0])[1] if eligible else None
    if recent is None:
        return 'phase unavailable'
    message = str(recent.get('message', '')).lower()
    if 'observing px4 abort landing' in message or 'mission stopped:' in message:
        return 'ABORT_LAND'
    phase = recent.get('phase') or recent.get('mission_phase')
    if not phase:
        message = str(recent.get('message', ''))
        for name in ('TAKEOFF', 'FIRST_CORRIDOR', 'MAZE', 'GATE_A', 'TUNNEL_ALIGN',
                     'TUNNEL', 'CHICANE', 'WINDOW_APPROACH', 'WINDOW_ALIGN',
                     'WINDOW_CROSS', 'LANDING_APPROACH'):
            if name in message:
                phase = name
                break
    return str(phase or 'mission')


def _events(camera_rows, telemetry_rows, max_gap):
    frames = sorted(camera_rows, key=lambda x: float(x['capture_sim_s']))
    if not frames:
        raise ValueError('camera frame index is empty')
    events = []
    previous = None
    for row in frames:
        t = float(row['capture_sim_s'])
        if not math.isfinite(t) or (previous is not None and t <= previous):
            raise ValueError('capture_sim_s must be finite and strictly increasing')
        if previous is not None and t - previous > max_gap:
            gap_t = previous + max_gap
            events.append({'t': gap_t, 'gap': True,
                           'phase': _phase_at(telemetry_rows, gap_t)})
        events.append({'t': t, 'gap': False, 'frame': row,
                       'phase': _phase_at(telemetry_rows, t)})
        previous = t
    if len(events) > 1:
        events[-1]['duration'] = min(max_gap, max(.04, events[-1]['t'] - events[-2]['t']))
        for i in range(len(events) - 1):
            events[i]['duration'] = max(.04, events[i + 1]['t'] - events[i]['t'])
    else:
        events[0]['duration'] = .2
    return events


def _inside(archive, frames_dir, max_gap):
    import cv2
    import numpy as np

    archive = Path(archive)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=False)
    camera = archive / 'camera'
    indexed = _load_jsonl(camera / 'frames.jsonl')
    telemetry_path = archive / 'telemetry.jsonl'
    telemetry = _load_jsonl(telemetry_path) if telemetry_path.exists() else []
    mode, result = _mission_label(telemetry)
    metadata = next((r for r in telemetry if r.get('type') == 'metadata'), {})
    if mode == 'LEARNED POLICY' and metadata.get('policy_boundary_brake_enabled'):
        mode = 'LEARNED + BRAKING GUARD'
    events = _events(indexed, telemetry, max_gap)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, event in enumerate(events):
        canvas = np.zeros((384, 768, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (767, 41), (24, 29, 36), -1)
        cv2.putText(canvas, f"COURSE  |  {mode}  |  MISSION {result}",
                    (12, 28), font, .58, (235, 240, 245), 1, cv2.LINE_AA)
        if event['gap']:
            cv2.rectangle(canvas, (8, 64), (759, 319), (8, 8, 8), -1)
            cv2.putText(canvas, 'CAMERA GAP - NO CAPTURE RECORDED', (58, 180),
                        font, .85, (30, 150, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'No prior image is held across this interval', (120, 210),
                        font, .5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            row = event['frame']
            rgb_path = camera / row['rgb']
            depth_path = camera / row['depth_z_mm_u16']
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path.exists() else None
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
            if rgb is None or depth is None or depth.dtype != np.uint16:
                cv2.putText(canvas, 'FRAME FILE MISSING / INVALID', (62, 165), font, .8,
                            (30, 150, 255), 2, cv2.LINE_AA)
            else:
                if rgb.shape[:2] != (256, 384) or depth.shape[:2] != (256, 384):
                    raise ValueError(f"unexpected camera dimensions: RGB {rgb.shape[:2]}, depth {depth.shape[:2]}")
                # Fixed 0–5 m optical-Z scale. Invalid zero values remain black.
                depth_m = depth.astype(np.float32) * .001
                depth_gray = np.clip(depth_m * (255.0 / 5.0), 0, 255).astype(np.uint8)
                depth_gray[depth == 0] = 0
                depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                canvas[64:320, 0:384] = rgb
                canvas[64:320, 384:768] = depth_bgr
                cv2.putText(canvas, 'RGB', (10, 59), font, .48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, 'OPTICAL Z 0-5 m', (392, 59), font, .48,
                            (255, 255, 255), 1, cv2.LINE_AA)
                diag = row.get('diagnostics', {}).get('tof', {})
                tof = 'ToF unavailable'
                age = diag.get('signed_age_ms')
                if (diag.get('status') == 'paired' and diag.get('range_m') is not None
                        and age is not None and abs(float(age)) <= float(diag.get('tolerance_ms', 15))):
                    tof = f"ToF paired {float(diag['range_m']):.3f} m  age {float(age):+.1f} ms"
                elif diag.get('reason'):
                    tof += f" ({diag['reason']})"
                cv2.putText(canvas, tof, (12, 370), font, .48, (205, 220, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"SIM {event['t']:.3f} s   |   {event['phase']}",
                    (12, 344), font, .53, (100, 240, 180), 1, cv2.LINE_AA)
        dest = frames_dir / f'{i:06d}.jpg'
        if not cv2.imwrite(str(dest), canvas, [cv2.IMWRITE_JPEG_QUALITY, 78]):
            raise RuntimeError(f'could not write {dest}')
        event['path'] = dest.name
    (frames_dir / 'events.json').write_text(json.dumps(events))


def _read_events(directory):
    return json.loads((Path(directory) / 'events.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--max-gap-s', type=float, default=.6)
    p.add_argument('--container', default=DEFAULT_CONTAINER)
    p.add_argument('--docker-context', default=DEFAULT_CONTEXT)
    p.add_argument('--_inside', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--_frames-dir', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.max_gap_s <= 0:
        p.error('--max-gap-s must be positive')
    if args._inside:
        _inside(args.archive, args._frames_dir, args.max_gap_s)
        return 0
    archive = args.archive.resolve()
    output = (args.output or archive / 'replay.mp4').resolve()
    ffmpeg = shutil.which('ffmpeg') or ('/opt/homebrew/bin/ffmpeg' if Path('/opt/homebrew/bin/ffmpeg').exists() else None)
    if not ffmpeg:
        p.error('ffmpeg executable not found')
    docker = shutil.which('docker')
    if not docker:
        p.error('docker executable not found; OpenCV renderer requires the existing sim container')
    repo = Path(__file__).resolve().parents[1]
    # Container mount is /workspace; temp composites are removed after encoding.
    try:
        relative = archive.relative_to(repo)
    except ValueError:
        p.error('archive must be inside the mounted repository')
    container_archive = Path('/workspace') / relative
    frames_dir = archive / '.render-flight-frames'
    if frames_dir.exists():
        p.error(f'temporary render directory already exists: {frames_dir}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.stem + '.rendering.mp4')
    try:
        subprocess.run([docker, '--context', args.docker_context, 'exec', args.container,
                        'python3', '/workspace/tools/render_flight_video.py',
                        str(container_archive), '--max-gap-s', str(args.max_gap_s),
                        '--_inside', '--_frames-dir', str(Path('/workspace') / relative / '.render-flight-frames')],
                       check=True)
        events = _read_events(frames_dir)
        if not events:
            raise RuntimeError('renderer produced no frames')
        # ffconcat durations are simulation-time deltas, giving VFR playback at
        # real sim time without duplicating frames at an arbitrary output rate.
        concat = frames_dir / 'replay.ffconcat'
        with concat.open('w') as f:
            f.write('ffconcat version 1.0\n')
            for event in events:
                f.write("file '" + str((frames_dir / event['path']).resolve()).replace("'", "'\\''") + "'\n")
                f.write(f"duration {float(event['duration']):.9f}\n")
            f.write("file '" + str((frames_dir / events[-1]['path']).resolve()).replace("'", "'\\''") + "'\n")
        duration = sum(float(e['duration']) for e in events)
        bitrate = max(120_000, min(1_200_000, int(MAX_OUTPUT_BYTES * 8 * .72 / max(duration, 1))))
        subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-safe', '0',
                        '-i', str(concat), '-fps_mode', 'vfr', '-an', '-c:v', 'libx264',
                        '-preset', 'veryfast', '-b:v', str(bitrate), '-maxrate', str(bitrate),
                        '-bufsize', str(2 * bitrate), '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(temp_output)], check=True)
        if temp_output.stat().st_size > MAX_OUTPUT_BYTES:
            temp_output.unlink(missing_ok=True)
            raise RuntimeError('encoded replay exceeded 20 MiB; reduce bitrate or shorten archive')
        os.replace(temp_output, output)
        print(f'Wrote {output} ({output.stat().st_size} bytes, {len(events)} frames, {duration:.3f} sim s)')
        return 0
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        temp_output.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
