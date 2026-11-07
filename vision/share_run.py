"""Render a recorded camera run beside its estimated flight path for review."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--score', type=Path, help='Optional matching offline geometry acceptance score')
    args = p.parse_args()
    rows = [json.loads(x) for x in args.telemetry.read_text().splitlines()]
    samples = [x for x in rows if x['type'] == 'sample']
    summary = next(x for x in reversed(rows) if x['type'] == 'summary')
    score = json.loads(args.score.read_text()) if args.score else None
    if score and (not score.get('run_id') or score['run_id'] != summary['run_id']):
        raise ValueError('score must belong to the recorded run')
    metadata = next(x for x in rows if x['type'] == 'metadata')
    stamps = [x['sim_time'] for x in samples]
    start = next(x['sim_time']-x['elapsed_sim_s'] for x in samples if x['elapsed_sim_s'] > 0)
    end = start+summary['elapsed_sim_s']
    frames = [json.loads(x) for x in (args.camera/'frames.jsonl').read_text().splitlines()]
    info = json.loads((args.camera/'video.json').read_text())
    cap = cv2.VideoCapture(str(args.camera/'camera.avi'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), info['playback_fps'], (960, 520))
    if not out.isOpened():
        raise RuntimeError('cannot open output video')
    count = 0

    def text(img, value, pos, scale=.55, color=(235, 235, 235)):
        cv2.putText(img, value, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def xy(position):
        return (int(510+(position[0]+3.048)/6.096*420),
                int(445-(position[1]+3.048)/6.096*370))

    try:
        for frame in frames:
            ok, camera = cap.read()
            if not ok:
                raise RuntimeError('recorded video ended before its timestamp index')
            stamp = frame['capture_sim_s']
            if not start <= stamp <= end+.25:
                continue
            index = max(0, bisect_right(stamps, stamp)-1)
            sample = samples[index]
            canvas = np.full((520, 960, 3), (24, 27, 30), np.uint8)
            canvas[80:440, :480] = cv2.resize(camera, (480, 360))
            text(canvas, 'DRONEHACKS / RECORDED PX4 SIMULATION', (16, 28), .75)
            text(canvas, 'Rendered camera with added markers', (16, 62))
            text(canvas, 'Estimated flight path (not an obstacle map)', (510, 62), .45)
            cv2.rectangle(canvas, (510, 75), (930, 445), (85, 85, 85), 1)
            route = [xy(w['position']) for w in metadata['waypoints']]
            cv2.polylines(canvas, [np.array(route)], False, (90, 90, 90), 1)
            for point in route:
                cv2.circle(canvas, point, 3, (120, 120, 120), -1)
            path = [xy(s['position']) for s in samples[:index+1] if s['elapsed_sim_s'] > 0]
            if len(path) > 1:
                cv2.polylines(canvas, [np.array(path)], False, (150, 220, 100), 2)
            cv2.circle(canvas, xy(sample['position']), 6, (90, 220, 255), -1)
            elapsed = min(stamp-start, summary['elapsed_sim_s'])
            text(canvas, f'{elapsed:05.1f} s   {sample["phase"]}   Waypoint {sample["waypoint_index"]}/{len(metadata["waypoints"])}', (16, 471), .6)
            if score and score['accepted']:
                caption = (f'{summary["elapsed_sim_s"]:.1f} s with landing | CAD sphere clearance: '
                           f'{100*score["sampled_clearance_m"]:.1f} cm min | SIMULATION ONLY')
            else:
                caption = f'Run result: {summary["elapsed_sim_s"]:.1f} s including landing/disarm. Collision score not verified.'
            text(canvas, caption, (16, 500), .51)
            out.write(canvas)
            count += 1
    finally:
        cap.release()
        out.release()
    if not count:
        raise RuntimeError('no flight frames found')
    print(json.dumps({'frames': count, 'output': str(args.output), 'result': summary,
                      'playback_note': 'Constant frame rate; displayed timer uses recorded simulation timestamps.'}))


if __name__ == '__main__':
    main()
