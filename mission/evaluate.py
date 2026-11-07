"""Run one SITL mission with camera and independent truth recording attached."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def main():
    if os.environ.get('DRONEHACKS_SITL') != '1':
        raise RuntimeError('This runner only operates inside the dedicated SITL container')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('mission_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    mission_args = args.mission_args
    if mission_args[:1] == ['--']:
        mission_args = mission_args[1:]
    if any(x == '--output' or x.startswith('--output=') for x in mission_args):
        parser.error('mission output is managed by this runner')
    args.output.mkdir(parents=True, exist_ok=False)
    commands = [
        [sys.executable, '-m', 'vision.record', '--output', str(args.output/'camera'),
         '--duration', '3600', '--fps', '5'],
        [sys.executable, '-m', 'vision.observe', '--output', str(args.output/'truth.jsonl'),
         '--duration', '3600'],
        [sys.executable, '-m', 'mission.px4_run', '--task', 'course',
         '--finish-action', 'land', '--exit-on-finish', '--output', str(args.output/'telemetry.jsonl'),
         *mission_args],
    ]
    children, streams = [], []
    interrupted = False
    try:
        for name, command in zip(('camera', 'truth', 'mission'), commands):
            stream = (args.output/(name+'.log')).open('x')
            streams.append(stream)
            children.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        code = children[-1].wait()
    except KeyboardInterrupt:
        interrupted = True
        code = 130
    finally:
        # Mission first: its own handler requests PX4 Land. Observers only read.
        for child in reversed(children):
            stop(child)
        for stream in streams:
            stream.close()
    summary_path = args.output/'telemetry.summary.json'
    result = json.loads(summary_path.read_text()) if summary_path.exists() else None
    recording = args.output/'camera/video.json'
    video = json.loads(recording.read_text()) if recording.exists() else None
    report = {'mission_exit_code': code, 'interrupted': interrupted,
              'mission': result, 'camera': video,
              'observer_exit_codes': [child.returncode for child in children[:2]],
              'truth_for_evaluation_only': True}
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return code if code else (0 if video and video['frames'] > 0 and
                              all(c.returncode == 0 for c in children[:2]) else 2)


if __name__ == '__main__':
    raise SystemExit(main())
