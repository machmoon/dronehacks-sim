"""Attach separate evidence recorders to a marker-free SITL mission."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def stop(process, timeout=12):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def main():
    if os.environ.get('DRONEHACKS_SITL') != '1':
        raise RuntimeError('this runner is for the isolated simulator only')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-timeout', type=float, default=1800,
                        help='mission wall-clock budget; allows slow software-rendered simulation')
    parser.add_argument('mission_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 60 <= args.wall_timeout <= 3400:
        parser.error('wall-timeout must be between 60 and 3400 seconds')
    options = args.mission_args[1:] if args.mission_args[:1] == ['--'] else args.mission_args
    if any(x == '--output' or x.startswith('--output=') for x in options):
        parser.error('mission output is managed by the evidence runner')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('output directory must be new or empty')
    children, streams = [], []
    commands = [
        ('camera', [sys.executable, '-m', 'vision.flight_record', '--output', str(args.output/'camera'),
                    '--duration', str(args.wall_timeout+180), '--fps-sim', '5',
                    '--max-bytes', str(64*1024*1024)]),
        ('observer', [sys.executable, '-m', 'vision.flight_observe', '--output', str(args.output/'observer'),
                      '--duration', str(args.wall_timeout+180)]),
    ]
    code, failure = 1, None
    try:
        for name, command in commands:
            streams.append((args.output/(name+'.log')).open('x'))
            children.append(subprocess.Popen(command, stdout=streams[-1], stderr=subprocess.STDOUT))
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in children):
                raise RuntimeError('an evidence recorder exited before the mission')
            try:
                camera = json.loads((args.output/'camera/status.json').read_text())
                observer = json.loads((args.output/'observer/status.json').read_text())
            except (OSError, ValueError):
                time.sleep(.1)
                continue
            # Only readiness/counts are used here; observed pose never enters
            # the flight-control subprocess or its command arguments.
            if camera['frames'] and observer['truth_samples']:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('camera or evaluation recorder has no samples')
        streams.append((args.output/'mission.log').open('x'))
        children.append(subprocess.Popen([sys.executable, '-m', 'mission.markerfree_run',
                                         '--output', str(args.output/'telemetry.jsonl'), *options],
                                        stdout=streams[-1], stderr=subprocess.STDOUT))
        print('Independent camera and evaluation recorders ready; mission started.', flush=True)
        code = children[-1].wait(timeout=args.wall_timeout)
    except KeyboardInterrupt:
        code, failure = 130, 'operator_interrupt'
    except Exception as error:
        failure = str(error)
    finally:
        for index in range(len(children)-1, -1, -1):
            # Let the flight subprocess observe its abort landing while both
            # independent recorders remain alive.
            stop(children[index], timeout=150 if index == 2 else 12)
        for stream in streams:
            stream.close()
        report = {'mission_exit_code': code, 'runner_error': failure,
                  'process_exit_codes': [p.returncode for p in children],
                  'truth_used_for_navigation': False,
                  'note': 'Read independent scoring output before making a completion or collision claim.'}
        (args.output/'recording.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
