#!/usr/bin/env python3
"""Launch one bounded, local learning experiment and retain its process metadata."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'artifacts' / 'learning'


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def supervise(run_dir, python, seconds, options):
    """A separate process enforces the wall-clock cap even if training hangs."""
    started = time.time()
    started_monotonic = time.monotonic()
    command = [python, '-u', '-m', 'learning.train', '--output', str(run_dir),
               '--max-seconds', str(seconds), *options]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONUNBUFFERED='1')
    # The trainer owns its fresh output directory; logs live alongside it.
    with run_dir.with_suffix('.log').open('a', buffering=1) as log:
        child = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '/usr/bin/nice', '-n', '10',
                                  *command], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info = {'run_directory': run_dir.name, 'supervisor_pid': os.getpid(),
                'process_group': child.pid, 'started_at': utc(started),
                'deadline_at': utc(started + seconds), 'max_seconds': seconds,
                'command': command, 'log': str(run_dir.with_suffix('.log')),
                'hardware': 'local Mac; no cloud or hardware flight'}
        write_json(RUNS / 'current.json', info)
        stopped = [False]
        signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
        reason = None
        while child.poll() is None:
            if stopped[0]:
                reason = 'user_stop'
                break
            if time.time() >= started + seconds or time.monotonic() >= started_monotonic + seconds:
                reason = 'wall_clock_cap'
                break
            available = os.statvfs(RUNS).f_bavail * os.statvfs(RUNS).f_frsize
            if available < 256 * 1024 * 1024:
                reason = 'low_disk_space'
                break
            time.sleep(1)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        info.update(finished_at=utc(time.time()), exit_code=child.returncode,
                    stop_reason=reason or 'trainer_exited')
        write_json(run_dir.with_suffix('.supervisor.json'), info)
        write_json(RUNS / 'current.json', info)
        return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=18000)
    parser.add_argument('--python', default=os.environ.get('SIM_TRAIN_PYTHON',
                        str(ROOT.parent / 'research/sotapilot-env/bin/python')))
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--stop', action='store_true')
    args, options = parser.parse_known_args()
    if options[:1] == ['--']:
        options = options[1:]
    if not 1 <= args.seconds <= 18000:
        parser.error('seconds must be between 1 and 18000')
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.supervise:
        return supervise(args.supervise, args.python, args.seconds, options)
    with (RUNS / 'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current_file = RUNS / 'current.json'
        current = json.loads(current_file.read_text()) if current_file.exists() else {}
        pid = current.get('supervisor_pid')
        alive = False
        if pid and not current.get('finished_at'):
            try:
                command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
                alive = str(Path(__file__).resolve()) in command and '--supervise' in command
            except subprocess.CalledProcessError:
                pass
        if args.stop:
            if alive:
                os.kill(pid, signal.SIGTERM)
                print('Stop requested; the trainer will save its checkpoint.')
            else:
                print('No running overnight experiment.')
            return 0
        if alive:
            raise SystemExit(f'A run is already active: {current["run_directory"]}')
        # Preserve a venv's python symlink path so its pyvenv.cfg is honored.
        executable = os.path.abspath(args.python) if Path(args.python).is_file() else args.python
        subprocess.run([executable, '-c', 'import torch,numpy; print("Torch",torch.__version__)'], check=True)
        run = RUNS / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
        with run.with_suffix('.launcher.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--supervise', str(run),
                                      '--python', executable, '--seconds', str(args.seconds), *options],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            if current_file.exists():
                current = json.loads(current_file.read_text())
                if current.get('supervisor_pid') == child.pid:
                    print(json.dumps(current, indent=2))
                    return 0
            if child.poll() is not None:
                raise SystemExit(f'Supervisor failed. Read {run.with_suffix(".launcher.log")}')
            time.sleep(.1)
        raise SystemExit('Supervisor startup timed out; inspect artifacts/learning before retrying.')


if __name__ == '__main__':
    sys.exit(main())
