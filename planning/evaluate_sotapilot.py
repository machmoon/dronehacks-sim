"""Exercise the pinned upstream SOTApilot on synthetic depth. No flight output."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

COMMIT = '51c4144a2ba33a7bba63e0cbe13bc3689ec27d9e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise ValueError('Upstream revision changed; review before running')
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('upstream_sotapilot', source/'extracted/drone_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pilot = module.DroneFlightController()
    results = []
    for name, metres in [('open_space', 10.0), ('wall_ahead', 1.0)]:
        pilot.reset()
        depth = np.full((128, 128, 1), (metres-.5)/19.5, dtype=np.float32)
        # State fields directly consumed by upstream source, with all other
        # features zeroed. This is a synthetic contract test, not sensor replay.
        state = np.zeros(141, dtype=np.float32)
        state[:3] = [0, 0, .95]
        state[-4] = .95/20
        state[-3:] = [2, 0, 0]
        times = []
        for _ in range(20):
            start = time.perf_counter()
            action = np.asarray(pilot.act({'state': state, 'depth': depth})).reshape(-1)
            times.append(1000*(time.perf_counter()-start))
        if action.shape != (5,) or not np.all(np.isfinite(action)):
            raise ValueError('upstream returned invalid action')
        nav = np.asarray(pilot.collipe_avoidance(depth, state, np.array([1, 0, 0], dtype=np.float32)))
        results.append({'case': name, 'synthetic_depth_m': metres,
                        'raw_action': action.tolist(), 'upstream_mode': pilot._mode,
                        'navigation_direction': nav.tolist(),
                        'median_inference_ms_excluding_first': statistics.median(times[1:]),
                        'interpreted_velocity_mps': (action[:3]*action[3]*3).tolist()})
    report = {'upstream': 'https://github.com/swarm-subnet/sotapilot', 'commit': revision,
              'files_sha256': {name: hashlib.sha256((source/'extracted'/name).read_bytes()).hexdigest()
                               for name in ('drone_agent.py', 'model.pt', 'navigation_model.pt')},
              'torch_version': torch.__version__, 'runtime': 'Mac CPU, one Torch thread',
              'evaluation': 'synthetic inference contract only', 'cases': results,
              'course_success_tested': False, 'real_sensor_tested': False,
              'flight_output_published': False,
              'limitations': ['Other state features are zero placeholders, not reconstructed benchmark observations.',
                              'Metric depth is synthetic; relative monocular depth is not interchangeable.',
                              'Upstream takeoff/goal detection and wide clearances are not configured for our ordered low-altitude course.',
                              'No hardware timing, collision guarantee, or benchmark success rate follows from this test.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
