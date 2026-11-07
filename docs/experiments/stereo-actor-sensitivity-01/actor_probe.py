"""Prepare capture-aligned archived estimates and run paired CPU actor inference."""
import argparse
from bisect import bisect_right
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from learning.inference import policy_depth, policy_state, validate_observation, validate_velocity


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE/'protocol.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def bracket(topic, capture, max_gap):
    # PX4 streams expose publication timestamp / 1000 as time_boot_ms.
    t = (topic['timestamp'].astype(np.int64)//1000)*1000
    index = int(np.searchsorted(t, capture))
    if index < len(t) and t[index] == capture:
        return [index]
    if index == 0 or index == len(t) or t[index]-t[index-1] > max_gap:
        raise ValueError(f'no admissible ULog bracket at {capture}')
    return [index-1, index]


def topic_evidence(topic, indices):
    return [{'publication_usec': int(topic['timestamp'][i]),
             'mavlink_time_boot_usec': int(topic['timestamp'][i]//1000)*1000,
             'uorb_timestamp_sample_usec': int(topic['timestamp_sample'][i])} for i in indices]


def ordered_mavlink_samples(topic):
    stamps = topic['timestamp'].astype(np.int64)//1000
    delta = np.diff(stamps)
    if np.any(delta < 0):
        raise ValueError('regressing ULog state publication timestamp')
    # PolicyStateHistory retains the first value for each boot timestamp.
    keep = np.r_[True, delta > 0]
    return {key: values[keep] for key, values in topic.items()}, int((~keep).sum())


def attitude_quaternion_rpy(topic, index):
    q = [float(topic[f'q[{i}]'][index]) for i in range(4)]
    if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > .001:
        raise ValueError('invalid logged PX4 quaternion')
    w, x, y, z = q
    return np.asarray([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                       math.asin(max(-1., min(1., 2*(w*y-z*x)))),
                       math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))], dtype=np.float32)


def prepare(protocol):
    import gzip
    import runpy
    from types import SimpleNamespace

    import cv2
    from pyulog.core import ULog
    from mission.state_history import MAX_BRACKET_USEC, PolicyStateHistory

    if MAX_BRACKET_USEC != protocol['state']['bracket_max_usec']:
        raise ValueError('the production bracket gate differs from the preregistered limit')
    source_dir = ROOT/protocol['input_experiment']
    stereo_protocol_path, stereo_report_path = (source_dir/name for name in ('protocol.json','results.json'))
    for path, field in ((stereo_protocol_path,'input_experiment_protocol_sha256'),
                        (stereo_report_path,'input_experiment_results_sha256')):
        if digest(path) != protocol[field]:
            raise ValueError(f'frozen stereo input changed: {path}')
    report, stereo = read(stereo_report_path), read(stereo_protocol_path)
    prepared_hashes = read(source_dir/'sources.json')['source_sha256']
    transforms = runpy.run_path(str(source_dir/'probe.py'))
    ulog_path = ROOT/protocol['ulog']
    if digest(ulog_path) != prepared_hashes[protocol['ulog']]:
        raise ValueError('the prepared stereo ULog source hash changed')
    telemetry_path = ROOT/protocol['archive']/'telemetry.jsonl'
    if digest(telemetry_path) != prepared_hashes[str(telemetry_path.relative_to(ROOT))]:
        raise ValueError('the prepared stereo telemetry source hash changed')
    with gzip.open(ulog_path, 'rb') as f:
        ulog = ULog(io.BytesIO(f.read()), ['vehicle_local_position','vehicle_attitude'])
    topics = {t.name:t.data for t in ulog.data_list}
    local, local_duplicates = ordered_mavlink_samples(topics['vehicle_local_position'])
    attitude, attitude_duplicates = ordered_mavlink_samples(topics['vehicle_attitude'])
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    meta = telemetry[0]
    if meta['policy_checkpoint_sha256'] != protocol['model_sha256'] or meta['policy_depth_mode'] != 'raw':
        raise ValueError('flight checkpoint or depth mode differs from protocol')
    samples = [row for row in telemetry if row['type']=='sample']
    times = [round(row['sim_time']*1e6) for row in samples]
    sources, all_depth, all_state = [], [], []
    images = [r for r in report['sources'] if r['source']['kind']=='archive_png']
    if [int(r['source']['id'].rsplit('-',1)[-1]) for r in images] != protocol['window_frame_ids']:
        raise ValueError('archive frame selection differs from protocol')
    inputs = [Path(__file__), PROTOCOL, stereo_protocol_path, stereo_report_path, source_dir/'probe.py',
              source_dir/'sources.json', ulog_path, telemetry_path, ROOT/'mission/state_history.py',
              ROOT/'learning/inference.py', ROOT/protocol['historical_training_depth_source']]
    for entry in images:
        s = entry['source']
        capture = s['capture_usec']
        left = bracket(local, capture, MAX_BRACKET_USEC)
        right = bracket(attitude, capture, MAX_BRACKET_USEC)
        if any(not all(local[key][i] for i in left) for key in ('xy_valid','z_valid','v_xy_valid','v_z_valid')):
            raise ValueError(f'position/velocity invalid at {capture}')
        for topic, indices, counters in ((local,left,('xy_reset_counter','z_reset_counter','vxy_reset_counter','vz_reset_counter')),
                                          (attitude,right,('quat_reset_counter',))):
            if any(topic[key][indices[0]] != topic[key][indices[-1]] for key in counters):
                raise ValueError(f'estimator reset within bracket at {capture}')
        history = PolicyStateHistory()
        for i in left:
            message = SimpleNamespace(time_boot_ms=int(local['timestamp'][i]//1000),
                                      **{k:float(local[k][i]) for k in ('x','y','z','vx','vy','vz')})
            history.record(message, None)
        for i in right:
            roll, pitch, yaw = map(float, attitude_quaternion_rpy(attitude,i))
            message = SimpleNamespace(time_boot_ms=int(attitude['timestamp'][i]//1000),roll=roll,pitch=pitch,yaw=yaw)
            history.record(None, message)
        estimate = history.at(capture)
        if estimate is None:
            raise ValueError(f'production state history could not align {capture}')
        i = bisect_right(times,capture)
        if i == 0 or i == len(times):
            raise ValueError(f'no telemetry waypoint bracket at {capture}')
        before, after = samples[i-1:i+1]
        waypoint_index = before['waypoint_index']
        waypoint = meta['waypoints'][waypoint_index]
        target = waypoint['position']
        if (before['phase'] != s['phase'] or after['phase'] != s['phase'] or waypoint['phase'] != s['phase']
                or after['waypoint_index'] != waypoint_index or before['target'] != target or after['target'] != target):
            raise ValueError(f'waypoint changed or disagreed at {capture}')
        speed_cap = waypoint['speed']*meta['speed_scale']
        state, _ = policy_state(estimate.velocity_enu, np.asarray(target)-estimate.position_enu,
                                *estimate.attitude_rpy_frd, speed_cap)
        raw = transforms['load_depth'](s)
        case = next(c for c in entry['cases'] if c['mode']=='finite_far_edge_mismatch')
        params = stereo['stereo_diagnostic']
        corrupt = transforms['finite_mismatch'](raw,params['far_mismatch_search_radius_px'],
                                                params['far_mismatch_min_difference_m'])
        changed = int(np.count_nonzero(np.isfinite(raw) & np.isfinite(corrupt) & (raw!=corrupt)))
        if changed != case['changed_finite_pixels']:
            raise ValueError(f'finite mismatch disagrees with frozen experiment at {capture}')
        pair = [policy_depth(image).depth for image in (raw,corrupt)]
        treatment = transforms['policy_input'](corrupt,pair[0],stereo['metrics'])
        if treatment != case['policy']:
            raise ValueError(f'policy input disagrees with frozen experiment at {capture}')
        path = ROOT/s['path']
        if digest(path) != prepared_hashes[s['path']]:
            raise ValueError(f'archive input hash changed at {capture}')
        inputs.append(path)
        all_depth.append(pair)
        all_state.append(state)
        sources.append(dict(frame_id=int(s['id'].rsplit('-',1)[-1]), capture_usec=capture, phase=s['phase'],
                            waypoint_index=waypoint_index, target_enu_m=target, speed_cap_m_s=speed_cap,
                            target_bracket_usec=times[i-1:i+1], local_bracket=topic_evidence(local,left),
                            attitude_bracket=topic_evidence(attitude,right), position_enu_m=estimate.position_enu,
                            velocity_enu_m_s=estimate.velocity_enu, attitude_frd_rpy_rad=estimate.attitude_rpy_frd,
                            state=state.tolist(), changed_finite_pixels=changed, policy_depth_delta=treatment))
    with (HERE/'inputs.npz').open('xb') as f:
        np.savez(f, depth=np.asarray(all_depth,dtype=np.float32),state=np.asarray(all_state,dtype=np.float32))
    inputs.append(HERE/'inputs.npz')
    write(HERE/'inputs.json', dict(protocol_sha256=digest(PROTOCOL), opencv=cv2.__version__, numpy=np.__version__,
                                  ulog_dropouts=len(ulog.dropouts), inputs=sources,
                                  duplicate_boot_stamps_discarded=dict(local=local_duplicates, attitude=attitude_duplicates),
                                  sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs}))


def infer(protocol):
    import torch
    from learning.inference import checkpoint_camera_forward
    from learning.policy import DepthPolicy

    report = read(HERE/'inputs.json')
    if report['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('the preregistered actor protocol changed after input preparation')
    for relative, value in report['sha256'].items():
        if digest(ROOT/relative) != value:
            raise ValueError(f'prepared actor source changed: {relative}')
    checkpoint = ROOT/protocol['model_file']
    if digest(checkpoint) != protocol['model_sha256']:
        raise ValueError('the packaged actor checkpoint hash differs from protocol')
    torch.set_num_threads(1)
    payload = torch.load(io.BytesIO(checkpoint.read_bytes()),map_location='cpu',weights_only=True)
    if (payload.get('format') != 1 or payload.get('architecture') != 'DepthPolicy'
            or payload.get('selection_split') != 'validation' or checkpoint_camera_forward(payload) != .34):
        raise ValueError('unexpected checkpoint observation contract')
    model = DepthPolicy().eval()
    model.load_state_dict(payload['model'],strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite packaged parameters')
    with np.load(HERE/'inputs.npz',allow_pickle=False) as data:
        depths, states = data['depth'],data['state']
    if len(depths) != len(states) or len(depths) != protocol['treatment']['number_of_pairs']:
        raise ValueError('all preregistered pairs must have estimated states')
    results = []
    for entry, pair, state in zip(report['inputs'],depths,states):
        values = []
        goal_direction = state[3:6]/np.linalg.norm(state[3:6])
        for raw in pair:
            depth, shared = validate_observation(raw,state)
            with torch.inference_mode():
                action = model(torch.from_numpy(depth)[None,None],torch.from_numpy(shared)[None])[0].numpy()
            values.append(np.asarray(validate_velocity(action, float(shared[-1]))))
        control, corrupt = values
        a, b = float(np.linalg.norm(control)),float(np.linalg.norm(corrupt))
        results.append(dict(**entry,control_velocity_flu_m_s=control.tolist(),finite_mismatch_velocity_flu_m_s=corrupt.tolist(),
                            delta_velocity_flu_m_s=(corrupt-control).tolist(),control_forward_m_s=float(control[0]),
                            finite_mismatch_forward_m_s=float(corrupt[0]),delta_forward_m_s=float(corrupt[0]-control[0]),
                            control_waypoint_closing_m_s=float(control@goal_direction),
                            finite_mismatch_waypoint_closing_m_s=float(corrupt@goal_direction),
                            delta_waypoint_closing_m_s=float((corrupt-control)@goal_direction),
                            control_speed_m_s=a,finite_mismatch_speed_m_s=b,delta_speed_m_s=b-a,
                            vector_delta_norm_m_s=float(np.linalg.norm(corrupt-control))))
    archive = ROOT/protocol['historical_training_depth_source']
    write(HERE/'results.json', dict(protocol_sha256=digest(PROTOCOL),checkpoint_sha256=digest(checkpoint),
                                   source_sha256={'learning/policy.py':digest(ROOT/'learning/policy.py'),
                                                  str(archive.relative_to(ROOT)):digest(archive),
                                                  str((HERE/'inputs.json').relative_to(ROOT)):digest(HERE/'inputs.json')},
                                   device='cpu',torch=torch.__version__,numpy=np.__version__,pairs=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','infer'))
    cmd = parser.parse_args().command
    spec = read(PROTOCOL)
    if cmd == 'prepare':
        try:
            prepare(spec)
        except ValueError as error:
            write(HERE/'pairing-failed.json',dict(protocol_sha256=digest(PROTOCOL),reason=str(error)))
            raise
    else:
        infer(spec)
