"""Prepare fixed stereo-stress replay inputs from native EV and camera archives."""
from bisect import bisect_right
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from pyulog.core import ULog

from vision.rgbd_odometry import TiltHistory, calibration, rotation_from_quaternion
from vision.rgbd_scene import RGBD_BODY_FROM_OPTICAL, RGBD_TRANSLATION


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENU_FROM_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
FLU_TO_FRD = np.diag([1, -1, -1])
BODY_FROM_OPTICAL = np.asarray(RGBD_BODY_FROM_OPTICAL)
MOUNT = np.asarray(RGBD_TRANSLATION)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def archive_pose(visual, index):
    position_ned = np.array([visual[f'position[{i}]'][index] for i in range(3)])
    q = [visual[f'q[{i}]'][index] for i in range(4)]
    rotation = ENU_FROM_NED@rotation_from_quaternion(q)@FLU_TO_FRD
    pose = np.eye(4)
    pose[:3, :3] = rotation@BODY_FROM_OPTICAL
    pose[:3, 3] = ENU_FROM_NED@position_ned + rotation@MOUNT
    return pose


def px4_tilt(attitude, stamp):
    times = attitude['timestamp']
    indices = range(np.searchsorted(times, stamp-60_000), np.searchsorted(times, stamp+60_001))
    history = TiltHistory()
    for index in indices:
        w, x, y, z = [float(attitude[f'q[{i}]'][index]) for i in range(4)]
        roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
        history.record(int(times[index]), roll, pitch)
    prior = history.at(stamp)
    if prior is None:
        return None
    return dict(capture_usec=prior.capture_usec, up_optical=prior.up_optical.tolist(),
                std_rad=prior.std_rad, bracket_usec=prior.bracket_usec)


def main():
    protocol_path = HERE/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    selection = protocol['source_selection']
    native_path = ROOT/selection['native']
    alias_path = ROOT/selection['synthetic_alias']
    with np.load(native_path, allow_pickle=False) as source:
        capture = json.loads(source['metadata_json'].item())
    native_usec = capture['capture_stamp_ns']//1000
    previous_usec = capture['previous_capture_sim_usec']
    max_gap = protocol['estimator']['max_inter_capture_gap_usec']
    if not 0 < native_usec-previous_usec <= max_gap:
        raise ValueError('the native source does not have the preregistered preceding EV gap')
    native_tof = dict(distance_m=capture['tof_range_m'], origin=capture['tof_origin_optical_m'],
                      direction=capture['tof_ray_optical'], std_m=capture['tof_range_std_m'])
    native = dict(id='policy-06-native', phase='MAZE', kind='native_npz', path=selection['native'],
                  capture_usec=native_usec, prior_usec=previous_usec, now_usec=capture['clock_sim_usec'],
                  pose=capture['previous_world_from_optical'], intrinsics=capture['camera_intrinsics'],
                  tof=native_tof, tilt=capture['px4_tilt_prior'],
                  prior_source='previous accepted RGBD pose saved before rejected capture')
    pose = np.asarray(native['pose'], dtype=float)
    body_rotation = pose[:3, :3]@BODY_FROM_OPTICAL.T
    body_position = pose[:3, 3]-body_rotation@MOUNT
    prior_yaw = math.atan2(body_rotation[1, 0], body_rotation[0, 0])
    synthetic_body_rotation = yaw_matrix(math.radians(69.)-prior_yaw)@body_rotation
    alias_pose = np.eye(4)
    alias_pose[:3, :3] = synthetic_body_rotation@BODY_FROM_OPTICAL
    alias_pose[:3, 3] = body_position+synthetic_body_rotation@MOUNT
    alias = dict(native, id='synthetic-alias-69deg', kind='synthetic_npz', path=selection['synthetic_alias'],
                 now_usec=native_usec, pose=alias_pose.tolist(),
                 prior_source='native EV position and tilt, with fixed synthetic heading of 69 degrees')
    sources = [native, alias]
    window = ROOT/selection['window_run']
    ulog_path = ROOT/selection['window_ulog_gzip']
    telemetry_path = window/'telemetry.jsonl'
    camera_path = window/'camera/frames.jsonl'
    telemetry = rows(telemetry_path)
    samples = [r for r in telemetry if r['type']=='sample']
    sample_usec = [round(r['sim_time']*1e6) for r in samples]
    window_times = [t for r,t in zip(samples,sample_usec) if r['phase'].startswith('WINDOW_')]
    bounds = [window_times[0], window_times[-1]]
    archive = [r for r in rows(camera_path) if bounds[0] <= r['capture_ns']//1000 <= bounds[-1]]
    selected = [archive[(2*i+1)*len(archive)//12] for i in range(6)]
    if (bounds != selection['window_bounds_sim_usec'] or len(archive) != selection['window_candidate_count']
            or [r['frame'] for r in selected] != selection['window_archive_frame_indices']):
        raise ValueError('selected archive does not match frozen sample frame IDs')
    with gzip.open(ulog_path, 'rb') as compressed:
        ulog = ULog(io.BytesIO(compressed.read()), ['vehicle_visual_odometry', 'vehicle_attitude', 'actuator_armed'])
    topics = {item.name: item.data for item in ulog.data_list}
    ev, att, armed = (topics[name] for name in ('vehicle_visual_odometry', 'vehicle_attitude', 'actuator_armed'))
    ev_raw = ev['timestamp_sample'].astype(np.int64)
    ev_grid = ((ev_raw+2_000)//4_000)*4_000
    arm_usec = int(armed['timestamp'][np.flatnonzero(armed['armed'])[0]])
    arm_event = next(round(r['sim_time']*1e6) for r in telemetry if
                     r.get('message') == 'PX4 armed in Offboard with marker-free RGBD aiding')
    last_telemetry_vision = max(round((r.get('vision_stamp_sim_s') or 0)*1e6) for r in samples)
    if not (0 <= arm_event-arm_usec <= 1_200_000 and abs(ev_raw[-1]-last_telemetry_vision) <= 150_000):
        raise ValueError('selected PX4 log does not match policy-07 arm and EV timeline')
    inputs = [Path(__file__), protocol_path, native_path, alias_path, ulog_path, telemetry_path, camera_path]
    for frame in selected:
        stamp = frame['capture_ns']//1000
        phase = samples[bisect_right(sample_usec, stamp)-1]['phase']
        paths = {name: window/'camera'/frame[name] for name in ('rgb', 'depth_z_mm_u16')}
        inputs.extend(paths.values())
        source = dict(id=f"policy-07-window-{frame['frame']}", phase=phase, kind='archive_png',
                      path=str(paths['depth_z_mm_u16'].relative_to(ROOT)), rgb=str(paths['rgb'].relative_to(ROOT)),
                      capture_usec=stamp, now_usec=stamp+protocol['estimator']['archive_processing_age_usec'],
                      intrinsics=calibration().tolist(), prior_source='PX4 ULog vehicle_visual_odometry NED/FRD')
        sources.append(source)
        candidates = np.flatnonzero(ev_grid < stamp)
        if not len(candidates):
            source['excluded'] = 'no preceding native EV entry'
            continue
        index = int(candidates[np.argmax(ev_grid[candidates])])
        prior = int(ev_grid[index])
        source.update(prior_usec=prior, prior_ulog_timestamp_sample_usec=int(ev_raw[index]),
                      prior_ulog_receive_usec=int(ev['timestamp'][index]),
                      prior_clock_grid_correction_usec=int(prior-ev_raw[index]))
        if not (ev['pose_frame'][index] == 1 and 0 < stamp-prior <= max_gap
                and abs(prior-ev_raw[index]) <= 1_000):
            source['excluded'] = 'preceding EV entry fails native stamp, frame or gap check'
            continue
        pose = archive_pose(ev, index)
        if not np.isfinite(pose).all():
            source['excluded'] = 'invalid preceding native EV pose'
            continue
        tof = frame['diagnostics']['tof']
        source.update(pose=pose.tolist(), tilt=px4_tilt(att, stamp),
                      tof=dict(native_tof, distance_m=tof['range_m'],
                               std_m=math.hypot(.012, tof['signed_age_ms']*.001)) if tof['status']=='paired' else None)
    report = dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), sources=sources,
                  policy07_ulog_arm_usec=arm_usec, policy07_ulog_last_vision_sample_usec=int(ev_raw[-1]),
                  policy07_ulog_dropouts=len(ulog.dropouts),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    with (HERE/'sources.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
