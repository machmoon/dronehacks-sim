"""Offline stereo perturbation replay with production survey and policy input gates."""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from learning.inference import PolicyError, policy_depth
from tools.score_markerfree import Truth
from vision.depth_odometry import (DepthFrame, DepthGates, DepthSurvey, DownwardRange, FalsSurveyNormals,
                                   InertialTilt, TrackingFailure, register_depth)
from vision.rgbd_odometry import Gates, RGBDOdometry, SURVEY_NORMAL_RADIUS_PX
from vision.rgbd_scene import RGBD_BODY_FROM_OPTICAL, RGBD_TRANSLATION


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BODY_FROM_OPTICAL = np.asarray(RGBD_BODY_FROM_OPTICAL)
MOUNT = np.asarray(RGBD_TRANSLATION)
SURVEY_PATH = ROOT/'models/course_v3/model.sdf'
DEPTH_GATES = DepthGates()
GATES = Gates()


def finite_returns(depth):
    return np.isfinite(depth) & (depth > 0.)


def quantize(depth, focal_baseline, params):
    valid = finite_returns(depth)
    step = params['quantization_disparity_step_px']
    metric = params['quantization_metric_step_m']
    disparity = np.rint(focal_baseline / depth[valid] / step)*step
    ranges = np.full(disparity.shape, np.nan)
    np.divide(focal_baseline, disparity, out=ranges, where=disparity > 0.)
    output = depth.copy()
    output[valid] = np.rint(ranges/metric)*metric
    return output


def occlusion_mask(depth, focal_baseline, difference):
    row, left = np.nonzero(finite_returns(depth))
    disparity = focal_baseline / depth[row, left]
    right = np.rint(left-disparity).astype(np.intp)
    in_image = (0 <= right) & (right < depth.shape[1])
    z_buffer = np.full(depth.shape, -np.inf, dtype=np.float32)
    np.maximum.at(z_buffer, (row[in_image], right[in_image]), disparity[in_image])
    unavailable = ~in_image
    unavailable[in_image] = z_buffer[row[in_image], right[in_image]] - disparity[in_image] > difference
    mask = np.zeros(depth.shape, dtype=bool)
    mask[row[unavailable], left[unavailable]] = True
    return mask


def finite_mismatch(depth, radius, distance):
    # Rank same-row candidates by increasing pixel distance, then lower column.
    width = depth.shape[1]
    columns = np.arange(width)
    valid = finite_returns(depth)
    changed = np.zeros(depth.shape, dtype=bool)
    output = depth.copy()
    for offset in range(1, radius+1):
        for delta in (-offset, offset):
            neighbor = np.clip(columns+delta, 0, width-1)
            farther = depth[:, neighbor]
            replace = valid & valid[:, neighbor] & ~changed & (farther >= depth+distance)
            output[replace] = farther[replace]
            changed |= replace
    return output


def transforms(depth, focal, params):
    scale = focal*params['baseline_m_assumed']
    mask = occlusion_mask(depth, scale, params['projection_occlusion_difference_px'])
    width = 1 + 2*params['horizontal_dilation_radius_px']
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((1, width), np.uint8)).astype(bool) & finite_returns(depth)
    marked = np.where(mask, np.float32(np.nan), depth)
    expanded = np.where(dilated, np.float32(np.nan), depth)
    wrong = finite_mismatch(depth, params['far_mismatch_search_radius_px'], params['far_mismatch_min_difference_m'])
    return {'control': depth, 'disparity_quantized': quantize(depth, scale, params),
            'geometric_occlusion': marked, 'occlusion_dilated': expanded, 'finite_far_edge_mismatch': wrong}, dilated


def policy_input(depth, base, limits):
    try:
        measured = policy_depth(depth)
    except PolicyError as error:
        return {'admitted': False, 'reason': str(error)}
    result = dict(admitted=True, reason=None, coverage=measured.coverage, center_coverage=measured.center_coverage)
    if base is not None:
        near = base >= 1.-limits['policy_near_z_ray_m']/4.
        clear = measured.depth <= 1.-limits['policy_clear_z_ray_m']/4.
        rows, cols = near.shape
        danger_to_clear = near & clear
        result.update(near_to_far_bins=int(danger_to_clear.sum()),
                      near_to_far_center_bins=int(danger_to_clear[rows//4:3*rows//4, cols//4:3*cols//4].sum()),
                      near_to_max_range_bins=int(np.count_nonzero(near & (measured.depth == 0.))))
    return result


def body_position(pose):
    return pose[:3, 3] - pose[:3, :3]@BODY_FROM_OPTICAL.T@MOUNT


def fit(source, frame, estimator, survey, east):
    initial = np.array(source['pose'], dtype=float)
    initial[0, 3] += east
    dt = (source['capture_usec']-source['prior_usec'])*1e-6
    age = (source['now_usec']-source['capture_usec'])*1e-6
    downward_range = source['tof']
    if downward_range is not None:
        downward_range = DownwardRange(**{**downward_range, 'origin': np.asarray(downward_range['origin']),
                                          'direction': np.asarray(downward_range['direction'])})
    tilt = source['tilt']
    if tilt is not None:
        tilt = InertialTilt(**{**tilt, 'up_optical': np.asarray(tilt['up_optical'])})
    if tilt is not None and tilt.capture_usec != source['capture_usec']:
        tilt = None
    started = time.perf_counter()
    pose = None
    try:
        pose, details = register_depth(frame, survey, initial, DEPTH_GATES, survey=True,
                                       downward_range=downward_range, inertial_tilt=tilt)
        fit_reason = None
    except TrackingFailure as error:
        fit_reason, details = str(error), error.details
    registration_wall_ms = (time.perf_counter()-started)*1e3
    gate_reason = fit_reason
    if not 0. <= age <= GATES.max_age_s:
        gate_reason = 'stale_capture'
    elif dt <= 0.:
        gate_reason = 'nonmonotonic_capture'
    elif dt > GATES.max_gap_s:
        gate_reason = 'capture_gap'
    elif pose is not None:
        estimator.world_from_optical = initial
        try:
            estimator._check_motion(pose, dt)
            if max(np.maximum(.018, details['translation_std'])) > .20 or max(np.maximum(.013, details['rotation_std'])) > .14:
                raise TrackingFailure('accumulated_odometry_uncertainty')
        except TrackingFailure as error:
            gate_reason = str(error)
    row = dict(east_prior_offset_m=east, capture_gap_s=dt, capture_processing_age_s=age,
               register_depth_accepted=pose is not None, registration_reason=fit_reason,
               survey_runtime_gates_accepted=gate_reason is None, runtime_gate_reason=gate_reason,
               source_samples=len(frame.sample_points), registration_wall_ms=registration_wall_ms,
               depth_matches=details.get('depth_matches'), depth_overlap=details.get('depth_overlap'),
               depth_observability=details.get('depth_observability'), rms_m=details.get('depth_plane_rms_m'),
               tof_range_residual_m=details.get('tof_range_residual_m'),
               px4_tilt_prior_status=details.get('px4_tilt_prior_status'),
               position_enu_m=body_position(pose).tolist() if pose is not None else None)
    if 'scaled_step_hessian' in details:
        eigenvalues, eigenvectors = np.linalg.eigh(details['scaled_step_hessian'])
        row.update(rejected_hessian_eigenvalues=eigenvalues.tolist(),
                   rejected_weakest_eigenvector_scaled_enu=eigenvectors[:, 0].tolist())
    return row


def load_depth(source):
    path = ROOT/source['path']
    if source['kind'] == 'archive_png':
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise ValueError(f'invalid archive depth PNG {path}')
        metric = depth.astype(np.float32)/1000.
        # The archive uses zero for any non-finite source ray and cannot
        # recover simulator near/far sentinel polarity at those pixels.
        metric[depth == 0] = np.nan
        return metric
    with np.load(path, allow_pickle=False) as archive:
        return archive['depth_z_m' if source['kind'] == 'native_npz' else 'noisy']


def load_truth(run):
    path = ROOT/'artifacts/markerfree'/run/'observer/truth.jsonl.gz'
    with gzip.open(path, 'rt') as archive:
        truth = Truth([json.loads(line) for line in archive], .1)
    return truth, path


def score_truth(entries, limits):
    truth06, path06 = load_truth('policy-06')
    truth07, path07 = load_truth('policy-07')
    alias_truth = None
    for entry in entries:
        source = entry['source']
        if source.get('excluded'):
            continue
        if source['kind'] == 'synthetic_npz':
            with np.load(ROOT/source['path'], allow_pickle=False) as archive:
                alias_truth = archive['truth_evaluation_only']
            evaluation = alias_truth
        else:
            truth = truth06 if source['kind']=='native_npz' else truth07
            sample = truth.at(source['capture_usec']*1e-6)
            evaluation = sample[0] if sample is not None else None
        entry['body_position_evaluation_only_m'] = evaluation.tolist() if evaluation is not None else None
        for case in entry['cases']:
            for row in case['fits']:
                if row['position_enu_m'] is None or evaluation is None:
                    row.update(rawfit_error_m=None, rawfit_horizontal_error_m=None, accepted_pose_error_m=None,
                               accepted_pose_warning_gt_30mm=None, accepted_pose_counterexample_gt_60mm=None)
                    continue
                delta = np.asarray(row['position_enu_m'])-evaluation
                error = float(np.linalg.norm(delta))
                accepted = row['survey_runtime_gates_accepted']
                row.update(rawfit_error_m=error, rawfit_horizontal_error_m=float(np.linalg.norm(delta[:2])),
                           accepted_pose_error_m=error if accepted else None,
                           accepted_pose_warning_gt_30mm=error > limits['accepted_pose_warning_gt_m'] if accepted else None,
                           accepted_pose_counterexample_gt_60mm=error > limits['accepted_pose_counterexample_gt_m'] if accepted else None)
    return path06, path07


def summarize(entries):
    cases = [case for entry in entries for case in entry['cases']]
    summary = {}
    for mode in dict.fromkeys(case['mode'] for case in cases):
        selected = [case for case in cases if case['mode'] == mode]
        rows = [row for case in selected for row in case['fits']]
        errors = [row['accepted_pose_error_m'] for row in rows if row['accepted_pose_error_m'] is not None]
        summary[mode] = dict(fit_attempts=len(rows), registered=sum(r['register_depth_accepted'] for r in rows),
                             survey_runtime_accepted=sum(r['survey_runtime_gates_accepted'] for r in rows),
                             survey_runtime_rejections=dict(Counter(r['runtime_gate_reason'] for r in rows if r['runtime_gate_reason'])),
                             accepted_with_truth=len(errors), accepted_pose_warnings=sum(r['accepted_pose_warning_gt_30mm'] is True for r in rows),
                             accepted_pose_counterexamples=sum(r['accepted_pose_counterexample_gt_60mm'] is True for r in rows),
                             maximum_accepted_pose_error_m=max(errors) if errors else None,
                             policy_images_admitted=sum(c['policy']['admitted'] for c in selected),
                             policy_near_to_far_bins=sum(c['policy'].get('near_to_far_bins',0) for c in selected))
    return summary


def main():
    cv2.setNumThreads(1)
    protocol_path = HERE/'protocol.json'
    sources_path = HERE/'sources.json'
    protocol = json.loads(protocol_path.read_text())
    prepared = json.loads(sources_path.read_text())
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if digest != prepared['protocol_sha256']:
        raise ValueError('preregistered protocol changed after source preparation')
    for relative, expected in prepared['source_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'prepared source hash changed: {relative}')
    assert (round(GATES.max_gap_s*1e6) == protocol['estimator']['max_inter_capture_gap_usec']
            and round(GATES.max_age_s*1e6) == protocol['estimator']['max_processing_age_usec'])
    survey = DepthSurvey.from_course(SURVEY_PATH)
    entries = []
    for source in prepared['sources']:
        entry = dict(source=source, cases=[])
        entries.append(entry)
        if source.get('excluded'):
            continue
        intrinsics = np.asarray(source['intrinsics'])
        estimator = RGBDOdometry((0., 0., 0.), intrinsics=intrinsics, survey=survey, survey_normals='fals7')
        fals = FalsSurveyNormals(intrinsics)
        raw = load_depth(source)
        if raw.shape != (256,384) or raw.dtype != np.float32:
            raise ValueError(f"unexpected depth shape or format for {source['id']}")
        depths, dilated = transforms(raw, intrinsics[0,0], protocol['stereo_diagnostic'])
        try:
            actor_base = policy_depth(raw).depth
        except PolicyError:
            actor_base = None
        for mode in protocol['matrix']:
            modified = depths[mode]
            valid_before, valid_after = finite_returns(raw), finite_returns(modified)
            frame = DepthFrame(modified, intrinsics, DEPTH_GATES, stride=estimator.survey_normals['sample_stride_px'],
                               normal_radius_px=SURVEY_NORMAL_RADIUS_PX, survey_fals=fals)
            fits = [fit(source, frame, estimator, survey, offset) for offset in protocol['source_selection']['prior_east_offsets_m']]
            entry['cases'].append(dict(mode=mode, fits=fits, policy=policy_input(modified, actor_base, protocol['metrics']),
                                       finite_returns_before=int(valid_before.sum()), finite_returns_after=int(valid_after.sum()),
                                       newly_invalid_pixels=int(np.count_nonzero(valid_before & ~valid_after)),
                                       changed_finite_pixels=int(np.count_nonzero(valid_before & valid_after & (raw != modified)))))
        wrong_policy = np.where(dilated, np.float32(np.inf), raw)
        entry['policy_negative_control'] = dict(replaced_hole_pixels=int(dilated.sum()),
                                                **policy_input(wrong_policy, actor_base, protocol['metrics']))
    truth_paths = score_truth(entries, protocol['metrics'])
    files = [Path(__file__), sources_path, SURVEY_PATH, *truth_paths,
             ROOT/'vision/depth_odometry.py', ROOT/'vision/rgbd_odometry.py', ROOT/'learning/inference.py']
    files += sorted((SURVEY_PATH.parent/'meshes').glob('*.obj'))
    report = dict(protocol_sha256=digest, opencv_version=cv2.__version__, numpy_version=np.__version__,
                  survey_normal_config=estimator.survey_normals, depth_gates=asdict(DEPTH_GATES),
                  vision_gates=asdict(GATES), sources=entries, summary=summarize(entries),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with (HERE/'results.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
