"""Recompute the rank diagnostic from a preserved first-rejected capture."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
EXPECTED_CAPTURE_SHA256 = '87668f266e3eecee49b53c98432c4e2c969b73d82c2c67a79ad4be3be19e2827'


def npy_shape(path, key):
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError('expected NumPy format v1.0 array headers')
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        if dtype.hasobject:
            raise ValueError(f'{key} must not use pickle-backed object dtype')
        return shape


def analyze(capture):
    import numpy as np
    digest = hashlib.sha256(capture.read_bytes()).hexdigest()
    if digest != EXPECTED_CAPTURE_SHA256:
        raise ValueError('capture hash does not match the registered local artifact')
    with np.load(capture, allow_pickle=False) as archive:
        if set(archive.files) != {'rgb', 'depth_z_m', 'metadata_json'}:
            raise ValueError(f'unexpected NPZ entries: {archive.files}')
        rgb_shape, depth_shape = npy_shape(capture, 'rgb'), npy_shape(capture, 'depth_z_m')
        if rgb_shape != (256, 384, 3) or depth_shape != (256, 384):
            raise ValueError('unexpected registered capture dimensions')
        value = archive['metadata_json']
        if value.shape != () or value.dtype.kind not in 'US':
            raise ValueError('metadata_json must be a non-object scalar string')
        meta = json.loads(str(value.item()))
    fit = meta['rejected_fits']['survey']
    h = np.asarray(fit['scaled_step_hessian'], dtype=np.float64)
    if h.shape != (6, 6) or not np.isfinite(h).all():
        raise ValueError('scaled_step_hessian must be finite 6x6')
    symmetry = float(np.max(np.abs(h-h.T)))
    if symmetry > 1e-12:
        raise ValueError('scaled_step_hessian is not symmetric')
    eig = np.linalg.eigvalsh(h); y = np.array([0., 1., 0., 0., 0., 0.]); residual = h @ y
    source = HERE.parents[2]/'vision/depth_odometry.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != '173a8dbff1a64c90ac9f770a37425c65136cb3dfa2a546e0fdb845a8d6e50821':
        raise ValueError('depth registration source differs from recorded experiment')
    order = 'jac = np.column_stack((normal, np.cross(rotated[ids], normal)/2))'
    if order not in source.read_text():
        raise ValueError('depth_odometry Jacobian ordering changed')
    return dict(schema='dronehacks.net_visual_ablation.observability.v1',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture=dict(file='first-rejected.npz', sha256=digest, rgb_shape=list(rgb_shape),
                     depth_shape=list(depth_shape), metadata_pickle_deserialization=False),
        timing=dict(capture_sim_s=meta['capture_stamp_ns']/1e9, clock_sim_s=meta['clock_sim_usec']/1e6,
                    tof_capture_sim_s=meta['tof_stamp_ns']/1e9,
                    tilt_capture_sim_s=meta['px4_tilt_prior']['capture_usec']/1e6),
        fit=dict(reason=fit['reason'], depth_matches=fit['depth_matches'], depth_overlap=fit['depth_overlap'],
                 depth_observability_recorded=fit['depth_observability'], hessian_symmetric=True,
                 hessian_symmetry_max_abs=symmetry, hessian_eigenvalues=eig.tolist(),
                 eigenvalue_ratio=float(eig[0]/eig[-1]), enu_y_translation_probe=y.tolist(),
                 hessian_times_enu_y=residual.tolist(), enu_y_residual_norm=float(np.linalg.norm(residual))),
        auxiliary=dict(tof_range_m=meta['tof_range_m'], tof_range_residual_m=fit['tof_range_residual_m'],
                       tilt_status=meta['px4_tilt_pairing_status'], tilt_prior_status=fit['px4_tilt_prior_status'],
                       tilt_bracket_usec=fit['px4_tilt_bracket_usec'],
                       tilt_disagreement_rad=fit['px4_tilt_disagreement_rad']),
        source=dict(file='vision/depth_odometry.py', sha256=source_hash,
                    jacobian_order='[normal ENU translation xyz, cross(rotated optical point, normal)/2 rotation xyz]'),
        interpretation='This recorded linearized fit has an unconstrained ENU-Y translation direction. It does not identify the physical cause or establish failure of every future estimator fit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'first-rejected.npz')
    parser.add_argument('--output', type=Path, required=True, help='new JSON path; existing files are refused')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new path')
    report = analyze(args.input); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=str(args.output), sha256=report['capture']['sha256'],
                          eigenvalue_ratio=report['fit']['eigenvalue_ratio'],
                          enu_y_residual_norm=report['fit']['enu_y_residual_norm'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
