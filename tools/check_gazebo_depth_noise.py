"""Configure and verify the isolated Gazebo forward-depth noise diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from vision.rgbd_scene import DEPTH_TOPIC, RGBD_HEIGHT, RGBD_TOPIC, RGBD_WIDTH


SEED = 4417
MODES = {'control': 0., 'gaussian-10mm': .010}
RENDERER = 'ogre2'
DEFAULT_PATCH = (176, 64, 32, 32)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def configure_world(world_path, mode):
    """Patch only the generated forward camera; the control keeps exact bytes."""
    raw = world_path.read_bytes()
    root = ET.fromstring(raw)
    world = root.find('./world[@name="dronehacks"]')
    if world is None or world.findtext('./plugin[@filename="gz-sim-sensors-system"]/render_engine') != RENDERER:
        raise ValueError('expected the generated Ogre2 dronehacks world')
    sensors = world.findall('./model[@name="course_quad"]/link[@name="down_camera_link"]/sensor')
    if (len(sensors) != 1 or sensors[0].get('type') != 'rgbd_camera'
            or sensors[0].findtext('topic') != RGBD_TOPIC):
        raise ValueError('expected one generated forward RGBD sensor')
    camera = sensors[0].find('camera')
    if camera is None or camera.find('noise') is not None:
        raise ValueError('expected the unmodified forward camera')
    stddev = MODES[mode]
    configured = raw
    if stddev:
        noise = ET.SubElement(camera, 'noise')
        for key, value in (('type', 'gaussian'), ('mean', '0'), ('stddev', str(stddev))):
            ET.SubElement(noise, key).text = value
        ET.indent(root, space='  ')
        configured = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        world_path.write_bytes(configured)
    return dict(mode=mode, seed=SEED, configured_renderer=RENDERER, depth_topic=DEPTH_TOPIC,
                noise_mean_m=0., noise_stddev_m=stddev, world_path=str(world_path.resolve()),
                configuration_tool_sha256=sha256(Path(__file__).read_bytes()),
                source_world_sha256=sha256(raw), world_sha256=sha256(configured))


def load_capture(path, patch):
    import numpy as np

    x, y, width, height = patch
    if (min(x, y) < 0 or min(width, height) < 8
            or x+width > RGBD_WIDTH or y+height > RGBD_HEIGHT):
        raise ValueError('patch must lie within the forward image and cover at least 8 by 8 pixels')
    with np.load(path, allow_pickle=False) as capture:
        depth, stamps = capture['depth_z_m'], capture['stamp_ns']
    if (depth.ndim != 3 or depth.shape[1:] != (RGBD_HEIGHT, RGBD_WIDTH)
            or depth.dtype != np.float32 or not 12 <= len(depth) <= 48
            or stamps.shape != (len(depth),) or stamps.dtype.kind not in 'iu'
            or not np.all(stamps[1:] > stamps[:-1])):
        raise ValueError(f'expected 12–48 stamped float32 forward depth frames in {path}')
    return depth[:, y:y+height, x:x+width], stamps


def compare_plane(control, noisy):
    import numpy as np

    for frames in (control, noisy):
        if not np.all(np.isfinite(frames) & (frames > .10) & (frames < 4.)):
            raise ValueError('select a static plane patch with finite depths between 0.1 and 4 m')
    reference, actual = (frames.mean(axis=0, dtype=float) for frames in (control, noisy))
    y, x = np.indices(reference.shape)
    design = np.column_stack((x.ravel(), y.ravel(), np.ones(reference.size)))
    # Inverse optical depth is affine in image coordinates for a plane.
    inverse = design @ np.linalg.lstsq(design, 1/reference.ravel(), rcond=None)[0]
    plane_rms = float(np.sqrt(np.mean((reference.ravel()-1/inverse)**2)))
    control_sigma, noisy_sigma = (frames.std(axis=0, ddof=1, dtype=float) for frames in (control, noisy))
    off, on = (float(np.median(sigma)) for sigma in (control_sigma, noisy_sigma))
    added = float(np.sqrt(max(0., on**2-off**2)))
    off_p95 = float(np.quantile(control_sigma, .95))
    mean_shift_p95 = float(np.quantile(np.abs(actual-reference), .95))
    frame_mean_jitter = float(noisy.mean(axis=(1, 2), dtype=float).std(ddof=1))
    checks = dict(control_plane=plane_rms <= .003, control_stationary=off_p95 <= .002,
                  same_view=mean_shift_p95 <= .010, added_noise_in_range=.006 <= added <= .016,
                  not_global_frame_jitter=frame_mean_jitter <= .003)
    return dict(passed=all(checks.values()), checks=checks, control_plane_rms_m=plane_rms,
                control_temporal_sigma_p95_m=off_p95, control_temporal_sigma_median_m=off,
                noisy_temporal_sigma_median_m=on, added_temporal_sigma_m=added,
                mean_shift_p95_m=mean_shift_p95, noisy_frame_mean_jitter_m=frame_mean_jitter)


def verify(directory, patch):
    captures, provenance, sources = {}, {}, {}
    for mode, stddev in MODES.items():
        config = json.loads((directory/f'{mode}.config.json').read_text())
        audit = json.loads((directory/f'{mode}.audit.json').read_text())
        paths = {row['path']: row['sha256'] for row in audit['world_and_model_sources']}
        world_hash = paths.pop(config['world_path'], None)
        if (config.get('mode') != mode or config.get('seed') != SEED
                or config.get('noise_mean_m') != 0. or config.get('noise_stddev_m') != stddev
                or config.get('configured_renderer') != RENDERER or config.get('depth_topic') != DEPTH_TOPIC
                or world_hash is None or world_hash != config.get('world_sha256')
                or audit.get('truth_topic_publishers') != 0 or audit.get('artificial_landmark_models') != 0
                or audit.get('live_simulator_position_used') is not False):
            raise ValueError(f'{mode} provenance did not match the isolated noise configuration')
        capture = directory/f'{mode}.npz'
        captures[mode], stamps = load_capture(capture, patch)
        provenance[mode] = dict(config=config, capture_sha256=sha256(capture.read_bytes()),
                                frames=len(stamps), first_capture_ns=int(stamps[0]), last_capture_ns=int(stamps[-1]))
        sources[mode] = paths
    control, noisy = (provenance[mode]['config'] for mode in MODES)
    if (control['world_sha256'] != control['source_world_sha256']
            or control['world_sha256'] != noisy['source_world_sha256']
            or control['world_sha256'] == noisy['world_sha256']
            or sources['control'] != sources['gaussian-10mm']):
        raise ValueError('control and noisy world/model provenance differs beyond the injected world change')
    return dict(patch_xywh=patch, captures=provenance, model_sources=sources['control'],
                **compare_plane(captures['control'], captures['gaussian-10mm']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    config = sub.add_parser('configure')
    config.add_argument('mode', choices=MODES)
    config.add_argument('--world', type=Path, required=True)
    config.add_argument('--manifest', type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('directory', type=Path)
    check.add_argument('--patch', type=int, nargs=4, default=DEFAULT_PATCH, metavar=('X', 'Y', 'W', 'H'))
    args = parser.parse_args()
    try:
        if args.command == 'configure':
            row = configure_world(args.world, args.mode)
            args.manifest.write_text(json.dumps(row, indent=2)+'\n')
            print(SEED)
            return 0
        report = verify(args.directory, args.patch)
        with (args.directory/'verification.json').open('x') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['passed'] else 1
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
