"""Reproduce the offline x500 window view sweep with the production depth gate.

Run with python3 -m tools.check_window_observability --output window-check.json.
The diagnostic runs offline using NumPy, OpenCV and SciPy.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import scipy
from scipy.spatial import ConvexHull

from mission.heading import route_yaw
from vision.depth_odometry import (DepthFrame, DepthGates, DepthSurvey, DownwardRange,
                                   InertialTilt, TrackingFailure, _mesh_triangles,
                                   register_depth)
from vision.rgbd_odometry import calibration
from vision.rgbd_scene import (RGBD_BODY_FROM_OPTICAL, RGBD_HEIGHT, RGBD_TRANSLATION,
                               RGBD_WIDTH, TOF_TRANSLATION)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE = ROOT / 'mission/course-window-x500.json'
MODEL = ROOT / 'models/course_v3/model.sdf'
LOOKAHEAD_M = 1.2
YAW_OFFSETS_DEG = (-7, 0, 7)
TURN_DISTANCES_M = (-.06, 0., .10, .20, .30)
TILT_STD_RAD = .01003
TILT_BRACKET_USEC = 32_000
SEED_POSITION_M = np.array([.025, -.025, .015])
SEED_ROTATION_DEG = np.array([1., 1., 4.])
BOULDER_DIRECTIONS = 96
GATES = DepthGates()


def _clip_near(triangle):
    clipped = []
    start = triangle[-1]
    for end in triangle:
        if (start[2] >= .035) != (end[2] >= .035):
            clipped.append(start + (end-start)*(.035-start[2])/(end[2]-start[2]))
        if end[2] >= .035:
            clipped.append(end)
        start = end
    return np.asarray(clipped)


class StaticDepth:
    """Approximate optical-Z raster with course triangles and decimated rocks."""

    def __init__(self, mesh_paths, intrinsics):
        triangles = []
        boulders = []
        draw_prefixes = ('a1', 'barrier', 'c1', 'c2', 'm1', 'm2', 'm3', 'tube', 'net_side', 'net_roof')
        indices = np.arange(BOULDER_DIRECTIONS, dtype=float)
        theta = math.pi*(3-np.sqrt(5))*indices
        z = 1-(indices+.5)/(BOULDER_DIRECTIONS/2)
        r = np.sqrt(1-z*z)
        directions = np.column_stack((r*np.cos(theta), r*np.sin(theta), z))
        for path in mesh_paths:
            if path.stem.startswith('boulder_'):
                points = np.unique(_mesh_triangles(path).reshape(-1, 3), axis=0)
                subset = points[np.unique(np.argmax(points@directions.T, axis=0))]
                boulders.append(subset[ConvexHull(subset).simplices])
            elif path.stem.startswith(draw_prefixes):
                triangles.append(_mesh_triangles(path))
        self.triangles = np.concatenate([*triangles, *boulders])
        self.intrinsics = intrinsics
        u, v = np.meshgrid(np.arange(RGBD_WIDTH), np.arange(RGBD_HEIGHT))
        self.rays = np.stack(((u-intrinsics[0, 2])/intrinsics[0, 0],
                              (v-intrinsics[1, 2])/intrinsics[1, 1],
                              np.ones_like(u)), axis=-1)

    def render(self, pose):
        rotation, position = pose[:3, :3], pose[:3, 3]
        triangles = (self.triangles-position) @ rotation
        vertical = np.sum(self.rays*rotation[2], axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = np.where(vertical < 0, -position[2]/vertical, np.inf)
        x, y, z = triangles.transpose(2, 0, 1)
        visible = ((z.max(axis=1) > .035) & ((z+x).max(axis=1) > 0) & ((z-x).max(axis=1) > 0)
                   & ((.67*z+y).max(axis=1) > 0) & ((.67*z-y).max(axis=1) > 0))
        for triangle in triangles[visible]:
            polygon = triangle if triangle[:, 2].min() > .035 else _clip_near(triangle)
            if len(polygon) < 3:
                continue
            uv = polygon @ self.intrinsics.T
            uv = uv[:, :2]/uv[:, 2:]
            u0, v0 = np.maximum(0, np.floor(uv.min(axis=0)).astype(int))
            u1, v1 = np.minimum((RGBD_WIDTH-1, RGBD_HEIGHT-1), np.ceil(uv.max(axis=0)).astype(int))
            if u0 > u1 or v0 > v1:
                continue
            mask = np.zeros((v1-v0+1, u1-u0+1), np.uint8)
            cv2.fillConvexPoly(mask, np.rint((uv-[u0, v0])*256).astype(np.int32), 1, shift=8)
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            v, u = np.nonzero(mask)
            v, u = v+v0, u+u0
            with np.errstate(divide='ignore', invalid='ignore'):
                distance = (triangle[0]@normal)/np.sum(self.rays[v, u]*normal, axis=1)
            depth[v, u] = np.minimum(depth[v, u], np.where(distance > .034, distance, np.inf))
        return depth.astype(np.float32)


@dataclass(frozen=True)
class View:
    position: np.ndarray
    yaw: float
    details: dict
    seed_sign: int = 0


def _heading(position, waypoints, index):
    following = waypoints[index+1]['position'] if index+1 < len(waypoints) else None
    return route_yaw(position, waypoints[index]['position'], following,
                     waypoints[index]['phase'], LOOKAHEAD_M)


def _view_sets(waypoints):
    phases = ('WINDOW_APPROACH', 'WINDOW_ALIGN', 'WINDOW_CROSS', 'LANDING_APPROACH')
    first = next((i for i, waypoint in enumerate(waypoints) if waypoint['phase'] == phases[0]), None)
    if first is None or first == 0 or tuple(w['phase'] for w in waypoints[first:first+4]) != phases:
        raise ValueError('the route needs four consecutive window approach/align/cross/landing phases')
    align, cross = first+1, first+2
    entry = np.asarray(waypoints[align]['position'])
    radius = float(waypoints[align]['arrival_radius'])
    continuity = {
        'exact_entry_align_deg': math.degrees(_heading(entry, waypoints, align)),
        'exact_entry_cross_deg': math.degrees(_heading(entry, waypoints, cross)),
        'max_entry_disk_step_deg': max(
            abs(math.degrees(math.remainder(_heading(p, waypoints, align)-_heading(p, waypoints, cross),
                                            2*math.pi)))
            for r in np.linspace(0., radius, 7)
            for angle in np.linspace(0., 2*math.pi, 721)
            for p in [entry+r*np.array([math.cos(angle), math.sin(angle), 0.])]
        ),
        'entry_arrival_radius_m': radius,
        'nominal_handoff_steps_deg': {},
    }
    groups = {key: [] for key in ('nominal', 'swept_turns', 'entry_capture', 'seed_perturbations')}
    held_heading = None
    for i in range(first, first+4):
        start = np.asarray(waypoints[i-1]['position'])
        goal = np.asarray(waypoints[i]['position'])
        phase = waypoints[i]['phase']
        for fraction in np.linspace(0., 1., 11):
            position = start+fraction*(goal-start)
            heading = _heading(position, waypoints, i)
            heading = held_heading if heading is None else heading
            if fraction == 0. and held_heading is not None:
                turn = math.remainder(heading-held_heading, 2*math.pi)
                continuity['nominal_handoff_steps_deg'][phase] = abs(math.degrees(turn))
            held_heading = heading
            for delta in YAW_OFFSETS_DEG:
                groups['nominal'].append(View(position, heading+math.radians(delta),
                                              dict(phase=phase, fraction=float(fraction), yaw_offset_deg=delta)))
    for i in (align, cross+1):
        before = np.asarray(waypoints[i-2]['position'])
        begin = np.asarray(waypoints[i-1]['position'])
        delta = np.asarray(waypoints[i]['position'])-begin
        direction = delta/np.linalg.norm(delta)
        phase = waypoints[i]['phase']
        heading_before = math.atan2(begin[1]-before[1], begin[0]-before[0])
        for distance in TURN_DISTANCES_M:
            position = begin+distance*direction
            turn = math.remainder(_heading(position, waypoints, i)-heading_before, 2*math.pi)
            for fraction in np.linspace(0., 1., 7):
                groups['swept_turns'].append(View(position, heading_before+fraction*turn,
                                                  dict(phase=phase, distance_m=distance,
                                                       turn_fraction=float(fraction))))
    offsets = [[radius*math.cos(a), radius*math.sin(a), 0.] for a in np.arange(8)*math.pi/4]
    offsets += [[0., 0., radius], [0., 0., -radius]]
    for offset in offsets:
        position = entry+offset
        for i in (align, cross):
            heading = _heading(position, waypoints, i)
            for delta in YAW_OFFSETS_DEG:
                groups['entry_capture'].append(View(
                    position, heading+math.radians(delta),
                    dict(phase=waypoints[i]['phase'], offset_enu_m=offset, yaw_offset_deg=delta)))
    # The same five deliberately perturbed seeds used in the original review.
    seeds = [
        ('approach minimum', (.849, -1.84, .95), 179.23708),
        ('course04 pose with repaired heading', (.26557524, -2.00255931, .99830087), 200.08697),
        ('align', (.30, -1.90, .95), 212.7523),
        ('cross exit', (-.987769, -2.083784, .9), 167.),
        ('landing', (-2.0422, -.7925, .7), 122.23424),
    ]
    for label, position, heading in seeds:
        for sign in (-1, 1):
            groups['seed_perturbations'].append(View(np.asarray(position), math.radians(heading),
                                                     dict(label=label, seed_sign=sign), sign))
    return groups, continuity


def _fit(frame, survey, initial, tof, tilt):
    try:
        pose, info = register_depth(frame, survey, initial, GATES, survey=True,
                                    downward_range=tof, inertial_tilt=tilt)
    except TrackingFailure as error:
        return None, dict(accepted=False, reason=str(error),
                          observability=error.details.get('depth_observability'))
    return pose, dict(accepted=True, reason=None, observability=info['depth_observability'])


def _evaluate(view, renderer, survey, intrinsics):
    body_rotation = cv2.Rodrigues(np.array([0., 0., view.yaw]))[0]
    pose = np.eye(4)
    pose[:3, :3] = body_rotation @ np.asarray(RGBD_BODY_FROM_OPTICAL)
    pose[:3, 3] = view.position+body_rotation@RGBD_TRANSLATION
    frame = DepthFrame(renderer.render(pose), intrinsics, GATES)
    optical_from_body = np.asarray(RGBD_BODY_FROM_OPTICAL).T
    origin = optical_from_body @ (np.asarray(TOF_TRANSLATION)-RGBD_TRANSLATION)
    ray = optical_from_body @ np.array([0., 0., -1.])
    hit = survey.surface.intersect(pose[:3, 3]+pose[:3, :3]@origin, pose[:3, :3]@ray)
    tof = DownwardRange(hit[0], origin, ray) if hit is not None else None
    tilt = InertialTilt(0, pose[:3, :3].T@np.array([0., 0., 1.]), TILT_STD_RAD, TILT_BRACKET_USEC)
    seed = pose.copy()
    if view.seed_sign:
        seed[:3, 3] += view.seed_sign*SEED_POSITION_M
        seed[:3, :3] = cv2.Rodrigues(view.seed_sign*np.radians(SEED_ROTATION_DEG))[0]@seed[:3, :3]
    found, result = _fit(frame, survey, seed, tof, tilt)
    return {**view.details, **result, 'position_enu_m': view.position.tolist(),
            'yaw_deg': math.degrees(view.yaw)%360,
            'optical_position_error_m': float(np.linalg.norm(found[:3, 3]-pose[:3, 3]))
            if found is not None else None}


def _recorded_rejection(path, survey):
    with np.load(path, allow_pickle=False) as saved:
        data = json.loads(saved['metadata_json'].item())
        frame = DepthFrame(saved['depth_z_m'], np.asarray(data['camera_intrinsics']), GATES)
    tof, tilt = None, None
    if data.get('tof_range_m') is not None:
        tof = DownwardRange(data['tof_range_m'], np.asarray(data['tof_origin_optical_m']),
                            np.asarray(data['tof_ray_optical']), data['tof_range_std_m'])
    if data.get('px4_tilt_prior') is not None:
        prior = data['px4_tilt_prior']
        tilt = InertialTilt(prior['capture_usec'], np.asarray(prior['up_optical']),
                            prior['std_rad'], prior['bracket_usec'])
    _, result = _fit(frame, survey, np.asarray(data['previous_world_from_optical']), tof, tilt)
    return {**result, 'recorded_reason': data['reason'],
            'recorded_observability': data.get('rejected_fits', {}).get('survey', {}).get('depth_observability'),
            'capture_stamp_ns': data['capture_stamp_ns']}


def _summary(rows):
    finite = [row for row in rows if row['observability'] is not None]
    worst = min(finite, key=lambda row: row['observability']) if finite else None
    return {'samples': len(rows), 'accepted': sum(row['accepted'] for row in rows),
            'minimum_observability': worst['observability'] if worst else None,
            'minimum_case': worst, 'failures': [row for row in rows if not row['accepted']]}


def _sha256(paths):
    hashes = {}
    for path in paths:
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def check(route, *, smoke=False, rejected_frame=None):
    route = Path(route)
    plan = json.loads(route.read_text())
    groups, continuity = _view_sets(plan['waypoints'])
    mesh_paths = sorted((MODEL.parent/'meshes').glob('*.obj'))
    intrinsics = calibration()
    renderer, survey = StaticDepth(mesh_paths, intrinsics), DepthSurvey.from_course(MODEL)
    rows = {group: [{'group': group, **_evaluate(view, renderer, survey, intrinsics)}
                    for view in (views[:1] if smoke else views)] for group, views in groups.items()}
    rank_rows = rows['nominal']+rows['swept_turns']+rows['entry_capture']
    phases = dict.fromkeys(row['phase'] for row in rows['nominal'])
    sources = [Path(__file__), route, MODEL, ROOT/'mission/heading.py', *mesh_paths]
    sources.extend(ROOT/'vision'/file for file in ('depth_odometry.py', 'rgbd_odometry.py', 'rgbd_scene.py'))
    if rejected_frame is not None:
        sources.append(rejected_frame)
    return {
        'schema_version': 1,
        'mode': 'smoke' if smoke else 'full',
        'scope': ('Static level-attitude CAD views with ideal down range and tilt. The renderer uses decimated '
                  'convex boulders, no aircraft self-occlusion or visual decals, and analytic ground. '
                  'Rank sweeps start at the sampled pose. Flight and temporal tracking require live validation.'),
        'source_sha256': _sha256(sources),
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'opencv': cv2.__version__, 'scipy': scipy.__version__},
        'parameters': {
            'production_depth_gates': asdict(GATES), 'lookahead_m': LOOKAHEAD_M,
            'image_size_px': [RGBD_WIDTH, RGBD_HEIGHT], 'intrinsics': intrinsics.tolist(),
            'camera_body_from_optical': RGBD_BODY_FROM_OPTICAL,
            'camera_translation_flu_m': RGBD_TRANSLATION, 'tof_translation_flu_m': TOF_TRANSLATION,
            'nominal_positions_per_phase': 11, 'yaw_offsets_deg': YAW_OFFSETS_DEG,
            'turn_distances_m': TURN_DISTANCES_M, 'yaw_samples_per_turn': 7,
            'entry_offsets': '8 azimuths on the arrival circle and +/- arrival-radius altitude',
            'tilt_std_rad': TILT_STD_RAD, 'tilt_bracket_usec': TILT_BRACKET_USEC,
            'boulder_decimation_directions': BOULDER_DIRECTIONS,
            'renderer_floor_z_m': 0., 'renderer_near_clip_m': .035,
            'seed_perturbation_position_enu_m': SEED_POSITION_M.tolist(),
            'seed_perturbation_rotation_vector_enu_deg': SEED_ROTATION_DEG.tolist(),
            'full_sample_counts': {key: len(views) for key, views in groups.items()},
        },
        'heading_continuity': continuity,
        'rank_checks': {**_summary(rank_rows), 'groups': {key: _summary(rows[key]) for key in
                         ('nominal', 'swept_turns', 'entry_capture')},
                        'nominal_by_phase': {phase: _summary(
                            [row for row in rows['nominal'] if row['phase'] == phase]) for phase in phases},
                        'cases': rank_rows},
        'seed_perturbations': {**_summary(rows['seed_perturbations']), 'cases': rows['seed_perturbations']},
        'recorded_rejection_replay': _recorded_rejection(rejected_frame, survey)
        if rejected_frame is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', type=Path, default=DEFAULT_ROUTE)
    parser.add_argument('--output', type=Path, help='new JSON artifact path; default stdout')
    parser.add_argument('--rejected-frame', type=Path, help='optional saved first-rejected.npz from a flight')
    parser.add_argument('--smoke', action='store_true', help='evaluate one case from each group')
    args = parser.parse_args()
    result = json.dumps(check(args.route, smoke=args.smoke, rejected_frame=args.rejected_frame),
                        indent=2, allow_nan=False)+'\n'
    if args.output is None:
        print(result, end='')
    else:
        with args.output.open('x') as output:
            output.write(result)


if __name__ == '__main__':
    main()
