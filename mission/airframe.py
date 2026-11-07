"""Conservative x500 collision envelope from the pinned simulator SDF.

Positions are base_link in ENU and quaternions are ENU-from-FLU in xyzw
order. Checks assume straight translation and shortest-arc attitude between
samples. Callers must split telemetry gaps before supplying a path.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


X500_SDF = Path(__file__).resolve().parents[1] / 'models/flight_x500_base/model.sdf'
_EPS = 1e-9
_MAX_SWEEP_ANGLE = math.radians(5.)
_SIGNS = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])


def _vector(value, n):
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f'expected {n} finite coordinates')
    return result


def quaternion_xyzw_from_rpy(roll, pitch, yaw):
    """Convert FLU roll, pitch and yaw in radians to ENU-from-FLU xyzw."""
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw])/2)
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw])/2)
    return np.array((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy))


def _rotation(q):
    x, y, z, w = q
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def _sdf_pose(element):
    pose = element.find('pose')
    if pose is None:
        return np.zeros(3), np.eye(3)
    if pose.attrib:
        raise ValueError('unsupported SDF pose frame or rotation representation')
    values = _vector(pose.text.split(), 6)
    return values[:3], _rotation(quaternion_xyzw_from_rpy(*values[3:]))


def _box_half(collision):
    geometry = collision.find('geometry')
    if geometry is None or [e.tag for e in geometry] != ['box']:
        raise ValueError('unsupported x500 collision geometry')
    size = _vector(geometry.findtext('box/size', '').split(), 3)
    if np.any(size <= 0):
        raise ValueError('invalid x500 collision box size')
    return size/2


@dataclass(frozen=True)
class Component:
    name: str
    center_flu: np.ndarray
    rotation_flu: np.ndarray
    half_size: np.ndarray  # box x/y/z, or rotor radius/radius/half thickness
    swept_rotor: bool = False

    def __post_init__(self):
        rotation = np.asarray(self.rotation_flu, dtype=float)
        half = _vector(self.half_size, 3)
        if (not self.name or rotation.shape != (3, 3)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10, rtol=0)
                or np.any(half <= 0) or (self.swept_rotor and half[0] != half[1])):
            raise ValueError('invalid envelope component')
        for name, value in (('center_flu', _vector(self.center_flu, 3)),
                            ('rotation_flu', rotation), ('half_size', half)):
            fixed = value.copy()
            fixed.flags.writeable = False
            object.__setattr__(self, name, fixed)

    @property
    def max_body_radius(self):
        radius = (math.hypot(self.half_size[0], self.half_size[2]) if self.swept_rotor
                  else np.linalg.norm(self.half_size))
        return float(np.linalg.norm(self.center_flu) + radius)

    def half_support(self, world_from_flu, axes_enu):
        """Uninflated support radii along one or more unit world axes."""
        local = np.asarray(axes_enu) @ world_from_flu @ self.rotation_flu
        if self.swept_rotor:
            return (self.half_size[0]*np.hypot(local[..., 0], local[..., 1])
                    + self.half_size[2]*np.abs(local[..., 2]))
        return np.abs(local) @ self.half_size


@dataclass(frozen=True)
class Envelope:
    components: tuple
    margin_m: float = .015

    def __post_init__(self):
        if not self.components or not math.isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError('envelope needs collision components and a nonnegative finite margin')


def load_x500_envelope(path=X500_SDF, *, margin_m=.015):
    """Read five base_link boxes and the revolute sweep of each rotor box.

    The simulated RGBD/ToF links have no collision geometry. No sensor
    housing proxy is included. The enclosing model's launch height is not a
    base_link offset. Unexpected collision links, frames or joints fail closed.
    """
    model = ET.parse(path).getroot().find('model')
    if model is None:
        raise ValueError('x500 SDF must contain a model')
    base = model.find("link[@name='base_link']")
    if base is None:
        raise ValueError('x500 SDF is missing base_link')
    base_center, base_rotation = _sdf_pose(base)
    components = []
    body_collisions = base.findall('collision')
    if len(body_collisions) != 5:
        raise ValueError('expected five x500 body/gear collisions')
    for collision in body_collisions:
        center, rotation = _sdf_pose(collision)
        components.append(Component(collision.get('name'), center, rotation, _box_half(collision)))

    rotor_names = {f'rotor_{index}' for index in range(4)}
    for link in model.findall('link'):
        name = link.get('name')
        if name not in rotor_names:
            if name != 'base_link' and link.find('collision') is not None:
                raise ValueError(f'unsupported collision link {name}')
            continue
        joint = next((j for j in model.findall('joint') if j.findtext('child') == name), None)
        axis = joint.find('axis/xyz') if joint is not None else None
        if (joint is None or joint.get('type') != 'revolute' or joint.findtext('parent') != 'base_link'
                or joint.find('pose') is not None or axis is None or axis.attrib
                or not np.allclose(_vector(axis.text.split(), 3), (0., 0., 1.))):
            raise ValueError(f'unsupported joint geometry for {name}')
        collisions = link.findall('collision')
        if len(collisions) != 1:
            raise ValueError(f'expected one rotor collision for {name}')
        collision = collisions[0]
        center, rotation = _sdf_pose(collision)
        corners = center + (_SIGNS*_box_half(collision)) @ rotation.T
        radius = float(np.hypot(corners[:, 0], corners[:, 1]).max())
        zmin, zmax = corners[:, 2].min(), corners[:, 2].max()
        link_center, link_rotation = _sdf_pose(link)
        cylinder_center = link_center + link_rotation[:, 2]*(zmin+zmax)/2
        components.append(Component(collision.get('name'), base_rotation.T @ (cylinder_center-base_center),
                                    base_rotation.T @ link_rotation, np.array([radius, radius, (zmax-zmin)/2]),
                                    swept_rotor=True))
        rotor_names.remove(name)
    if rotor_names:
        raise ValueError(f'x500 SDF missing rotor links {sorted(rotor_names)}')
    return Envelope(tuple(components), margin_m)


def _path(positions, quaternions_xyzw):
    positions = np.array(positions, dtype=float)
    q = np.array(quaternions_xyzw, dtype=float)
    if positions.size == q.size == 0:
        return positions.reshape(0, 3), q.reshape(0, 4)
    if (positions.ndim != 2 or positions.shape[1] != 3 or q.shape != (len(positions), 4)
            or not np.isfinite(positions).all() or not np.isfinite(q).all()):
        raise ValueError('path needs paired finite ENU positions and xyzw attitudes')
    norm = np.linalg.norm(q, axis=1)
    if np.any(np.abs(norm-1) > .001):
        raise ValueError('path contains non-unit attitude')
    return positions, q/norm[:, None]


def _sweeps(positions, q):
    for i, (a, b) in enumerate(zip(positions, positions[1:])):
        qa, qb = q[i:i+2]
        dot = float(qa @ qb)
        if dot < 0:
            qb, dot = -qb, -dot
        angle = math.acos(min(1., dot))  # half the SO(3) rotation angle
        pieces = max(1, math.ceil(2*angle/_MAX_SWEEP_ANGLE))
        delta = b-a
        padding_per_radius = 2*math.sin(angle/(2*pieces))
        for j in range(pieces):
            f = (j+.5)/pieces
            if pieces == 1:
                midpoint = qa+qb
            else:
                midpoint = math.sin((1-f)*angle)*qa + math.sin(f*angle)*qb
            yield (i, a + delta*(j/pieces), a + delta*((j+1)/pieces),
                   _rotation(midpoint/np.linalg.norm(midpoint)), padding_per_radius, j+1 == pieces)


def _overlap(start, end, half):
    """Time intervals in [0, 1] where linear coordinates overlap slabs."""
    delta = end-start
    stationary = np.abs(delta) < 1e-14
    safe_delta = np.where(stationary, 1., delta)
    enter, leave = (-half-start)/safe_delta, (half-start)/safe_delta
    inside = np.abs(start) <= half
    lower = np.where(stationary, np.where(inside, 0., np.inf), np.minimum(enter, leave))
    upper = np.where(stationary, np.where(inside, 1., -np.inf), np.maximum(enter, leave))
    return lower.clip(0., np.inf), upper.clip(-np.inf, 1.)


def _scene_axes(scene):
    centers, obstacle_axes, half = map(np.asarray, (scene.centers, scene.axes, scene.half))
    if (centers.ndim != 2 or centers.shape[1] != 3 or half.shape != centers.shape
            or obstacle_axes.shape != (len(centers), 3, 3)
            or not np.isfinite(centers).all() or not np.isfinite(half).all() or np.any(half < 0)
            or not np.allclose(obstacle_axes.transpose(0, 2, 1) @ obstacle_axes, np.eye(3), atol=1e-10, rtol=0)):
        raise ValueError('invalid static obstacle boxes')
    return centers, obstacle_axes.transpose(0, 2, 1), half


def face_clearance_at_poses(positions, rotations_enu_from_flu, scene, envelope, *, check_ground=True):
    """Batched face-axis clearance lower bound, in metres, for sampled poses.

    Accepts positions (..., 3) and broadcastable rotation matrices (..., 3,
    3). Positive results certify separation at those poses; zero/negative
    means unresolved, with no interpretation as actual penetration depth.
    This does not check motion between candidates or samples.
    """
    p = np.asarray(positions, dtype=float)
    rotation = np.asarray(rotations_enu_from_flu, dtype=float)
    if (p.shape[-1:] != (3,) or not np.isfinite(p).all() or rotation.shape[-2:] != (3, 3)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-10, rtol=0)):
        raise ValueError('expected finite positions and proper rotation matrices')
    centers, axes, half = _scene_axes(scene)
    support_axes = axes.reshape(-1, 3)
    obstacle_origins = np.einsum('nij,nj->ni', axes, centers)
    result = np.full(np.broadcast_shapes(p.shape[:-1], rotation.shape[:-2]), np.inf)
    inflation = envelope.margin_m + _EPS
    for part in envelope.components:
        center = p + rotation @ part.center_flu
        if len(centers):
            support = part.half_support(rotation, support_axes).reshape(*rotation.shape[:-2], len(centers), 3)
            projection = (center @ support_axes.T).reshape(*center.shape[:-1], len(centers), 3)
            local = projection-obstacle_origins
            gap = np.abs(local)-half-support-inflation
            result = np.minimum(result, gap.max(axis=-1).min(axis=-1))
        if check_ground:
            support = part.half_support(rotation, (0., 0., 1.))
            result = np.minimum(result, center[..., 2]-support-inflation)
    return result


def check_scene_path(positions, quaternions_xyzw, scene, envelope, *, check_ground=True):
    """Certify separation from learning.sim Scene boxes and optional floor.

    A component whose projections overlap all three obstacle face axes at
    the same instant is unresolved. This is an outer-bound check, so that
    result does not distinguish a genuine collision from a false positive.
    """
    positions, q = _path(positions, quaternions_xyzw)
    centers, axes, half = _scene_axes(scene)
    support_axes = axes.reshape(-1, 3)
    if len(positions) == 1:
        positions = np.repeat(positions, 2, axis=0)
        q = np.repeat(q, 2, axis=0)
    unresolved = None
    checked = 0
    parts = [(part, part.max_body_radius) for part in envelope.components]
    for index, a, b, rotation, padding, _ in _sweeps(positions, q):
        checked += 1
        for part, radius in parts:
            offset = rotation @ part.center_flu
            inflation = envelope.margin_m + padding*radius + _EPS
            if check_ground:
                height = part.half_support(rotation, (0., 0., 1.)) + inflation
                if min(a[2], b[2])+offset[2] <= height:
                    unresolved = dict(segment=index, component=part.name, obstacle='floor')
                    break
            support = part.half_support(rotation, support_axes).reshape(-1, 3)
            start = np.einsum('nij,nj->ni', axes, a+offset-centers)
            end = np.einsum('nij,nj->ni', axes, b+offset-centers)
            lower, upper = _overlap(start, end, half+support+inflation)
            ambiguous = np.flatnonzero(lower.max(axis=1) <= upper.min(axis=1))
            if len(ambiguous):
                unresolved = dict(segment=index, component=part.name, obstacle=int(ambiguous[0]))
                break
        if unresolved is not None:
            break
    return dict(certified_clear=checked > 0 and unresolved is None, checked_subsegments=checked,
                first_unresolved=unresolved, margin_m=envelope.margin_m)


def evaluate_window_crossing(positions, quaternions_xyzw, window, envelope):
    """Count passages with every inflated component entirely clear of the slab."""
    positions, q = _path(positions, quaternions_xyzw)
    center = _vector(window['center'], 3)
    normal = _vector(window['normal_toward_entry'], 3)
    horizontal = _vector(window['horizontal_axis'], 3)
    norms = np.array([np.linalg.norm(normal), np.linalg.norm(horizontal)])
    if np.any(norms <= _EPS):
        raise ValueError('window axes must be nonzero')
    normal, horizontal = normal/norms[0], horizontal/norms[1]
    if abs(normal @ horizontal) > 1e-6:
        raise ValueError('window axes must be perpendicular')
    horizontal -= (normal @ horizontal)*normal
    horizontal /= np.linalg.norm(horizontal)
    axes = np.stack((normal, horizontal, np.cross(normal, horizontal)))
    width, height, thickness = _vector([window[k] for k in ('width', 'height', 'thickness')], 3)
    if min(width, height) <= 0 or thickness < 0:
        raise ValueError('invalid window dimensions')
    slab = thickness/2

    def side(position, rotation):
        projections = [normal @ (position+rotation @ part.center_flu-center) for part in envelope.components]
        support = [part.half_support(rotation, normal)+envelope.margin_m for part in envelope.components]
        if all(p-h > slab+_EPS for p, h in zip(projections, support)):
            return 1
        if all(p+h < -slab-_EPS for p, h in zip(projections, support)):
            return -1
        return 0

    origin = side(positions[0], _rotation(q[0])) if len(positions) else 0
    margin = math.inf
    forward, reverse, partial = [], [], []
    parts = [(part, part.max_body_radius) for part in envelope.components]
    for index, a, b, rotation, padding, is_sample_boundary in _sweeps(positions, q):
        if origin:
            for part, radius in parts:
                offset = rotation @ part.center_flu
                inflation = envelope.margin_m + padding*radius
                n, h, v = part.half_support(rotation, axes) + inflation
                start, end = axes @ (a+offset-center), axes @ (b+offset-center)
                lower, upper = _overlap(start[0], end[0], slab+n+_EPS)
                if lower <= upper:
                    endpoints = start[1:] + np.array([lower, upper])[:, None]*(end-start)[1:]
                    gap = np.array([width/2-h, height/2-v]) - np.abs(endpoints).max(axis=0)
                    margin = min(margin, float(gap.min()))
        if is_sample_boundary:
            end_side = side(positions[index+1], _rotation(q[index+1]))
            if end_side:
                if math.isfinite(margin):
                    if origin == 1:
                        (forward if end_side == -1 else partial).append(margin)
                    elif origin == -1 and end_side == 1:
                        reverse.append(margin)
                origin, margin = end_side, math.inf
    if origin == 1 and math.isfinite(margin):
        partial.append(margin)

    valid = [clearance for clearance in forward if clearance > _EPS]
    valid_reverse = [clearance for clearance in reverse if clearance > _EPS]
    if valid:
        minimum, direction = min(valid), 'entry_to_exit'
    elif forward or partial:
        minimum, direction = max(forward+partial), 'entry_to_exit_blocked' if forward else 'incomplete'
    elif valid_reverse:
        minimum, direction = min(valid_reverse), 'exit_to_entry'
    else:
        minimum, direction = None, 'incomplete'
    return dict(full_envelope_crossed=bool(valid), min_aperture_clearance_m=minimum, direction=direction,
                valid_forward_crossings=len(valid), forward_attempts=len(forward),
                valid_reverse_crossings=len(valid_reverse), margin_m=envelope.margin_m,
                path_assumption='Linear translation, shortest-arc attitude, conservative inflated component supports.')
