"""Point-to-plane depth registration against a previous frame or a static survey."""
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


class TrackingFailure(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = {} if details is None else details


@dataclass(frozen=True)
class DepthGates:
    min_depth: float = .04
    max_depth: float = 7.98
    min_overlap: float = .40
    min_matches: int = 100
    min_observability: float = .004
    max_plane_rms: float = .035


@dataclass(frozen=True)
class DownwardRange:
    """A measured range along a calibrated ray in the forward optical frame."""
    distance_m: float
    origin: np.ndarray
    direction: np.ndarray
    std_m: float = .012


MIN_TILT_STD_RAD = .01
MAX_TILT_BRACKET_USEC = 60_000
TILT_MAX_DISAGREEMENT_RAD = .10
TILT_CORRELATION_NOTE = ('PX4 roll/pitch is inertial-driven but can correlate with vision '
                         'through EKF state updates; treated as a correlated prior.')


@dataclass(frozen=True)
class InertialTilt:
    """World up in optical coordinates, derived from timestamped PX4 tilt only."""
    capture_usec: int
    up_optical: np.ndarray
    std_rad: float
    bracket_usec: int

    def valid(self):
        try:
            up = np.asarray(self.up_optical, dtype=float)
            return (up.shape == (3,) and np.isfinite(up).all()
                    and abs(np.linalg.norm(up)-1.) < 1e-6
                    and math.isfinite(self.std_rad) and self.std_rad >= MIN_TILT_STD_RAD
                    and 0 <= self.bracket_usec <= MAX_TILT_BRACKET_USEC)
        except (TypeError, ValueError):
            return False


def tilt_residual(rotation, up_optical):
    """Gravity alignment residual and Jacobian for the scaled left SE(3) step."""
    world_up = np.array((0., 0., 1.))
    predicted = rotation.T @ world_up
    # The optical-frame residual is invariant to world yaw, including when
    # there is nonzero tilt disagreement. The yaw column is identically zero.
    jacobian = np.zeros((3, 6))
    jacobian[:, 3:] = rotation.T @ np.cross(world_up, np.eye(3)).T/2
    angle = math.atan2(np.linalg.norm(np.cross(predicted, up_optical)), predicted @ up_optical)
    return predicted-up_optical, jacobian, angle


class SurveySurface:
    """Nearest downward ray intersection against course triangles and ground."""

    def __init__(self, triangles, floor_z):
        self.vertices = triangles[:, 0]
        self.edge1 = triangles[:, 1] - self.vertices
        self.edge2 = triangles[:, 2] - self.vertices
        self.low = np.ascontiguousarray(triangles.min(axis=1).T)
        self.high = np.ascontiguousarray(triangles.max(axis=1).T)
        self.floor_z = floor_z

    def intersect(self, origin, ray):
        if ray[2] > -.5 or origin[2] <= self.floor_z:
            return None
        floor_range = (self.floor_z - origin[2]) / ray[2]
        floor_hit = origin + floor_range * ray
        lo, hi = np.minimum(origin, floor_hit), np.maximum(origin, floor_hit)
        candidates = np.flatnonzero(
            (self.low[0] <= hi[0]) & (self.high[0] >= lo[0]) &
            (self.low[1] <= hi[1]) & (self.high[1] >= lo[1]) &
            (self.low[2] <= hi[2]) & (self.high[2] >= lo[2]))
        if len(candidates):
            edge1, edge2 = self.edge1[candidates], self.edge2[candidates]
            normal = np.cross(edge1, edge2)
            p = np.cross(ray, edge2)
            determinant = np.sum(edge1*p, axis=1)
            usable = np.abs(determinant) > 1e-12
            denominator = np.where(usable, determinant, 1.)
            offset = origin - self.vertices[candidates]
            u = np.sum(offset*p, axis=1) / denominator
            q = np.cross(offset, edge1)
            v = np.sum(q*ray, axis=1) / denominator
            distance = np.sum(q*edge2, axis=1) / denominator
            keep = usable & (u >= 0.) & (v >= 0.) & (u+v <= 1.) & (distance > .02) & (distance <= floor_range)
            if np.any(keep):
                best = np.argmin(np.where(keep, distance, np.inf))
                return float(distance[best]), normal[best]/np.linalg.norm(normal[best])
        return float(floor_range), np.array((0., 0., 1.))


FALS_SURVEY_WINDOW_PX = 7
FALS_SURVEY_STRIDE_PX = 8
FALS_SUPPORT_CURVATURE_M = .085
CENTER_CURVATURE_M = .014
CURVATURE_PER_DEPTH = .018


def survey_normal_config(method):
    """Describe the configured absolute survey source normals for evidence logs."""
    if method not in ('central2', 'fals7'):
        raise ValueError('survey normals must be central2 or fals7')
    fals = method == 'fals7'
    return {
        'method': method,
        'opencv_version': cv2.__version__,
        'sample_stride_px': FALS_SURVEY_STRIDE_PX if fals else 6,
        'sample_origin_uv_px': [2, 2],
        'normal_radius_px': FALS_SURVEY_WINDOW_PX//2 if fals else 2,
        'center_curvature_radii_px': [1, 2],
        'center_curvature_limit_m': [CENTER_CURVATURE_M, CURVATURE_PER_DEPTH],
        'full_support_window_px': FALS_SURVEY_WINDOW_PX if fals else None,
        'full_support_requires_valid_depth': fals,
        'full_support_curvature_limit_m': [FALS_SUPPORT_CURVATURE_M, CURVATURE_PER_DEPTH] if fals else None,
    }


class FalsSurveyNormals:
    """Cache OpenCV FALS for one calibrated survey camera geometry."""

    def __init__(self, intrinsics):
        rgbd = getattr(cv2, 'rgbd', None)
        if not all(hasattr(rgbd, name) for name in (
                'RgbdNormals_create', 'RgbdNormals_RGBD_NORMALS_METHOD_FALS', 'depthTo3d')):
            raise RuntimeError('fals7 survey normals require OpenCV contrib cv2.rgbd')
        self._rgbd = rgbd
        self._intrinsics = np.array(intrinsics, dtype=np.float64, copy=True)
        self._shape = None
        self._normals = None

    def __call__(self, depth):
        if depth.shape != self._shape:
            self._normals = self._rgbd.RgbdNormals_create(
                *depth.shape, cv2.CV_32F, self._intrinsics, FALS_SURVEY_WINDOW_PX,
                self._rgbd.RgbdNormals_RGBD_NORMALS_METHOD_FALS)
            self._shape = depth.shape
        return self._normals.apply(self._rgbd.depthTo3d(depth, self._intrinsics))


class DepthFrame:
    """Optical-Z points and local surface normals, evaluated at queried pixels."""

    def __init__(self, depth, intrinsics, gates, stride=6, *, normal_radius_px=1, survey_fals=None):
        if type(normal_radius_px) is not int or normal_radius_px not in (1, 2):
            raise ValueError('normal radius must be one or two pixels')
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or min(depth.shape) < 8:
            raise TrackingFailure('invalid_depth_shape')
        self.normal_radius_px = normal_radius_px
        self.fx, self.fy = intrinsics[0, 0], intrinsics[1, 1]
        self.cx, self.cy = intrinsics[0, 2], intrinsics[1, 2]
        # Finite optical-Z depth, in metres. Clipped returns have no surface.
        self.depth = np.where((depth > gates.min_depth) & (depth < gates.max_depth), depth, np.nan)
        self._fals_normals = self._fals_support = None
        if survey_fals is not None:
            if normal_radius_px != 2:
                raise ValueError('survey FALS requires both central curvature radii')
            self._fals_normals = survey_fals(self.depth)
            support = np.isfinite(self.depth)
            # The coarse full-window mask adds rejection around deeper steps
            # and holes. Each center still must pass the tight checks below.
            horizontal = self.depth[:, 1:-1]
            vertical = self.depth[1:-1]
            x_curvature = np.abs(self.depth[:, :-2] + self.depth[:, 2:] - 2*horizontal)
            y_curvature = np.abs(self.depth[:-2] + self.depth[2:] - 2*vertical)
            support[:, 1:-1] &= x_curvature < FALS_SUPPORT_CURVATURE_M + CURVATURE_PER_DEPTH*horizontal
            support[1:-1] &= y_curvature < FALS_SUPPORT_CURVATURE_M + CURVATURE_PER_DEPTH*vertical
            kernel = np.ones((FALS_SURVEY_WINDOW_PX, FALS_SURVEY_WINDOW_PX), np.uint8)
            self._fals_support = cv2.erode(support.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                           borderValue=0).astype(bool)
        h, w = self.depth.shape
        u, v = np.meshgrid(np.arange(2, w-2, stride), np.arange(2, h-2, stride))
        points, normals = self._surface_at(u.ravel(), v.ravel())
        good = np.isfinite(normals[:, 0])
        self.sample_points = points[good].astype(np.float64)
        self.sample_normals = normals[good].astype(np.float64)

    def _surface_at(self, u, v):
        radius = self.normal_radius_px
        center = self.depth[v, u]
        left, right = self.depth[v, u-radius], self.depth[v, u+radius]
        top, bottom = self.depth[v-radius, u], self.depth[v+radius, u]
        x = (u.astype(np.float32)-self.cx)/self.fx
        y = (v.astype(np.float32)-self.cy)/self.fy
        if self._fals_normals is None:
            dz_x, dz_y = right-left, bottom-top
            # Central differences of back-projected neighboring 3D points.
            dx = np.column_stack((x*dz_x+(right+left)*radius/self.fx, y*dz_x, dz_x))
            dy = np.column_stack((x*dz_y, y*dz_y+(bottom+top)*radius/self.fy, dz_y))
            normals = np.cross(dx, dy)
        else:
            normals = self._fals_normals[v, u]
        norm = np.linalg.norm(normals, axis=1)
        # Perspective changes depth even on a plane near the horizon. Reject
        # abrupt asymmetric jumps, which are usually surface discontinuities.
        curvature = np.maximum(np.abs(right+left-2*center), np.abs(bottom+top-2*center))
        curvature_limit = CENTER_CURVATURE_M + CURVATURE_PER_DEPTH * center
        valid = (norm > 1e-8) & (curvature < curvature_limit)
        if radius == 2:
            # Reject a hole or depth jump at intermediate pixels that the
            # outer derivative could skip, using the same curvature test.
            short_x = np.abs(self.depth[v, u-1]+self.depth[v, u+1]-2*center)
            short_y = np.abs(self.depth[v-1, u]+self.depth[v+1, u]-2*center)
            valid &= np.maximum(short_x, short_y) < curvature_limit
        if self._fals_support is not None:
            valid &= self._fals_support[v, u] & np.isfinite(norm) & (norm > 1e-6)
        normals /= np.where(valid, norm, np.nan)[:, None]
        return np.column_stack((x*center, y*center, center)), normals

    def correspond(self, points, normals, max_distance):
        visible = np.flatnonzero(points[:, 2] > .03)
        p = points[visible]
        u = np.rint(self.fx * p[:, 0] / p[:, 2] + self.cx).astype(int)
        v = np.rint(self.fy * p[:, 1] / p[:, 2] + self.cy).astype(int)
        h, w = self.depth.shape
        radius = self.normal_radius_px
        inside = (radius <= u) & (u < w-radius) & (radius <= v) & (v < h-radius)
        ids = visible[inside]
        target, target_normal = self._surface_at(u[inside], v[inside])
        close = np.sum((target - points[ids])**2, axis=1) < max_distance**2
        aligned = np.abs(np.sum(target_normal * normals[ids], axis=1)) > .86
        keep = close & aligned
        return ids[keep], target[keep], target_normal[keep]


def _mesh_triangles(path):
    vertices, faces = [], []
    with path.open() as mesh:
        for line in mesh:
            if line.startswith('v '):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith('f '):
                face = [int(value.split('/')[0]) - 1 for value in line.split()[1:]]
                faces.extend((face[0], a, b) for a, b in zip(face[1:-1], face[2:]))
    return np.asarray(vertices)[np.asarray(faces)]


class DepthSurvey:
    """Local surveyed course surfaces indexed by position with OpenCV FLANN.

    The survey is an a priori set of static geometry. The nearest local point
    on the ground is analytic. No simulator state is involved.
    """

    def __init__(self, points, normals, floor_z=0., panels=(), triangles=None):
        self.panels = panels
        if triangles is None:
            triangles = np.empty((0, 3, 3))
        self.surface = SurveySurface(triangles, floor_z)
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        self.floor_z = float(floor_z)
        self.index = cv2.flann_Index(self.points, dict(algorithm=1, trees=2))

    @classmethod
    def from_course(cls, model_path):
        """Load this course's mesh vertices, which are already in ENU metres."""
        path = Path(model_path)
        root = ET.parse(path)
        mesh_paths = set()
        panels = []
        for visual in root.findall('./model/link/visual'):
            uri = visual.findtext('geometry/mesh/uri')
            if not uri or visual.get('name') == 'floor_20_x_20_ft':
                continue
            if visual.find('pose') is not None or visual.find('geometry/mesh/scale') is not None:
                raise ValueError('course survey expects untransformed mesh vertices')
            mesh_path = path.parent / uri.removeprefix('model://course_v3/')
            if visual.get('name', '').startswith('net_'):
                # The course boundary nets are thin rectangular axis-aligned
                # boxes. Project analytically to their two large faces.
                vertices = _mesh_triangles(mesh_path).reshape(-1, 3)
                lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                axis = int(np.argmin(upper-lower))
                if upper[axis]-lower[axis] > .03:
                    raise ValueError('course net survey expects a thin axis-aligned box')
                panels.append((axis, lower, upper))
            else:
                mesh_paths.add(mesh_path)
        triangles = np.concatenate([_mesh_triangles(mesh) for mesh in sorted(mesh_paths)])
        edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edge1, edge2)
        area_twice = np.linalg.norm(normals, axis=1)
        count = int(area_twice.sum() / (2 * .037**2))
        rng = np.random.default_rng(1701)
        selected = rng.choice(len(triangles), count, p=area_twice/area_twice.sum())
        a, b = rng.random((2, count))
        flipped = a+b > 1
        a[flipped], b[flipped] = 1-a[flipped], 1-b[flipped]
        points = triangles[selected, 0] + a[:, None]*edge1[selected] + b[:, None]*edge2[selected]
        return cls(points, normals[selected]/area_twice[selected, None], panels=panels,
                   triangles=triangles)

    def correspond(self, points, normals, max_distance):
        floor_distance = (points[:, 2] - self.floor_z)**2
        floor_facing = np.abs(normals[:, 2]) > .94
        floor = floor_facing & (floor_distance < .013**2)
        target, target_normal = points.copy(), np.zeros_like(points)
        distances = np.full(len(points), np.inf)
        search = ~floor
        if np.any(search):
            neighbors, squared = self.index.knnSearch(points[search].astype(np.float32), 1,
                                                       params={'checks': 12})
            neighbors = neighbors.ravel()
            target[search] = self.points[neighbors]
            target_normal[search] = self.normals[neighbors]
            aligned = np.abs(np.sum(target_normal[search] * normals[search], axis=1)) > .86
            distances[search] = np.where(aligned, squared.ravel(), np.inf)
        floor = floor_facing & (floor_distance < distances)
        target[floor] = points[floor]
        target[floor, 2] = self.floor_z
        target_normal[floor] = (0, 0, 1)
        distances[floor] = floor_distance[floor]
        for axis, lower, upper in self.panels:
            coordinate = points[:, axis]
            face = np.where(coordinate < (lower[axis]+upper[axis])/2, lower[axis], upper[axis])
            ids = np.flatnonzero((np.abs(normals[:, axis]) > .86) & ((coordinate-face)**2 < distances))
            closest = np.clip(points[ids], lower, upper)
            closest[:, axis] = face[ids]
            squared = np.sum((closest-points[ids])**2, axis=1)
            keep = squared < distances[ids]
            ids = ids[keep]
            distances[ids] = squared[keep]
            target[ids] = closest[keep]
            target_normal[ids] = 0.
            target_normal[ids, axis] = 1.
        keep = distances < max_distance**2
        return np.flatnonzero(keep), target[keep], target_normal[keep]


def register_depth(frame, reference, initial, gates=DepthGates(), *, survey=False,
                   downward_range=None, inertial_tilt=None):
    """Find reference_from_frame using robust point-to-plane Gauss–Newton.

    Returns a transform and local fit diagnostics. The Hessian must constrain
    all six degrees of freedom before a pose is accepted. A single plane can
    fit with zero error while leaving three of those directions unobservable.
    """
    points, normals = frame.sample_points, frame.sample_normals
    if len(points) < gates.min_matches:
        raise TrackingFailure('insufficient_depth_samples')
    tilt = inertial_tilt if survey and inertial_tilt is not None and inertial_tilt.valid() else None
    tilt_used = False
    tilt_status = 'not_available' if tilt is None else 'not_needed'
    tilt_angle = None
    depth_ratio = None

    def with_tilt(hessian, gradient, rotation, residual):
        nonlocal tilt_used, tilt_status, tilt_angle, depth_ratio
        depth_eigen = np.linalg.eigvalsh(hessian)
        if tilt is None:
            return hessian, gradient, depth_eigen
        depth_ratio = float(depth_eigen[0]/depth_eigen[-1])
        error, rows, tilt_angle = tilt_residual(rotation, tilt.up_optical)
        if tilt_angle > TILT_MAX_DISAGREEMENT_RAD:
            tilt_status = 'disagreement'
            return hessian, gradient, depth_eigen
        if not tilt_used and depth_ratio >= gates.min_observability:
            return hessian, gradient, depth_eigen
        depth_noise = max(.009, np.sqrt(np.mean(residual**2))) * 5
        strength = (depth_noise/tilt.std_rad)**2
        augmented = hessian + strength * (rows.T @ rows)
        tilt_used, tilt_status = True, 'applied'
        return augmented, gradient + strength * (rows.T @ error), np.linalg.eigvalsh(augmented)

    def equations(rotation, translation, distance):
        # Rotation is updated about the optical center, so its lever arm does
        # not grow with the ENU position of the aircraft.
        rotated = points @ rotation.T
        source = rotated + translation
        ids, target, normal = reference.correspond(source, normals @ rotation.T, distance)
        if len(ids) < max(gates.min_matches, gates.min_overlap*len(points)):
            raise TrackingFailure('insufficient_depth_overlap')
        residual = np.sum((source[ids] - target) * normal, axis=1)
        # A smooth robust loss keeps edge and occlusion residuals from
        # overpowering intact surfaces. Scale rotation by a 2 m lever arm.
        weight = 1 / np.sqrt(1 + (residual/.025)**2)
        jac = np.column_stack((normal, np.cross(rotated[ids], normal)/2))
        weighted_jac = jac * weight[:, None]
        hessian, gradient = jac.T @ weighted_jac, weighted_jac.T @ residual
        range_residual = None
        if survey and downward_range is not None:
            origin = rotation @ downward_range.origin
            ray = rotation @ downward_range.direction
            hit = reference.surface.intersect(translation + origin, ray)
            if hit is not None:
                predicted_range, normal = hit
                incidence = float(normal @ ray)
                error = downward_range.distance_m - predicted_range
                if abs(incidence) > .5 and abs(error) < min(.14, distance):
                    lever_arm = origin + downward_range.distance_m * ray
                    row = np.r_[normal, np.cross(lever_arm, normal)/2]
                    # The camera fit treats its nearby pixels as correlated.
                    # A separate ToF contributes one independently measured ray.
                    variance = (incidence*downward_range.std_m)**2 + .006**2
                    strength = 25 * max(.009, np.sqrt(np.mean(residual**2)))**2 / variance
                    hessian += strength * np.outer(row, row)
                    gradient += strength * row * (incidence * error)
                    range_residual = error
        return hessian, gradient, residual, ids, range_residual

    rotation, translation = initial[:3, :3].copy(), initial[:3, 3].copy()

    def rejected(reason, hessian, eigenvalues, residual, ids, range_residual):
        rms = float(np.sqrt(np.mean(residual**2)))
        weakest = float(eigenvalues[0])
        diagnostics = {
            'reference_from_optical_3x4': np.column_stack((rotation, translation)).tolist(),
            'depth_matches': len(ids),
            'depth_overlap': len(ids)/len(points),
            'depth_plane_rms_m': rms,
            'depth_observability': float(weakest/eigenvalues[-1]),
            'scaled_step_hessian': hessian.tolist(),
            'scaled_step_std_max': 5 * max(.009, rms)/np.sqrt(weakest) if weakest > 0 else None,
            'tof_range_residual_m': range_residual,
        }
        if tilt is not None:
            diagnostics.update(px4_tilt_prior_status=tilt_status, px4_tilt_capture_usec=tilt.capture_usec,
                               px4_tilt_std_rad=tilt.std_rad, px4_tilt_bracket_usec=tilt.bracket_usec,
                               px4_tilt_disagreement_rad=tilt_angle,
                               depth_observability_without_tilt=depth_ratio)
        return TrackingFailure(reason, diagnostics)

    for _ in range(6):
        hessian, gradient, residual, ids, range_residual = equations(
            rotation, translation, .20 if survey else .16)
        hessian, gradient, eigenvalues = with_tilt(hessian, gradient, rotation, residual)
        if eigenvalues[0] < 1e-7 * eigenvalues[-1]:
            raise rejected(f'unobservable_depth_geometry:{eigenvalues[0]/eigenvalues[-1]:.5g}',
                           hessian, eigenvalues, residual, ids, range_residual)
        step = np.linalg.solve(hessian, -gradient)
        if np.linalg.norm(step[:3]) > .25 or np.linalg.norm(step[3:]) > .30:
            raise rejected('implausible_depth_step', hessian, eigenvalues, residual, ids, range_residual)
        rotation = cv2.Rodrigues(step[3:]/2)[0] @ rotation
        translation += step[:3]
        if np.linalg.norm(step) < .00025:
            break

    hessian, gradient, residual, ids, range_residual = equations(rotation, translation, .095 if survey else .08)
    depth_hessian = hessian
    hessian, gradient, eigenvalues = with_tilt(hessian, gradient, rotation, residual)
    if tilt_used and (tilt_status != 'applied' or tilt_angle > max(.03, 3*tilt.std_rad)):
        raise rejected('px4_tilt_disagrees_with_depth', hessian, eigenvalues, residual, ids, range_residual)
    conditioning = float(eigenvalues[0] / eigenvalues[-1])
    if conditioning < gates.min_observability:
        raise rejected(f'unobservable_depth_geometry:{conditioning:.5g}',
                       hessian, eigenvalues, residual, ids, range_residual)
    # Correspondences change across iterations. Reject a result that still
    # demands significant motion under the final, tighter correspondence gate.
    correction = np.linalg.solve(hessian, -gradient)
    if np.linalg.norm(correction) > .027:
        raise rejected('depth_registration_not_converged', hessian, eigenvalues, residual, ids, range_residual)
    rms = float(np.sqrt(np.mean(residual**2)))
    if rms > gates.max_plane_rms:
        raise rejected('depth_plane_residual', hessian, eigenvalues, residual, ids, range_residual)
    # Treat nearby pixels as correlated. This is a local precision estimate;
    # RGBD odometry additionally accumulates drift between survey matches.
    covariance = np.linalg.inv(hessian) * max(.009, rms)**2 * 25.
    if np.max(np.sqrt(np.diag(covariance))) > .075:
        raise rejected('depth_uncertainty', hessian, eigenvalues, residual, ids, range_residual)
    if tilt_used:
        # Derive claimed precision from depth/ToF only. Directions below the
        # unassisted observability gate get an inflated .15 scaled-step std,
        # rather than the small covariance of an independent tilt observation.
        values, directions = np.linalg.eigh(depth_hessian)
        independent = values >= gates.min_observability*values[-1]
        scaled_variance = np.full(6, .15**2)
        scaled_variance[independent] = (max(.009, rms)**2 * 25) / values[independent]
        covariance = (directions * scaled_variance) @ directions.T
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    axes_std = np.sqrt(np.diag(covariance))
    diagnostics = dict(depth_matches=len(ids), depth_overlap=len(ids)/len(points),
                       depth_plane_rms_m=rms, depth_observability=conditioning,
                       translation_std=axes_std[:3], rotation_std=axes_std[3:]/2)
    if range_residual is not None:
        diagnostics['tof_range_residual_m'] = float(range_residual)
    if tilt is not None:
        diagnostics.update(px4_tilt_prior_status=tilt_status, px4_tilt_capture_usec=tilt.capture_usec,
                           px4_tilt_std_rad=tilt.std_rad,
                           px4_tilt_disagreement_rad=tilt_angle, px4_tilt_bracket_usec=tilt.bracket_usec,
                           depth_observability_without_tilt=depth_ratio,
                           px4_tilt_prior_correlation=TILT_CORRELATION_NOTE)
    return transform, diagnostics
