"""Marker-free forward RGBD localization with an optional static course survey.

The initial pose must be independently known at rest. Subsequent pose updates
use synchronized image and depth measurements. Tracking loss latches until an
explicit reset at a newly established stationary pose.
"""
from collections import deque
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import struct
from threading import Lock
import time

import cv2
import numpy as np

from .depth_odometry import (DepthFrame, DepthGates, DepthSurvey, DownwardRange, FalsSurveyNormals,
                             InertialTilt, MAX_TILT_BRACKET_USEC, MIN_TILT_STD_RAD, TrackingFailure,
                             register_depth, survey_normal_config)
from .rgbd_scene import (COLOR_TOPIC, DEPTH_TOPIC, RGBD_BODY_FROM_OPTICAL,
                         RGBD_HEIGHT, RGBD_HFOV, RGBD_TRANSLATION, RGBD_WIDTH,
                         TOF_MAX_DEPTH, TOF_MIN_DEPTH, TOF_TOPIC, TOF_TRANSLATION)

CLOCK_TOPIC = '/world/dronehacks/clock'
TOF_SYNC_NS = 15_000_000
TILT_WAIT_NS = MAX_TILT_BRACKET_USEC*1000
TILT_WAIT_WALL_S = .3
SURVEY_NORMAL_RADIUS_PX = 2
RELATIVE_DEPTH_NORMALS = {'method': 'central1', 'sample_stride_px': 6, 'normal_radius_px': 1}


@dataclass(frozen=True)
class Gates:
    min_inliers: int = 24
    min_ratio: float = .65
    max_reprojection: float = 1.5
    max_gap_s: float = .12
    max_age_s: float = .3
    max_speed: float = 3.0
    max_angular_speed: float = 3.0
    min_depth: float = .04
    max_depth: float = 8.0
    min_image_span: float = .12


def calibration(width=RGBD_WIDTH, height=RGBD_HEIGHT):
    focal = width / (2 * math.tan(RGBD_HFOV / 2))
    return np.array(((focal, 0, width/2), (0, focal, height/2), (0, 0, 1.)), dtype=float)


def rotation_from_quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError('launch quaternion must be finite, unit, and wxyz')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def quaternion_from_rotation(rotation):
    vector = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(vector)
    q = np.array([1., 0, 0, 0]) if angle < 1e-10 else np.r_[math.cos(angle/2), vector*math.sin(angle/2)/angle]
    if not np.isfinite(q).all():
        raise TrackingFailure('nonfinite_orientation')
    return (q / np.linalg.norm(q)).tolist()


def estimate_motion(points, pixels, intrinsics, dt, gates=Gates()):
    """Robust optical-frame transform from metric previous points/current pixels.

    Returns current_from_previous, diagnostics. Rejects collinear geometry,
    small image support, excessive motion, and poorly explained correspondences.
    Planar floors are allowed; this is VO, not global relocalization.
    """
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(xyz) != len(uv) or len(xyz) < gates.min_inliers:
        raise TrackingFailure('insufficient_correspondences')
    if not 0 < dt <= gates.max_gap_s:
        raise TrackingFailure('capture_gap')
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise TrackingFailure('nonfinite_correspondences')
    if np.any((xyz[:, 2] < gates.min_depth) | (xyz[:, 2] > gates.max_depth)):
        raise TrackingFailure('invalid_depth')
    def spread(indices):
        pts = xyz[indices]
        singular = np.linalg.svd(pts-pts.mean(axis=0), compute_uv=False)
        if singular[1] < max(1e-5, singular[0]*.015):
            raise TrackingFailure('degenerate_geometry')
        normalized = (uv[indices] - intrinsics[:2, 2]) / np.diag(intrinsics)[:2]
        if np.min(np.ptp(normalized, axis=0)) < gates.min_image_span:
            raise TrackingFailure('insufficient_image_support')
    spread(np.arange(len(xyz)))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz, uv, intrinsics, None, iterationsCount=200, reprojectionError=2.,
        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:
        raise TrackingFailure('pnp_failed')
    ids = inliers.ravel()
    if len(ids) < gates.min_inliers or len(ids)/len(xyz) < gates.min_ratio:
        raise TrackingFailure('insufficient_inliers')
    spread(ids)
    ok, rvec, tvec = cv2.solvePnP(xyz[ids], uv[ids], intrinsics, None, rvec, tvec,
                                 useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(np.r_[rvec.ravel(), tvec.ravel()]).all():
        raise TrackingFailure('refinement_failed')
    reprojection = cv2.projectPoints(xyz[ids], rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
    errors = np.linalg.norm(reprojection-uv[ids], axis=1)
    if np.sqrt(np.mean(errors**2)) > gates.max_reprojection:
        raise TrackingFailure('reprojection_error')
    rotation = cv2.Rodrigues(rvec)[0]
    transformed = xyz[ids] @ rotation.T + tvec.ravel()
    if np.any(transformed[:, 2] <= gates.min_depth):
        raise TrackingFailure('behind_camera')
    if np.linalg.norm(tvec) > gates.max_speed*dt + .005:
        raise TrackingFailure('implausible_translation')
    if np.linalg.norm(rvec) > gates.max_angular_speed*dt + .01:
        raise TrackingFailure('implausible_rotation')
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, tvec.ravel()
    return transform, {'inliers': len(ids), 'matches': len(xyz),
                       'reprojection_rms': float(np.sqrt(np.mean(errors**2)))}


class RGBDOdometry:
    def __init__(self, launch_position_enu, launch_orientation_wxyz=(1, 0, 0, 0),
                 intrinsics=None, gates=Gates(), survey=None, depth_gates=DepthGates(), *,
                 survey_normals='central2'):
        cv2.setNumThreads(1)
        self.intrinsics = calibration() if intrinsics is None else np.array(intrinsics, dtype=float)
        self.inverse_intrinsics = np.linalg.inv(self.intrinsics)
        self.gates = gates
        self.depth_gates = depth_gates
        self.survey = survey
        self.survey_normals = survey_normal_config(survey_normals)
        if survey_normals == 'fals7' and survey is None:
            raise ValueError('fals7 normals require an absolute depth survey')
        self._survey_fals = FalsSurveyNormals(self.intrinsics) if survey_normals == 'fals7' else None
        self.body_from_optical = np.array(RGBD_BODY_FROM_OPTICAL)
        self.camera_translation = np.array(RGBD_TRANSLATION)
        self.reset(launch_position_enu, launch_orientation_wxyz)

    def reset(self, known_stationary_position, known_stationary_orientation=(1, 0, 0, 0)):
        position = np.asarray(known_stationary_position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('an independently known launch base_link position is required')
        body_rotation = rotation_from_quaternion(known_stationary_orientation)
        self.world_from_optical = np.eye(4)
        self.world_from_optical[:3, :3] = body_rotation @ self.body_from_optical
        self.world_from_optical[:3, 3] = position + body_rotation @ self.camera_translation
        self.position_variance = np.full(3, .02**2)
        self.orientation_variance = np.full(3, .02**2)
        self.previous = None
        self.lost = None
        self.last_usec = None

    def _record(self, sim_usec, reason=None, **diagnostics):
        record = dict(valid=False, sim_usec=int(sim_usec), stamp_sim_s=sim_usec*1e-6,
                      position_enu=None, orientation_enu_flu_wxyz=None,
                      orientation_xyzw=None, quality=0, reason=reason,
                      survey_normals=self.survey_normals if self.survey is not None else None,
                      **diagnostics)
        if reason is None:
            rotation = self.world_from_optical[:3, :3] @ self.body_from_optical.T
            position = self.world_from_optical[:3, 3] - rotation @ self.camera_translation
            q = quaternion_from_rotation(rotation)
            position_std = np.sqrt(self.position_variance)
            orientation_std = np.sqrt(self.orientation_variance)
            quality = 100 * (1 - max(position_std.max()/.28, orientation_std.max()/.2))
            record.update(valid=True, position_enu=position.tolist(),
                          orientation_enu_flu_wxyz=q, orientation_xyzw=q[1:]+q[:1],
                          position_std_m=position_std.tolist(), orientation_std_rad=orientation_std.tolist(),
                          quality=max(1, min(100, int(quality))))
        return record

    def _features(self, gray, depth):
        mask = (np.isfinite(depth) & (depth >= self.gates.min_depth) &
                (depth <= self.gates.max_depth)).astype('uint8')*255
        return cv2.goodFeaturesToTrack(gray, maxCorners=600, qualityLevel=.01,
                                       minDistance=6, mask=mask, blockSize=5)

    def _rgb_motion(self, old_gray, gray, old_depth, depth, old_features, dt):
        if old_features is None or len(old_features) < self.gates.min_inliers:
            raise TrackingFailure('insufficient_texture')
        current, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_features, None,
                                                    winSize=(21, 21), maxLevel=3)
        if current is None:
            raise TrackingFailure('optical_flow_failed')
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, current, None,
                                                              winSize=(21, 21), maxLevel=3)
        if backward is None:
            raise TrackingFailure('optical_flow_failed')
        old_uv, uv = old_features.reshape(-1, 2), current.reshape(-1, 2)
        keep = status.ravel().astype(bool) & reverse_status.ravel().astype(bool)
        keep &= np.linalg.norm(backward.reshape(-1, 2)-old_uv, axis=1) < .8
        keep &= np.isfinite(uv).all(axis=1)
        keep &= (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < gray.shape[1]-1) & (uv[:, 1] < gray.shape[0]-1)
        previous_uv, uv = old_uv[keep], uv[keep]
        integer = np.rint(previous_uv).astype(int)
        z = old_depth[integer[:, 1], integer[:, 0]]
        rays = np.c_[previous_uv, np.ones(len(previous_uv))] @ self.inverse_intrinsics.T
        points = rays*z[:, None]
        transform, diagnostics = estimate_motion(points, uv, self.intrinsics, dt, self.gates)
        predicted = points @ transform[:3, :3].T + transform[:3, 3]
        projected = predicted @ self.intrinsics.T
        projected = projected[:, :2] / projected[:, 2:]
        geometric_inliers = np.linalg.norm(projected-uv, axis=1) <= 2.
        current_pixels = np.rint(uv).astype(int)
        measured_z = depth[current_pixels[:, 1], current_pixels[:, 0]]
        agrees = np.isfinite(measured_z) & (np.abs(predicted[:, 2]-measured_z) <= .03+.03*predicted[:, 2])
        depth_inliers = int(np.count_nonzero(agrees & geometric_inliers))
        if (depth_inliers < self.gates.min_inliers or
                depth_inliers/max(1, np.count_nonzero(geometric_inliers)) < self.gates.min_ratio):
            raise TrackingFailure('inconsistent_depth')
        diagnostics['depth_inliers'] = depth_inliers
        return np.linalg.inv(transform), diagnostics

    def _check_motion(self, pose, dt):
        change = self.world_from_optical[:3, :3].T @ pose[:3, :3]
        angle = np.linalg.norm(cv2.Rodrigues(change)[0])
        displacement = np.linalg.norm(pose[:3, 3] - self.world_from_optical[:3, 3])
        if dt is None:
            if displacement > .12 or angle > .09:
                raise TrackingFailure('survey_disagrees_with_launch')
        elif displacement > self.gates.max_speed*dt + .025:
            raise TrackingFailure('implausible_translation')
        elif angle > self.gates.max_angular_speed*dt + .025:
            raise TrackingFailure('implausible_rotation')

    def _update_depth(self, frame, reference, initial, dt, survey, downward_range=None,
                      inertial_tilt=None):
        transform, diagnostics = register_depth(frame, reference, initial, self.depth_gates,
                                                 survey=survey, downward_range=downward_range,
                                                 inertial_tilt=inertial_tilt)
        normals = self.survey_normals if survey else RELATIVE_DEPTH_NORMALS
        diagnostics.update(depth_normals=normals, depth_normal_radius_px=normals['normal_radius_px'])
        pose = transform if survey else self.world_from_optical @ transform
        try:
            self._check_motion(pose, dt)
        except TrackingFailure as error:
            error.details.update(reference_from_optical_3x4=transform[:3].tolist())
            error.details.update({name: value for name, value in diagnostics.items()
                                  if name not in ('translation_std', 'rotation_std')})
            raise
        # Independent survey matches bound accumulated drift. Frame-to-frame
        # fits add uncertainty and cannot re-anchor a position.
        translation_std = diagnostics.pop('translation_std')
        rotation_std = diagnostics.pop('rotation_std')
        if survey:
            self.position_variance = np.maximum(.018, translation_std)**2
            self.orientation_variance = np.maximum(.013, rotation_std)**2
        else:
            self.position_variance += (pose[:3, :3]**2) @ np.maximum(.002, translation_std)**2
            self.orientation_variance += (pose[:3, :3]**2) @ np.maximum(.001, rotation_std)**2
        self.world_from_optical = pose
        if survey:
            diagnostics['source'] = ('depth_survey_tof' if 'tof_range_residual_m' in diagnostics
                                     else 'depth_survey')
            if diagnostics.get('px4_tilt_prior_status') == 'applied':
                diagnostics['source'] += '_px4_tilt'
        else:
            diagnostics['source'] = 'depth_odometry'
        return diagnostics

    def estimate(self, rgb, depth, sim_usec, now_usec=None, *, downward_range=None, inertial_tilt=None):
        """Consume synchronized RGB uint8 and optical-Z float32 depth in metres."""
        if self.lost:
            return self._record(sim_usec, 'tracking_lost:'+self.lost)
        attempts = {}
        if inertial_tilt is not None and inertial_tilt.capture_usec != sim_usec:
            inertial_tilt = None
        try:
            if now_usec is not None and not 0 <= now_usec-sim_usec <= self.gates.max_age_s*1e6:
                raise TrackingFailure('stale_capture')
            if self.last_usec is not None and sim_usec <= self.last_usec:
                raise TrackingFailure('nonmonotonic_capture')
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
                raise TrackingFailure('invalid_image_shape')
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            frame = DepthFrame(depth, self.intrinsics, self.depth_gates)
            dt = None if self.last_usec is None else (sim_usec-self.last_usec)*1e-6
            if dt is not None and not 0 < dt <= self.gates.max_gap_s:
                raise TrackingFailure('capture_gap')
            diagnostics, survey_failure = None, None
            if self.survey is not None:
                # Only the survey uses the wider measured normal. Its target
                # normals come from static geometry; keep frame fallback and
                # its observability based on the original image derivative.
                survey_frame = DepthFrame(depth, self.intrinsics, self.depth_gates,
                                          stride=self.survey_normals['sample_stride_px'],
                                          normal_radius_px=SURVEY_NORMAL_RADIUS_PX,
                                          survey_fals=self._survey_fals)
                try:
                    diagnostics = self._update_depth(survey_frame, self.survey, self.world_from_optical, dt,
                                                     True, downward_range=downward_range,
                                                     inertial_tilt=inertial_tilt)
                except TrackingFailure as error:
                    survey_failure = str(error)
                    attempts['survey'] = {'reason': survey_failure,
                                          'depth_normals': self.survey_normals,
                                          'depth_normal_radius_px': self.survey_normals['normal_radius_px'],
                                          **error.details}
            if self.previous is not None and diagnostics is None:
                old_gray, old_depth, old_frame = self.previous
                try:
                    diagnostics = self._update_depth(frame, old_frame, np.eye(4), dt, False)
                except TrackingFailure as error:
                    depth_failure = str(error)
                    attempts['frame_depth'] = {'reason': depth_failure,
                                               'depth_normals': RELATIVE_DEPTH_NORMALS,
                                               'depth_normal_radius_px': frame.normal_radius_px, **error.details}
                if diagnostics is None:
                    try:
                        old_features = self._features(old_gray, old_depth)
                        transform, diagnostics = self._rgb_motion(old_gray, gray, old_depth, depth, old_features, dt)
                    except TrackingFailure as error:
                        attempts['frame_rgb'] = {'reason': str(error), **error.details}
                        cause = f'depth:{depth_failure};rgb:{error}'
                        if survey_failure is not None:
                            cause = f'survey:{survey_failure};{cause}'
                        raise TrackingFailure(cause) from error
                    pose = self.world_from_optical @ transform
                    self._check_motion(pose, dt)
                    self.world_from_optical = pose
                    self.position_variance += .004**2
                    self.orientation_variance += .002**2
                    diagnostics['source'] = 'rgbd_odometry'
            if diagnostics is None:
                features = self._features(gray, depth)
                if features is None or len(features) < self.gates.min_inliers:
                    register_depth(frame, frame, np.eye(4), self.depth_gates)
                diagnostics = {'source': 'stationary_launch'}
            if survey_failure is not None:
                diagnostics['survey_rejected'] = survey_failure
            if self.position_variance.max() > .20**2 or self.orientation_variance.max() > .14**2:
                raise TrackingFailure('accumulated_odometry_uncertainty')
            bootstrap = self.previous is None
            self.previous = (gray, depth, frame)
            self.last_usec = sim_usec
            return self._record(sim_usec, bootstrap=bootstrap, **diagnostics)
        except (TrackingFailure, cv2.error, np.linalg.LinAlgError) as error:
            if self.previous is None:
                return self._record(sim_usec, 'waiting_for_observable_capture:'+str(error))
            self.lost = str(error)
            return self._record(sim_usec, 'tracking_lost:'+self.lost, rejected_fits=attempts)


class CapturePairs:
    """Small exact-nanosecond pairing buffer; never pair 'latest' different times."""
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.color, self.depth = {}, {}
        self.last = -1

    def push(self, kind, stamp_ns, frame):
        if stamp_ns <= self.last:
            return
        target = self.color if kind == 'color' else self.depth
        target[stamp_ns] = frame
        while len(target) > self.capacity:
            del target[min(target)]

    def take(self, through_ns=None, *, ready=None):
        common = self.color.keys() & self.depth.keys()
        if through_ns is not None:
            common = {stamp for stamp in common if stamp <= through_ns}
        stamp = next((stamp for stamp in sorted(common, reverse=True)
                      if ready is None or ready(stamp)), None)
        if stamp is None:
            return None
        result = stamp, self.color[stamp], self.depth[stamp]
        self.last = stamp
        for target in (self.color, self.depth):
            for old in list(target):
                if old <= stamp:
                    del target[old]
        return result


class TiltHistory:
    """Bounded PX4 ATTITUDE history. Interpolate gravity, never extrapolate."""

    def __init__(self):
        self._lock = Lock()
        self._samples = deque(maxlen=64)
        self._optical_from_body = np.asarray(RGBD_BODY_FROM_OPTICAL).T

    def record(self, stamp_usec, roll, pitch):
        if not isinstance(stamp_usec, (int, np.integer)) or stamp_usec < 0:
            return
        with self._lock:
            if self._samples and stamp_usec <= self._samples[-1][0]:
                return
            up = None
            try:
                roll, pitch = float(roll), float(pitch)
            except (TypeError, ValueError):
                roll, pitch = math.nan, math.nan
            if abs(roll) <= math.pi and abs(pitch) <= math.pi/2:
                # NED/FRD roll and pitch define world up in body FLU; yaw cancels.
                cp = math.cos(pitch)
                up = self._optical_from_body @ (math.sin(pitch), math.sin(roll)*cp, math.cos(roll)*cp)
            self._samples.append((stamp_usec, up))

    def at(self, stamp_usec):
        with self._lock:
            samples = tuple(self._samples)
        later = None
        for stamp, up in reversed(samples):
            if stamp == stamp_usec:
                return InertialTilt(stamp_usec, up, MIN_TILT_STD_RAD, 0) if up is not None else None
            if stamp < stamp_usec:
                if up is None or later is None or later[1] is None:
                    return None
                span = later[0]-stamp
                if span > MAX_TILT_BRACKET_USEC:
                    return None
                fraction = (stamp_usec-stamp)/span
                other = later[1]
                angular_change = math.atan2(np.linalg.norm(np.cross(up, other)), up @ other)
                if angular_change > .20:
                    return None
                direction = (1-fraction)*up + fraction*other
                nearest_s = min(stamp_usec-stamp, later[0]-stamp_usec)*1e-6
                skew_std = nearest_s * max(.1, angular_change/(span*1e-6))
                return InertialTilt(stamp_usec, direction/np.linalg.norm(direction),
                                    math.hypot(MIN_TILT_STD_RAD, skew_std), span)
            later = stamp, up
        return None


class Vision:
    """CameraWorker-compatible Gazebo adapter. No pose/odometry subscription."""
    def __init__(self, launch_position_enu, launch_orientation_wxyz=(1, 0, 0, 0),
                 *, rejected_frame_dir=None, use_px4_tilt=False, survey_normals='central2'):
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.image_pb2 import Image, RGB_INT8, R_FLOAT32
        from gz.transport13 import Node
        survey = DepthSurvey.from_course(Path(__file__).resolve().parents[1] / 'models/course_v3/model.sdf')
        self.estimator = RGBDOdometry(launch_position_enu, launch_orientation_wxyz, survey=survey,
                                      survey_normals=survey_normals)
        self._formats = (RGB_INT8, R_FLOAT32)
        self._lock, self._pairs = Lock(), CapturePairs()
        self._clock_usec = None
        self._ranges = {}
        self._tilts = TiltHistory() if use_px4_tilt else None
        self._tilt_wait_started = {}
        self._rejected_frame_dir = Path(rejected_frame_dir) if rejected_frame_dir is not None else None
        self._rejected_frame = None
        optical_from_body = np.asarray(RGBD_BODY_FROM_OPTICAL).T
        self._tof_origin = optical_from_body @ (np.asarray(TOF_TRANSLATION)-RGBD_TRANSLATION)
        self._tof_direction = optical_from_body @ np.array((0., 0., -1.))
        self.node = Node()
        self._subscriptions = []
        for message_type, topic, callback in ((Image, COLOR_TOPIC, self._on_color),
                                              (Image, DEPTH_TOPIC, self._on_depth),
                                              (Image, TOF_TOPIC, self._on_range),
                                              (Clock, CLOCK_TOPIC, self._on_clock)):
            if not self.node.subscribe(message_type, topic, callback):
                self.close()
                raise RuntimeError('cannot subscribe to '+topic)
            self._subscriptions.append(topic)

    def _capture(self, kind, message):
        stamp = message.header.stamp
        with self._lock:
            self._pairs.push(kind, stamp.sec*1_000_000_000+stamp.nsec, message)

    def _on_color(self, message):
        self._capture('color', message)

    def _on_depth(self, message):
        self._capture('depth', message)

    def _on_range(self, message):
        if ((message.width, message.height, message.step) != (1, 1, 4) or
                message.pixel_format_type != self._formats[1] or len(message.data) != 4):
            return
        distance, = struct.unpack('<f', message.data)
        if not TOF_MIN_DEPTH < distance < TOF_MAX_DEPTH:
            return
        stamp = message.header.stamp
        with self._lock:
            self._ranges[stamp.sec*1_000_000_000 + stamp.nsec] = distance
            while len(self._ranges) > 8:
                del self._ranges[min(self._ranges)]

    def _on_clock(self, message):
        with self._lock:
            self._clock_usec = message.sim.sec*1_000_000 + message.sim.nsec//1000

    @property
    def sim_usec_now(self):
        with self._lock:
            return self._clock_usec

    def record_attitude(self, stamp_usec, roll, pitch):
        """Receive PX4 ATTITUDE roll/pitch in FRD and boot-clock microseconds."""
        if self._tilts is not None:
            self._tilts.record(stamp_usec, roll, pitch)

    def _take_capture(self, clock_ns, now_wall):
        """Consume the latest pair eligible for range and optional tilt alignment."""
        through_ns = clock_ns - TOF_SYNC_NS
        if self._tilts is None:
            return self._pairs.take(through_ns), None, 'disabled'

        tilt = None

        def ready(stamp_ns):
            nonlocal tilt
            tilt = self._tilts.at(stamp_ns//1000)
            if tilt is not None:
                return True
            started = self._tilt_wait_started.setdefault(stamp_ns, now_wall)
            return clock_ns-stamp_ns >= TILT_WAIT_NS or now_wall-started >= TILT_WAIT_WALL_S

        pair = self._pairs.take(through_ns, ready=ready)
        self._tilt_wait_started = {
            stamp: started for stamp, started in self._tilt_wait_started.items()
            if stamp in self._pairs.color and stamp in self._pairs.depth
        }
        if pair is None:
            return None, None, 'waiting_for_bracket'
        return pair, tilt, 'capture_aligned' if tilt is not None else 'missing_after_deadline'

    def poll(self):
        with self._lock:
            now = self._clock_usec
            if now is None:
                return None
            clock_ns = now*1000 + 999
            # Leave time for the independently delivered ToF sample. After
            # this deadline, the RGBD pair can still support a camera-only fit.
            pair, tilt, tilt_pairing = self._take_capture(clock_ns, time.monotonic())
            if pair is None:
                return None
            stamp_ns, color, depth = pair
            range_ns = min(self._ranges, key=lambda stamp: abs(stamp-stamp_ns), default=None)
            sample = self._ranges.get(range_ns)
        downward_range = None
        if sample is not None and abs(range_ns-stamp_ns) <= TOF_SYNC_NS:
            skew_s = (range_ns-stamp_ns)*1e-9
            downward_range = DownwardRange(sample, self._tof_origin, self._tof_direction,
                                           math.hypot(.012, skew_s))
        for message, expected, stride in ((color, self._formats[0], 3), (depth, self._formats[1], 4)):
            if (message.width, message.height) != (RGBD_WIDTH, RGBD_HEIGHT) or message.pixel_format_type != expected or message.step != RGBD_WIDTH*stride:
                raise ValueError('unexpected RGBD dimensions, format, or stride')
        rgb = np.frombuffer(color.data, dtype=np.uint8).reshape(RGBD_HEIGHT, RGBD_WIDTH, 3)
        z = np.frombuffer(depth.data, dtype='<f4').reshape(RGBD_HEIGHT, RGBD_WIDTH)
        should_remember = (self._rejected_frame_dir is not None and self._rejected_frame is None
                           and self.estimator.lost is None)
        if should_remember:
            previous_pose = self.estimator.world_from_optical
            previous_usec = self.estimator.last_usec
        record = self.estimator.estimate(rgb, z, stamp_ns//1000, now, downward_range=downward_range,
                                         inertial_tilt=tilt)
        record['px4_tilt_pairing_status'] = tilt_pairing
        if tilt_pairing == 'missing_after_deadline' and not record.get('valid') and record.get('reason'):
            record['reason'] += ';px4_tilt_bracket_missing_after_deadline'
        if should_remember and self.estimator.lost is not None:
            self._rejected_frame = {
                'rgb': rgb, 'depth_z_m': z,
                'metadata': {
                    'capture_stamp_ns': stamp_ns, 'clock_sim_usec': now,
                    'previous_capture_sim_usec': previous_usec,
                    'previous_world_from_optical': previous_pose.tolist(),
                    'camera_intrinsics': self.estimator.intrinsics.tolist(),
                    'camera_body_from_optical': RGBD_BODY_FROM_OPTICAL,
                    'camera_translation_flu_m': RGBD_TRANSLATION,
                    'survey_normals': record.get('survey_normals'),
                    'reason': record['reason'], 'rejected_fits': record.get('rejected_fits', {}),
                    'tof_stamp_ns': range_ns if downward_range is not None else None,
                    'tof_range_m': downward_range.distance_m if downward_range is not None else None,
                    'tof_range_std_m': downward_range.std_m if downward_range is not None else None,
                    'tof_origin_optical_m': self._tof_origin.tolist(),
                    'tof_ray_optical': self._tof_direction.tolist(),
                    'px4_tilt_pairing_status': tilt_pairing,
                    'px4_tilt_prior': {
                        'capture_usec': tilt.capture_usec, 'bracket_usec': tilt.bracket_usec,
                        'up_optical': tilt.up_optical.tolist(), 'std_rad': tilt.std_rad,
                    } if tilt is not None else None,
                },
            }
        record['depth_z_m'] = z
        if downward_range is not None:
            record['tof_range_m'] = downward_range.distance_m
            record['tof_offset_usec'] = (range_ns-stamp_ns)/1000
        return record

    def close(self):
        for topic in self._subscriptions:
            self.node.unsubscribe(topic)
        self._subscriptions.clear()
        if self._rejected_frame is not None:
            self._write_rejected()

    def _write_rejected(self):
        frame, self._rejected_frame = self._rejected_frame, None
        try:
            self._rejected_frame_dir.mkdir(parents=True, exist_ok=True)
            with (self._rejected_frame_dir / 'first-rejected.npz').open('xb') as output:
                np.savez(output, rgb=frame['rgb'], depth_z_m=frame['depth_z_m'],
                         metadata_json=json.dumps(frame['metadata'], allow_nan=False))
        except (OSError, ValueError) as error:
            logging.getLogger(__name__).warning('could not save first rejected RGBD frame: %s', error)


def main():
    """Read-only sensor smoke test; does not connect to or command PX4."""
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-position', type=float, nargs=3, required=True,
                        metavar=('X', 'Y', 'Z'), help='known stationary base_link ENU position, metres')
    parser.add_argument('--launch-quaternion', type=float, nargs=4, default=(1., 0, 0, 0),
                        metavar=('W', 'X', 'Y', 'Z'))
    parser.add_argument('--seconds', type=float, default=20., help='wall-clock observation duration')
    args = parser.parse_args()
    vision = Vision(args.launch_position, args.launch_quaternion)
    end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < end:
            record = vision.poll()
            if record is not None:
                del record['depth_z_m']
                print(json.dumps(record, allow_nan=False), flush=True)
            else:
                time.sleep(.002)
    finally:
        vision.close()


if __name__ == '__main__':
    main()
