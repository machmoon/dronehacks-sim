"""Marker-free RGBD VO. Metric depth is a camera measurement, not a pose feed.

Frame convention: solvePnP estimates current_optical <- previous_optical;
world <- current_optical is accumulated using its inverse. A known stationary
launch pose is REQUIRED. Tracking loss latches until an explicit reset, which
must only be used at a new independently established stationary pose.
"""
from dataclasses import dataclass
import math
from threading import Lock

import cv2
import numpy as np

from .course import (BODY_FROM_OPTICAL, CAMERA_TRANSLATION, CAMERA_WIDTH,
                     CAMERA_HEIGHT, CAMERA_HFOV)
from .rgbd_scene import COLOR_TOPIC, DEPTH_TOPIC

CLOCK_TOPIC = '/world/dronehacks/clock'


@dataclass(frozen=True)
class Gates:
    min_inliers: int = 24
    min_ratio: float = .65
    max_reprojection: float = 1.5
    max_gap_s: float = .3
    max_age_s: float = .3
    max_speed: float = 3.0
    max_angular_speed: float = 3.0
    min_depth: float = .04
    max_depth: float = 8.0
    min_image_span: float = .12


class TrackingFailure(ValueError):
    pass


def calibration(width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    focal = width / (2 * math.tan(CAMERA_HFOV / 2))
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
                 intrinsics=None, gates=Gates()):
        cv2.setNumThreads(1)
        self.intrinsics = calibration() if intrinsics is None else np.array(intrinsics, dtype=float)
        self.gates = gates
        self.reset(launch_position_enu, launch_orientation_wxyz)

    def reset(self, known_stationary_position, known_stationary_orientation=(1, 0, 0, 0)):
        position = np.asarray(known_stationary_position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('an independently known launch base_link position is required')
        body_rotation = rotation_from_quaternion(known_stationary_orientation)
        self.world_from_optical = np.eye(4)
        self.world_from_optical[:3, :3] = body_rotation @ np.array(BODY_FROM_OPTICAL)
        self.world_from_optical[:3, 3] = position + body_rotation @ np.array(CAMERA_TRANSLATION)
        self.previous = None
        self.lost = None
        self.last_usec = None

    def _record(self, sim_usec, reason=None, **diagnostics):
        record = dict(valid=False, sim_usec=int(sim_usec), stamp_sim_s=sim_usec*1e-6,
                      position_enu=None, orientation_enu_flu_wxyz=None,
                      orientation_xyzw=None, reason=reason, source='rgbd_vo', **diagnostics)
        if reason is None:
            rotation = self.world_from_optical[:3, :3] @ np.array(BODY_FROM_OPTICAL).T
            position = self.world_from_optical[:3, 3] - rotation @ np.array(CAMERA_TRANSLATION)
            q = quaternion_from_rotation(rotation)
            record.update(valid=True, position_enu=position.tolist(),
                          orientation_enu_flu_wxyz=q, orientation_xyzw=q[1:]+q[:1])
        return record

    def _features(self, gray, depth):
        mask = (np.isfinite(depth) & (depth >= self.gates.min_depth) &
                (depth <= self.gates.max_depth)).astype('uint8')*255
        features = cv2.goodFeaturesToTrack(gray, maxCorners=600, qualityLevel=.01,
                                           minDistance=6, mask=mask, blockSize=5)
        if features is None or len(features) < self.gates.min_inliers:
            raise TrackingFailure('insufficient_texture')
        return features

    def estimate(self, rgb, depth, sim_usec, now_usec=None):
        """Consume synchronized RGB uint8 and optical-Z float32 depth in metres."""
        if self.lost:
            return self._record(sim_usec, 'tracking_lost:'+self.lost)
        try:
            if now_usec is not None and not 0 <= now_usec-sim_usec <= self.gates.max_age_s*1e6:
                raise TrackingFailure('stale_capture')
            if self.last_usec is not None and sim_usec <= self.last_usec:
                raise TrackingFailure('nonmonotonic_capture')
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
                raise TrackingFailure('invalid_image_shape')
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            next_features = self._features(gray, depth)
            diagnostics = {'bootstrap': self.previous is None}
            if self.previous is not None:
                old_gray, old_depth, old_features = self.previous
                dt = (sim_usec-self.last_usec)*1e-6
                if not 0 < dt <= self.gates.max_gap_s:
                    raise TrackingFailure('capture_gap')
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
                rays = np.c_[previous_uv, np.ones(len(previous_uv))] @ np.linalg.inv(self.intrinsics).T
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
                if depth_inliers < self.gates.min_inliers or depth_inliers/max(1, np.count_nonzero(geometric_inliers)) < self.gates.min_ratio:
                    raise TrackingFailure('inconsistent_depth')
                diagnostics['depth_inliers'] = depth_inliers
                self.world_from_optical = self.world_from_optical @ np.linalg.inv(transform)
            self.previous = (gray, np.array(depth, copy=True), next_features)
            self.last_usec = sim_usec
            return self._record(sim_usec, **diagnostics)
        except (TrackingFailure, cv2.error) as error:
            self.lost = str(error)
            return self._record(sim_usec, 'tracking_lost:'+self.lost)


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

    def take(self):
        common = self.color.keys() & self.depth.keys()
        if not common:
            return None
        stamp = max(common)
        result = stamp, self.color[stamp], self.depth[stamp]
        self.last = stamp
        for target in (self.color, self.depth):
            for old in list(target):
                if old <= stamp:
                    del target[old]
        return result


class Vision:
    """CameraWorker-compatible Gazebo adapter. No pose/odometry subscription."""
    def __init__(self, launch_position_enu, launch_orientation_wxyz=(1, 0, 0, 0)):
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.image_pb2 import Image, RGB_INT8, R_FLOAT32
        from gz.transport13 import Node
        self.estimator = RGBDOdometry(launch_position_enu, launch_orientation_wxyz)
        self._formats = (RGB_INT8, R_FLOAT32)
        self._lock, self._pairs = Lock(), CapturePairs()
        self._clock_usec = None
        self.node = Node()
        self._subscriptions = []
        for message_type, topic, callback in ((Image, COLOR_TOPIC, self._on_color),
                                              (Image, DEPTH_TOPIC, self._on_depth),
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

    def _on_clock(self, message):
        with self._lock:
            self._clock_usec = message.sim.sec*1_000_000 + message.sim.nsec//1000

    @property
    def sim_usec_now(self):
        with self._lock:
            return self._clock_usec

    def poll(self):
        with self._lock:
            now = self._clock_usec
            pair = self._pairs.take() if now is not None else None
        if pair is None:
            return None
        stamp_ns, color, depth = pair
        for message, expected, stride in ((color, self._formats[0], 3), (depth, self._formats[1], 4)):
            if (message.width, message.height) != (CAMERA_WIDTH, CAMERA_HEIGHT) or message.pixel_format_type != expected or message.step != CAMERA_WIDTH*stride:
                raise ValueError('unexpected RGBD dimensions, format, or stride')
        rgb = np.frombuffer(color.data, dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        z = np.frombuffer(depth.data, dtype='<f4').reshape(CAMERA_HEIGHT, CAMERA_WIDTH)
        return self.estimator.estimate(rgb, z, stamp_ns//1000, now)

    def close(self):
        for topic in self._subscriptions:
            self.node.unsubscribe(topic)
        self._subscriptions.clear()


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
                print(json.dumps(record, allow_nan=False), flush=True)
            else:
                time.sleep(.002)
    finally:
        vision.close()


if __name__ == '__main__':
    main()
