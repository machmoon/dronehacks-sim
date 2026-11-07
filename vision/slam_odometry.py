"""SITL pose bridge for the bounded RGBD sparse map.

A known stationary launch anchors the initial optical map to course ENU. This
adapter never loads the course geometry or any simulator pose. Relocalization
is intentionally not forwarded: any tracking break latches invalid until a
new, independently established stationary launch. Uncertainty is a conservative
engineering budget, not calibrated real-camera covariance.
"""
import math

import cv2
import numpy as np

from .rgbd_odometry import calibration, rotation_from_quaternion, quaternion_from_rotation
from .rgbd_scene import RGBD_BODY_FROM_OPTICAL, RGBD_TRANSLATION
from .sparse_slam import SparseMap, Config, inverse_pose


class SLAMOdometry:
    def __init__(self, launch_position_enu, launch_orientation_wxyz=(1, 0, 0, 0),
                 intrinsics=None, config=Config(), *, max_age_s=.3):
        cv2.setNumThreads(1)
        self.intrinsics = calibration() if intrinsics is None else np.asarray(intrinsics, dtype=float)
        self.tracker = SparseMap(self.intrinsics, config)
        self.body_from_optical = np.asarray(RGBD_BODY_FROM_OPTICAL)
        self.camera_translation = np.asarray(RGBD_TRANSLATION)
        rotation = rotation_from_quaternion(launch_orientation_wxyz)
        position = np.asarray(launch_position_enu, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('a finite, independently known launch position is required')
        self.world_from_map = np.eye(4)
        self.world_from_map[:3, :3] = rotation @ self.body_from_optical
        self.world_from_map[:3, 3] = position + rotation @ self.camera_translation
        self.world_from_optical = self.world_from_map.copy()
        self.optical_from_body = np.eye(4)
        self.optical_from_body[:3, :3] = self.body_from_optical.T
        self.optical_from_body[:3, 3] = -self.body_from_optical.T @ self.camera_translation
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('positive maximum capture age required')
        self.max_age_s = max_age_s
        self.lost = self.last_usec = self.first_usec = None
        self.path_length_m = 0.

    def estimate(self, rgb, depth, sim_usec, now_usec=None, **unused_priors):
        # PX4 tilt and ToF may be logged by the transport but are not consumed
        # by this independent RGBD mapper; no feedback of PX4 position/yaw.
        stamp_ok = isinstance(sim_usec, (int, np.integer)) and sim_usec >= 0
        record = dict(valid=False, sim_usec=int(sim_usec) if stamp_ok else 0,
                      stamp_sim_s=sim_usec*1e-6 if stamp_ok else None,
                      source='sparse_rgbd_slam', position_enu=None,
                      orientation_enu_flu_wxyz=None, orientation_xyzw=None,
                      quality=0, reason=None, static_survey_used=False)
        reason = self.lost
        if not reason:
            if not stamp_ok:
                reason = 'invalid_capture_timestamp'
            elif self.last_usec is not None and sim_usec <= self.last_usec:
                reason = 'nonmonotonic_capture'
            elif now_usec is not None and not 0 <= now_usec-sim_usec <= self.max_age_s*1e6:
                reason = 'stale_capture'
            elif self.last_usec is not None and sim_usec-self.last_usec > self.tracker.cfg.max_gap_s*1e6:
                reason = 'capture_gap'
        if reason:
            self.lost = reason
            record['reason'] = 'tracking_lost:'+reason
            return record
        result = self.tracker.estimate(rgb, depth, sim_usec*1e-6)
        record['slam'] = {k: v for k, v in result.items() if k != 'map_from_camera'}
        if not result['valid']:
            reason = result['reason']
            if self.first_usec is None and reason == 'insufficient_texture':
                record['reason'] = 'waiting_for_observable_capture:'+reason
                return record
        elif result['relocalized']:
            reason = 'relocalization_requires_stationary_reset'
        else:
            world_from_optical = self.world_from_map @ np.asarray(result['map_from_camera'])
            delta = inverse_pose(self.world_from_optical) @ world_from_optical
            distance = float(np.linalg.norm(delta[:3, 3]))
            angle = float(np.linalg.norm(cv2.Rodrigues(delta[:3, :3])[0]))
            dt = 0 if self.last_usec is None else (sim_usec-self.last_usec)*1e-6
            if (distance > self.tracker.cfg.max_speed_m_s*dt+.005 or
                    angle > self.tracker.cfg.max_angular_speed_rad_s*dt+.01):
                reason = 'discontinuous_pose'
            else:
                self.path_length_m += distance
                elapsed = 0 if self.first_usec is None else (sim_usec-self.first_usec)*1e-6
                p_std = .03 + .02*self.path_length_m + .001*elapsed
                r_std = .04 + .003*self.path_length_m + .0003*elapsed
                if p_std > .20 or r_std > .12:
                    reason = 'odometry_uncertainty_budget'
                else:
                    self.world_from_optical = world_from_optical
                    body = world_from_optical @ self.optical_from_body
                    q = quaternion_from_rotation(body[:3, :3])
                    record.update(valid=True, position_enu=body[:3, 3].tolist(),
                                  orientation_enu_flu_wxyz=q, orientation_xyzw=q[1:]+q[:1],
                                  position_std_m=[p_std]*3, orientation_std_rad=[r_std]*3,
                                  quality=max(1, int(100*(1-p_std/.25))),
                                  uncertainty_model='uncalibrated_distance_time_budget')
                    if self.first_usec is None:
                        self.first_usec = sim_usec
                    self.last_usec = sim_usec
        if reason:
            self.lost = reason
            record['reason'] = 'tracking_lost:'+reason
        return record
