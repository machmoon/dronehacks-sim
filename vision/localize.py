"""Estimate body pose from the camera image and the fixed AprilTag map."""

import argparse
import json
import math
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np

from .course import (
    BODY_FROM_OPTICAL, CAMERA_HFOV, CAMERA_HEIGHT, CAMERA_TOPIC, CAMERA_TRANSLATION,
    CAMERA_WIDTH, course_markers,
)

CLOCK_TOPIC = '/world/dronehacks/clock'


def calibration():
    focal = CAMERA_WIDTH / (2 * math.tan(CAMERA_HFOV / 2))
    return np.array(((focal, 0, CAMERA_WIDTH / 2), (0, focal, CAMERA_HEIGHT / 2), (0, 0, 1)))


def orientation_wxyz(rotation):
    axis_angle = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    return [math.cos(angle / 2), *(axis_angle * (math.sin(angle / 2) / angle)).tolist()]


class PoseEstimator:
    def __init__(self):
        cv2.setNumThreads(1)
        self.intrinsics = calibration()
        self.map = {tag.id: np.array(tag.corners()) for tag in course_markers().values()}
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(dictionary, self._set_refinement(parameters))
            self.detect = self.detector.detectMarkers
        else:  # Ubuntu 24.04 packages OpenCV 4.6
            parameters = self._set_refinement(cv2.aruco.DetectorParameters_create())
            self.detect = lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    @staticmethod
    def _set_refinement(parameters):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        return parameters

    def estimate(self, bgr, sim_usec):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect(gray)
        # A tag must resolve roughly two pixels per cell for a stable pose.
        detections = [] if ids is None else [
            (int(tag_id), quad.reshape(4, 2))
            for tag_id, quad in zip(ids.ravel(), corners)
            if int(tag_id) in self.map and cv2.arcLength(quad, True) >= 60
        ]
        record = {
            'sim_usec': sim_usec,
            'stamp_sim_s': sim_usec * 1e-6,
            'valid': False,
            'position_enu': None,
            'orientation_enu_flu_wxyz': None,
            'orientation_xyzw': None,
            'tag_count': len(detections),
            'tag_ids': [tag_id for tag_id, _ in detections],
            'reprojection_error_px': None,
            'reason': 'no usable mapped tags',
        }
        if not detections:
            return record, detections

        world = np.concatenate([self.map[tag_id] for tag_id, _ in detections])
        origin = world.mean(axis=0)
        object_points = world - origin
        image = np.concatenate([quad for _, quad in detections]).astype(np.float64)
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, image, self.intrinsics, None, flags=cv2.SOLVEPNP_SQPNP,
        )
        if len(detections) == 1:
            # Include the two planar hypotheses explicitly for a lone tag.
            # SQPnP also handles the near-nadir numerical cases where IPPE can fail.
            _, planar_rvecs, planar_tvecs, _ = cv2.solvePnPGeneric(
                object_points, image, self.intrinsics, None, flags=cv2.SOLVEPNP_IPPE,
            )
            rvecs, tvecs = (*rvecs, *planar_rvecs), (*tvecs, *planar_tvecs)
        solutions = []
        for rvec, tvec in zip(rvecs, tvecs):
            if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
                continue
            optical_from_world = cv2.Rodrigues(rvec)[0]
            camera_points = (optical_from_world @ object_points.T) + tvec
            world_from_optical = optical_from_world.T
            world_from_body = world_from_optical @ np.asarray(BODY_FROM_OPTICAL).T
            camera = origin - world_from_optical @ tvec.ravel()
            body = camera - world_from_body @ np.asarray(CAMERA_TRANSLATION)
            if camera_points[2].min() <= 0 or camera[2] <= world[:, 2].max():
                continue
            if world_from_body[2, 2] < math.cos(math.radians(45)):
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.intrinsics, None)
            residuals = projected.reshape(-1, 2) - image
            rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
            solutions.append((rms, body, world_from_body))
        if not solutions:
            record['reason'] = 'no physically plausible pose'
            return record, detections

        solutions.sort(key=lambda solution: solution[0])
        error, position, rotation = solutions[0]
        record['reprojection_error_px'] = error
        if error > 2.0:
            record['reason'] = 'reprojection residual above 2 px'
            return record, detections
        if any(alternate_error - error < 0.25 and np.linalg.norm(position - alternate_position) > 0.10
               for alternate_error, alternate_position, _ in solutions[1:]):
            record['reason'] = 'ambiguous planar pose'
            return record, detections

        w, x, y, z = orientation_wxyz(rotation)
        record.update(valid=True, position_enu=position.tolist(), reason=None,
                      orientation_enu_flu_wxyz=[w, x, y, z], orientation_xyzw=[x, y, z, w])
        return record, detections


class Vision:
    """Return each new camera pose at most once, with simulation clock for time sync."""

    def __init__(self, output_dir=None):
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.image_pb2 import Image, RGB_INT8
        from gz.transport13 import Node

        self.estimator = PoseEstimator()
        self._rgb_format = RGB_INT8
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._next_preview = 0.0
        self._lock = Lock()
        self._frame = None
        self._processed_usec = -1
        self._clock_usec = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for name in ('camera.jpg', 'latest.json'):
                (self.output_dir / name).unlink(missing_ok=True)
        self.node = Node()
        if not self.node.subscribe(Image, CAMERA_TOPIC, self._on_image):
            raise RuntimeError(f'cannot subscribe to {CAMERA_TOPIC}')
        if not self.node.subscribe(Clock, CLOCK_TOPIC, self._on_clock):
            self.close()
            raise RuntimeError(f'cannot subscribe to {CLOCK_TOPIC}')

    def _on_image(self, message):
        with self._lock:
            self._frame = message

    def _on_clock(self, message):
        with self._lock:
            self._clock_usec = message.sim.sec * 1_000_000 + message.sim.nsec // 1_000

    @property
    def sim_usec_now(self):
        with self._lock:
            return self._clock_usec

    def close(self):
        self.node.unsubscribe(CAMERA_TOPIC)
        self.node.unsubscribe(CLOCK_TOPIC)

    def poll(self):
        with self._lock:
            frame, self._frame = self._frame, None
        if frame is None:
            return None
        stamp = frame.header.stamp
        sim_usec = stamp.sec * 1_000_000 + stamp.nsec // 1_000
        if sim_usec <= self._processed_usec:
            return None
        self._processed_usec = sim_usec
        if (frame.width, frame.height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            raise ValueError(f'unexpected camera dimensions {frame.width}x{frame.height}')
        if frame.pixel_format_type != self._rgb_format or frame.step != CAMERA_WIDTH * 3:
            raise ValueError(f'unexpected camera format {frame.pixel_format_type}, stride {frame.step}')
        rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        record, detections = self.estimator.estimate(bgr, sim_usec)
        now = time.monotonic()
        if self.output_dir is not None and now >= self._next_preview:
            self._write_preview(bgr, record, detections)
            self._next_preview = now + 0.1
        return record

    def _write_preview(self, bgr, record, detections):
        for tag_id, quad in detections:
            polygon = np.rint(quad).astype(np.int32)
            cv2.polylines(bgr, [polygon], True, (36, 224, 66), 2, cv2.LINE_AA)
            cv2.putText(bgr, str(tag_id), tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (36, 224, 66), 1, cv2.LINE_AA)
        status = f"{record['tag_count']} tags | {record['stamp_sim_s']:.2f} s | "
        status += 'pose valid' if record['valid'] else record['reason']
        cv2.rectangle(bgr, (0, 0), (CAMERA_WIDTH, 25), (24, 24, 24), -1)
        cv2.putText(bgr, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 240, 230), 1,
                    cv2.LINE_AA)
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if ok:
            self._replace('camera.jpg', jpeg.tobytes())
        self._replace('latest.json', (json.dumps(record, allow_nan=False) + '\n').encode())

    def _replace(self, name, content):
        temp = self.output_dir / (name + '.tmp')
        temp.write_bytes(content)
        temp.replace(self.output_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/vision'))
    parser.add_argument('--frames', type=int, default=0, help='stop after this many camera frames')
    args = parser.parse_args()
    vision = Vision(args.output_dir)
    count = 0
    try:
        while args.frames == 0 or count < args.frames:
            record = vision.poll()
            if record is None:
                time.sleep(0.005)
                continue
            print(json.dumps(record, allow_nan=False), flush=True)
            count += 1
    finally:
        vision.close()


if __name__ == '__main__':
    main()
