"""Calibration and landmark survey shared by the renderer and pose estimator."""

from dataclasses import dataclass
import math
import os


CAMERA_TOPIC = '/dronehacks/down_camera'
CAMERA_WIDTH = int(os.environ.get('SIM_CAMERA_WIDTH', '480'))
CAMERA_HEIGHT = int(os.environ.get('SIM_CAMERA_HEIGHT', '360'))
if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
    raise ValueError('SIM_CAMERA_WIDTH and SIM_CAMERA_HEIGHT must be positive')
CAMERA_HFOV = 1.95
CAMERA_RATE = 30
# Gazebo camera +X looks down at +pi/2 pitch. Relative to x500 base_link (FLU).
CAMERA_TRANSLATION = (0.0, 0.0, -0.075)
CAMERA_PITCH = math.pi / 2
# Columns are optical +X (right), +Y (down), +Z (forward) expressed in body FLU.
BODY_FROM_OPTICAL = ((0, -1, 0), (-1, 0, 0), (0, 0, -1))
LAUNCH_XY = (-2.5146, 0.9144)
LANDING_XY = (-2.0422, -0.7925)


@dataclass(frozen=True)
class Marker:
    id: int
    x: float
    y: float
    z: float
    size: float

    def corners(self):
        """World ENU corners in the AprilTag decoder's canonical TL, TR, BR, BL order."""
        h = self.size / 2
        return ((self.x - h, self.y + h, self.z), (self.x + h, self.y + h, self.z),
                (self.x + h, self.y - h, self.z), (self.x - h, self.y - h, self.z))


def course_markers():
    # Reserve the ground directly under the vehicle for four near-field tags at
    # each pad. Keep quiet margins free of neighboring tags.
    specs = [(x + dx, y + dy, 0.006, 0.058)
             for x, y in (LAUNCH_XY, LANDING_XY)
             for dx in (-0.080, 0.080)
             for dy in (-0.075, 0.075)]
    for i in range(12):
        for j in range(12):
            x, y = -2.75 + i * 0.5, -2.75 + j * 0.5
            if any(abs(x - px) < 0.25 and abs(y - py) < 0.25
                   for px, py in (LAUNCH_XY, LANDING_XY)):
                continue
            if x >= 1.75 and -0.25 <= y <= 1.25:
                continue  # underneath the oval tunnel, occluded from inside
            specs.append((x, y, 0.006, 0.20))
    # Tunnel mesh has a curved inner bottom. These flat, collisionless decals
    # sit above the inner surface across their full width.
    specs.extend((2.3114, y, 0.053, 0.20) for y in (-0.14, 0.24, 0.62, 1.00))
    return {i: Marker(i, *spec) for i, spec in enumerate(specs)}
