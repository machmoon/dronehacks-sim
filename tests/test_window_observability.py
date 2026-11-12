"""Check the forward camera bearing and the window-entry heading handoff."""
import json
import math
from pathlib import Path
import unittest

from mission.heading import route_yaw
from vision.rgbd_scene import RGBD_HFOV, RGBD_TRANSLATION


class WindowHeadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        route = Path(__file__).resolve().parents[1] / 'mission/course-window-x500.json'
        course = json.loads(route.read_text())
        waypoints = {waypoint['phase']: waypoint for waypoint in course['waypoints']}
        cls.entry = waypoints['WINDOW_ALIGN']['position']
        cls.exit = waypoints['WINDOW_CROSS']['position']
        cls.arrival_radius = waypoints['WINDOW_ALIGN']['arrival_radius']
        cls.center = course['window']['center']

    def test_crossing_handoff_is_continuous_at_entry_and_bounded_when_early(self):
        def change(position):
            before = route_yaw(position, self.entry, self.exit, 'WINDOW_ALIGN')
            after = route_yaw(position, self.exit, None, 'WINDOW_CROSS')
            return abs(math.remainder(after-before, 2*math.pi))

        self.assertAlmostEqual(change(self.entry), 0.)
        for degree in range(360):
            angle = math.radians(degree)
            position = (self.entry[0] + self.arrival_radius*math.cos(angle),
                        self.entry[1] + self.arrival_radius*math.sin(angle),
                        self.entry[2])
            self.assertLess(change(position), math.radians(4.))

    def test_center_bearing_brings_the_rejected_view_into_the_nose_camera(self):
        # Base-link candidate at the course-04 rejection, from its saved RGBD fit.
        position = (.26557524, -2.00255931, .99830087)

        def camera_angle(heading):
            fx, fy = math.cos(heading), math.sin(heading)
            rx = self.center[0] - position[0] - RGBD_TRANSLATION[0]*fx
            ry = self.center[1] - position[1] - RGBD_TRANSLATION[0]*fy
            return math.atan2(fx*ry-fy*rx, fx*rx+fy*ry)

        self.assertGreater(abs(camera_angle(math.radians(155.23))), RGBD_HFOV/2)
        heading = route_yaw(position, self.entry, self.exit, 'WINDOW_ALIGN')
        self.assertAlmostEqual(camera_angle(heading), 0., places=6)


if __name__ == '__main__':
    unittest.main()
