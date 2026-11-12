import math
import unittest

from mission.px4_frames import camera_pose_to_px4, pose_covariance


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


class Px4FrameTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=places)

    def test_position_enu_becomes_local_ned(self):
        position, _ = camera_pose_to_px4((3.0, 5.0, 2.0), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(position, (5.0, 3.0, -2.0))

    def test_identity_camera_orientation_has_map_heading(self):
        _, rpy = camera_pose_to_px4((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        self.assertTupleAlmostEqual(rpy, (0.0, 0.0, math.pi / 2))

    def test_enu_yaw_pi_over_two_maps_to_ned_yaw_zero(self):
        q = quaternion_from_euler(0.0, 0.0, math.pi / 2)
        _, rpy = camera_pose_to_px4((0.0, 0.0, 0.0), q)
        self.assertTupleAlmostEqual(rpy, (0.0, 0.0, 0.0))

    def test_roll_is_preserved_as_frd_roll(self):
        q = quaternion_from_euler(math.pi / 2, 0.0, 0.0)
        _, rpy = camera_pose_to_px4((0.0, 0.0, 0.0), q)
        self.assertTupleAlmostEqual(rpy, (math.pi / 2, 0.0, math.pi / 2))

    def test_pose_covariance_uses_mavlink_upper_triangle_diagonal_slots(self):
        covariance = pose_covariance(position_std=0.3, angle_std=0.2)
        self.assertEqual(len(covariance), 21)
        expected = {0: 0.09, 6: 0.09, 11: 0.09,
                    15: 0.04, 18: 0.04, 20: 0.04}
        for index, value in enumerate(covariance):
            self.assertAlmostEqual(value, expected.get(index, 0.0))


if __name__ == '__main__':
    unittest.main()
