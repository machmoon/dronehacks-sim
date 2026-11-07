import unittest

from vision.velocity import CameraOdometry


def observation(t, x=0, y=0, z=1):
    return dict(valid=True, sim_usec=round(t*1e6), position_enu=[x,y,z],
                orientation_enu_flu_wxyz=[1,0,0,0])


class CameraVelocityTests(unittest.TestCase):
    def test_stationary_and_moving_camera_with_centered_timestamp(self):
        for speed in (0., .6):
            fit = CameraOdometry()
            for i in range(7):
                result = fit.update(observation(i/30, x=speed*i/30))
            self.assertAlmostEqual(result['velocity_enu'][0], speed, places=5)
            self.assertLess(result['sim_usec'], 200000)
            self.assertAlmostEqual(result['sim_usec']/1e6, .1, places=5)

    def test_gap_and_jump_do_not_emit_fake_velocity(self):
        fit = CameraOdometry()
        for i in range(7):fit.update(observation(i/30))
        self.assertIsNone(fit.update(observation(1)))
        for i in range(1,7):fit.update(observation(1+i/30))
        self.assertIsNone(fit.update(observation(1+7/30, x=1)))
