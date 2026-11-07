"""The estimator's optical geometry must agree with the airframe in Gazebo."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class CameraSceneTests(unittest.TestCase):
    def test_skybeta_camera_extrinsics_intrinsics_and_noise_match_generated_world(self):
        with tempfile.TemporaryDirectory() as directory:
            world = Path(directory)/'world.sdf'
            script = '''
import json
from pathlib import Path
from vision.rgbd_scene import generate, RGBD_TRANSLATION, TOF_TRANSLATION, RGBD_HFOV
from vision.rgbd_odometry import calibration
from vehicle.skybeta import load, properties
import sys
generate(Path('worlds/course.sdf'), Path(sys.argv[1]))
p,g=load()
print(json.dumps(dict(camera=RGBD_TRANSLATION,tof=TOF_TRANSLATION,hfov=RGBD_HFOV,
                     k=calibration().tolist(),mass=properties(p,g)['mass'])))
'''
            result = subprocess.run([sys.executable, '-c', script, str(world)], cwd=ROOT,
                                    env={**os.environ, 'SIM_AIRFRAME': 'skybeta'},
                                    capture_output=True, text=True, check=True)
            calibration = json.loads(result.stdout)
            tree = ET.parse(world)
            body = tree.find(".//model[@name='course_quad']/link[@name='base_link']")
            sensor = body.find("sensor[@name='oak_left']")
            self.assertEqual(sensor.get('type'), 'rgbd_camera')
            self.assertEqual(list(map(float, sensor.findtext('pose').split()[:3])), calibration['camera'])
            self.assertAlmostEqual(float(sensor.findtext('camera/horizontal_fov')), calibration['hfov'])
            self.assertEqual(list(map(float, body.findtext("sensor[@name='down_tof']/pose").split()[:3])), calibration['tof'])
            self.assertIsNone(body.find("sensor[@name='oak_right']"))
            self.assertFalse(tree.findall('.//plugin[@name="gz::sim::systems::OdometryPublisher"]'))
            self.assertIsNone(tree.find('.//model[@name="vision_landmarks"]'))
            for kind in ('angular_velocity', 'linear_acceleration'):
                for axis in 'xyz':
                    std = body.findtext(f"sensor[@name='imu_sensor']/imu/{kind}/{axis}/noise/stddev")
                    self.assertGreater(float(std), 0.)
            self.assertEqual(len(tree.findall(".//model[@name='course_quad']/link")), 5)


if __name__ == '__main__':
    unittest.main()
