"""Generate a marker-free RGBD variant; adds a sensor, never visual landmarks."""
import argparse
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from .scene import generate as generate_camera_scene

RGBD_TOPIC = '/dronehacks/rgbd'
COLOR_TOPIC = RGBD_TOPIC + '/image'
DEPTH_TOPIC = RGBD_TOPIC + '/depth_image'
RGBD_WIDTH = 384
RGBD_HEIGHT = 256
RGBD_HFOV = math.pi / 2
RGBD_RATE = 20
RGBD_TRANSLATION = (.34, 0., 0.)
RGBD_BODY_FROM_OPTICAL = ((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.))
LAUNCH_YAW = math.pi / 2
TOF_TOPIC = '/dronehacks/tof'
TOF_TRANSLATION = (0., 0., -.075)
TOF_MIN_DEPTH = .025
TOF_MAX_DEPTH = 4.


def generate(world_path, output_path, launch_z=0.0):
    output_path = Path(output_path)
    generate_camera_scene(Path(world_path), output_path, launch_z, with_test_markers=False)
    tree = ET.parse(output_path)
    factor = float(os.environ.get('SIM_REAL_TIME_FACTOR', '.3'))
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('SIM_REAL_TIME_FACTOR must be in (0, 1]')
    tree.find('./world/physics/real_time_factor').text = str(factor)
    sensor = tree.find(".//model[@name='course_quad']/link[@name='down_camera_link']/sensor")
    if sensor is None:
        raise ValueError('camera fixture not found')
    sensor.set('type', 'rgbd_camera')
    sensor.find('topic').text = RGBD_TOPIC
    sensor.find('update_rate').text = str(RGBD_RATE)
    sensor.find('camera/horizontal_fov').text = str(RGBD_HFOV)
    sensor.find('camera/image/width').text = str(RGBD_WIDTH)
    sensor.find('camera/image/height').text = str(RGBD_HEIGHT)
    # Gazebo looks along link +X. Optical +Z is forward, +X right and +Y down.
    link = tree.find(".//model[@name='course_quad']/link[@name='down_camera_link']")
    link.find('pose').text = '0.34 0 0 0 0 0'
    model = tree.find(".//model[@name='course_quad']")
    model.find('include/uri').text = 'model://flight_x500'
    pose = model.find('pose').text.split()
    pose[-1] = str(LAUNCH_YAW)
    model.find('pose').text = ' '.join(pose)
    # A single downward range ray supplies height when the forward image
    # sees only vertical surfaces. Image messages contain no vehicle pose.
    model.append(ET.fromstring(f'''<link name="down_tof_link">
      <pose relative_to="base_link">0 0 -0.075 0 {math.pi/2} 0</pose>
      <inertial><mass>0.001</mass><inertia><ixx>1e-7</ixx><iyy>1e-7</iyy><izz>1e-7</izz></inertia></inertial>
      <sensor name="down_tof" type="depth_camera">
        <always_on>true</always_on><update_rate>{RGBD_RATE}</update_rate><topic>{TOF_TOPIC}</topic>
        <camera><horizontal_fov>0.1</horizontal_fov>
          <image><width>1</width><height>1</height><format>R_FLOAT32</format></image>
          <clip><near>{TOF_MIN_DEPTH}</near><far>{TOF_MAX_DEPTH}</far></clip>
        </camera>
      </sensor>
    </link>'''))
    model.append(ET.fromstring('''<joint name="down_tof_joint" type="fixed">
      <parent>base_link</parent><child>down_tof_link</child></joint>'''))
    # No texture, landmark or collision geometry is introduced here. The
    # starting yaw is a surveyed placement, not a simulator pose measurement.
    ET.indent(tree, space='  ')
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, default=Path('worlds/course.sdf'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch-z', type=float, default=0.0)
    args = parser.parse_args()
    generate(args.world, args.output, args.launch_z)


if __name__ == '__main__':
    main()
