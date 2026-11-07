"""Generate a marker-free RGBD variant; adds a sensor, never visual landmarks."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from .scene import generate as generate_camera_scene

RGBD_TOPIC = '/dronehacks/rgbd'
COLOR_TOPIC = RGBD_TOPIC + '/image'
DEPTH_TOPIC = RGBD_TOPIC + '/depth_image'


def generate(world_path, output_path, launch_z=0.0):
    output_path = Path(output_path)
    # This existing RGBD odometry diagnostic uses the downward calibration.
    generate_camera_scene(Path(world_path), output_path, launch_z, with_test_markers=False,
                          airframe='x500')
    tree = ET.parse(output_path)
    sensor = tree.find(".//model[@name='course_quad']/link[@name='down_camera_link']/sensor")
    if sensor is None:
        raise ValueError('camera fixture not found')
    sensor.set('type', 'rgbd_camera')
    sensor.find('topic').text = RGBD_TOPIC
    # Gazebo RGBD optical-depth images share the color camera intrinsics and time.
    # Keep existing link pose, image size, FOV, and near/far clipping unchanged.
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
