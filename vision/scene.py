"""Generate the x500 camera-equipped variant of the existing Gazebo course."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from .course import (
    CAMERA_HFOV, CAMERA_HEIGHT, CAMERA_PITCH, CAMERA_RATE, CAMERA_TOPIC,
    CAMERA_TRANSLATION, CAMERA_WIDTH, LAUNCH_XY, course_markers,
)


def value(parent, name, text):
    child = ET.SubElement(parent, name)
    child.text = str(text)
    return child


def mesh_quad(stream, first, left, right, bottom, top, z):
    for x, y in ((left, bottom), (right, bottom), (right, top), (left, top)):
        stream.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
    indices = ' '.join(f'{i}//1' for i in range(first, first + 4))
    stream.write(f'f {indices}\n')
    return first + 4


def marker_meshes(directory):
    """Render printable marker bit patterns as two batched colored meshes."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    make_bits = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
    cells = 8  # 36h11's six data cells plus a black border cell on each side
    directory.mkdir(parents=True, exist_ok=True)
    black_path, white_path = (directory / f'tags-{color}.obj' for color in ('black', 'white'))
    (directory / 'tags.mtl').write_text('newmtl black\nKd 0.008 0.008 0.008\n'
                                        'newmtl white\nKd 1 1 1\n')
    with ExitStack() as stack:
        black = stack.enter_context(black_path.open('w'))
        white = stack.enter_context(white_path.open('w'))
        for stream, color in ((black, 'black'), (white, 'white')):
            stream.write(f'mtllib tags.mtl\nusemtl {color}\nvn 0 0 1\n')
        black_index = white_index = 1
        for marker in course_markers().values():
            cell = marker.size / cells
            quiet_half_size = marker.size / 2 + cell
            white_index = mesh_quad(
                white, white_index, marker.x - quiet_half_size, marker.x + quiet_half_size,
                marker.y - quiet_half_size, marker.y + quiet_half_size, marker.z - 0.0005,
            )
            bits = make_bits(dictionary, marker.id, cells)
            left = marker.x - marker.size / 2
            top = marker.y + marker.size / 2
            for row, values in enumerate(bits):
                edges = [i for i, (a, b) in enumerate(zip((255, *values), (*values, 255)))
                         if a != b]
                for start, end in zip(edges[::2], edges[1::2]):
                    y = top - row * cell
                    black_index = mesh_quad(black, black_index, left + start * cell,
                                            left + end * cell, y - cell, y, marker.z)
    return black_path, white_path


def add_markers(world, directory):
    model = ET.SubElement(world, 'model', name='vision_landmarks')
    value(model, 'static', 'true')
    link = ET.SubElement(model, 'link', name='decals')
    for path, rgba in zip(marker_meshes(directory), ('0.008 0.008 0.008 1', '1 1 1 1')):
        visual = ET.SubElement(link, 'visual', name=path.stem)
        value(visual, 'cast_shadows', 'false')
        mesh = ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh')
        value(mesh, 'uri', path.as_uri())
        material = ET.SubElement(visual, 'material')
        value(material, 'ambient', rgba)
        value(material, 'diffuse', rgba)
        value(material, 'emissive', rgba)
        value(material, 'specular', '0 0 0 1')


def add_x500(world, launch_z):
    original = next((include for include in world.findall('include')
                     if include.findtext('name') == 'course_quad'), None)
    if original is None:
        raise ValueError('the course world must include a vehicle named course_quad')
    world.remove(original)
    model = ET.SubElement(world, 'model', name='course_quad', canonical_link='base_link')
    value(model, 'pose', f'{LAUNCH_XY[0]} {LAUNCH_XY[1]} {launch_z} 0 0 0')
    value(ET.SubElement(model, 'include', merge='true'), 'uri', 'model://x500')
    link = ET.SubElement(model, 'link', name='down_camera_link')
    pose = value(link, 'pose', ' '.join(map(str, (*CAMERA_TRANSLATION, 0, CAMERA_PITCH, 0))))
    pose.set('relative_to', 'base_link')
    inertial = ET.SubElement(link, 'inertial')
    value(inertial, 'mass', 0.005)
    inertia = ET.SubElement(inertial, 'inertia')
    for diagonal in ('ixx', 'iyy', 'izz'):
        value(inertia, diagonal, 0.000001)
    joint = ET.SubElement(model, 'joint', name='down_camera_joint', type='fixed')
    value(joint, 'parent', 'base_link')
    value(joint, 'child', 'down_camera_link')
    sensor = ET.SubElement(link, 'sensor', name='down_camera', type='camera')
    value(sensor, 'always_on', 'true')
    value(sensor, 'update_rate', CAMERA_RATE)
    value(sensor, 'topic', CAMERA_TOPIC)
    camera = ET.SubElement(sensor, 'camera')
    value(camera, 'horizontal_fov', CAMERA_HFOV)
    image = ET.SubElement(camera, 'image')
    value(image, 'width', CAMERA_WIDTH)
    value(image, 'height', CAMERA_HEIGHT)
    value(image, 'format', 'R8G8B8')
    clip = ET.SubElement(camera, 'clip')
    value(clip, 'near', 0.02)
    value(clip, 'far', 8)


def generate(world_path, output_path, launch_z, *, with_test_markers=False):
    tree = ET.parse(world_path)
    world = tree.getroot().find('world')
    view_pose = world.find('gui/plugin[@filename="MinimalScene"]/camera_pose')
    if view_pose is not None:
        view_pose.text = '-6 -6 8 0 0.724 0.785398163'
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid an additional shadow-map pass for the software-rendered onboard camera.
    for element in (*world.findall('scene/shadows'), *world.findall('light/cast_shadows')):
        element.text = 'false'
    if world.find('spherical_coordinates') is None:
        # Same synthetic reference as the upstream PX4 default world. Gazebo's
        # magnetometer needs it even though PX4 navigates without GNSS.
        geo = ET.SubElement(world, 'spherical_coordinates')
        for key, val in (
            ('surface_model', 'EARTH_WGS84'), ('world_frame_orientation', 'ENU'),
            ('latitude_deg', 47.397971057728974), ('longitude_deg', 8.546163739800146),
            ('elevation', 0),
        ):
            value(geo, key, val)
    for filename, system in (
        ('imu', 'Imu'), ('air-pressure', 'AirPressure'), ('magnetometer', 'Magnetometer'),
        ('sensors', 'Sensors'),
    ):
        plugin = ET.SubElement(world, 'plugin', filename=f'gz-sim-{filename}-system',
                               name=f'gz::sim::systems::{system}')
        if system == 'Sensors':
            value(plugin, 'render_engine', 'ogre2')
    add_x500(world, launch_z)
    if with_test_markers:
        add_markers(world, output_path.with_suffix('.assets'))
    ET.indent(tree, space='  ')
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, default=Path('worlds/course.sdf'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch-z', type=float, default=0.0,
                        help='x500 model root launch height in metres (base_link is 0.24 m above)')
    parser.add_argument('--with-test-markers', action='store_true',
                        help='add synthetic AprilTags for diagnostics (also enabled by SIM_TEST_MARKERS=1)')
    args = parser.parse_args()
    with_test_markers = args.with_test_markers or os.environ.get('SIM_TEST_MARKERS') == '1'
    if with_test_markers:
        print('Adding synthetic AprilTag fixtures for diagnostic use.', file=sys.stderr)
    generate(args.world, args.output, args.launch_z, with_test_markers=with_test_markers)


if __name__ == '__main__':
    main()
