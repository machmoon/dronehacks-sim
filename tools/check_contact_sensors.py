"""Generate and score isolated Gazebo contact-sensor positive controls.

These fixtures verify only the flight_x500 contact-sensor wiring. They are
not flight tests and do not establish collision-free flight or airframe safety.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FLIGHT_MODEL = ROOT / 'models/flight_x500/model.sdf'
BASE_MODEL = ROOT / 'models/flight_x500_base/model.sdf'
COURSE_WORLD = ROOT / 'worlds/course.sdf'
TARGET_MODEL = 'course_quad'
STEP_SIZE = '0.004'
PROBE_SIZE = (0.004, 0.004, 0.004)
INITIAL_CLEARANCE_M = 0.015
NEGATIVE_OFFSET_XY_M = (0.50, 0.50)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers(element, default=(0., 0., 0., 0., 0., 0.)):
    if element is None or not (element.text or '').strip():
        return tuple(default)
    values = tuple(float(value) for value in element.text.split())
    if len(values) not in (3, 6):
        raise ValueError(f'expected 3 or 6 SDF pose values, got {len(values)}')
    return values if len(values) == 6 else values + (0., 0., 0.)


def _rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
            (-sp, cp*sr, cp*cr))


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _rotate(rotation, point):
    return tuple(sum(rotation[i][j]*point[j] for j in range(3)) for i in range(3))


def _vector_add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def extract_contact_targets(base_model=BASE_MODEL):
    """Read the named box collisions and their exact contact sensor topics."""
    root = ET.parse(base_model).getroot()
    model = root.find('./model')
    base_pose = _numbers(model.find('pose'))
    model_rotation = _rotation(base_pose[3:])
    targets = []
    for link in model.findall('link'):
        link_name = link.get('name')
        link_pose = _numbers(link.find('pose'))
        link_rotation = _matmul(model_rotation, _rotation(link_pose[3:]))
        for sensor in link.findall('sensor'):
            if sensor.get('type') != 'contact':
                continue
            contact = sensor.find('contact')
            collision_name = contact.findtext('collision')
            topic = contact.findtext('topic')
            collision = link.find(f"collision[@name='{collision_name}']")
            if collision is None:
                raise ValueError(f'{sensor.get("name")} references missing {collision_name}')
            size_text = collision.findtext('geometry/box/size')
            if size_text is None:
                raise ValueError(f'{collision_name} is not an SDF box collision')
            size = tuple(float(value) for value in size_text.split())
            if len(size) != 3 or any(value <= 0 for value in size):
                raise ValueError(f'{collision_name} has invalid box dimensions')
            collision_pose = _numbers(collision.find('pose'))
            collision_rotation = _rotation(collision_pose[3:])
            rotation = _matmul(link_rotation, collision_rotation)
            center = _vector_add(
                base_pose[:3],
                _vector_add(_rotate(model_rotation, link_pose[:3]),
                            _rotate(link_rotation, collision_pose[:3])))

            # Pick the lower box vertex. For axis-aligned bottom faces, zero
            # components remain centered, yielding a stable interior probe.
            local_corner = tuple((-1 if rotation[2][axis] > 0 else 1 if rotation[2][axis] < 0 else 0)
                                 * size[axis]/2 for axis in range(3))
            bottom_point = _vector_add(center, _rotate(rotation, local_corner))
            bottom_z = center[2] - sum(abs(rotation[2][i])*size[i]/2 for i in range(3))
            fixture_id = collision_name
            targets.append({
                'fixture_id': fixture_id,
                'kind': 'positive',
                'link_name': link_name,
                'sensor_name': sensor.get('name'),
                'topic': topic,
                'collision_name': collision_name,
                'collider_scoped_name': f'{TARGET_MODEL}::{link_name}::{collision_name}',
                'collision_size_m': list(size),
                'collision_center_model_m': list(center),
                'collision_rotation_model': [list(row) for row in rotation],
                'underside_contact_point_model_m': list(bottom_point),
                'underside_z_model_m': bottom_z,
            })
    if len(targets) != 9:
        raise ValueError(f'expected 9 flight contact sensors, found {len(targets)}')
    ids = [target['fixture_id'] for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate flight collision names')
    return targets


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _colliders_at_pose(targets, model_pose_rpy_rad):
    model_rotation = _rotation(model_pose_rpy_rad)
    colliders = []
    for target in targets:
        local_rotation = tuple(tuple(row) for row in target['collision_rotation_model'])
        colliders.append({
            'collision_name': target['collision_name'],
            'center': _rotate(model_rotation, target['collision_center_model_m']),
            'rotation': _matmul(model_rotation, local_rotation),
            'size': target['collision_size_m'],
        })
    return colliders


def _vertical_interval(collider, x, y):
    """Return bottom/top z on a world-vertical ray through an oriented box."""
    rotation = collider['rotation']
    inverse = _transpose(rotation)
    center = collider['center']
    origin = _rotate(inverse, (x-center[0], y-center[1], 10.-center[2]))
    direction = _rotate(inverse, (0., 0., -1.))
    half = tuple(value/2 for value in collider['size'])
    enter, leave = -float('inf'), float('inf')
    for axis in range(3):
        if abs(direction[axis]) < 1e-10:
            if abs(origin[axis]) > half[axis] + 1e-9:
                return None
            continue
        near = (-half[axis]-origin[axis])/direction[axis]
        far = (half[axis]-origin[axis])/direction[axis]
        enter, leave = max(enter, min(near, far)), min(leave, max(near, far))
    if enter > leave:
        return None
    return 10.-leave, 10.-enter


def _support_face_points(target, model_pose_rpy_rad):
    colliders = _colliders_at_pose((target,), model_pose_rpy_rad)
    collider = colliders[0]
    rotation, size = collider['rotation'], collider['size']
    vertical_axis = max(range(3), key=lambda axis: abs(rotation[2][axis]))
    side = -1. if rotation[2][vertical_axis] > 0 else 1.
    other_axes = [axis for axis in range(3) if axis != vertical_axis]

    # Keep rotor probes on an exposed outer blade segment, beyond the body
    # footprint. Probe the blade box dimensions and link transform from SDF.
    if target['link_name'].startswith('rotor_'):
        local_x = .08 if target['collision_center_model_m'][0] > 0 else -.08
        local = [0., 0., 0.]
        local[0], local[1], local[vertical_axis] = local_x, 0., side*size[vertical_axis]/2
        return [_vector_add(collider['center'], _rotate(rotation, local))]

    if target['collision_name'] in ('base_link_collision_0',):
        fractions = ((0., 0.),)
    elif target['collision_name'] in ('base_link_collision_3', 'base_link_collision_4'):
        # Move along the bar away from the overlapping vertical-leg boxes.
        local = [0., 0., 0.]
        local[0], local[vertical_axis] = .10, side*size[vertical_axis]/2
        return [_vector_add(collider['center'], _rotate(rotation, local))]
    else:
        fractions = tuple((x, y) for x in (0., -.4, .4, -.8, .8)
                          for y in (0., -.4, .4, -.8, .8))

    points = []
    for first, second in fractions:
        local = [0., 0., 0.]
        local[vertical_axis] = side*size[vertical_axis]/2
        local[other_axes[0]] = first*size[other_axes[0]]/2
        local[other_axes[1]] = second*size[other_axes[1]]/2
        points.append(_vector_add(collider['center'], _rotate(rotation, local)))
    return points


def _orientation_candidates():
    yield (0., 0., 0.)
    for degrees in range(5, 66, 5):
        angle = math.radians(degrees)
        for pose in ((-angle, 0., 0.), (angle, 0., 0.),
                     (0., -angle, 0.), (0., angle, 0.)):
            yield pose


def _probe_samples(x, y):
    half_x, half_y = PROBE_SIZE[0]/2, PROBE_SIZE[1]/2
    return [(x+dx, y+dy) for dx in (-half_x, 0., half_x)
            for dy in (-half_y, 0., half_y)]


def validate_probe_geometry(targets, fixture):
    """Check clearance and reject a probe intercepted by another source box."""
    colliders = _colliders_at_pose(targets, fixture['model_pose_rpy_rad'])
    target_collider = next(box for box in colliders
                           if box['collision_name'] == fixture['collision_name'])
    x, y = fixture['probe_contact_xy_world_m']
    samples = _probe_samples(x, y)
    target_intervals = [_vertical_interval(target_collider, sx, sy) for sx, sy in samples]
    if any(interval is None for interval in target_intervals):
        return {'valid': False, 'reason': 'probe_footprint_misses_target'}
    lowest_target_z = min(interval[0] for interval in target_intervals)
    highest_target_z = max(interval[0] for interval in target_intervals)
    probe_top = fixture['probe_pose_world_m'][2] + PROBE_SIZE[2]/2
    probe_bottom = fixture['probe_pose_world_m'][2] - PROBE_SIZE[2]/2
    clearance = lowest_target_z - probe_top
    if abs(clearance-INITIAL_CLEARANCE_M) > .0005:
        return {'valid': False, 'reason': 'incorrect_initial_clearance',
                'clearance_m': clearance}
    if highest_target_z-lowest_target_z > .003:
        return {'valid': False, 'reason': 'uneven_target_support',
                'surface_spread_m': highest_target_z-lowest_target_z}

    blockers = set()
    for other in colliders:
        if other['collision_name'] == fixture['collision_name']:
            continue
        for sx, sy in samples:
            interval = _vertical_interval(other, sx, sy)
            if interval is None:
                continue
            # A box between the target's first contact surface and the probe
            # bottom will touch the probe before the intended source box.
            if interval[0] < lowest_target_z-1e-6 and interval[1] > probe_bottom+1e-6:
                blockers.add(other['collision_name'])
                break
    if blockers:
        return {'valid': False, 'reason': 'probe_occluded_by_source_collision',
                'blocking_collisions': sorted(blockers), 'clearance_m': clearance}
    return {'valid': True, 'clearance_m': clearance,
            'support_spread_m': highest_target_z-lowest_target_z,
            'blocking_collisions': []}


def _select_exposed_probe(target, targets):
    """Choose a small-footprint underside point with 15mm free-drop clearance."""
    for model_pose in _orientation_candidates():
        colliders = _colliders_at_pose(targets, model_pose)
        target_box = next(box for box in colliders
                          if box['collision_name'] == target['collision_name'])
        for point in _support_face_points(target, model_pose):
            x, y = point[:2]
            samples = _probe_samples(x, y)
            intervals = [_vertical_interval(target_box, sx, sy) for sx, sy in samples]
            if any(interval is None for interval in intervals):
                continue
            lowest_target_z = min(interval[0] for interval in intervals)
            highest_target_z = max(interval[0] for interval in intervals)
            if highest_target_z-lowest_target_z > .003:
                continue
            probe_top = lowest_target_z-INITIAL_CLEARANCE_M
            probe_bottom = probe_top-PROBE_SIZE[2]
            blocked = False
            for other in colliders:
                if other['collision_name'] == target['collision_name']:
                    continue
                for sx, sy in samples:
                    interval = _vertical_interval(other, sx, sy)
                    if interval is not None and interval[0] < lowest_target_z-1e-6 and interval[1] > probe_bottom+1e-6:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            fixture = dict(target)
            fixture.update(
                model_pose_rpy_rad=list(model_pose),
                probe_contact_xy_world_m=[x, y],
                target_surface_min_z_at_probe_m=lowest_target_z,
                initial_clearance_m=INITIAL_CLEARANCE_M,
            )
            fixture['probe_pose_world_m'] = [x, y, probe_top-PROBE_SIZE[2]/2]
            report = validate_probe_geometry(targets, fixture)
            if report['valid']:
                fixture['geometry_validation'] = report
                return fixture
    raise ValueError(f'no unoccluded 15mm-drop probe point for {target["collision_name"]}')


def _add(parent, tag, text=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    element.text = text
    return element


def _add_probe(world, target, *, negative=False, reference=False):
    fixture_id = target['fixture_id']
    suffix = '__reference' if reference else ''
    probe_model = f'contact_probe_{fixture_id}{suffix}'
    probe_collision = f'probe_{fixture_id}{suffix}_collision'
    x, y = target['probe_contact_xy_world_m']
    if negative and not reference:
        x += NEGATIVE_OFFSET_XY_M[0]
        y += NEGATIVE_OFFSET_XY_M[1]
    probe_top = target['target_surface_min_z_at_probe_m'] - INITIAL_CLEARANCE_M
    probe_pose = (x, y, probe_top-PROBE_SIZE[2]/2)
    model = _add(world, 'model', name=probe_model)
    _add(model, 'static', 'true')
    _add(model, 'pose', ' '.join(f'{value:.12g}' for value in probe_pose) + ' 0 0 0')
    link = _add(model, 'link', name='probe_link')
    collision = _add(link, 'collision', name=probe_collision)
    box = _add(_add(collision, 'geometry'), 'box')
    _add(box, 'size', ' '.join(f'{value:.12g}' for value in PROBE_SIZE))
    _add(_add(_add(_add(collision, 'surface'), 'contact'), 'ode'), 'min_depth', '0.001')
    _add(_add(collision.find('surface'), 'friction'), 'ode')
    return {
        'probe_model_name': probe_model,
        'probe_link_name': 'probe_link',
        'probe_collision_name': probe_collision,
        'probe_scoped_name': f'{probe_model}::probe_link::{probe_collision}',
        'probe_pose_world_m': list(probe_pose),
        'probe_size_m': list(PROBE_SIZE),
        'probe_top_z_world_m': probe_pose[2]+PROBE_SIZE[2]/2,
        'pedestal_displacement_xy_m': (list(NEGATIVE_OFFSET_XY_M)
                                       if negative and not reference else [0., 0.]),
    }


def _make_world(fixture, *, negative=False):
    fixture = dict(fixture)
    fixture['kind'] = 'negative_displaced_pedestal' if negative else 'positive'
    if negative:
        fixture['source_fixture_id'] = fixture['fixture_id']
        fixture['fixture_id'] += '__negative_displaced'
    sdf = ET.Element('sdf', {'version': '1.9'})
    world = _add(sdf, 'world', name=f'contact_{fixture["fixture_id"]}' + ('_negative' if negative else ''))
    physics = _add(world, 'physics', name='course_4ms', type='ignored')
    _add(physics, 'max_step_size', STEP_SIZE)
    _add(physics, 'real_time_factor', '1.0')
    _add(world, 'gravity', '0 0 -9.8')
    physics_plugin = _add(world, 'plugin', filename='gz-sim-physics-system',
                          name='gz::sim::systems::Physics')
    engine = _add(physics_plugin, 'engine')
    _add(engine, 'filename', 'gz-physics-dartsim-plugin')
    _add(world, 'plugin', filename='gz-sim-contact-system', name='gz::sim::systems::Contact')
    _add(world, 'plugin', filename='gz-sim-scene-broadcaster-system',
         name='gz::sim::systems::SceneBroadcaster')
    _add(world, 'plugin', filename='gz-sim-user-commands-system',
         name='gz::sim::systems::UserCommands')

    include = _add(world, 'include')
    _add(include, 'uri', 'model://flight_x500')
    _add(include, 'name', TARGET_MODEL)
    pose = fixture['model_pose_rpy_rad']
    _add(include, 'pose', '0 0 0 ' + ' '.join(f'{value:.12g}' for value in pose))

    fixture.update(_add_probe(world, fixture, negative=negative))
    if negative:
        reference = _add_probe(world, fixture, negative=True, reference=True)
        fixture['reference_pedestal'] = {
            'model_name': reference['probe_model_name'],
            'link_name': reference['probe_link_name'],
            'collision_name': reference['probe_collision_name'],
            'scoped_name': reference['probe_scoped_name'],
            'pose_world_m': reference['probe_pose_world_m'],
            'displacement_xy_m': [0., 0.],
        }
        fixture['displaced_pedestal'] = {
            'model_name': fixture['probe_model_name'],
            'link_name': fixture['probe_link_name'],
            'collision_name': fixture['probe_collision_name'],
            'scoped_name': fixture['probe_scoped_name'],
            'pose_world_m': fixture['probe_pose_world_m'],
            'displacement_xy_m': list(NEGATIVE_OFFSET_XY_M),
        }
    ET.indent(sdf, space='  ')
    return ET.tostring(sdf, encoding='unicode', xml_declaration=True), fixture


def generate(output_dir):
    """Create nine source-derived positive worlds and one negative control."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    targets = extract_contact_targets()
    flight_xml = ET.parse(FLIGHT_MODEL).getroot()
    if flight_xml.find('./model/include[@merge="true"]/uri') is None:
        raise ValueError('flight_x500 must retain its merge=true flight_x500_base include')
    if len(flight_xml.findall('./model/plugin')) != 4:
        raise ValueError('flight_x500 motor plugin wiring changed; review before generating controls')

    sources = {str(path.relative_to(ROOT)): _sha256(path)
               for path in (FLIGHT_MODEL, BASE_MODEL, COURSE_WORLD)}
    fixtures = []
    for target in targets:
        selected = _select_exposed_probe(target, targets)
        report = validate_probe_geometry(targets, selected)
        if not report['valid']:
            raise ValueError(f'unsafe generated fixture for {target["collision_name"]}: {report}')
        world_text, fixture = _make_world(selected)
        filename = f'{target["fixture_id"]}.sdf'
        world_path = output_dir / filename
        world_path.write_text(world_text + '\n')
        fixture['world_file'] = filename
        fixture['world_sha256'] = _sha256(world_path)
        fixtures.append(fixture)

    negative_target = _select_exposed_probe(targets[0], targets)
    world_text, fixture = _make_world(negative_target, negative=True)
    filename = f'{negative_target["fixture_id"]}__negative_displaced.sdf'
    world_path = output_dir / filename
    world_path.write_text(world_text + '\n')
    fixture['world_file'] = filename
    fixture['world_sha256'] = _sha256(world_path)
    fixtures.append(fixture)

    manifest = {
        'schema_version': 1,
        'purpose': 'Gazebo contact sensor wiring positive controls only; no flight safety claim',
        'world_step_size_s': float(STEP_SIZE),
        'physics_engine': 'DART',
        'gravity_m_s2': [0., 0., -9.8],
        'target_model_name': TARGET_MODEL,
        'flight_include_uri': 'model://flight_x500',
        'px4_plugins_or_models': [],
        'motor_commands_sent': False,
        'source_sha256': sources,
        'fixtures': fixtures,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def _name_has_scoped_suffix(actual, expected):
    if not isinstance(actual, str):
        return False
    actual_parts = actual.split('::')
    expected_parts = expected.split('::')
    return len(actual_parts) >= len(expected_parts) and actual_parts[-len(expected_parts):] == expected_parts


def matching_expected_pairs(fixture, contact_rows):
    """Count only expected collider/probe pairs recorded on the mapped topic."""
    matches = 0
    for row in contact_rows:
        if row.get('topic') != fixture['topic']:
            continue
        for pair in row.get('pairs', ()):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            first, second = pair
            direct = (_name_has_scoped_suffix(first, fixture['collider_scoped_name']) and
                      _name_has_scoped_suffix(second, fixture['probe_scoped_name']))
            reverse = (_name_has_scoped_suffix(second, fixture['collider_scoped_name']) and
                       _name_has_scoped_suffix(first, fixture['probe_scoped_name']))
            matches += int(direct or reverse)
    return matches


def score_fixture(fixture, contact_rows, status=None):
    """Score expected pairs; a negative requires observed reference contact."""
    matches = matching_expected_pairs(fixture, contact_rows)
    topic_status = (status or {}).get('contact_topic_observation', {}).get(fixture['topic'], {})
    topic_messages = topic_status.get('transport_messages_received')
    if fixture['kind'] == 'positive':
        outcome = 'positive_pair_observed' if matches else 'expected_pair_not_observed'
    else:
        reference_fixture = dict(fixture, probe_scoped_name=fixture['reference_pedestal']['scoped_name'])
        reference_matches = matching_expected_pairs(reference_fixture, contact_rows)
        if matches:
            outcome = 'negative_control_failed_displaced_pair_observed'
        elif reference_matches:
            outcome = 'negative_control_passed_reference_only'
        else:
            outcome = 'negative_control_inconclusive_reference_pair_not_observed'
    return {
        'fixture_id': fixture['fixture_id'],
        'kind': fixture['kind'],
        'topic': fixture['topic'],
        'expected_collider': fixture['collider_scoped_name'],
        'expected_probe': fixture['probe_scoped_name'],
        'expected_pair_observations': matches,
        'reference_pair_observations': (matching_expected_pairs(
            dict(fixture, probe_scoped_name=fixture['reference_pedestal']['scoped_name']), contact_rows)
            if fixture['kind'] != 'positive' else None),
        'reference_probe': (fixture['reference_pedestal']['scoped_name']
                            if fixture['kind'] != 'positive' else None),
        'transport_messages_on_expected_topic': topic_messages,
        'outcome': outcome,
        'interpretation': 'sensor wiring check only; no flight safety claim',
    }


def _read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    generate_parser = subparsers.add_parser('generate', help='write fixture worlds and manifest')
    generate_parser.add_argument('--output', type=Path, required=True,
                                 help='new output directory; it must not already exist')
    score_parser = subparsers.add_parser('score', help='score one fixture against observer evidence')
    score_parser.add_argument('--manifest', type=Path, required=True)
    score_parser.add_argument('--fixture', required=True, help='fixture_id from manifest')
    score_parser.add_argument('--contacts', type=Path, required=True, help='contacts.jsonl')
    score_parser.add_argument('--status', type=Path, help='observer status.json, if available')
    args = parser.parse_args()
    if args.command == 'generate':
        result = generate(args.output)
        print(json.dumps({'manifest': str(args.output/'manifest.json'),
                          'fixture_worlds': len(result['fixtures']),
                          'positive_worlds': sum(f['kind'] == 'positive' for f in result['fixtures']),
                          'negative_worlds': sum(f['kind'] != 'positive' for f in result['fixtures'])}, indent=2))
        return
    manifest = json.loads(args.manifest.read_text())
    fixture = next((item for item in manifest['fixtures'] if item['fixture_id'] == args.fixture), None)
    if fixture is None:
        parser.error(f'fixture {args.fixture!r} not found in manifest')
    status = json.loads(args.status.read_text()) if args.status else None
    print(json.dumps(score_fixture(fixture, _read_jsonl(args.contacts), status), indent=2))


if __name__ == '__main__':
    main()
