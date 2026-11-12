"""Create paired runtime base worlds with the course net visuals on/off.

The treatment removes only the five named net <visual> elements. It preserves
all collisions and all other world/model fields. This is a rendered-net absence
stress case, not a realistic woven-net optical model.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NET_VISUAL_NAMES = (
    'net_side_east', 'net_side_north', 'net_side_south', 'net_side_west', 'net_roof',
)
EXPECTED_VEHICLE_OVERRIDE = {
    'uri': 'model://course_x3',
    'name': 'course_quad',
    'pose': '-2.5146 0.9144 0.056 0 0 1.570796327',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_text(text):
    return ' '.join(text.split()) if text and text.strip() else ''


def canonical(element):
    """Whitespace-insensitive semantic tuple; preserves tags, attrs, and order."""
    return (element.tag, tuple(sorted(element.attrib.items())), _normalized_text(element.text),
            tuple(canonical(child) for child in element))


def _manifest_path(path: Path) -> str:
    """Return a stable repo-relative path when possible, otherwise only basename."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _direct_includes(world):
    includes = world.findall('include')
    all_includes = world.findall('.//include')
    if len(includes) != 2 or len(all_includes) != 2:
        raise ValueError('expected exactly the two direct course source includes')
    by_uri = {}
    for include in includes:
        uri = include.findtext('uri')
        if uri in by_uri:
            raise ValueError(f'duplicate source include: {uri}')
        by_uri[uri] = include
    if set(by_uri) != {'model://course_v3', 'model://course_x3'}:
        raise ValueError('unexpected or missing source world include')
    course_include = by_uri['model://course_v3']
    if course_include.attrib or [child.tag for child in course_include] != ['uri']:
        raise ValueError('unexpected course_v3 include override')
    vehicle_include = by_uri['model://course_x3']
    if vehicle_include.attrib:
        raise ValueError('unexpected course_quad include attributes')
    fields = {child.tag: _normalized_text(child.text) for child in vehicle_include}
    if (len(vehicle_include) != len(EXPECTED_VEHICLE_OVERRIDE)
            or fields != EXPECTED_VEHICLE_OVERRIDE):
        raise ValueError('unexpected course_quad include override')
    return course_include


def _load_course_model(path: Path):
    root = ET.parse(path).getroot()
    models = root.findall('model')
    if len(models) != 1:
        raise ValueError('expected exactly one model in course model SDF')
    model = models[0]
    if model.get('name') != 'course_v3':
        raise ValueError('expected course_v3 model SDF')
    links = model.findall('link')
    if len(links) != 1 or links[0].get('name') != 'layout':
        raise ValueError('expected the single course_v3 layout link')
    link = links[0]
    visuals = link.findall('visual')
    named = [visual.get('name') for visual in visuals
             if visual.get('name', '').startswith('net_')]
    if len(named) != len(NET_VISUAL_NAMES) or set(named) != set(NET_VISUAL_NAMES):
        raise ValueError('expected exactly the five named net visuals')
    collisions = link.findall('collision')
    collision_names = [collision.get('name') for collision in collisions
                       if collision.get('name', '').startswith('net_')]
    if len(collision_names) != len(NET_VISUAL_NAMES) or set(collision_names) != set(NET_VISUAL_NAMES):
        raise ValueError('expected one matching collision for each named net visual')
    collision_by_name = {collision.get('name'): collision for collision in collisions}
    for visual in visuals:
        name = visual.get('name')
        if name in NET_VISUAL_NAMES:
            collision = collision_by_name[name]
            visual_geometry = visual.find('geometry')
            collision_geometry = collision.find('geometry')
            if visual_geometry is None or collision_geometry is None:
                raise ValueError(f'missing geometry for net visual/collision {name}')
            if canonical(visual_geometry) != canonical(collision_geometry):
                raise ValueError(f'net visual/collision geometry differs for {name}')
    return root, model, link


def _remove_named_net_visuals(model):
    links = model.findall('link')
    if len(links) != 1 or links[0].get('name') != 'layout':
        raise ValueError('expected the single course_v3 layout link')
    link = links[0]
    visuals = [visual for visual in link.findall('visual')
               if visual.get('name') in NET_VISUAL_NAMES]
    found = [visual.get('name') for visual in visuals]
    if len(found) != len(NET_VISUAL_NAMES) or set(found) != set(NET_VISUAL_NAMES):
        raise ValueError('expected exactly the five named net visuals before ablation')
    for visual in visuals:
        link.remove(visual)


def _flatten_course(world_root, course_model):
    world = world_root.find('world')
    if world is None or world.get('name') != 'dronehacks':
        raise ValueError('expected the dronehacks source world')
    include = _direct_includes(world)
    index = list(world).index(include)
    world.remove(include)
    world.insert(index, copy.deepcopy(course_model))
    return world


def _net_collisions(model):
    link = model.find("link[@name='layout']")
    return [collision for collision in link.findall('collision')
            if collision.get('name') in NET_VISUAL_NAMES]


def generate_pair(world_path: Path, course_model_path: Path, output_dir: Path):
    world_path, course_model_path, output_dir = map(Path, (world_path, course_model_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite existing output directory: {output_dir.name}')
    source_world_bytes = world_path.read_bytes()
    source_model_bytes = course_model_path.read_bytes()
    original_world = ET.fromstring(source_world_bytes)
    _source_model_root, source_model, _ = _load_course_model(course_model_path)
    world = original_world.find('world')
    if world is None or world.get('name') != 'dronehacks':
        raise ValueError('expected the dronehacks source world')
    _direct_includes(world)  # Validate before either variant is constructed.

    control_root = copy.deepcopy(original_world)
    treatment_root = copy.deepcopy(original_world)
    control_world = _flatten_course(control_root, source_model)
    treatment_world = _flatten_course(treatment_root, source_model)
    treatment_model = treatment_world.find("model[@name='course_v3']")
    if treatment_model is None:
        raise ValueError('failed to flatten course_v3 in treatment')
    _remove_named_net_visuals(treatment_model)

    control_model = control_world.find("model[@name='course_v3']")
    if control_model is None:
        raise ValueError('failed to flatten course_v3 in control')
    before_collisions = [canonical(c) for c in control_model.findall('.//collision')]
    after_collisions = [canonical(c) for c in treatment_model.findall('.//collision')]
    if before_collisions != after_collisions:
        raise ValueError('course collision semantics changed in treatment')
    before_net_collisions = [canonical(c) for c in _net_collisions(control_model)]
    after_net_collisions = [canonical(c) for c in _net_collisions(treatment_model)]
    if before_net_collisions != after_net_collisions:
        raise ValueError('net collision semantics changed in treatment')

    control_without_targets = copy.deepcopy(control_root)
    control_course = control_without_targets.find("./world/model[@name='course_v3']")
    _remove_named_net_visuals(control_course)
    if canonical(control_without_targets) != canonical(treatment_root):
        raise ValueError('treatment contains changes beyond removal of the five net visuals')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    outputs = {}
    for name, root in (('control.sdf', control_root), ('treatment.sdf', treatment_root)):
        ET.indent(root, space='  ')
        data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        (output_dir / name).write_bytes(data)
        outputs[name] = {'path': name, 'sha256': sha256(data), 'bytes': len(data)}

    canonical_collisions = json.dumps(before_collisions, separators=(',', ':')).encode()
    manifest = {
        'schema_version': 1,
        'inputs': {
            'source_world': {'path': _manifest_path(world_path), 'sha256': sha256(source_world_bytes)},
            'course_model': {'path': _manifest_path(course_model_path), 'sha256': sha256(source_model_bytes)},
        },
        'outputs': outputs,
        'tool': {'path': 'tools/make_net_visual_ablation.py',
                 'sha256': sha256(Path(__file__).read_bytes())},
        'ablation': {
            'removed_visual_names': list(NET_VISUAL_NAMES),
            'net_collision_names_retained': list(NET_VISUAL_NAMES),
            'all_course_collision_semantics_equal': True,
            'net_collision_semantics_equal': True,
            'canonical_all_course_collisions_sha256': sha256(canonical_collisions),
            'canonical_net_collision_sha256': sha256(json.dumps(before_net_collisions, separators=(',', ':')).encode()),
            'all_other_world_semantics_equal_after_flattening': True,
            'worlds_are_base_inputs_for_existing_vision_rgbd_scene': True,
        },
        'claims_and_limits': [
            'Treatment removes only the five named net visual elements; all collisions are retained.',
            'Control retains all source visuals and flattens the unoverridden course_v3 include.',
            'The expected course_quad include name and pose override is retained unchanged.',
            'RGB and depth rendering both change; rays can return background or far-limit depth.',
            'This is a rendered net-absence stress case, not a realistic wire-net rendering or real sensor model.',
            'The estimator static survey still loads the original course model with analytic net panels; this intentional map/sensor mismatch remains.',
            'No truth topic, segmentation labels, controller, flight policy, or collision geometry is added or altered.',
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    (output_dir / 'manifest.json').write_bytes(manifest_bytes)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, default=ROOT / 'worlds/course.sdf')
    parser.add_argument('--course-model', type=Path, default=ROOT / 'models/course_v3/model.sdf')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='new run-specific directory; existing directories are never overwritten')
    args = parser.parse_args()
    manifest = generate_pair(args.world, args.course_model, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
