#!/usr/bin/env python3
"""Tessellate official Sky Beta STEP + mount into SI/FLU component meshes.

Build-time dependency: cadquery==2.8.0 (not needed to run Gazebo).
Usage: python tools/import_skybeta.py --source ../official-drone-files
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]


def write_obj(path, vertices, faces):
    with path.open('w') as f:
        for v in vertices:
            f.write('v ' + ' '.join(f'{x:.8f}' for x in v) + '\n')
        for face in faces:
            f.write('f ' + ' '.join(str(i + 1) for i in face) + '\n')


def build(source, output):
    import cadquery as cq
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    (output / 'meshes').mkdir(exist_ok=True)
    assembly = cq.Assembly.load(str(source / 'sky-beta.step'))
    parts = [(shape.located(loc), name.split('/')[-1]) for shape, name, loc, _ in assembly]
    # The exact primary IMU location is not encoded in CAD. Use the Pixhawk
    # enclosure centre as the explicitly documented body-frame reference.
    pixhawk = next(shape for shape, name in parts if 'Pixhawk 6C' in name)
    b = pixhawk.BoundingBox()
    origin = np.array([(b.xmin+b.xmax)/2, (b.ymin+b.ymax)/2, (b.zmin+b.zmax)/2])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    def record(name, vertices, faces, volume=None):
        vertices = (np.asarray(vertices) - origin) @ rotation.T * .001
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        write_obj(output / 'meshes' / f'{slug}.obj', vertices, faces)
        return {'name': name, 'mesh': f'meshes/{slug}.obj',
                'min_m': vertices.min(axis=0).tolist(), 'max_m': vertices.max(axis=0).tolist(),
                'solid_volume_m3': volume, 'triangles': len(faces)}

    components = []
    for shape, name in parts:
        if not shape.Faces():
            continue
        vertices, faces = shape.tessellate(.35, .3)
        if not vertices:
            continue
        components.append(record(name, [v.toTuple() for v in vertices], faces, shape.Volume()*1e-9))
        print(name, components[-1]['min_m'], components[-1]['max_m'], flush=True)
    # Official mount is an already positioned binary STL in millimetres.
    raw = (source / 'oak-d-camera-mount.stl').read_bytes()
    count, = struct.unpack_from('<I', raw, 80)
    if len(raw) != 84 + 50 * count:
        raise ValueError('Expected a binary STL with a valid triangle count')
    vertices = [struct.unpack_from('<3f', raw, 84 + 50*i + 12 + 12*j)
                for i in range(count) for j in range(3)]
    components.append(record('OAK-D Camera Mount', vertices,
                             [(i, i+1, i+2) for i in range(0,len(vertices),3)]))
    report = {'source_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                              for name in ('sky-beta.step', 'oak-d-camera-mount.stl')},
              'units': 'metres', 'axes': 'FLU: +X forward, +Y left, +Z up',
              'cad_origin_mm': origin.tolist(), 'cad_to_body_rotation': rotation.tolist(),
              'origin_note': 'Pixhawk enclosure bounding-box centre; primary IMU offset unmeasured',
              'components': components}
    (output/'geometry.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'models/skybeta')
    args = parser.parse_args()
    build(args.source, args.output)
