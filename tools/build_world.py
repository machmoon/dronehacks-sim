#!/usr/bin/env python3
"""Build one static Gazebo mesh for each named part in the course OBJ."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/course-v3.zip'
MODEL = ROOT / 'models/course_v3'


def parse_obj(content):
    vertices, normals, groups = [], [], {}
    for line in content.splitlines():
        fields = line.split()
        if not fields:
            continue
        kind, *values = fields
        if kind == 'g':
            faces = groups.setdefault(' '.join(values), [])
        elif kind == 'v':
            vertices.append(tuple(map(float, values[:3])))
        elif kind == 'vn':
            normals.append(tuple(map(float, values[:3])))
        elif kind == 'f':
            face = []
            for value in values:
                vertex, _, normal = value.split('/')
                face.append((int(vertex) - 1, int(normal) - 1))
            for i in range(1, len(face) - 1):
                faces.append((face[0], face[i], face[i + 1]))
    return vertices, normals, groups


def write_mesh(path, vertices, normals, faces):
    used_vertices = sorted({vertex for face in faces for vertex, _ in face})
    used_normals = sorted({normal for face in faces for _, normal in face})
    vertex_ids = {old: new for new, old in enumerate(used_vertices, 1)}
    normal_ids = {old: new for new, old in enumerate(used_normals, 1)}
    with path.open('w') as out:
        print('mtllib course.mtl\nusemtl course', file=out)
        for index in used_vertices:
            print('v', *vertices[index], file=out)
        for index in used_normals:
            print('vn', *normals[index], file=out)
        for face in faces:
            print('f', *(f'{vertex_ids[v]}//{normal_ids[n]}' for v, n in face), file=out)
    return list(zip(*[vertices[i] for i in used_vertices]))


def add(parent, tag, text=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    child.text = text
    return child


def geometry(parent, uri):
    add(add(add(parent, 'geometry'), 'mesh'), 'uri', uri)


def visual_style(name):
    for prefix, color in (
        ('Net', '#85b3cb'),
        ('Floor', '#536171'),
        ('Landing pad 5', '#246260'),
        ('Landing', '#f4df9a'),
        ('Approach ring', '#92d8c5'),
        ('Launch', '#ffd681'),
        ('Guideline', '#e7ad5c'),
        ('Barrier', '#c46440'),
        ('Tube', '#388c93'),
        ('Boulder', '#987958'),
        ('C2a', '#e2a343'),
        ('C2b', '#829aab'),
        ('C1', '#829aab'),
    ):
        if name.startswith(prefix):
            return color
    return '#809bae'


def build():
    with ZipFile(SOURCE) as archive:
        vertices, normals, groups = parse_obj(archive.read('dronehacks-course-v3.obj').decode())
    (MODEL / 'meshes').mkdir(parents=True, exist_ok=True)
    (MODEL / 'meshes/course.mtl').write_text('newmtl course\nKa 1 1 1\nKd 1 1 1\n')

    sdf = ET.Element('sdf', version='1.9')
    model = add(sdf, 'model', name='course_v3')
    add(model, 'static', 'true')
    link = add(model, 'link', name='layout')
    metadata = []
    for name, faces in groups.items():
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        mesh_path = MODEL / 'meshes' / f'{slug}.obj'
        axes = write_mesh(mesh_path, vertices, normals, faces)
        # The separate ground plane carries landing loads. CAD floor markings
        # are decorative, with z positions preserved to prevent overlap.
        collides = not name.startswith(('Floor', 'Landing', 'Approach', 'Guideline', 'Launch'))
        transparency = 0.94 if name.startswith('Net') else 0.0
        color = visual_style(name)
        uri = f'model://course_v3/meshes/{slug}.obj'
        if collides:
            geometry(add(link, 'collision', name=slug), uri)
        visual = add(link, 'visual', name=f'{slug}_visual')
        geometry(visual, uri)
        if transparency:
            add(visual, 'transparency', str(transparency))
            add(visual, 'cast_shadows', 'false')
        material = add(visual, 'material')
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        for field in ('ambient', 'diffuse'):
            add(material, field, ' '.join(f'{v:.4f}' for v in rgb) + ' 1')
        metadata.append({
            'group': name,
            'name': slug,
            'mesh': f'meshes/{slug}.obj',
            'bounds': {'min': list(map(min, axes)), 'max': list(map(max, axes))},
            'triangles': len(faces),
            'collidable': collides,
            'color': color,
            'transparency': transparency,
        })
    ET.indent(sdf)
    ET.ElementTree(sdf).write(MODEL / 'model.sdf', encoding='utf-8', xml_declaration=True)
    (MODEL / 'groups.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'coordinates': 'Metres, Z up, centre of course floor at origin',
        'groups': metadata,
    }, indent=2) + '\n')


if __name__ == '__main__':
    build()
