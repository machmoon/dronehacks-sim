"""Single source for Sky Beta inertial properties, Gazebo model, and PX4 mixer.

Only geometry and hardware identities are sourced from the official aircraft.
See skybeta.json for every unmeasured physical parameter. Coordinates are FLU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(__file__).with_suffix('.json')
GEOMETRY = ROOT / 'models/skybeta/geometry.json'
G = 9.80665


def load(profile_path=PROFILE, geometry_path=GEOMETRY):
    profile = json.loads(Path(profile_path).read_text())
    geometry = json.loads(Path(geometry_path).read_text())
    for key in ('thrust_coefficient', 'moment_ratio_m', 'max_speed_rad_s',
                'rise_time_s', 'fall_time_s'):
        if not math.isfinite(profile['motor'][key]) or profile['motor'][key] <= 0:
            raise ValueError(f'motor.{key} must be finite and positive')
    for key in ('radius_m', 'mass_kg', 'thickness_m'):
        if not math.isfinite(profile['propeller'][key]) or profile['propeller'][key] <= 0:
            raise ValueError(f'propeller.{key} must be finite and positive')
    idle, maximum = (profile['motor'][k] for k in ('idle_speed_rad_s', 'max_speed_rad_s'))
    if not 0 <= idle < maximum:
        raise ValueError('idle speed must be below maximum speed')
    return profile, geometry


def components(profile, geometry):
    result = []
    for part in geometry['components']:
        name = part['name']
        lo, hi = np.array(part['min_m']), np.array(part['max_m'])
        size = hi-lo
        # A placeholder in the assembly points at the separate STL mount.
        if 'Supplied STL' in name:
            continue
        mass = next((m for prefix, m in profile['component_mass_kg'].items() if prefix in name), None)
        if mass is None:
            density = profile['standoff_density_kg_m3'] if 'Standoff' in name else profile['frame_density_kg_m3']
            mass = part['solid_volume_m3'] * density
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f'invalid mass for {name}')
        inertia = np.diag(mass / 12 * (np.sum(size**2)-size**2))
        result.append(dict(part, mass_kg=mass, center=(hi+lo)/2, size=size, inertia=inertia))
    return result


def rotors(profile, parts):
    # PX4 motor order: front-right, rear-left, front-left, rear-right.
    quadrants = ((1,-1),(-1,1),(1,1),(-1,-1))
    rows = []
    for i, (sx,sy) in enumerate(quadrants):
        part = next(p for p in parts if 'X2807' in p['name']
                    and p['center'][0]*sx > 0 and p['center'][1]*sy > 0)
        xyz = part['center'].copy()
        xyz[2] = part['max_m'][2] + profile['propeller']['thickness_m']/2
        rows.append({'index': i, 'position': xyz, 'direction': 'ccw' if i < 2 else 'cw',
                     'reaction_sign': -1 if i < 2 else 1})
    return rows


def aggregate(items):
    mass = sum(m for m, _, _ in items)
    center = sum(m*p for m, p, _ in items)/mass
    inertia = np.zeros((3,3))
    for m, p, local in items:
        d = p-center
        inertia += local + m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    return mass, center, inertia


def properties(profile, geometry):
    parts = components(profile, geometry)
    motors = rotors(profile, parts)
    extra = profile['unmodelled_mass_kg']
    if not math.isfinite(extra) or extra <= 0:
        raise ValueError('unmodelled_mass_kg must be positive')
    base = [(p['mass_kg'],p['center'],p['inertia']) for p in parts]
    base.append((extra,np.zeros(3),np.eye(3)*extra*.025**2/6))
    m, r, h = (profile['propeller'][k] for k in ('mass_kg','radius_m','thickness_m'))
    rotor_inertia = np.diag([m*(3*r*r+h*h)/12]*2+[m*r*r/2])
    all_items = base + [(m,row['position'],rotor_inertia) for row in motors]
    total, com, inertia = aggregate(all_items)
    offsets = np.array([row['position']-com for row in motors])
    # Per-rotor thrusts balancing gravity and roll/pitch/yaw moments.
    allocation = np.array([np.ones(4), offsets[:,1], -offsets[:,0],
                           [row['reaction_sign']*profile['motor']['moment_ratio_m'] for row in motors]])
    thrust = np.linalg.solve(allocation,np.array([total*G,0,0,0]))
    k = profile['motor']['thrust_coefficient']
    maximum = profile['motor']['max_speed_rad_s']
    if np.any(thrust <= 0) or np.any(thrust >= k*maximum**2):
        raise ValueError('profile cannot hover within motor thrust limits')
    return {'parts':parts, 'rotors':motors, 'base':aggregate(base), 'mass':total,
            'com':com,'inertia':inertia, 'rotor_inertia':rotor_inertia,
            'hover_thrust':thrust,'hover_speed':np.sqrt(thrust/k)}


def add(parent, tag, text=None, **attrs):
    e = ET.SubElement(parent,tag,attrs)
    if text is not None:
        e.text = str(text)
    return e


def numbers(values):
    return ' '.join(f'{x:.10g}' for x in values)


def inertial(link, mass, center, tensor):
    e=add(link,'inertial')
    add(e,'pose',numbers([*center,0,0,0])); add(e,'mass',mass)
    i=add(e,'inertia')
    for key,a,b in [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]:
        add(i,key,tensor[a,b])


def primitive(link, kind, name, xyz, shape, dimensions, color=None):
    e=add(link,kind,name=name); add(e,'pose',numbers([*xyz,0,0,0]))
    g=add(add(e,'geometry'),shape)
    for key,val in dimensions.items(): add(g,key,val)
    if color:
        mat=add(e,'material'); add(mat,'ambient',color); add(mat,'diffuse',color)
    return e


def build_model(profile, geometry):
    p=properties(profile,geometry)
    sdf=ET.Element('sdf',version='1.9'); model=add(sdf,'model',name='skybeta',canonical_link='base_link')
    # Default spawn rests just above the lowest CAD component.
    ground_z=-min(part['min_m'][2] for part in p['parts'])+.003
    add(model,'pose',numbers([0,0,ground_z,0,0,0])); add(model,'self_collide','false')
    body=add(model,'link',name='base_link'); inertial(body,*p['base'])
    for n,part in enumerate(p['parts']):
        visual=add(body,'visual',name=f'cad_{n}')
        mesh=add(add(visual,'geometry'),'mesh'); add(mesh,'uri','model://skybeta/'+part['mesh'])
        color=('0.12 0.14 0.16 1' if any(x in part['name'] for x in ('Plate','Arm','Motor'))
               else '0.35 0.38 0.42 1')
        if 'OVONIC' in part['name']: color='0.16 0.30 0.65 1'
        mat=add(visual,'material'); add(mat,'ambient',color); add(mat,'diffuse',color)
        # Per-component AABBs are conservative contact proxies; no giant solid
        # box spanning the gaps between arms. Props use swept cylinders below.
        for j, box in enumerate(part.get('collision_boxes', [part])):
            lo,hi=np.array(box['min_m']),np.array(box['max_m'])
            primitive(body,'collision',f'component_{n}_{j}',(lo+hi)/2,'box',{'size':numbers(np.maximum(.001,hi-lo))})
    for name,kind,rate in [('imu_sensor','imu',250),('air_pressure_sensor','air_pressure',50),('magnetometer_sensor','magnetometer',100)]:
        sensor=add(body,'sensor',name=name,type=kind); add(sensor,'always_on','true'); add(sensor,'update_rate',rate)
    for row in p['rotors']:
        i=row['index']; xyz=row['position']; prop=profile['propeller']; motor=profile['motor']
        link=add(model,'link',name=f'rotor_{i}'); add(link,'pose',numbers([*xyz,0,0,0]))
        inertial(link,prop['mass_kg'],np.zeros(3),p['rotor_inertia'])
        primitive(link,'visual','propeller',[0,0,0],'box',{'size':numbers([2*prop['radius_m'],.014,prop['thickness_m']])},'0.04 0.04 0.04 1')
        primitive(link,'collision','swept_propeller',[0,0,0],'cylinder',{'radius':prop['radius_m'],'length':prop['thickness_m']})
        joint=add(model,'joint',name=f'rotor_{i}_joint',type='revolute')
        add(joint,'parent','base_link'); add(joint,'child',f'rotor_{i}')
        axis=add(joint,'axis'); add(axis,'xyz','0 0 1')
        plugin=add(model,'plugin',filename='gz-sim-multicopter-motor-model-system',name='gz::sim::systems::MulticopterMotorModel')
        fields={'jointName':f'rotor_{i}_joint','linkName':f'rotor_{i}','turningDirection':row['direction'],
                'timeConstantUp':motor['rise_time_s'],'timeConstantDown':motor['fall_time_s'],
                'maxRotVelocity':motor['max_speed_rad_s'],'motorConstant':motor['thrust_coefficient'],
                'momentConstant':motor['moment_ratio_m'],'commandSubTopic':'command/motor_speed','motorNumber':i,
                'rotorDragCoefficient':motor['rotor_drag_coefficient'],'rollingMomentCoefficient':motor['rolling_moment_coefficient'],
                'rotorVelocitySlowdownSim':10,'motorType':'velocity'}
        for key,val in fields.items(): add(plugin,key,val)
    return sdf,p


def px4_parameters(profile, props):
    motor=profile['motor']; idle= motor['idle_speed_rad_s']; maximum=motor['max_speed_rad_s']
    params={'CA_ROTOR_COUNT':4, 'MPC_THR_HOVER':float(np.mean((props['hover_speed']-idle)/(maximum-idle)))}
    for row in props['rotors']:
        i=row['index']; offset=row['position']-props['com']
        for axis,val in zip('XYZ',offset*np.array([1,-1,-1])):
            params[f'CA_ROTOR{i}_P{axis}']=float(val)
        params[f'CA_ROTOR{i}_KM']=-row['reaction_sign']*motor['moment_ratio_m']
        params[f'SIM_GZ_EC_MIN{i+1}']=idle
        params[f'SIM_GZ_EC_MAX{i+1}']=maximum
    return params


def report(profile, geometry):
    p=properties(profile,geometry)
    return {'name':profile['name'],'status':profile['status'],
            'profile_sha256':hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
            'geometry_sha256':hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest(),
            'mass_kg':p['mass'],
            'center_of_mass_flu_m':p['com'].tolist(),'inertia_kg_m2':p['inertia'].tolist(),
            'hover_motor_rad_s':p['hover_speed'].tolist(),'hover_thrust_n':p['hover_thrust'].tolist(),
            'max_thrust_to_weight':4*profile['motor']['thrust_coefficient']*profile['motor']['max_speed_rad_s']**2/(p['mass']*G),
            'horizontal_radius_m':max(np.linalg.norm(row['position'][:2])+profile['propeller']['radius_m'] for row in p['rotors']),
            'px4_parameters':px4_parameters(profile,p)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=PROFILE)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--shell',action='store_true')
    args=parser.parse_args(); profile,geometry=load(args.profile)
    if args.build:
        sdf,_=build_model(profile,geometry); ET.indent(sdf,space='  ')
        ET.ElementTree(sdf).write(ROOT/'models/skybeta/model.sdf',encoding='utf-8',xml_declaration=True)
    if args.shell:
        for key,val in px4_parameters(profile,properties(profile,geometry)).items():
            print(f'export PX4_PARAM_{key}={val:.10g}')
    else: print(json.dumps(report(profile,geometry),indent=2))


if __name__=='__main__': main()
