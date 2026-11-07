#!/usr/bin/env python3
"""Exercise the actual Gazebo motor/rigid-body plugins in an isolated world.

Run inside the simulator image: python3 -m tools.check_skybeta_physics.
Never arms PX4 or connects to aircraft. Each probe has its own GZ_PARTITION.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from vehicle.skybeta import G, add, build_model, load, report


def probe(mode, profile, geometry):
    # Set the transport partition BEFORE constructing any transport nodes.
    os.environ['GZ_PARTITION']='skybeta-probe-'+uuid.uuid4().hex
    from gz.msgs10.actuators_pb2 import Actuators
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.odometry_pb2 import Odometry
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.transport13 import Node

    sdf,props=build_model(profile,geometry)
    model=sdf.find('model'); model.find('pose').text='0 0 10 0 0 0'
    if mode in ('roll','yaw'):
        # Isolate inertia/allocation from aerodynamic damping when comparing
        # against the undamped Euler equation below. Flight model is unchanged.
        for plugin in model.findall('plugin'):
            for field in ('rotorDragCoefficient','rollingMomentCoefficient'):
                value=plugin.find(field)
                if value is not None:value.text='0'
    # Visual meshes and sensors do not participate in this dynamics probe.
    for link in model.findall('link'):
        for child in list(link):
            if child.tag in ('visual','sensor'): link.remove(child)
    odom=add(model,'plugin',filename='gz-sim-odometry-publisher-system',name='gz::sim::systems::OdometryPublisher')
    add(odom,'odom_publish_frequency',200); add(odom,'dimensions',3); add(odom,'odom_topic','/probe/odometry')
    root=ET.Element('sdf',version='1.9'); world=add(root,'world',name='probe')
    physics=add(world,'physics',name='fixed',type='ignored')
    add(physics,'max_step_size',.004); add(physics,'real_time_factor',1)
    add(world,'gravity',f'0 0 {-G}')
    add(world,'plugin',filename='gz-sim-physics-system',name='gz::sim::systems::Physics')
    world.append(model)
    node=Node(); samples=[]
    def receive(msg):
        t=msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
        q=msg.pose.orientation; v=msg.twist.linear; w=msg.twist.angular
        samples.append([t,msg.pose.position.z,v.x,v.y,v.z,w.x,w.y,w.z,q.x,q.y,q.z,q.w])
    if not node.subscribe(Odometry,'/probe/odometry',receive): raise RuntimeError('odometry subscribe failed')
    publisher=node.advertise('/skybeta/command/motor_speed',Actuators)
    message=Actuators()
    speeds=props['hover_speed'].copy() if mode!='freefall' else np.zeros(4)
    if mode=='climb': speeds*=1.1
    applied_torque=np.zeros(3)
    if mode in ('roll','yaw'):
        applied_torque[0 if mode=='roll' else 2]=.001
        arms=np.array([r['position']-props['com'] for r in props['rotors']])
        allocation=np.array([np.ones(4),arms[:,1],-arms[:,0],
                             [r['reaction_sign']*profile['motor']['moment_ratio_m'] for r in props['rotors']]])
        thrust=props['hover_thrust']+np.linalg.solve(allocation,np.r_[0,applied_torque])
        speeds=np.sqrt(thrust/profile['motor']['thrust_coefficient'])
    message.velocity.extend(speeds.tolist())
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'probe.sdf'; ET.ElementTree(root).write(path)
        with (Path(temp)/'gazebo.log').open('w+') as log:
            server=subprocess.Popen(['gz','sim','-s',str(path)],stdout=log,stderr=subprocess.STDOUT)
            try:
                control=WorldControl(); control.pause=True
                for _ in range(30):
                    ok,_=node.request('/world/probe/control',control,WorldControl,Boolean,1000)
                    if ok: break
                    if server.poll() is not None: raise RuntimeError('Gazebo exited during startup')
                else: raise TimeoutError('Gazebo control service did not appear')
                # Re-publish through discovery before allowing simulation time.
                for _ in range(15): publisher.publish(message); time.sleep(.02)
                control.pause=False
                ok,_=node.request('/world/probe/control',control,WorldControl,Boolean,2000)
                if not ok: raise RuntimeError('could not start Gazebo physics')
                deadline=time.monotonic()+25
                while time.monotonic()<deadline:
                    publisher.publish(message)
                    if samples and samples[-1][0]>=1.5: break
                    if server.poll() is not None: raise RuntimeError('Gazebo stopped during probe')
                    time.sleep(.01)
                else: raise TimeoutError('no complete physics trace')
            except Exception:
                log.flush(); log.seek(0); print(log.read()[-6000:])
                raise
            finally:
                server.terminate()
                try: server.wait(timeout=5)
                except subprocess.TimeoutExpired: server.kill(); server.wait()
                node.unsubscribe('/probe/odometry')
    data=np.array(samples); stable=data[(data[:,0]>.6)&(data[:,0]<1.4)]
    if len(stable)<40: raise RuntimeError('not enough steady-state samples')
    acceleration=float(np.polyfit(stable[:,0],stable[:,4],1)[0])
    expected={'freefall':-G,'hover':0,'climb':.21*G,'roll':0,'yaw':0}[mode]
    # Rotor drag affects vertical motion only weakly; allow discretization and
    # open-loop rotor gyroscopic transients but catch coefficient/frame errors.
    error=abs(acceleration-expected)
    tilt=float(2*np.arcsin(np.clip(np.linalg.norm(stable[:,8:10],axis=1),0,1)).max())
    result={'probe':mode,'samples':len(data),'vertical_acceleration_m_s2':acceleration,
            'expected_m_s2':expected,'error_m_s2':error,'max_tilt_rad':tilt,
            'passed':bool(error<.12 and tilt<.04)}
    if np.any(applied_torque):
        # Initial angular acceleration should agree with Euler's equation;
        # use the early settled-motor interval to limit nonlinear coupling.
        early=data[(data[:,0]>.20)&(data[:,0]<.55)]
        alpha=np.array([np.polyfit(early[:,0],early[:,j],1)[0] for j in (5,6,7)])
        predicted=np.linalg.solve(props['inertia'],applied_torque)
        angular_error=float(np.linalg.norm(alpha-predicted))
        result.update(angular_acceleration_rad_s2=alpha.tolist(),expected_angular_rad_s2=predicted.tolist(),
                      angular_error_rad_s2=angular_error,
                      aero_drag_disabled=True,
                      passed=bool(error<.12 and angular_error<.01))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',choices=('freefall','hover','climb','roll','yaw'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/physics/skybeta-check.json'))
    args=parser.parse_args();profile,geometry=load()
    if args.probe:
        print(json.dumps(probe(args.probe,profile,geometry)));return
    # Separate processes prevent Gazebo transport's process-wide discovery
    # singleton from reusing a previous probe's partition.
    import sys
    results=[]
    for mode in ('freefall','hover','climb','roll','yaw'):
        result=subprocess.run([sys.executable,'-m','tools.check_skybeta_physics','--probe',mode],capture_output=True,text=True,timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    summary={'airframe':report(profile,geometry),'probes':results,'passed':all(r['passed'] for r in results),
             'scope':'Gazebo rigid body and motor dynamics only; not PX4 flight or hardware calibration'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    if not summary['passed']: raise SystemExit(1)


if __name__=='__main__':main()
