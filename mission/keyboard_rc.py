"""Local-only keyboard velocity control for the isolated PX4/Gazebo simulator."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

DIRECTORY = Path('artifacts/keyboard')
KEYS = frozenset(('KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown'))


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


def movement(keys, yaw):
    forward = .6*(('KeyW' in keys)-('KeyS' in keys))
    left = .6*(('KeyA' in keys)-('KeyD' in keys))
    length = math.hypot(forward, left)
    if length > .6:
        forward, left = forward*.6/length, left*.6/length
    return (forward*math.cos(yaw)-left*math.sin(yaw),
            forward*math.sin(yaw)+left*math.cos(yaw),
            .3*(('ArrowUp' in keys)-('ArrowDown' in keys)),
            .5*(('KeyQ' in keys)-('KeyE' in keys)))


class Logger:
    def __init__(self):
        self.file = (DIRECTORY/'events.jsonl').open('a', buffering=1)
    def event(self, message, **fields):
        self.file.write(json.dumps(dict(time=time.time(), message=message, **fields))+'\n')


def run():
    if os.environ.get('DRONEHACKS_SITL') != '1' or os.environ.get('SIM_KEYBOARD_RC') != '1':
        raise RuntimeError('Keyboard RC requires the dedicated local simulator')
    lock = open('/tmp/dronehacks-keyboard.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from mission.markerfree_run import MarkerFreeLink, record_px4_tilt, observe_abort_landing
    from mission.sitl_health import SitlHealthMonitor
    from mission.keyboard_input import InputMailbox
    from mission.px4_frames import enu_to_ned
    from vision.worker import CameraWorker
    from vision.rgbd_odometry import Vision

    DIRECTORY.mkdir(parents=True, exist_ok=True)
    log = Logger()
    link = MarkerFreeLink(log)
    camera = CameraWorker(Vision((-2.5146, .9144, .24),
        (math.sqrt(.5), 0., 0., math.sqrt(.5)), use_px4_tilt=True, survey_normals='fals7'))
    health = SitlHealthMonitor()
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    phase, reason = 'CONNECTING', 'Waiting for camera and PX4'
    requested = False
    frame = None
    valid_wall = valid_sim = None
    sent = 0
    last_hb = last_send = last_status = last_land = last_retry = 0.
    priming = None
    action_seen = None
    hold = None
    yaw = math.pi/2
    last_boot = None
    last_advance = time.monotonic()
    motion = (0., 0., 0.)
    previous_phase = None
    input_lease = InputMailbox()
    try:
        while not stop:
            now = time.monotonic()
            link.sim_usec_now = camera.sim_usec_now
            link.drain()
            record_px4_tilt(link, camera)
            sim = link.boot_s
            dt = min(.1, max(0., sim-last_boot)) if last_boot is not None else 0.
            if last_boot != sim:
                last_advance = now
            last_boot = sim
            fresh = camera.poll()
            if fresh:
                frame = fresh
                if frame.get('valid') and link.ready:
                    link.vision(frame)
                    valid_wall, valid_sim = now, frame['sim_usec']/1e6
                    sent += 1
            if now-last_hb > 1:
                link.gcs_heartbeat()
                last_hb = now
            p, a = link.position, link.attitude
            position = list(enu_to_ned((p.x,p.y,p.z))) if p else None
            velocity = list(enu_to_ned((p.vx,p.vy,p.vz))) if p else None
            camera_ok = (frame and frame.get('valid') and valid_sim is not None
                         and -.15 <= sim-valid_sim <= .6 and now-valid_wall < 3)
            state_ok = (p and a and link.telemetry_fresh(now) and
                        -.1 <= sim-p.time_boot_ms/1000 <= .4 and
                        -.1 <= sim-a.time_boot_ms/1000 <= .4 and now-last_advance < 3
                        and all(math.isfinite(v) for v in (*position,*velocity,a.roll,a.pitch,a.yaw)))
            healthy = health.status(sim, now)['healthy']
            sensors = bool(camera_ok and state_ok and link.params_ok and healthy)
            grounded = (link.landed_state == link.m.MAV_LANDED_STATE_ON_GROUND and
                        link.landed_state_sim_s is not None and 0 <= sim-link.landed_state_sim_s <= 1)
            ready = bool(sensors and not link.armed and grounded and sent >= 60 and
                         link.timesync_responses >= 650 and sim >= 12 and not requested)
            row = input_lease.poll()
            enabled = bool(row and row['enabled'])
            action = row.get('action', 'none') if row else 'none'
            action_id = row.get('action_id') if row else None
            if action_id != action_seen and action != 'none':
                action_seen = action_id
                if action == 'takeoff' and enabled and ready:
                    requested = True
                    priming = sim
                    hold = [*position[:2], .95]
                    yaw = math.pi/2-a.yaw
                    phase, reason = 'PRIMING', 'Preparing takeoff'
                elif action == 'land' and (requested or link.armed):
                    phase, reason = 'LANDING', 'Operator requested landing'
            if requested and (not enabled or not sensors):
                if phase != 'LANDING':
                    log.event('Keyboard flight interrupted', enabled=enabled, camera_ok=bool(camera_ok),
                              state_ok=bool(state_ok), params_ok=link.params_ok, power_ok=healthy,
                              sim_time=sim, previous_phase=phase)
                phase = 'LANDING'
                reason = 'Controls released or connection lost' if not enabled else 'Sensor tracking unavailable'
            if link.armed and not requested:
                phase, reason = 'LANDING', 'Unexpected armed state'
                requested = True
            if phase == 'LANDING':
                if now-last_land > 1:
                    link.command(link.m.MAV_CMD_NAV_LAND)
                    last_land = now
                if not link.armed and grounded:
                    requested = False
                    priming = None
                    phase, reason = 'DISARMED', 'Landed and disarmed'
            elif requested:
                if not link.armed:
                    if now-last_send >= .04:
                        link.velocity((0.,0.,0.), yaw)
                        last_send = now
                    if sim-priming > 2 and sim-last_retry > 2:
                        link.command(link.m.MAV_CMD_DO_SET_MODE,1,6)
                        link.command(link.m.MAV_CMD_COMPONENT_ARM_DISARM,1)
                        last_retry = sim
                    if sim-priming > 15:
                        phase, reason = 'LANDING', 'Arming timed out'
                elif not link.offboard:
                    phase, reason = 'LANDING', 'PX4 left Offboard mode'
                else:
                    if phase == 'PRIMING':
                        phase, reason = 'TAKEOFF', 'Climbing to 0.95 m'
                    if phase == 'TAKEOFF' and abs(position[2]-.95) < .08:
                        phase, reason = 'FLYING', 'Keyboard control active'
                    keys = set(row['keys']) if enabled and phase == 'FLYING' else set()
                    vx,vy,vz,yr = movement(keys, math.pi/2-a.yaw)
                    yaw += yr*dt
                    horizontal = bool(keys & {'KeyW','KeyA','KeyS','KeyD'})
                    if horizontal:
                        hold[:2] = position[:2]
                    if vz:
                        hold[2] = min(2.2,max(.35,hold[2]+vz*dt))
                    desired = [vx,vy,0.]
                    for i in range(3):
                        if i == 2 or not horizontal:
                            desired[i] = max(-.5,min(.5,1.7*(hold[i]-position[i])-.8*velocity[i]))
                    motion = tuple(old+max(-dt,min(dt,new-old)) for old,new in zip(motion,desired))
                    if now-last_send >= .04:
                        link.velocity(motion,yaw)
                        last_send = now
            else:
                phase = 'READY' if ready else 'CONNECTING'
                reason = 'Enable keyboard RC, then take off' if ready else 'Waiting for camera and PX4'
                motion = (0.,0.,0.)
            if phase != previous_phase:
                log.event('Keyboard flight phase', phase=phase, reason=reason, sim_time=sim, position=position, armed=link.armed)
                previous_phase = phase
            if now-last_status > .2:
                write_json(DIRECTORY/'status.json', dict(phase=phase,ready=ready,armed=link.armed,
                    position=position,yaw=yaw,sim_time=sim,reason=reason,updated_at=time.time(),
                    enabled=enabled,action_id=action_seen,vision_valid=bool(camera_ok),
                    command_velocity=motion,target_position=hold))
                last_status = now
            time.sleep(.005)
    finally:
        if link.armed or requested:
            link.command(link.m.MAV_CMD_NAV_LAND)
            observe_abort_landing(link,camera,log)
        input_lease.close()
        camera.close()
        health.close()
        write_json(DIRECTORY/'status.json',dict(phase='STOPPED',ready=False,armed=link.armed,
            reason='Controller stopped',updated_at=time.time()))
        log.file.close()

if __name__ == '__main__':
    run()
