"""SITL-only image pose -> PX4 EKF2 -> Offboard route.

No Gazebo pose/odometry subscription exists here. Position feedback is PX4's
LOCAL_POSITION_NED; its external vision input comes from rendered image pixels.
Only a loopback MAVLink endpoint inside the dedicated simulator is supported.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import time
from uuid import uuid4

from .core import CourseController, ZERO
from .px4_frames import camera_pose_to_px4, enu_to_ned, pose_covariance
from .landing import LandingMonitor
from vision.course import CAMERA_HEIGHT, CAMERA_RATE, CAMERA_WIDTH

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PARAMS = {
    'EKF2_GPS_CTRL': 0, 'EKF2_EV_CTRL': 11, 'EKF2_HGT_REF': 3,
    'EKF2_BARO_CTRL': 0, 'EKF2_RNG_CTRL': 0, 'EKF2_OF_CTRL': 0,
    'EKF2_MAG_TYPE': 5,
}


def px4_parameter_value(message):
    """PX4 uses MAVLink's bytewise parameter encoding, including integer EKF flags."""
    if message.param_type == 6:  # MAV_PARAM_TYPE_INT32
        return struct.unpack('<i', struct.pack('<f', message.param_value))[0]
    return message.param_value


class Log:
    def __init__(self, path, course, task, speed_scale=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('w', buffering=1)
        self.summary_path = path.with_suffix('.summary.json')
        self.summary_path.unlink(missing_ok=True)
        self.run_id = uuid4().hex
        self.write(type='metadata', run_id=self.run_id, world='dronehacks',
                   vehicle='course_quad', vehicle_model='PX4 x500',
                   source='PX4 EKF2 + camera marker localization',
                   navigation_source='camera',
                   simulator_position_used_for_control=False,
                   camera_method='OpenCV AprilTag map + solvePnP',
                   camera_resolution_px=[CAMERA_WIDTH, CAMERA_HEIGHT],
                   camera_rate_hz=CAMERA_RATE,
                   task=task, waypoint_count=len(course['waypoints']),
                   waypoints=course['waypoints'], course_limits=course.get('limits', {}),
                   speed_scale=speed_scale, footprint_radius=course.get('footprint_radius'))

    def write(self, **row):
        self.file.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')

    def event(self, message, **kwargs):
        print(message, flush=True)
        self.write(type='event', message=message, **kwargs)

    def finish(self, result, **kwargs):
        row = dict(type='summary', run_id=self.run_id, result=result, **kwargs)
        self.write(**row)
        temp = self.summary_path.with_suffix('.tmp')
        temp.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        temp.replace(self.summary_path)


class Link:
    def __init__(self, log):
        # Set before import: covariance extensions require MAVLink 2.
        os.environ['MAVLINK20'] = '1'
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink
        self.link = mavutil.mavlink_connection('udpin:127.0.0.1:14540',
                                              source_system=245, source_component=191)
        self.log = log
        self.heartbeat = None
        self.position = None
        self.attitude = None
        self.estimator = None
        self.params = {}
        self.position_wall = 0.0
        self.heartbeat_wall = 0.0
        self.heartbeat_sim_s = 0.0
        self.boot_s = 0.0
        self.sim_usec_now = None
        self.timesync_responses = 0
        self.ready = False
        self.landed_state = None
        self.landed_state_sim_s = None

    def command(self, command, *params):
        values = list(params) + [0.0] * (7-len(params))
        self.link.mav.command_long_send(1, 1, command, 0, *values)

    def drain(self):
        for _ in range(150):
            msg = self.link.recv_match(blocking=False)
            if msg is None:
                break
            if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                continue
            kind = msg.get_type()
            if kind == 'HEARTBEAT':
                if msg.autopilot != self.m.MAV_AUTOPILOT_PX4:
                    raise RuntimeError('The loopback vehicle is not PX4')
                self.heartbeat, self.heartbeat_wall = msg, time.monotonic()
                self.heartbeat_sim_s = self.boot_s
                if not self.ready:
                    self.ready = True
                    for name in REQUIRED_PARAMS:
                        self.link.mav.param_request_read_send(1, 1, name.encode(), -1)
                    for message_id, rate in ((32, 30), (30, 30), (230, 5), (245, 5), (2, 5), (111, 50)):
                        self.command(self.m.MAV_CMD_SET_MESSAGE_INTERVAL,
                                     message_id, 1e6/rate)
                    self.log.event('PX4 heartbeat connected on container loopback')
            elif kind == 'LOCAL_POSITION_NED':
                self.position, self.position_wall = msg, time.monotonic()
                self.boot_s = max(self.boot_s, msg.time_boot_ms / 1000.0)
            elif kind == 'SYSTEM_TIME':
                self.boot_s = max(self.boot_s, msg.time_boot_ms / 1000.0)
            elif kind == 'ATTITUDE':
                self.attitude = msg
                self.boot_s = max(self.boot_s, msg.time_boot_ms / 1000.0)
            elif kind == 'ESTIMATOR_STATUS':
                self.estimator = msg
            elif kind == 'EXTENDED_SYS_STATE':
                self.landed_state = msg.landed_state
                self.landed_state_sim_s = self.boot_s
            elif kind == 'TIMESYNC' and msg.tc1 == 0 and self.sim_usec_now is not None:
                self.link.mav.timesync_send(int(self.sim_usec_now) * 1000, msg.ts1)
                self.timesync_responses += 1
            elif kind == 'PARAM_VALUE':
                name = msg.param_id
                if isinstance(name, bytes):
                    name = name.decode().rstrip('\0')
                if name in REQUIRED_PARAMS:
                    value = px4_parameter_value(msg)
                    self.params[name] = value
                    self.log.event(f'{name}={value:g}', parameter=name, value=value)
            elif kind == 'STATUSTEXT':
                self.log.event(str(msg.text), severity=msg.severity)
            elif kind == 'COMMAND_ACK':
                if msg.command != self.m.MAV_CMD_SET_MESSAGE_INTERVAL:
                    self.log.event(f'PX4 command {msg.command}: result {msg.result}',
                                   command=msg.command, result=msg.result)

    @property
    def armed(self):
        return bool(self.heartbeat and
                    self.heartbeat.base_mode & self.m.MAV_MODE_FLAG_SAFETY_ARMED)

    @property
    def offboard(self):
        return bool(self.heartbeat and (self.heartbeat.custom_mode >> 16) & 255 == 6)

    @property
    def params_ok(self):
        return all(self.params.get(k) == v for k, v in REQUIRED_PARAMS.items())

    def telemetry_fresh(self, now):
        # Heartbeats run at 1 Hz of simulated time, which can be much slower
        # than wall time under software rendering. Position has a wall watchdog.
        return (self.position is not None and now-self.position_wall < 2.0
                and self.boot_s-self.heartbeat_sim_s <= 3.0)

    def vision(self, observation):
        p, rpy = camera_pose_to_px4(observation['position_enu'],
                                   observation['orientation_enu_flu_wxyz'])
        self.link.mav.vision_position_estimate_send(
            int(observation['sim_usec']), *p, *rpy, pose_covariance(), 0)

    def velocity(self, enu_velocity, yaw_enu=0.0):
        # Ignore position, acceleration and yaw rate. Hold a fixed map heading.
        mask = 1 | 2 | 4 | 64 | 128 | 256 | 2048
        self.link.mav.set_position_target_local_ned_send(
            int(self.boot_s * 1000) & 0xffffffff, 1, 1,
            self.m.MAV_FRAME_LOCAL_NED, mask,
            0, 0, 0, *enu_to_ned(enu_velocity), 0, 0, 0,
            math.pi/2-yaw_enu, 0)

    def gcs_heartbeat(self):
        self.link.mav.heartbeat_send(self.m.MAV_TYPE_GCS,
                                     self.m.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def run(args):
    if os.environ.get('DRONEHACKS_SITL') != '1':
        raise RuntimeError('Run inside the dedicated simulator with DRONEHACKS_SITL=1')
    from vision.localize import Vision
    from vision.worker import CameraWorker
    course = json.loads(args.course.read_text())
    if args.task == 'hover':
        p = list(course['start']); p[2] = .95
        course['waypoints'] = [dict(position=p, phase='CAMERA_HOVER', speed=.25)]
        course['landing'] = dict(position=[p[0], p[1], 0.0], approach_height=p[2])
    # Stock x500 is a different airframe from Skybeta. No window claim here.
    course['footprint_radius'] = .36
    course['limits']['odom_timeout'] = .6
    course['limits']['state_timeout'] = 2.0
    course['limits']['max_yaw_rate'] = .4
    controller = CourseController(course, speed_scale=args.speed_scale)
    log = Log(args.output, course, args.task, args.speed_scale)
    try:
        link = Link(log)
        vision = CameraWorker(Vision(output_dir=ROOT / 'artifacts/vision'))
    except Exception as error:
        detail = f'{type(error).__name__}: {error}'
        log.event('Bridge initialization failed', error=detail)
        log.finish('failed', fault='bridge_initialization_failed', error=detail)
        log.file.close()
        raise
    wall_start = time.monotonic()
    last_hb = last_setpoint = last_retry = 0.0
    last_log = -100.0
    valid_wall = 0.0
    observation = None
    valid_sim = None
    priming = None
    start = None
    completed = None
    last_control_stamp = None
    command = None
    fault = None
    sent = 0
    blackout = False
    finish_action = getattr(args, 'finish_action', 'hold')
    landing = LandingMonitor(course['landing']['position'],
                             getattr(args, 'pad_radius', .762),
                             course['footprint_radius'],
                             getattr(args, 'landing_timeout', 30.0))
    last_land_request = None
    route_time = None
    log.event('Finish policy selected', finish_action=finish_action,
              pad_radius_m=landing.pad_radius, aircraft_radius_m=landing.aircraft_radius)
    log.event('Waiting for image-derived pose and PX4 EKF2; GPS fusion must be disabled')
    try:
        while True:
            now = time.monotonic()
            link.sim_usec_now = getattr(vision, 'sim_usec_now', None)
            link.drain()
            frame = vision.poll()
            if frame is not None:
                observation = frame
                if (args.blackout_after is not None and start is not None and
                        link.boot_s-start >= args.blackout_after):
                    if not blackout:
                        log.event('Injected camera dropout: no more vision measurements sent')
                    blackout = True
                if frame.get('valid') and link.ready and not blackout:
                    link.vision(frame)
                    valid_wall, valid_sim = now, frame['sim_usec']/1e6
                    sent += 1
            if now-last_hb >= 1.0 and link.ready:
                link.gcs_heartbeat()
                last_hb = now
            position = link.position
            sim_s = link.boot_s
            # Sim capture time is used for freshness; wall timeout also catches a dead bridge.
            camera_fresh = (valid_sim is not None and -.15 <= sim_s-valid_sim <= .6
                            and now-valid_wall < 3.0)
            state_fresh = position is not None and now-link.position_wall < 2.0
            if start is None:
                phase = 'WAIT_FOR_CAMERA_AND_EKF'
                if (state_fresh and camera_fresh and link.params_ok and sent >= 60
                        and link.timesync_responses >= 650):
                    if priming is None:
                        priming = sim_s
                        log.event('Camera and estimator available; priming Offboard')
                    phase = 'PRIMING_OFFBOARD'
                    if now-last_setpoint >= .04:
                        link.velocity(ZERO)
                        last_setpoint = now
                    if sim_s-priming > 2.0 and sim_s-last_retry > 2.0:
                        link.command(link.m.MAV_CMD_DO_SET_MODE, 1, 6)
                        if not link.armed:
                            link.command(link.m.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                        last_retry = sim_s
                    if link.armed and link.offboard:
                        start = sim_s
                        log.event('PX4 armed in Offboard using camera-aided EKF position',
                                  sim_time=sim_s, verified_parameters=link.params)
                else:
                    priming = None
                    if sim_s-last_retry > 3 and link.ready:
                        for name in REQUIRED_PARAMS:
                            link.link.mav.param_request_read_send(1, 1, name.encode(), -1)
                        last_retry = sim_s
                if now-wall_start > args.startup_timeout:
                    fault = 'camera_or_estimator_startup_timeout'
                    break
            else:
                phase = command.phase if command else 'TAKEOFF'
                if landing.started is not None:
                    p = enu_to_ned((position.x, position.y, position.z))
                    ground_fresh = (link.landed_state_sim_s is not None and
                                    0 <= sim_s-link.landed_state_sim_s <= 1.0)
                    status, fault = landing.check(
                        sim_s, p, armed=link.armed,
                        on_ground=link.landed_state == link.m.MAV_LANDED_STATE_ON_GROUND,
                        telemetry_fresh=link.telemetry_fresh(now) and ground_fresh)
                    if status == 'failed':
                        break
                    if status == 'completed':
                        completed = sim_s
                        log.finish('completed', elapsed_sim_s=sim_s-start,
                                   route_elapsed_sim_s=route_time, route_completed=True,
                                   landed_and_disarmed=True, landing_position_enu=p,
                                   pad_radius_m=landing.pad_radius,
                                   aircraft_radius_m=landing.aircraft_radius,
                                   waypoints_reached=controller.waypoint_index,
                                   waypoint_count=len(controller.waypoints),
                                   navigation_source='camera + PX4 EKF2',
                                   vision_measurements_sent=sent,
                                   simulator_position_used_for_control=False,
                                   note='SITL ground/disarm and estimated pad containment; obstacle clearance not scored')
                        log.event('PX4 confirmed touchdown and disarm inside the modeled pad')
                        break
                    if not camera_fresh:
                        fault = 'camera_tracking_lost_during_landing'
                        break
                    if last_land_request is None or sim_s-last_land_request >= 2.0:
                        link.command(link.m.MAV_CMD_NAV_LAND)
                        last_land_request = sim_s
                    # Stop all Offboard velocity messages during AUTO_LAND.
                    if sim_s-last_log >= .1:
                        last_log = sim_s
                        log.write(type='sample', sim_time=sim_s, elapsed_sim_s=sim_s-start,
                                  position=p, phase='AUTO_LAND', px4_armed=link.armed,
                                  landed_state=link.landed_state, done=False,
                                  waypoint_index=controller.waypoint_index)
                    time.sleep(.002)
                    continue
                if not camera_fresh:
                    fault = 'camera_tracking_lost'
                    break
                if not link.telemetry_fresh(now):
                    fault = 'px4_telemetry_lost'
                    break
                if not link.offboard or not link.armed:
                    fault = 'px4_offboard_or_arm_lost'
                    break
                stamp = position.time_boot_ms / 1000
                if stamp != last_control_stamp:
                    p = enu_to_ned((position.x, position.y, position.z))
                    v = enu_to_ned((position.vx, position.vy, position.vz))
                    command = controller.update(p, v, stamp, yaw=None)
                    last_control_stamp = stamp
                    if command.fault:
                        fault = command.fault
                        break
                    if command.done and completed is None:
                        if finish_action == 'land':
                            route_time = sim_s-start
                            landing.start(sim_s)
                            log.event('Route complete; requesting SITL AUTO_LAND',
                                      route_elapsed_sim_s=route_time)
                            continue
                        completed = sim_s
                        log.finish('completed', elapsed_sim_s=sim_s-start,
                                   route_completed=True,
                                   waypoints_reached=controller.waypoint_index,
                                   waypoint_count=len(controller.waypoints),
                                   navigation_source='camera + PX4 EKF2',
                                   vision_measurements_sent=sent,
                                   simulator_position_used_for_control=False,
                                   landed_and_disarmed=False,
                                   note='Estimator-based waypoint completion; collision/clearance not scored here')
                        log.event('Camera/PX4 mission reached final position hold')
                if now-last_setpoint >= .04 and command:
                    link.velocity(command.velocity)
                    last_setpoint = now
                if completed is None and sim_s-start > args.max_sim_seconds:
                    fault = 'mission_timeout'
                    break
                if completed is not None and args.exit_on_finish:
                    break
            if sim_s-last_log >= .1 or (not sim_s and now-wall_start < .2):
                last_log = sim_s
                p = enu_to_ned((position.x, position.y, position.z)) if position else None
                v = enu_to_ned((position.vx, position.vy, position.vz)) if position else None
                log.write(type='sample', sim_time=sim_s,
                          elapsed_sim_s=max(0, sim_s-start) if start else 0,
                          position=p, velocity=v,
                          command_velocity=command.velocity if command else ZERO,
                          phase=phase, waypoint_index=controller.waypoint_index,
                          waypoint_count=len(controller.waypoints),
                          target=command.target if command else None,
                          done=bool(completed), fault=None,
                          px4_armed=link.armed, px4_offboard=link.offboard,
                          px4_gps_fusion=link.params.get('EKF2_GPS_CTRL'),
                          px4_ev_ctrl=link.params.get('EKF2_EV_CTRL'),
                          navigation_source='camera + PX4 EKF2',
                          vision_valid=camera_fresh,
                          vision_age_sim_s=sim_s-valid_sim if valid_sim is not None else None,
                          vision_measurements_sent=sent,
                          timesync_responses=link.timesync_responses,
                          tag_count=(observation or {}).get('tag_count', 0),
                          estimator_flags=getattr(link.estimator, 'flags', None))
            time.sleep(.002)
    except KeyboardInterrupt:
        fault = 'operator_stopped'
    except Exception as error:
        fault = 'bridge_exception'
        log.event('Bridge exception', error=f'{type(error).__name__}: {error}')
    finally:
        if link.armed:
            # Stop Offboard stream and request PX4 Land. Never force-disarm in air.
            try:
                link.command(link.m.MAV_CMD_NAV_LAND)
            except Exception as error:
                log.event('Land request could not be transmitted', error=str(error))
        if fault:
            log.event(f'Mission stopped: {fault}; requested PX4 Land')
            log.finish('failed', fault=fault,
                       elapsed_sim_s=max(0, link.boot_s-start) if start else 0,
                       route_completed=False, waypoints_reached=controller.waypoint_index,
                       vision_measurements_sent=sent, navigation_source='camera + PX4 EKF2')
        close_vision = getattr(vision, 'close', None)
        if close_vision:
            close_vision()
        log.file.close()
    return 1 if fault else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--course', type=Path, default=ROOT/'mission/course.json')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/px4/telemetry.jsonl')
    p.add_argument('--task', choices=('hover', 'course'), default='hover')
    p.add_argument('--speed-scale', type=float, default=.45)
    p.add_argument('--startup-timeout', type=float, default=240)
    p.add_argument('--max-sim-seconds', type=float, default=240)
    p.add_argument('--blackout-after', type=float)
    p.add_argument('--exit-on-finish', action='store_true')
    p.add_argument('--finish-action', choices=('hold', 'land'), default='hold',
                   help='SITL only: land waits for ground/disarm and estimated pad containment')
    p.add_argument('--pad-radius', type=float, default=.762,
                   help='Modeled outer radius in metres; preliminary CAD pad is 5 ft across')
    p.add_argument('--landing-timeout', type=float, default=30.0)
    raise SystemExit(run(p.parse_args()))


if __name__ == '__main__':
    main()
