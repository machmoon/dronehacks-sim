"""Marker-free RGBD -> PX4 EKF2 -> SITL hover, course, or learned policy.

Only camera measurements and PX4 estimates drive navigation. Any simulator
truth recorder must be a separate process used for evaluation only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

from learning.inference import (CAMERA_HEIGHT, CAMERA_WIDTH, PolicyClient, PolicyError,
                                action_to_enu, policy_depth, policy_state, write_json,
                                validate_transfer_contract)
from learning.sim import DT as POLICY_STEP
from .core import CourseController, ZERO, limited, norm, sub, vector, wrap
from .boundary_brake import BoundaryBrake
from .heading import route_yaw
from .landing import LandingMonitor
from .navigation_audit import audit_navigation_sources
from .px4_frames import camera_pose_to_px4, enu_to_ned
from .px4_run import Link, REQUIRED_PARAMS, ROOT
from .state_history import PolicyStateHistory
from .sitl_health import SitlHealthMonitor
from .runtime_provenance import px4_runtime_provenance
from .survey_startup import SurveyStartup

NAVIGATION = 'RGBD depth registration to supplied course map + PX4 EKF2'


def navigation_source(args):
    return ('Sparse RGBD SLAM + PX4 EKF2' if getattr(args, 'localizer', 'survey') == 'sparse'
            else NAVIGATION)
REQUIRED_PARAMS.update(EKF2_DELAY_MAX=500., EKF2_EVP_NOISE=.03)
HOVER_SECONDS = 3.0
POLICY_MAX_AGE_S = .45
POLICY_MAX_CAPTURE_AGE_S = .25
# The stopping budget depends on recent local position/velocity, even when
# the most recent camera action remains valid for its longer lifetime.
POLICY_BRAKE_MAX_ESTIMATE_AGE_S = .06
POLICY_MAX_SETPOINT_GAP_S = .10
# Horizontal enclosing radius of the SDF-derived swept propellers plus 15 mm.
# This bounds the course/pad footprint; obstacle/window scoring uses components.
AIRCRAFT_RADIUS = .401


class FlightFault(RuntimeError):
    pass


def record_px4_tilt(link, vision):
    """Forward timestamped inertial tilt only; never PX4 position or yaw."""
    attitude = getattr(link, 'attitude', None)
    record = getattr(getattr(vision, 'vision', None), 'record_attitude', None)
    if attitude is not None and record is not None:
        record(attitude.time_boot_ms*1000, attitude.roll, attitude.pitch)


def observe_abort_landing(link, vision, log, wall_timeout=120., *, allow_vision_aid=True):
    """Keep valid sensor aid alive while PX4 lands; never stream route commands."""
    deadline = time.monotonic() + wall_timeout
    last_heartbeat = last_log = 0.
    while time.monotonic() < deadline:
        now = time.monotonic()
        link.sim_usec_now = vision.sim_usec_now if vision is not None else None
        link.drain()
        if allow_vision_aid:
            record_px4_tilt(link, vision)
        if now-last_heartbeat >= 1:
            link.gcs_heartbeat()
            last_heartbeat = now
        if vision is not None and allow_vision_aid:
            try:
                frame = vision.poll()
                if frame and frame.get('valid') and -.15 <= link.boot_s-frame['sim_usec']/1e6 <= .4:
                    link.vision(frame)
            except Exception:
                # Invalid aid remains absent. PX4 handles the degraded landing.
                allow_vision_aid = False
        ground_fresh = (link.landed_state_sim_s is not None and
                        0 <= link.boot_s-link.landed_state_sim_s <= 1)
        if not link.armed and ground_fresh and link.landed_state == link.m.MAV_LANDED_STATE_ON_GROUND:
            log.event('PX4 confirmed touchdown and disarm after abort', sim_time=link.boot_s)
            return True
        if now-last_log >= 1:
            log.event('Observing PX4 abort landing', sim_time=link.boot_s,
                      armed=link.armed, landed_state=link.landed_state)
            last_log = now
        time.sleep(.01)
    log.event('Abort landing confirmation timed out', armed=link.armed, landed_state=link.landed_state)
    return False


def policy_camera_shift(physical, trained, mode):
    shift = physical - trained
    if shift < -1e-6 or (mode == 'raw' and abs(shift) > 1e-6):
        raise FlightFault('policy_checkpoint_camera_mount_mismatch')
    return max(shift, 0.)


class Log:
    def __init__(self, args, course, launch, camera_translation, boundary_brake=None):
        from vision.depth_odometry import survey_normal_config
        from vision.rgbd_odometry import Gates

        vision_gates = Gates()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = args.output.open('w', buffering=1)
        self.write_error = None
        self.summary_path = args.output.with_suffix('.summary.json')
        self.summary_path.unlink(missing_ok=True)
        self.run_id = uuid4().hex
        self.common = dict(run_id=self.run_id, world='dronehacks', vehicle='course_quad',
                           vehicle_model='PX4 '+(os.environ.get('SIM_AIRFRAME') or 'x500'),
                           localizer=getattr(args, 'localizer', 'survey'), task=args.task, launch=launch,
                           source=navigation_source(args), navigation_source=navigation_source(args),
                           simulator_position_used_for_control=False,
                           inertial_tilt_prior_enabled=args.use_px4_tilt,
                           require_survey_startup=getattr(args, 'require_survey_startup', False),
                           survey_startup_config={'min_span_s': 3.0, 'max_gap_s': .12}
                               if getattr(args, 'require_survey_startup', False) else None,
                           survey_normals=(survey_normal_config(args.survey_normals)
                                           if getattr(args, 'localizer', 'survey') == 'survey' else None),
                           static_survey_used=getattr(args, 'localizer', 'survey') == 'survey',
                           sitl_power_watchdog_enabled=True,
                           policy_boundary_brake_enabled=boundary_brake is not None,
                           policy_checkpoint_sha256=args.policy_sha256,
                           policy_depth_mode=args.policy_depth_mode if args.task == 'policy' else None,
                           vision_max_capture_gap_s=(.25 if getattr(args, 'localizer', 'survey') == 'sparse'
                                                     else vision_gates.max_gap_s),
                           vision_max_capture_age_s=vision_gates.max_age_s)
        diagnostic_path = Path('/tmp/dronehacks-rgbd-depth-noise.json')
        sensor_diagnostic = (json.loads(diagnostic_path.read_text()) if diagnostic_path.exists()
                             else {'mode': 'off'})
        self.write(type='metadata', **self.common,
                   sensor_diagnostic=sensor_diagnostic,
                   px4_runtime=px4_runtime_provenance(),
                   runtime_source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                          for folder in ('mission', 'vision', 'learning')
                                          for path in sorted((ROOT/folder).rglob('*.py'))},
                   policy_boundary_brake=(dict(response_s=boundary_brake.response_s,
                                               command_lead_s=boundary_brake.lead_s,
                                               position_reserve_m=boundary_brake.reserve_m,
                                               max_estimate_age_s=POLICY_BRAKE_MAX_ESTIMATE_AGE_S,
                                               max_setpoint_gap_s=POLICY_MAX_SETPOINT_GAP_S,
                                               scope='Predictive planar bound guard; no formal safety guarantee')
                                          if boundary_brake is not None else None),
                   camera_method=navigation_source(args),
                   camera_resolution_px=[CAMERA_WIDTH, CAMERA_HEIGHT],
                   camera_translation_flu_m=camera_translation,
                   waypoint_count=len(course['waypoints']), waypoints=course['waypoints'],
                   course_limits=course['limits'], speed_scale=args.speed_scale,
                   heading_lookahead_m=args.heading_lookahead,
                   footprint_radius=course['footprint_radius'], window=course.get('window'),
                   landing=course['landing'])

    def write(self, **row):
        if self.write_error is not None:
            return
        try:
            self.file.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
        except OSError as error:
            # A full evidence disk must not prevent the control loop from
            # requesting Land or monitoring touchdown. Abort on the next tick.
            self.write_error = str(error)
            self._console('Telemetry logging failed: '+self.write_error)

    def _console(self, message):
        try:
            print(message, flush=True)
        except OSError as error:
            self.write_error = self.write_error or str(error)

    def event(self, message, **fields):
        self._console(message)
        self.write(type='event', message=message, **fields)

    def finish(self, **fields):
        row = dict(type='summary', **self.common, **fields)
        self.write(**row)
        if self.write_error is None:
            try:
                write_json(self.summary_path, row)
            except OSError as error:
                self.write_error = str(error)
                self._console('Summary logging failed: '+self.write_error)


class MarkerFreeLink(Link):
    def __init__(self, log):
        super().__init__(log)
        self.policy_history = PolicyStateHistory()

    def drain(self):
        super().drain()
        self.policy_history.record(self.position, self.attitude)

    @property
    def params_ok(self):
        # MAVLink float32 parameter values need a small representation tolerance.
        return all(k in self.params and math.isclose(self.params[k], v, rel_tol=1e-6, abs_tol=1e-7)
                   for k, v in REQUIRED_PARAMS.items())

    def vision(self, observation):
        if not observation.get('valid'):
            raise ValueError('invalid localization must not reach PX4 external vision')
        p, rpy = camera_pose_to_px4(observation['position_enu'],
                                   observation['orientation_enu_flu_wxyz'])
        p, rpy = vector(p), vector(rpy)
        # The estimator reports axis-wise ENU/FLU standard deviations. Rotate
        # ENU position variances to NED and apply conservative covariance floors.
        pos_std = enu_to_ned(vector(observation.get('position_std_m', (.03,)*3)))
        ang_std = vector(observation.get('orientation_std_rad', (.04,)*3))
        covariance = [0.0]*21
        for i, std, floor in zip((0, 6, 11, 15, 18, 20), (*pos_std, *ang_std), (.03,)*3+(.04,)*3):
            covariance[i] = max(abs(std), floor)**2
        self.link.mav.vision_position_estimate_send(
            int(observation['sim_usec']), *p, *rpy, covariance, 0)


class LearnedRoute:
    """Bind every accepted action to its capture time, heading, and waypoint."""
    STEP_USEC = round(POLICY_STEP*1e6)
    # One 20 Hz capture period plus Gazebo's 4 ms sensor scheduling quantum.
    MAX_DEADLINE_LATENESS_USEC = 54_000

    def __init__(self, client, camera_shift_m=0., log=None):
        self.client = client
        self.camera_shift_m = camera_shift_m
        self.log = log
        self.segment = None
        self.pending_yaw = None
        self.pending_coverage = None
        self.action = None
        self.current_coverage = None
        self.segment_started = None
        self.actions = 0
        self.last_accepted_capture = None
        self.next_capture_due_usec = None
        self.preflight_verified = False

    def tick(self, segment, frame, history, target, cap, sim_s, now):
        if segment != self.segment:
            self.client.cancel()
            self.segment = segment
            self.segment_started = sim_s, now
            self.action = None
            self.current_coverage = None
        if self.client.pending is not None:
            reply = self.client.poll(now)
            if reply is not None:
                capture_age = sim_s - reply.capture_usec/1e6
                if not -.15 <= capture_age <= POLICY_MAX_AGE_S:
                    raise PolicyError('stale_policy_response')
                self.action = (reply, action_to_enu(reply.velocity_flu, self.pending_yaw), now)
                self.current_coverage = self.pending_coverage
                if segment == 'preflight':
                    self.preflight_verified = True
                else:
                    self.actions += 1
                    if self.log is not None:
                        previous = self.last_accepted_capture
                        self.log.write(type='policy_action', sim_time=sim_s, waypoint_index=segment+1,
                                       capture_usec=reply.capture_usec, capture_age_sim_s=capture_age,
                                       capture_interval_sim_s=(reply.capture_usec-previous)/1e6 if previous is not None else None,
                                       latency_wall_s=reply.latency_s)
                    self.last_accepted_capture = reply.capture_usec

        capture = frame['sim_usec'] if frame is not None else -1
        deadline = self.next_capture_due_usec
        eligible = (self.client.pending is None and capture > self.client.last_capture
                    and (deadline is None or capture >= deadline)
                    and -.15 <= sim_s-capture/1e6 <= POLICY_MAX_CAPTURE_AGE_S)
        aligned = history.at(capture) if eligible else None
        if aligned is not None:
            # Check real sensor input and exercise the service before arming,
            # but don't expect a ground-level frame to cover the virtual camera.
            # Preflight output never reaches a PX4 velocity setpoint.
            shift = 0. if segment == 'preflight' else self.camera_shift_m
            observation = policy_depth(frame.get('depth_z_m'), shift)
            state, self.pending_yaw = policy_state(aligned.velocity_enu, sub(target, aligned.position_enu),
                                                  *aligned.attitude_rpy_frd, cap)
            self.pending_coverage = observation.coverage, observation.center_coverage
            self.client.request(observation.depth, state, capture, now)
            # Keep the .16 s grid through ordinary frame quantization. After
            # a dropped/late frame, start a fresh grid without catching up.
            if deadline is None or capture-deadline > self.MAX_DEADLINE_LATENESS_USEC:
                deadline = capture
            self.next_capture_due_usec = deadline+self.STEP_USEC

        if self.action is None:
            since_sim, since_wall = self.segment_started
            if sim_s-since_sim > POLICY_MAX_AGE_S or now-since_wall > self.client.timeout_s:
                raise PolicyError('no_fresh_policy_action')
            # A short, bounded pause while the first action for this waypoint
            # arrives. The previous waypoint's output cannot cross the handoff.
            return ZERO, None

        reply, velocity_enu, received = self.action
        if sim_s - reply.capture_usec/1e6 > POLICY_MAX_AGE_S or now-received > self.client.timeout_s:
            raise PolicyError('stale_policy_action')
        return velocity_enu, reply

    def clear(self):
        self.client.cancel()
        self.segment = None
        self.action = None
        self.current_coverage = None
        self.next_capture_due_usec = None


def prepare_course(args, launch):
    course = json.loads(args.course.read_text())
    first = course['waypoints'][0]
    if (first['phase'] != 'TAKEOFF'
            or math.hypot(first['position'][0]-launch['position_enu'][0],
                          first['position'][1]-launch['position_enu'][1]) > .05
            or first['position'][2] < launch['position_enu'][2] + .3):
        raise ValueError('course must begin with a surveyed vertical takeoff waypoint')
    if args.task != 'hover' and (len(course['waypoints']) < 2 or not course.get('window')
                                 or not any(w['phase'] == 'WINDOW_CROSS' for w in course['waypoints'])):
        raise ValueError('course must include the full window route')
    if args.task != 'hover':
        proof = json.loads(args.hover_summary.read_text())
        if not (proof.get('task') == 'hover' and proof.get('result') == 'completed'
                and proof.get('landed_and_disarmed') is True and proof.get('route_completed') is True
                and proof.get('navigation_source') == navigation_source(args)
                and proof.get('vehicle_model', 'PX4 x500') == 'PX4 '+(os.environ.get('SIM_AIRFRAME') or 'x500')
                and proof.get('simulator_position_used_for_control') is False
                and proof.get('inertial_tilt_prior_enabled', False) == getattr(args, 'use_px4_tilt', False)
                and (proof.get('survey_normals') or {}).get('method', 'central2')
                == getattr(args, 'survey_normals', 'central2')
                and proof.get('require_survey_startup', False)
                == getattr(args, 'require_survey_startup', False)
                and proof.get('launch') == launch
                and proof.get('hover_settled_sim_s', 0) >= HOVER_SECONDS):
            raise ValueError('course requires a completed RGBD hover and landing at this launch pose')
    course['footprint_radius'] = AIRCRAFT_RADIUS
    course['limits'].update(odom_timeout=.6, state_timeout=2.0)
    takeoff = dict(course, waypoints=[first], landing={
        'position': [*first['position'][:2], 0.0], 'approach_height': first['position'][2]})
    return takeoff if args.task == 'hover' else course, takeoff


def run(args):
    if os.environ.get('DRONEHACKS_SITL') != '1':
        raise RuntimeError('Run inside the dedicated simulator with DRONEHACKS_SITL=1')
    from vision.rgbd_odometry import Vision
    from vision.rgbd_scene import RGBD_TRANSLATION
    from vision.worker import CameraWorker

    launch = dict(position_enu=list(vector(args.launch_position)), yaw_enu_deg=args.launch_yaw_deg)
    course, takeoff_course = prepare_course(args, launch)
    takeoff = CourseController(takeoff_course, speed_scale=args.speed_scale)
    route = None if args.task == 'hover' else CourseController(
        dict(course, waypoints=course['waypoints'][1:]), speed_scale=args.speed_scale)
    landing = LandingMonitor(course['landing']['position'], args.pad_radius,
                             AIRCRAFT_RADIUS, args.landing_timeout)
    # The default .40 s lead budgets .24 s actuator delay and the maximum
    # estimate age and inter-setpoint interval accepted below.
    boundary_brake = (BoundaryBrake(route.bound_min, route.bound_max, route.footprint_radius)
                      if args.task == 'policy' and not args.no_policy_boundary_brake else None)
    log = Log(args, course, launch, RGBD_TRANSLATION, boundary_brake)
    link = vision = client = learned = health = None
    yaw_command = math.radians(args.launch_yaw_deg)
    launch_yaw = yaw_command
    rotation = (math.cos(yaw_command/2), 0., 0., math.sin(yaw_command/2))
    wall_start = time.monotonic()
    last_hb = last_setpoint = last_retry = 0.0
    last_log = -100.0
    valid_wall = valid_sim = None
    frame = None
    start = priming = last_control_stamp = None
    motion = sent_velocity = ZERO
    last_setpoint_sim = None
    fault = None
    landed_after_fault = False
    sent = 0
    arm_requested = False
    blackout = False
    route_time = settled_at = None
    settled_seconds = 0.0
    last_land_request = None
    phase = 'WAIT_FOR_RGBD_AND_EKF'
    previous_boot = 0.0
    last_sim_advance_wall = wall_start
    position_enu = velocity_enu = None
    reply = command = None
    policy_setpoints_sent = 0
    braking_step = None
    braking_interventions = braking_setpoints = 0
    max_stop_budget_deficit = 0.
    sim_s = 0.0
    survey_startup = SurveyStartup() if getattr(args, 'require_survey_startup', False) else None
    survey_startup_at_arm = None
    try:
        try:
            audit = audit_navigation_sources(static_survey_used=getattr(args, 'localizer', 'survey') == 'survey')
        except Exception as error:
            raise FlightFault(f'navigation_source_audit_failed:{error}') from error
        log.event('Navigation source audit passed', **audit)
        health = SitlHealthMonitor()
        link = MarkerFreeLink(log)
        vision = CameraWorker(Vision(launch['position_enu'], rotation,
                                     rejected_frame_dir=args.output.parent/'vision-diagnostics',
                                     use_px4_tilt=args.use_px4_tilt, survey_normals=args.survey_normals,
                                     localizer=getattr(args, 'localizer', 'survey')))
        if args.task == 'policy':
            client = PolicyClient(args.policy_dir, args.policy_sha256, log.run_id, args.policy_timeout)
            if tuple(RGBD_TRANSLATION[1:]) != (0., 0.):
                raise FlightFault('unsupported_policy_camera_extrinsics')
            learned = LearnedRoute(client, log=log)
        log.event('Waiting for RGBD pose and PX4 EKF2; GPS fusion must be disabled')
        while True:
            now = time.monotonic()
            link.sim_usec_now = vision.sim_usec_now
            link.drain()
            if log.write_error is not None:
                raise FlightFault('telemetry_logging_failed')
            record_px4_tilt(link, vision)
            sim_s = link.boot_s
            power_health = health.status(sim_s, now)
            if sim_s > previous_boot:
                previous_boot, last_sim_advance_wall = sim_s, now
            fresh_frame = vision.poll()
            if (args.blackout_after is not None and start is not None
                    and sim_s-start >= args.blackout_after and not blackout):
                blackout = True
                log.event('Injected RGBD capture loss; no further measurements sent', sim_time=sim_s)
            if fresh_frame is not None and not blackout:
                frame = fresh_frame
                if start is None and survey_startup is not None:
                    survey_startup.update(fresh_frame)
                    log.write(type='survey_startup_capture', source=fresh_frame.get('source'),
                              valid=bool(fresh_frame.get('valid')), **survey_startup.status(sim_s))
                if fresh_frame.get('valid') and link.ready:
                    link.vision(fresh_frame)
                    valid_wall, valid_sim = now, fresh_frame['sim_usec']/1e6
                    sent += 1
            if now-last_hb >= 1.0 and link.ready:
                link.gcs_heartbeat()
                last_hb = now
            camera_fresh = (frame is not None and frame.get('valid') and valid_sim is not None
                            and -.15 <= sim_s-valid_sim <= .6 and now-valid_wall < 3.0)
            position, attitude = link.position, link.attitude
            stamp = None
            if position is not None:
                stamp = position.time_boot_ms/1000
                position_enu = vector(enu_to_ned((position.x, position.y, position.z)))
                velocity_enu = vector(enu_to_ned((position.vx, position.vy, position.vz)))
            attitude_fresh = (attitude is not None and -.10 <= sim_s-attitude.time_boot_ms/1000 <= .4
                              and all(math.isfinite(x) for x in (attitude.roll, attitude.pitch, attitude.yaw)))
            state_fresh = (stamp is not None and link.telemetry_fresh(now) and attitude_fresh
                           and -.10 <= sim_s-stamp <= .4)

            if start is None:
                if link.armed and not arm_requested:
                    raise FlightFault('px4_already_armed')
                sensors_ready = (state_fresh and camera_fresh and link.params_ok and power_health['healthy']
                                 and norm(sub(position_enu, launch['position_enu'])) <= .25
                                 and norm(velocity_enu) <= .18
                                 and (survey_startup is None or survey_startup.ready(sim_s)))
                if learned and sensors_ready and not learned.preflight_verified and client.connect():
                    if not client.contract_verified:
                        profile = validate_transfer_contract(
                            client.navigation_contract, args.course, ROOT/'mission/heading.py',
                            ROOT/'models/flight_x500_base/model.sdf', speed_scale=args.speed_scale,
                            lookahead=args.heading_lookahead)
                        client.contract_verified = True
                        log.event('Learned policy navigation contract verified', transfer_profile=profile)
                    learned.camera_shift_m = policy_camera_shift(RGBD_TRANSLATION[0], client.camera_forward_m,
                                                                 args.policy_depth_mode)
                    if (args.policy_depth_mode == 'raw' and client.heading_lookahead is not None
                            and abs(client.heading_lookahead-args.heading_lookahead) > 1e-6):
                        raise FlightFault('policy_checkpoint_heading_lookahead_mismatch')
                    first = route.waypoints[0]
                    learned.tick('preflight', frame, link.policy_history,
                                 first['position'], first['speed'], sim_s, now)
                if (sensors_ready and sent >= 60 and link.timesync_responses >= 650
                        and (learned is None or learned.preflight_verified)):
                    phase = 'PRIMING_OFFBOARD'
                    if priming is None:
                        priming = sim_s
                        log.event('RGBD and PX4 available; priming Offboard',
                                  sitl_power_health=power_health,
                                  policy_service_session=client.session if client else None,
                                  policy_training_camera_forward_m=client.camera_forward_m if client else None,
                                  policy_training_heading_lookahead_m=client.heading_lookahead if client else None)
                    if now-last_setpoint >= .04:
                        link.velocity(ZERO, launch_yaw)
                        last_setpoint = now
                    if sim_s-priming > 2.0 and sim_s-last_retry > 2.0:
                        if not link.armed and link.landed_state != link.m.MAV_LANDED_STATE_ON_GROUND:
                            raise FlightFault('px4_not_confirmed_on_ground')
                        link.command(link.m.MAV_CMD_DO_SET_MODE, 1, 6)
                        if not link.armed:
                            link.command(link.m.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                            arm_requested = True
                        last_retry = sim_s
                    if link.armed and link.offboard:
                        start = sim_s
                        survey_startup_at_arm = survey_startup.status(sim_s) if survey_startup else None
                        phase = 'TAKEOFF'
                        if learned:
                            learned.clear()
                        log.event('PX4 armed in Offboard with marker-free RGBD aiding',
                                  sim_time=sim_s, verified_parameters=link.params,
                                  survey_startup=survey_startup_at_arm)
                else:
                    if link.armed:
                        raise FlightFault('flight_inputs_lost_during_arming')
                    phase = 'WAIT_FOR_RGBD_AND_EKF'
                    priming = None
                    if sim_s-last_retry > 3 and link.ready and not link.params_ok:
                        for name in REQUIRED_PARAMS:
                            link.link.mav.param_request_read_send(1, 1, name.encode(), -1)
                        last_retry = sim_s
                if now-wall_start > args.startup_timeout:
                    log.event('Startup inputs unavailable', sitl_power_health=power_health,
                              survey_startup=survey_startup.status(sim_s) if survey_startup else None)
                    if survey_startup is not None and not survey_startup.ready(sim_s):
                        raise FlightFault('absolute_survey_startup_unqualified')
                    raise FlightFault('rgbd_ekf_or_policy_startup_timeout')
            else:
                if not power_health['healthy']:
                    log.event('PX4 battery or power feed failed', sitl_power_health=power_health)
                    raise FlightFault('px4_power_health:'+power_health['reason'])
                if not camera_fresh:
                    reason = frame.get('reason') if frame is not None else None
                    raise FlightFault(f'rgbd_pose_unavailable:{reason or "stale_capture"}')
                if not state_fresh or now-last_sim_advance_wall > 3.0:
                    raise FlightFault('px4_telemetry_or_clock_lost')

                if landing.started is not None:
                    phase = 'AUTO_LAND'
                    ground_fresh = (link.landed_state_sim_s is not None
                                    and 0 <= sim_s-link.landed_state_sim_s <= 1.0)
                    status, landing_fault = landing.check(
                        sim_s, position_enu, armed=link.armed,
                        on_ground=link.landed_state == link.m.MAV_LANDED_STATE_ON_GROUND,
                        telemetry_fresh=ground_fresh)
                    if status == 'failed':
                        raise FlightFault(landing_fault)
                    if status == 'completed':
                        log.event('PX4 confirmed touchdown and disarm inside the modeled pad')
                        break
                    if last_land_request is None or sim_s-last_land_request >= 2.0:
                        link.command(link.m.MAV_CMD_NAV_LAND)
                        last_land_request = sim_s
                else:
                    if not link.offboard or not link.armed:
                        raise FlightFault('px4_offboard_or_arm_lost')
                    if sim_s-start > args.max_sim_seconds:
                        raise FlightFault('mission_timeout')
                    if stamp != last_control_stamp:
                        dt = min(.1, max(0., stamp-last_control_stamp)) if last_control_stamp is not None else 0.
                        last_control_stamp = stamp
                        if settled_seconds < HOVER_SECONDS:
                            command = takeoff.update(position_enu, velocity_enu, stamp)
                            phase = 'HOVER_CHECK' if command.done else 'TAKEOFF'
                            if (command.done and norm(velocity_enu) <= .12
                                    and norm(sub(command.target, position_enu)) <= takeoff.arrival_radius):
                                settled_at = stamp if settled_at is None else settled_at
                                settled_seconds = stamp-settled_at
                            else:
                                settled_at, settled_seconds = None, 0.
                            if settled_seconds >= HOVER_SECONDS:
                                log.event('Sustained takeoff hover verified', sim_time=sim_s,
                                          settled_sim_s=settled_seconds, position=position_enu)
                                if route is None:
                                    route_time = sim_s-start
                                    landing.start(sim_s)
                                    continue
                        else:
                            command = route.update(position_enu, velocity_enu, stamp)
                            phase = command.phase
                            if command.done:
                                route_time = sim_s-start
                                if learned:
                                    learned.clear()
                                    reply = None
                                landing.start(sim_s)
                                log.event('Route complete; requesting PX4 AUTO_LAND',
                                          route_elapsed_sim_s=route_time)
                                continue
                            following_index = route.waypoint_index + 1
                            following = (route.waypoints[following_index]['position']
                                         if following_index < len(route.waypoints) else None)
                            goal_yaw = route_yaw(position_enu, command.target, following,
                                                 phase, args.heading_lookahead)
                            if goal_yaw is not None:
                                limit = (1.6 if learned else route.max_yaw_rate)*dt
                                yaw_command = wrap(yaw_command + max(-limit, min(limit, wrap(goal_yaw-yaw_command))))
                        if command.fault:
                            active = takeoff if settled_seconds < HOVER_SECONDS else route
                            log.event('Controller rejected estimated state', sim_time=sim_s,
                                      fault=command.fault, position=position_enu,
                                      previous_position=active.last_position,
                                      vision_position=(frame or {}).get('position_enu'),
                                      vision_capture_sim_s=(frame or {}).get('stamp_sim_s'))
                            raise FlightFault(command.fault)
                        motion = command.velocity
                    policy_active = (learned is not None and settled_seconds >= HOVER_SECONDS
                                     and phase not in ('HOVER_CHECK', 'TAKEOFF'))
                    if policy_active:
                        cap = route.waypoints[route.waypoint_index]['speed']
                        motion, reply = learned.tick(route.waypoint_index, frame, link.policy_history,
                                                     command.target, cap, sim_s, now)
                    if now-last_setpoint >= .04:
                        braking_step = None
                        if policy_active:
                            setpoint_gap = sim_s-last_setpoint_sim if last_setpoint_sim is not None else 0.
                            dt = max(0., min(POLICY_MAX_SETPOINT_GAP_S, setpoint_gap))
                            if boundary_brake is not None:
                                if sim_s-stamp > POLICY_BRAKE_MAX_ESTIMATE_AGE_S+1e-9:
                                    raise FlightFault('policy_boundary_estimate_stale')
                                if setpoint_gap > POLICY_MAX_SETPOINT_GAP_S+1e-9:
                                    raise FlightFault('policy_boundary_setpoint_gap')
                                braking_step = boundary_brake.step(position_enu, velocity_enu, motion, sent_velocity,
                                                                   slew=route.max_accel, dt=dt, cap=cap)
                                sent_velocity = braking_step.sent_velocity
                            else:
                                delta_v = limited(sub(motion, sent_velocity), route.max_accel*dt)
                                sent_velocity = limited(tuple(a+b for a, b in zip(sent_velocity, delta_v)), cap)
                        else:
                            sent_velocity = motion
                        link.velocity(sent_velocity, yaw_command)
                        if policy_active and reply is not None:
                            policy_setpoints_sent += 1
                        if braking_step is not None:
                            braking_setpoints += 1
                            if norm(sub(sent_velocity, braking_step.unfiltered_velocity)) > 1e-9:
                                braking_interventions += 1
                            max_stop_budget_deficit = max(max_stop_budget_deficit,
                                max((value for _, value in braking_step.stop_budget_deficits_m), default=0.))
                        last_setpoint, last_setpoint_sim = now, sim_s

            if sim_s-last_log >= .1:
                last_log = sim_s
                log.write(type='sample', sim_time=sim_s, elapsed_sim_s=sim_s-start if start is not None else 0.,
                          position=position_enu, velocity=velocity_enu,
                          position_stamp_sim_s=position.time_boot_ms/1000 if position is not None else None,
                          attitude_stamp_sim_s=attitude.time_boot_ms/1000 if attitude is not None else None,
                          vision_position_enu=(frame or {}).get('position_enu'),
                          vision_stamp_sim_s=(frame or {}).get('stamp_sim_s'),
                          tof_range_m=(frame or {}).get('tof_range_m'),
                          vision_source=(frame or {}).get('source'),
                          survey_startup=survey_startup.status(sim_s) if survey_startup and start is None else None,
                          attitude_rpy_frd=[attitude.roll, attitude.pitch, attitude.yaw] if attitude_fresh else None,
                          command_velocity=sent_velocity if landing.started is None else ZERO,
                          command_yaw_enu=yaw_command, phase=phase,
                          waypoint_index=takeoff.waypoint_index + (route.waypoint_index if route else 0),
                          waypoint_count=len(course['waypoints']), target=command.target if command else None,
                          px4_armed=link.armed, px4_offboard=link.offboard, landed_state=link.landed_state,
                          navigation_source=navigation_source(args), vision_valid=bool(camera_fresh),
                          vision_age_sim_s=sim_s-valid_sim if valid_sim is not None else None,
                          vision_measurements_sent=sent, timesync_responses=link.timesync_responses,
                          sitl_power_health=power_health,
                          vision_reason=(frame or {}).get('reason'),
                          vision_quality=(frame or {}).get('quality'),
                          vision_backend=(frame or {}).get('source'),
                          vision_slam=(frame or {}).get('slam'),
                          vision_depth_normals=(frame or {}).get('depth_normals'),
                          vision_position_std_m=(frame or {}).get('position_std_m'),
                          px4_tilt_prior_status=(frame or {}).get('px4_tilt_prior_status'),
                          px4_tilt_pairing_status=(frame or {}).get('px4_tilt_pairing_status'),
                          px4_tilt_std_rad=(frame or {}).get('px4_tilt_std_rad'),
                          px4_tilt_capture_usec=(frame or {}).get('px4_tilt_capture_usec'),
                          px4_tilt_bracket_usec=(frame or {}).get('px4_tilt_bracket_usec'),
                          px4_tilt_disagreement_rad=(frame or {}).get('px4_tilt_disagreement_rad'),
                          depth_observability_without_tilt=(frame or {}).get('depth_observability_without_tilt'),
                          px4_gps_fusion=link.params.get('EKF2_GPS_CTRL'),
                          estimator_flags=getattr(link.estimator, 'flags', None),
                          policy_capture_usec=reply.capture_usec if reply else None,
                          policy_actions_received=learned.actions if learned else 0,
                          policy_latency_wall_s=reply.latency_s if reply else None,
                          policy_velocity_flu=reply.velocity_flu if reply else None,
                          policy_boundary_braking_interventions=braking_interventions,
                          policy_boundary_guard_setpoints=braking_setpoints,
                          policy_boundary_braking_faces=braking_step.faces if braking_step and landing.started is None else None,
                          policy_boundary_budget_deficits_m=dict(braking_step.stop_budget_deficits_m)
                          if braking_step and landing.started is None else None,
                          policy_unfiltered_sent_velocity=braking_step.unfiltered_velocity
                          if braking_step and landing.started is None else None,
                          policy_depth_coverage=learned.current_coverage if learned else None)
            time.sleep(.002)
    except KeyboardInterrupt:
        fault = 'operator_stopped'
    except (FlightFault, PolicyError) as error:
        fault = str(error)
    except Exception as error:
        fault = 'bridge_exception'
        log.event('Bridge exception', error=f'{type(error).__name__}: {error}')
    finally:
        if fault:
            log.event(f'Mission stopped: {fault}')
            if link is not None and arm_requested:
                try:
                    # Stop streaming Offboard setpoints. Never force-disarm.
                    link.command(link.m.MAV_CMD_NAV_LAND)
                    log.event('Requested PX4 Land after mission stop')
                    landed_after_fault = observe_abort_landing(link, vision, log, allow_vision_aid=not blackout)
                except Exception as error:
                    log.event('Land request could not be transmitted', error=str(error))
        log.finish(result='failed' if fault else 'completed', fault=fault,
                   elapsed_sim_s=max(0., sim_s-start) if start is not None else 0.,
                   route_elapsed_sim_s=route_time, route_completed=route_time is not None,
                   landed_and_disarmed=fault is None,
                   landed_after_fault=landed_after_fault,
                   arm_request_sent=arm_requested,
                   survey_startup_at_arm=survey_startup_at_arm,
                   landing_position_enu=position_enu if fault is None else None,
                   pad_radius_m=landing.pad_radius, aircraft_radius_m=landing.aircraft_radius,
                   waypoints_reached=takeoff.waypoint_index + (route.waypoint_index if route else 0),
                   waypoint_count=len(course['waypoints']), hover_settled_sim_s=settled_seconds,
                   vision_measurements_sent=sent,
                   policy_actions_received=learned.actions if learned else 0,
                   policy_setpoints_sent=policy_setpoints_sent,
                   policy_boundary_braking_interventions=braking_interventions,
                   policy_boundary_guard_setpoints=braking_setpoints,
                   policy_boundary_budget_max_deficit_m=max_stop_budget_deficit,
                   policy_training_camera_forward_m=client.camera_forward_m if client else None,
                   policy_training_heading_lookahead_m=client.heading_lookahead if client else None,
                   policy_service_session=client.session if client else None)
        if client is not None:
            client.close()
        if health is not None:
            health.close()
        if vision is not None:
            vision.close()
        try:
            log.file.close()
        except OSError:
            pass  # the failure is already recorded; never interrupt cleanup
    return 1 if fault or log.write_error else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', type=Path, default=ROOT/'mission/course-window-x500.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/markerfree/telemetry.jsonl')
    parser.add_argument('--task', choices=('hover', 'course', 'policy'), default='hover')
    parser.add_argument('--launch-position', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--launch-yaw-deg', type=float, required=True)
    parser.add_argument('--hover-summary', type=Path)
    parser.add_argument('--policy-dir', type=Path)
    parser.add_argument('--policy-sha256')
    parser.add_argument('--policy-depth-mode', choices=('virtual', 'raw'), default='raw',
                        help='virtual reprojects to the checkpoint camera; raw requires matching camera mounts')
    parser.add_argument('--policy-timeout', type=float, default=2.0)
    parser.add_argument('--no-policy-boundary-brake', action='store_true',
                        help='Explicit diagnostic opt-out of predictive planar braking for the learned policy')
    parser.add_argument('--speed-scale', type=float, default=1.0)
    parser.add_argument('--heading-lookahead', type=float, default=1.8,
                        help='Metres before a turn to begin looking into the next corridor; 0 disables')
    parser.add_argument('--localizer', choices=('survey', 'sparse'), default='survey',
                        help='survey matches known course geometry; sparse builds its own bounded RGBD map')
    parser.add_argument('--use-px4-tilt', action='store_true',
                        help='Optional capture-aligned inertial roll/pitch prior for weak depth geometry')
    parser.add_argument('--require-survey-startup', action='store_true',
                        help='SITL opt-in: require 3 s of continuous absolute survey tracking before arming')
    parser.add_argument('--survey-normals', choices=('central2', 'fals7'), default='central2',
                        help='Absolute depth survey source normals. FALS requires OpenCV contrib.')
    parser.add_argument('--startup-timeout', type=float, default=240.0)
    parser.add_argument('--max-sim-seconds', type=float, default=240.0)
    parser.add_argument('--blackout-after', type=float,
                        help='SITL fault injection: stop forwarding camera captures after this many flight seconds')
    parser.add_argument('--pad-radius', type=float, default=.762)
    parser.add_argument('--landing-timeout', type=float, default=30.0)
    args = parser.parse_args()
    if args.localizer == 'sparse' and (args.use_px4_tilt or args.require_survey_startup or args.survey_normals != 'central2'):
        parser.error('sparse SLAM does not use survey startup, normals or PX4 tilt')
    if args.task == 'policy' and (os.environ.get('SIM_AIRFRAME') or 'x500') != 'x500':
        parser.error('the learned checkpoint is only validated for the x500 surrogate')
    if not math.isfinite(args.launch_yaw_deg):
        parser.error('--launch-yaw-deg must be finite')
    if not math.isfinite(args.heading_lookahead) or args.heading_lookahead < 0:
        parser.error('--heading-lookahead must be finite and nonnegative')
    for option in ('policy_timeout', 'startup_timeout', 'max_sim_seconds', 'landing_timeout', 'blackout_after'):
        value = getattr(args, option)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error('--' + option.replace('_', '-') + ' must be finite and positive')
    if args.task != 'hover' and args.hover_summary is None:
        parser.error('course and policy tasks require --hover-summary')
    if args.task == 'policy' and (args.policy_dir is None or args.policy_sha256 is None):
        parser.error('policy task requires --policy-dir and --policy-sha256')
    if args.task != 'policy' and (args.policy_dir is not None or args.policy_sha256 is not None):
        parser.error('policy options require --task policy')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
