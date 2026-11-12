"""Run a pinned DepthPolicy on the Mac via a local, file-based SITL exchange."""
import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from .sim import HEIGHT, HFOV_DEG, MAX_DEPTH, RAYS, WIDTH, yaw_rotation

VERSION = 2
CAMERA_WIDTH, CAMERA_HEIGHT = 384, 256
TRAINING_CAMERA_FORWARD_M = .08
_NATIVE_X = (np.arange(CAMERA_WIDTH)+.5-CAMERA_WIDTH/2)/(CAMERA_WIDTH/2)
_NATIVE_Y = (np.arange(CAMERA_HEIGHT)+.5-CAMERA_HEIGHT/2)/(CAMERA_WIDTH/2)
_NATIVE_RADIAL2 = 1 + _NATIVE_X[None, :]**2 + _NATIVE_Y[:, None]**2
_LOG = logging.getLogger(__name__)


class PolicyError(RuntimeError):
    pass


class PolicyExchangeReadError(PolicyError):
    pass


@dataclass(frozen=True)
class DepthObservation:
    depth: np.ndarray
    coverage: float
    center_coverage: float


def policy_depth(depth_z_m, camera_shift_m=0.0, near_clip_m=.02):
    """Convert optical Z into the trainer's 24x16 forward ray-range image.

    The grids share the 90-degree horizontal FOV and aspect ratio. Each
    training ray falls between four pixels of the 16x finer sensor grid. Use
    the closest of those returns rather than interpolate across object edges.
    If the camera is mounted farther forward, project measured 3D points into
    the virtual training camera and depth-buffer them. The rear viewpoint can
    see rays the sensor cannot; mask these as near hazards after coverage gates.
    A far-clip return (+inf) encodes clear range; a near-clip return (-inf)
    encodes a nearby surface. Missing/corrupt measurements fail explicitly.
    """
    z = np.asarray(depth_z_m)
    if z.shape != (CAMERA_HEIGHT, CAMERA_WIDTH) or z.dtype.kind != 'f':
        raise PolicyError('invalid_policy_depth_shape_or_format')
    finite = np.isfinite(z)
    if np.any(np.isnan(z) | (finite & (z <= 0))) or not finite.any():
        raise PolicyError('invalid_or_empty_policy_depth')
    if not 0 <= camera_shift_m < 1 or not 0 < near_clip_m < .2:
        raise PolicyError('unsupported_policy_camera_extrinsics')
    if camera_shift_m < 1e-6:
        rows = np.arange(HEIGHT) * (CAMERA_HEIGHT // HEIGHT) + CAMERA_HEIGHT // HEIGHT // 2 - 1
        cols = np.arange(WIDTH) * (CAMERA_WIDTH // WIDTH) + CAMERA_WIDTH // WIDTH // 2 - 1
        samples = np.minimum.reduce([z[np.ix_(rows+i, cols+j)] for i in (0, 1) for j in (0, 1)])
        coverage = center_coverage = 1.0
    else:
        # For far-clip rays use their intersection with the virtual 4 m range
        # sphere. The sensor itself saw no surface up to that range.
        dz = camera_shift_m
        far = (np.sqrt(dz**2+_NATIVE_RADIAL2*(MAX_DEPTH**2-dz**2))-dz)/_NATIVE_RADIAL2
        native_z = np.where(np.isposinf(z), far, np.where(np.isneginf(z), near_clip_m, z))
        virtual_z = native_z + dz
        scale = native_z / virtual_z
        u = np.floor((1 + _NATIVE_X[None, :]*scale) * WIDTH/2).astype(np.intp)
        v = np.floor(HEIGHT/2 + _NATIVE_Y[:, None]*scale*WIDTH/2).astype(np.intp)
        inside = (0 <= u) & (u < WIDTH) & (0 <= v) & (v < HEIGHT)
        # NaN would conceal a sensor error, and +inf is a valid far return, so
        # a separate sentinel distinguishes virtual rays without observations.
        buffer = np.full((HEIGHT, WIDTH), np.inf)
        vi, ui = v[inside], u[inside]
        far_at_bin = MAX_DEPTH * RAYS[:, 0].reshape(HEIGHT, WIDTH)[vi, ui]
        projected_z = np.where(np.isposinf(z[inside]), far_at_bin, virtual_z[inside])
        np.minimum.at(buffer, (vi, ui), projected_z)
        observed = np.isfinite(buffer)
        coverage = float(np.mean(observed))
        center_coverage = float(np.mean(observed[HEIGHT//4:3*HEIGHT//4, WIDTH//4:3*WIDTH//4]))
        if coverage < .70 or center_coverage < .90:
            raise PolicyError(f'insufficient_policy_depth_coverage:{coverage:.3f}:{center_coverage:.3f}')
        samples = np.where(observed, buffer, 0.)
    # Optical Z is along FLU forward. The forward component of a unit ray
    # is its cosine to this axis, so divide by it to recover Euclidean range.
    ranges = samples / RAYS[:, 0].reshape(HEIGHT, WIDTH)
    return DepthObservation((1 - ranges.clip(0, MAX_DEPTH) / MAX_DEPTH).astype(np.float32),
                            coverage, center_coverage)


def policy_state(velocity_enu, goal_delta_enu, roll_frd, pitch_frd, yaw_ned, speed_cap):
    """Map PX4 FRD/NED attitude and ENU vectors into the training state."""
    yaw_enu = math.pi / 2 - yaw_ned
    rotation = yaw_rotation(yaw_enu)
    state = np.array([*(np.asarray(velocity_enu) @ rotation),
                      *(np.asarray(goal_delta_enu) @ rotation),
                      roll_frd*5, -pitch_frd*5, speed_cap], dtype=np.float32)
    if state.shape != (9,) or not np.isfinite(state).all() or not 0 < speed_cap <= 2:
        raise PolicyError('invalid_policy_state')
    return state, yaw_enu


def action_to_enu(action, yaw_enu):
    return tuple(np.asarray(action, dtype=float) @ yaw_rotation(yaw_enu).T)


def write_json(path, data):
    """Flush complete contents before publishing a name across the shared mount."""
    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        encoded = json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n'
        with temp.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path):
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise PolicyExchangeReadError(
            f'invalid_policy_exchange_record:{path.name}:{type(error).__name__}: {error}') from error
    if not isinstance(row, dict):
        raise PolicyExchangeReadError(f'invalid_policy_exchange_record:{path.name}:expected object, got {type(row).__name__}')
    return row


def response_name(reply_id):
    return f'response-{reply_id}.json'


def take_lock(path):
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise PolicyError(f'policy_exchange_already_in_use:{path.name}') from error
    return handle


def validate_observation(depth, state):
    depth, state = np.asarray(depth, dtype=np.float32), np.asarray(state, dtype=np.float32)
    if (depth.shape != (HEIGHT, WIDTH) or state.shape != (9,)
            or not np.isfinite(depth).all() or not np.isfinite(state).all()
            or np.any((depth < 0) | (depth > 1)) or not 0 < state[-1] <= 2):
        raise PolicyError('invalid_policy_observation')
    return depth, state


def validate_velocity(velocity, cap):
    try:
        v = np.asarray(velocity, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise PolicyError('invalid_policy_velocity') from error
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) > cap + 1e-5:
        raise PolicyError('invalid_policy_velocity')
    return tuple(map(float, v))


def validate_camera_forward(value):
    if type(value) not in (int, float) or not 0 < value < 1:
        raise PolicyError('invalid_checkpoint_camera_forward_m')
    return float(value)


def checkpoint_camera_forward(payload):
    # Checkpoints before mount-specific training used .08 m and omitted the
    # observation description. Explicit metadata must match the running code.
    forward = validate_camera_forward(payload.get('camera_forward_m', TRAINING_CAMERA_FORWARD_M))
    spec = payload.get('depth')
    expected = dict(height=HEIGHT, width=WIDTH, horizontal_fov_degrees=HFOV_DEG,
                    max_range_m=MAX_DEPTH, encoding='1 - range / max_range', camera_forward_m=forward)
    if (spec is not None and (not isinstance(spec, dict) or any(spec.get(k) != v for k, v in expected.items()))
            or payload.get('state_size', 9) != 9):
        raise PolicyError('checkpoint_observation_contract_mismatch')
    return forward


def validate_heading_lookahead(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise PolicyError('invalid_checkpoint_heading_lookahead')
    return value


def validate_transfer_contract(contract, course_path, heading_path, model_path, *, speed_scale, lookahead):
    """Reject a stale route/model contract before allowing an x500 policy flight."""
    kind = contract.get('profile')
    if kind not in ('x500-transfer', 'x500-transfer-delay'):
        raise PolicyError('policy_checkpoint_transfer_profile_required')
    delayed = kind == 'x500-transfer-delay'
    expected_name = 'x500-transfer-v2' if delayed else 'x500-transfer-v1'
    profile = contract.get('transfer_profile')
    if not isinstance(profile, dict) or profile.get('name') != expected_name:
        raise PolicyError('policy_checkpoint_transfer_profile_invalid')
    for label, path, expected in (
            ('route', course_path, contract.get('route_sha256')),
            ('heading', heading_path, contract.get('heading_sha256')),
            ('airframe', model_path, profile.get('x500_sdf_sha256'))):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise PolicyError('policy_checkpoint_'+label+'_mismatch')
    if (profile.get('route_sha256') != contract.get('route_sha256')
            or profile.get('speed_scale') != speed_scale
            or profile.get('heading_lookahead_m') != lookahead
            or profile.get('velocity_setpoint_slew_mps2') != .8
            or profile.get('actor_step_s') != .16):
        raise PolicyError('policy_checkpoint_control_profile_mismatch')
    if delayed:
        dynamics = dict(command_delay_s=[.12, .24], command_delay_quantum_s=.04,
                        command_delay_location='post_slew', velocity_gain_xy=[1., 1.08],
                        velocity_response_s=[.50, .80], plant_acceleration_mps2=.8,
                        plant_substep_s=.04, substeps_per_action=4, yaw_rate_rad_s=1.6)
        if any(profile.get(key) != value for key, value in dynamics.items()):
            raise PolicyError('policy_checkpoint_delay_dynamics_mismatch')
        for source, key in (('transfer.py', 'transfer_source_sha256'),
                            ('transfer_delay.py', 'transfer_delay_source_sha256')):
            try:
                actual = hashlib.sha256(Path(__file__).with_name(source).read_bytes()).hexdigest()
            except OSError as error:
                raise PolicyError('policy_checkpoint_transfer_source_unavailable') from error
            if actual != profile.get(key):
                raise PolicyError('policy_checkpoint_transfer_source_mismatch')
    plan = json.loads(Path(course_path).read_text())
    radii = [w.get('arrival_radius', plan['limits']['arrival_radius']) for w in plan['waypoints'][1:]]
    if profile.get('estimated_waypoint_arrival_radii_m') != radii:
        raise PolicyError('policy_checkpoint_waypoint_tolerance_mismatch')
    return profile


@dataclass(frozen=True)
class Reply:
    capture_usec: int
    velocity_flu: tuple
    latency_s: float


class PolicyClient:
    """One pending request at a time, bound to one model and service session."""

    def __init__(self, exchange, checkpoint_sha256, run_id, timeout_s=2.0):
        self.exchange = Path(exchange)
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        if len(self.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.checkpoint_sha256):
            raise ValueError('checkpoint SHA256 must be 64 hexadecimal characters')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('policy timeout must be finite and positive')
        self.exchange.mkdir(parents=True, exist_ok=True)
        self.lock = take_lock(self.exchange/'client.lock')
        self.timeout_s = timeout_s
        self.run_id = run_id
        self.session = None
        self.camera_forward_m = None
        self.heading_lookahead = None
        self.navigation_contract = None
        self.contract_verified = False
        self.pending = None
        self.seq = 0
        self.last_capture = -1
        self.transport_read_errors = 0
        self.last_transport_error = None
        self._warn_after = 0.

    def _read(self, name):
        try:
            return read_json(self.exchange/name)
        except PolicyExchangeReadError as error:
            self.transport_read_errors += 1
            self.last_transport_error = str(error)
            now = time.monotonic()
            if now >= self._warn_after:
                _LOG.warning('Retrying policy exchange read: %s', error)
                self._warn_after = now+1.
            return None

    def connect(self):
        if self.session is not None:
            return True
        service = self._read('service.json')
        if service is None:
            return False
        if (service.get('version') != VERSION or service.get('checkpoint_sha256') != self.checkpoint_sha256
                or not isinstance(service.get('session'), str) or not service['session']):
            raise PolicyError('policy_service_or_checkpoint_mismatch')
        self.camera_forward_m = validate_camera_forward(service.get('camera_forward_m'))
        self.heading_lookahead = validate_heading_lookahead(service.get('heading_lookahead_m'))
        self.navigation_contract = {key: service.get(key) for key in
                                    ('profile', 'route_sha256', 'heading_sha256', 'transfer_profile')}
        self.session = service['session']
        return True

    def request(self, depth, state, capture_usec, now):
        if self.session is None or self.pending is not None:
            raise PolicyError('policy_client_not_ready_for_request')
        if type(capture_usec) is not int or capture_usec <= self.last_capture:
            raise PolicyError('nonmonotonic_policy_depth_stamp')
        depth, state = validate_observation(depth, state)
        self.seq += 1
        key = dict(version=VERSION, session=self.session, checkpoint_sha256=self.checkpoint_sha256,
                   run_id=self.run_id, seq=self.seq, capture_usec=capture_usec, reply_id=uuid4().hex)
        write_json(self.exchange/'request.json', dict(**key, depth=depth.tolist(), state=state.tolist()))
        self.last_transport_error = None
        self.pending = key, now, float(state[-1])
        self.last_capture = capture_usec

    def poll(self, now):
        if self.pending is None:
            return None
        key, sent, cap = self.pending
        if now - sent > self.timeout_s:
            detail = f':{self.last_transport_error}' if self.last_transport_error else ''
            raise PolicyError(f'policy_response_timeout{detail}')
        row = self._read(response_name(key['reply_id']))
        if row is None or any(row.get(k) != v for k, v in key.items()):
            return None
        self.pending = None
        if row.get('error') is not None:
            raise PolicyError(f"policy_inference_failed:{row['error']}")
        return Reply(key['capture_usec'], validate_velocity(row.get('velocity_flu'), cap), now-sent)

    def cancel(self):
        self.pending = None

    def close(self):
        self.lock.close()


def serve(checkpoint, exchange):
    # Keep torch out of the container process entirely.
    import torch
    from .policy import DepthPolicy

    exchange.mkdir(parents=True, exist_ok=True)
    with take_lock(exchange/'service.lock'):
        torch.set_num_threads(1)
        weights = checkpoint.read_bytes()
        sha256 = hashlib.sha256(weights).hexdigest()
        payload = torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True)
        if (payload.get('format') != 1 or payload.get('architecture') != 'DepthPolicy'
                or payload.get('selection_split') != 'validation'):
            raise PolicyError('expected a validation-selected DepthPolicy best.pt snapshot')
        camera_forward_m = checkpoint_camera_forward(payload)
        heading_lookahead = validate_heading_lookahead(payload.get('heading_lookahead_m'))
        model = DepthPolicy().eval()
        model.load_state_dict(payload['model'], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise PolicyError('nonfinite_policy_parameters')

        def infer(depth, state):
            with torch.inference_mode():
                velocity = model(torch.from_numpy(depth)[None, None], torch.from_numpy(state)[None])[0].numpy()
            return validate_velocity(velocity, float(state[-1]))

        infer(np.zeros((HEIGHT, WIDTH), np.float32), np.r_[np.zeros(8, np.float32), np.float32(.4)])
        fixed = dict(version=VERSION, session=uuid4().hex, checkpoint_sha256=sha256)
        ready = exchange/'service.json'
        write_json(ready, dict(**fixed, checkpoint=checkpoint.name, training_steps=payload.get('steps'),
                               camera_forward_m=camera_forward_m, heading_lookahead_m=heading_lookahead,
                               **{key: payload.get(key) for key in
                                  ('profile', 'route_sha256', 'heading_sha256', 'transfer_profile')}))
        print(f'DepthPolicy ready on CPU, SHA256 {sha256}', flush=True)
        previous = None
        warn_after = 0.
        try:
            while True:
                try:
                    row = read_json(exchange/'request.json')
                except PolicyExchangeReadError as error:
                    now = time.monotonic()
                    if now >= warn_after:
                        _LOG.warning('Ignoring unreadable policy request: %s', error)
                        warn_after = now+1.
                    row = None
                reply_id = row.get('reply_id') if row is not None else None
                if (row is not None and all(row.get(k) == v for k, v in fixed.items())
                        and isinstance(row.get('run_id'), str) and type(row.get('seq')) is int
                        and type(row.get('capture_usec')) is int and isinstance(reply_id, str)
                        and len(reply_id) == 32 and all(c in '0123456789abcdef' for c in reply_id)):
                    key = dict(**fixed, **{k: row[k] for k in ('run_id', 'seq', 'capture_usec', 'reply_id')})
                    if key != previous:
                        if previous is not None:
                            try:
                                (exchange/response_name(previous['reply_id'])).unlink(missing_ok=True)
                            except OSError as error:
                                _LOG.warning('Could not remove superseded policy response: %s', error)
                        previous = key
                        try:
                            depth, state = validate_observation(row.get('depth'), row.get('state'))
                            response = dict(**key, velocity_flu=infer(depth, state))
                        except (TypeError, ValueError, PolicyError, RuntimeError) as error:
                            response = dict(**key, error=str(error))
                        # Never reuse a response name. VirtioFS can briefly
                        # expose the size of the prior inode after replacement.
                        write_json(exchange/response_name(reply_id), response)
                time.sleep(.008)
        except KeyboardInterrupt:
            return
        finally:
            ready.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--exchange', type=Path, required=True)
    args = parser.parse_args()
    serve(args.checkpoint, args.exchange)


if __name__ == '__main__':
    main()
