"""Fast analytic depth and velocity dynamics over proxies of the supplied CAD.

World is ENU, observations/actions use local heading FLU (+x forward, +y left,
+z up). The actor never receives the scene, world position, or waypoint index.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from mission.crossing import evaluate_window_crossing
from mission.heading import route_yaw

ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH = 16, 24
HFOV_DEG, MAX_DEPTH = 90.0, 4.0
DEFAULT_CAMERA_FORWARD_M = .08
DT, ACCEL = .16, .8
GOAL_RADIUS = .11
MAX_STEPS = 640
Z = np.array([0., 0., 1.])
I = np.eye(3)
WIDTH_SCALE = math.tan(math.radians(HFOV_DEG / 2))
PIXEL_X = (2 * (np.arange(WIDTH) + .5) / WIDTH - 1) * WIDTH_SCALE
PIXEL_Y = (2 * (np.arange(HEIGHT) + .5) / HEIGHT - 1) * WIDTH_SCALE * HEIGHT / WIDTH
PIXEL_Y, PIXEL_X = np.meshgrid(PIXEL_Y, PIXEL_X, indexing="ij")
RAYS = np.stack([np.ones_like(PIXEL_X), -PIXEL_X, -PIXEL_Y], axis=-1).reshape(-1, 3)
RAYS /= np.linalg.norm(RAYS, axis=-1, keepdims=True)


def yaw_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def camera_rotation(yaw, roll, pitch):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return yaw_rotation(yaw) @ np.array([[cp, sp * sr, sp * cr], [0, cr, -sr],
                                         [-sp, cp * sr, cp * cr]])


def clip_norm(vector, limit):
    return vector * np.minimum(1, limit / np.maximum(1e-8, np.linalg.norm(vector, axis=-1, keepdims=True)))


def advance(p, v, target, response, wind):
    a = clip_norm((target - v) / response, ACCEL) + wind
    v1 = v + DT * a
    return p + DT * (v + v1) * .5, v1, a


def slab_interval(origin, direction, half):
    """Broadcast ray/box interval; parallel rays outside a slab never hit."""
    parallel = np.abs(direction) < 1e-10
    invalid = np.any(parallel & (np.abs(origin) > half), axis=-1)
    d = np.where(parallel, 1, direction)
    t0, t1 = (-half - origin) / d, (half - origin) / d
    near = np.max(np.where(parallel, -np.inf, np.minimum(t0, t1)), axis=-1)
    far = np.min(np.where(parallel, np.inf, np.maximum(t0, t1)), axis=-1)
    return near, np.where(invalid, -np.inf, far)


@dataclass(frozen=True)
class Scene:
    centers: np.ndarray
    axes: np.ndarray
    half: np.ndarray
    window: dict
    radius: float
    camera_forward_m: float = DEFAULT_CAMERA_FORWARD_M

    def __post_init__(self):
        # Pack local rotations so one BLAS multiply projects the entire ray grid.
        object.__setattr__(self, "packed_axes", np.ascontiguousarray(self.axes.transpose(1, 0, 2).reshape(3, -1)))
        object.__setattr__(self, "offsets", np.einsum("bi,bij->bj", self.centers, self.axes))

    def local(self, points):
        return (points @ self.packed_axes).reshape(*points.shape[:-1], -1, 3) - self.offsets

    def clearance(self, points):
        points = np.asarray(points)
        q = np.abs(self.local(points)) - self.half
        signed = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
        return np.minimum(signed.min(axis=-1), points[..., 2]) - self.radius

    def hits(self, start, end):
        """Continuous sphere/box collision along the linear integration step."""
        if min(start[2], end[2]) <= self.radius:
            return True
        a = self.local(start)
        d = self.local(end) - a
        near, far = slab_interval(a, d, self.half + self.radius)
        for i in np.flatnonzero(np.maximum(0, near) <= np.minimum(1, far)):
            # Between the six slab crossings, squared distance to the box
            # is quadratic; minimize each piece rather than inflating corners.
            h = self.half[i]
            axis = np.abs(d[i]) > 1e-10
            breaks = np.sort(np.r_[0., 1., ((-h[axis] - a[i, axis]) / d[i, axis]).clip(0, 1),
                                   ((h[axis] - a[i, axis]) / d[i, axis]).clip(0, 1)])
            mid = a[i] + ((breaks[1:] + breaks[:-1]) / 2)[:, None] * d[i]
            outside = np.abs(mid) > h
            b = (a[i] - np.sign(mid) * h) * outside
            slope = d[i] * outside
            t = (-np.sum(b * slope, axis=1) / np.maximum(1e-20, np.sum(slope * slope, axis=1)))
            nearest = b + t.clip(breaks[:-1], breaks[1:])[:, None] * slope
            if np.any(np.sum(nearest * nearest, axis=1) <= self.radius ** 2):
                return True
        return False

    def depth(self, position, rotation):
        rays = RAYS @ rotation.T
        origin = position + self.camera_forward_m * rotation[:, 0]
        # Cull objects entirely out of depth range before slab intersection.
        keep = np.linalg.norm(self.centers - origin, axis=1) <= MAX_DEPTH + np.linalg.norm(self.half, axis=1)
        axes = self.axes[keep]
        projected = (rays @ axes.transpose(1, 0, 2).reshape(3, -1)).reshape(len(rays), -1, 3)
        near, far = slab_interval(self.local(origin)[keep], projected, self.half[keep])
        t = np.where(near >= 0, near, far)
        t = np.where((near <= far) & (t >= 0), t, MAX_DEPTH)
        nearest = np.minimum(t.min(axis=1, initial=MAX_DEPTH), MAX_DEPTH)
        ground = -origin[2] / np.minimum(rays[:, 2], -1e-10)
        return np.minimum(nearest, ground).reshape(HEIGHT, WIDTH).astype(np.float32)


class Course:
    """Read the route and collision proxies directly from versioned assets."""

    def __init__(self, camera_forward_m=DEFAULT_CAMERA_FORWARD_M, heading_lookahead_m=0.):
        if not 0 <= camera_forward_m < 1:
            raise ValueError("camera forward offset must be in [0, 1) metres")
        if not math.isfinite(heading_lookahead_m) or not (heading_lookahead_m == 0 or heading_lookahead_m > .3):
            raise ValueError("heading lookahead must be zero (disabled) or greater than 0.3 metres")
        self.camera_forward_m = camera_forward_m
        self.heading_lookahead_m = heading_lookahead_m
        self.route_path = ROOT / "mission/course-window-fast.json"
        self.mesh_dir = ROOT / "models/course_v3"
        self.groups_path = self.mesh_dir / "groups.json"
        plan = json.loads(self.route_path.read_text())
        self.groups = json.loads(self.groups_path.read_text())
        waypoints = plan["waypoints"]
        self.start = np.array(waypoints[0]["position"])
        self.goals = np.array([w["position"] for w in waypoints[1:]])
        self.speeds = np.array([w["speed"] for w in waypoints[1:]])
        self.phases = [w["phase"] for w in waypoints[1:]]
        self.window = plan["window"]
        self.radius = plan["footprint_radius"]
        self.axes = np.array([self.window["normal_toward_entry"], self.window["horizontal_axis"], Z]).T
        self.boxes = []
        for group in self.groups["groups"]:
            if not group["collidable"]:
                continue
            name = group["name"]
            lo, hi = np.array(group["bounds"]["min"]), np.array(group["bounds"]["max"])
            if name == "c2a_chicane_wall_window":
                self.wall_limits = self._vertices(group) @ self.axes
            elif name.startswith("tube_oval"):
                self.tube_limits = lo, hi
            elif name in ("c1_chicane_wall", "c2b_chicane_wall"):
                local = self._vertices(group) @ self.axes
                lo, hi = local.min(axis=0), local.max(axis=0)
                self.boxes.append((name, self.axes @ ((lo + hi) / 2), self.axes, (hi - lo) / 2))
            else:
                self.boxes.append((name, (lo + hi) / 2, I, (hi - lo) / 2))

    def _vertices(self, group):
        lines = (self.mesh_dir / group["mesh"]).read_text().splitlines()
        return np.array([[float(s) for s in line.split()[1:4]] for line in lines if line.startswith("v ")])

    def scene(self, rng, difficulty):
        centers, axes, halves = [], [], []

        def add(center, orientation, half):
            centers.append(center)
            axes.append(orientation)
            halves.append(half)

        for name, center, orientation, half in self.boxes:
            if not name.startswith(("net_", "tube_cradle")):
                jitter = rng.uniform(-.035, .035, 3) * difficulty
                jitter[2] = 0
                center = center + jitter
            add(center, orientation, half)

        # Cut out the measured opening from the actual oriented CAD wall. The
        # rest of the wall stays fixed as the opening moves in stress variants.
        local_center = np.array(self.window["center"]) @ self.axes
        local_center[1:] += rng.uniform(-.20, .20, 2) * difficulty
        width = self.window["width"] + rng.uniform(-.06, .08) * difficulty
        height = self.window["height"] + rng.uniform(-.06, .08) * difficulty
        left, right = local_center[1] + np.array([-width, width]) / 2
        bottom, top = local_center[2] + np.array([-height, height]) / 2
        _, l0, z0 = self.wall_limits.min(axis=0)
        _, l1, z1 = self.wall_limits.max(axis=0)
        for a, b, c, d in ((l0, left, z0, z1), (right, l1, z0, z1),
                           (left, right, z0, bottom), (left, right, top, z1)):
            add(self.axes @ [local_center[0], (a + b) / 2, (c + d) / 2], self.axes,
                np.array([self.window["thickness"], b - a, d - c]) / 2)

        # Polygonal shell, not a solid tunnel AABB. Extrude sixteen overlapping
        # oriented panels over the measured length, with assumed 25 mm shell.
        lo, hi = self.tube_limits
        center, half = (lo + hi) / 2, (hi - lo) / 2
        center[0] += rng.uniform(-.04, .04) * difficulty
        theta = np.linspace(0, 2 * np.pi, 17)
        ring = np.stack([half[0] * np.cos(theta), theta * 0, half[2] * np.sin(theta)], axis=1)
        for a, b in zip(ring, ring[1:]):
            along = b - a
            length = np.linalg.norm(along)
            along /= length
            y = I[:, 1]
            outward = np.cross(y, along)
            add(center + (a + b) / 2 - outward * .0125, np.stack([along, y, outward], axis=1),
                np.array([length / 2 + .008, half[1], .0125]))

        window = {**self.window, "center": (self.axes @ local_center).tolist(),
                  "width": width, "height": height}
        return Scene(np.array(centers), np.array(axes), np.array(halves), window, self.radius,
                     self.camera_forward_m)


@dataclass(frozen=True)
class Observation:
    depth: np.ndarray
    state: np.ndarray  # estimated body velocity (3), body goal (3), roll/pitch, speed cap


def direct_velocity(goal, speed):
    distance = np.linalg.norm(goal)
    return clip_norm(1.75 * goal, min(speed, math.sqrt(1.25 * distance)))


class Flight:
    """A deterministic episode given scenario seed and the action sequence."""

    def __init__(self, course, seed, difficulty, sensor_stress=1.):
        self.course = course
        self.rng = np.random.default_rng(seed)
        self.sensor_rng = np.random.default_rng([seed, 1])
        self.camera_rng = np.random.default_rng([seed, 2])
        self.scene = course.scene(self.rng, difficulty)
        self.difficulty = difficulty
        self.sensor_stress = sensor_stress
        self.p = course.start + self.rng.uniform(-.045, .045, 3) * difficulty
        self.v = np.zeros(3)
        self.yaw = math.pi / 2 + self.rng.uniform(-.18, .18) * difficulty
        self.pitch = self.roll = 0.
        self.index = self.truth_index = self.steps = 0
        self.bias = self.rng.uniform(-.033, .033, 3) * difficulty
        self.yaw_bias = self.rng.uniform(-.06, .06) * difficulty
        self.wind = self.rng.uniform(-.045, .045, 3) * difficulty
        self.response = self.rng.uniform(.28, .40)
        self.path = [self.p.copy()]
        self.min_clearance = float(self.scene.clearance(self.p))
        self.collision = False
        self.rotation = yaw_rotation(self.yaw)
        self.yaw_command = self.yaw

    @property
    def done(self):
        return self.collision or self.index == len(self.course.goals) or self.steps >= MAX_STEPS

    def observe(self, *, render=True):
        rng = self.sensor_rng
        noise = self.difficulty * self.sensor_stress
        pose_est = self.p + self.bias + rng.normal(0, .008, 3) * noise
        v_est = self.v + rng.normal(0, .023, 3) * noise
        self.rotation = yaw_rotation(self.yaw + self.yaw_bias)
        index = min(self.index, len(self.course.goals) - 1)
        target = self.course.goals[index]
        delta = target - pose_est
        if self.course.heading_lookahead_m:
            following = self.course.goals[index + 1] if index + 1 < len(self.course.goals) else None
            heading = route_yaw(pose_est, target, following, self.course.phases[index],
                                lookahead=self.course.heading_lookahead_m)
            if heading is not None:
                self.yaw_command = heading - self.yaw_bias
        elif np.linalg.norm(delta[:2]) > .08:
            self.yaw_command = math.atan2(delta[1], delta[0]) - self.yaw_bias
        state = np.array([*(v_est @ self.rotation), *(delta @ self.rotation),
                          (self.roll + rng.normal(0, .01) * noise) * 5,
                          (self.pitch + rng.normal(0, .01) * noise) * 5,
                          self.course.speeds[index]], dtype=np.float32)
        if render:
            rng = self.camera_rng
            depth = self.scene.depth(self.p, camera_rotation(self.yaw, self.roll, self.pitch))
            depth += rng.normal(0, .018 * noise, depth.shape).astype(np.float32)
            depth[rng.random(depth.shape) < .018 * noise] = MAX_DEPTH
            if noise > 1.25 and rng.random() < .09 * (noise - 1.25):
                y, x = rng.integers([HEIGHT - 3, WIDTH - 5])
                depth[y:y + 4, x:x + 6] = MAX_DEPTH
            depth = 1 - depth.clip(0, MAX_DEPTH) / MAX_DEPTH
        else:
            depth = np.zeros((HEIGHT, WIDTH), np.float32)
        return Observation(depth, state)

    def step(self, action):
        cap = self.course.speeds[self.index]
        old_p = self.p
        target = self.rotation @ clip_norm(np.asarray(action, dtype=float), cap)
        self.p, self.v, a = advance(self.p, self.v, target, self.response, self.wind)
        delta_yaw = math.remainder(self.yaw_command - self.yaw, 2 * math.pi)
        self.yaw += np.clip(delta_yaw, -1.6 * DT, 1.6 * DT)
        local_a = a @ yaw_rotation(self.yaw)
        self.roll, self.pitch = -local_a[1] / 9.81, local_a[0] / 9.81
        self.bias = np.clip(.997 * self.bias + self.rng.normal(0, .002, 3) * self.difficulty, -.07, .07)
        self.steps += 1
        self.path.append(self.p.copy())
        self.min_clearance = min(self.min_clearance, float(self.scene.clearance(self.p)))
        self.collision = self.scene.hits(old_p, self.p)
        goal = self.course.goals[self.index]
        if np.linalg.norm(goal - self.p - self.bias) <= GOAL_RADIUS:
            self.index += 1
        if self.truth_index < len(self.course.goals):
            if np.linalg.norm(self.course.goals[self.truth_index] - self.p) <= .20:
                self.truth_index += 1

    def result(self):
        passage = evaluate_window_crossing(self.path, self.scene.window, self.scene.radius)
        count = len(self.course.goals)
        return {
            "success": not self.collision and self.truth_index == count and self.index == count
                       and passage["full_envelope_crossed"],
            "collision": self.collision,
            "window_crossed": passage["full_envelope_crossed"],
            "window_margin_m": passage["min_aperture_clearance_m"],
            "route_progress": self.truth_index / count,
            "sim_seconds": round(self.steps * DT, 3),
            "sampled_min_clearance_m": self.min_clearance,
            "phase_at_end": self.course.phases[min(self.index, count - 1)],
        }


# Give the expert alternate lateral and vertical lines of travel, including
# hover/braking, without giving its obstacle query or truth state to the actor.
SIDE, UP = np.meshgrid([-1., -.5, 0., .5, 1.], [-1., -.5, 0., .5, 1.])
SIDE, UP = SIDE.ravel()[:, None], UP.ravel()[:, None]


def teacher(flight):
    """Evaluate candidate setpoints with a short privileged dynamics rollout."""
    target = flight.course.goals[flight.index].copy()
    if flight.course.phases[flight.index] == "WINDOW_CROSS":
        center = np.array(flight.scene.window["center"])
        normal = flight.course.axes[:, 0]
        distance_from_wall = float((flight.p - center) @ normal)
        shift = center - flight.course.window["center"]
        # Leave the nominal guidance unchanged elsewhere. The true aperture
        # center is available to the expert only, and aligns the flight while
        # the vehicle can still overlap the frame.
        weight = np.clip((distance_from_wall + .50) / .15, 0., 1.)
        target += weight * shift
    goal = target - flight.p
    cap = flight.course.speeds[flight.index]
    direct = direct_velocity(goal, cap)
    h = goal[:2] / max(.01, np.linalg.norm(goal[:2]))
    side = np.array([-h[1], h[0], 0.])
    offsets = SIDE * side + UP * Z
    candidates = np.concatenate([direct[None, :], np.zeros((1, 3)),
                                 .50 * direct + cap * .65 * offsets,
                                 .86 * direct + cap * .43 * offsets], axis=0)
    candidates = clip_norm(candidates, cap)
    p, v = flight.p, flight.v
    clearance = np.full(len(candidates), np.inf)
    for _ in range(5):
        p, v, _ = advance(p, v, candidates, flight.response, flight.wind)
        clearance = np.minimum(clearance, flight.scene.clearance(p))
    # Do not encourage needless detours for already safe commands. Near a
    # surface, trade off progress against a modest positive clearance reserve.
    progress_cost = np.linalg.norm(target - p, axis=1)
    deviation = np.linalg.norm(direct - candidates, axis=1)
    # Direct targets still matter close to waypoint capture, where many
    # candidates have essentially the same immediate progress.
    safety = 7.5 * np.maximum(.09 - clearance, 0) ** 2 + 18 * np.maximum(-clearance, 0)
    score = progress_cost + .23 * deviation + safety
    return candidates[np.argmin(score)] @ flight.rotation
