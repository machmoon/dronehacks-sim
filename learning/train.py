"""Train a synthetic-depth velocity policy by expert imitation on the CAD course."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import signal
import statistics
import time

BOOT_TIME = time.monotonic()

import numpy as np
import torch
from torch.nn import functional as F

from .policy import DepthPolicy
from .sim import (DEFAULT_CAMERA_FORWARD_M, DT, HFOV_DEG, HEIGHT, MAX_DEPTH, WIDTH,
                  Course, Flight, direct_velocity, teacher)
from .transfer import (CAMERA_FORWARD_M, DEFAULT_LOOKAHEAD_M, InfeasibleScenario, TransferCourse,
                       X500_SDF)
from .transfer_delay import DelayCourse, TRAIN_PROFILE as DELAY_PROFILE

# Seeds for training are drawn below 1e9; these two banks are disjoint from
# training and from each other. Only validation selects the best checkpoint.
VALIDATION = [(2_000_061_001, 0., 1.), (2_000_061_002, .65, 1.),
              (2_000_061_003, .85, 1.), (2_000_061_004, 1., 1.)]
HOLDOUT = [(2_000_071_000 + i, difficulty, stress) for i, (difficulty, stress) in
           enumerate([(0., 1.), (.50, 1.), (.65, 1.), (.75, 1.), (.80, 1.),
                      (.90, 1.), (1., 1.60), (1., 2.)])]
ACTORS = ("depth_policy", "same_policy_blank_depth", "blind_waypoint", "geometry_teacher")
METHOD = "DAgger-style supervised imitation of a geometry-aware velocity teacher"
DEFAULT_SEED = 41
TRANSFER_COURSES = {'x500-transfer': TransferCourse, DELAY_PROFILE: DelayCourse}


def case_banks(seed, warm_start):
    if seed == DEFAULT_SEED and not warm_start:
        return VALIDATION, HOLDOUT
    # Leave the legacy banks intact and dedicate sixteen seeds per run seed.
    base = 2_100_000_000 + 16 * seed
    validation = [(base + i, difficulty, stress)
                  for i, (_, difficulty, stress) in enumerate(VALIDATION)]
    holdout = [(base + 8 + i, difficulty, stress)
               for i, (_, difficulty, stress) in enumerate(HOLDOUT)]
    return validation, holdout


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_torch(path, data):
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def load_initial_weights(model, path):
    """Hash the exact bytes loaded, even if a source latest.pt is replaced."""
    path = path.resolve(strict=True)
    contents = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != 1
            or checkpoint.get("architecture") != "DepthPolicy" or not isinstance(checkpoint.get("model"), dict)):
        raise ValueError("expected a format-1 DepthPolicy checkpoint")
    model.load_state_dict(checkpoint["model"])
    return {
        "mode": "weights_only_warm_start",
        "checkpoint": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "source_steps": checkpoint.get("steps"),
        "source_camera_forward_m": checkpoint.get("camera_forward_m", DEFAULT_CAMERA_FORWARD_M),
        "source_heading_lookahead_m": checkpoint.get("heading_lookahead_m", 0.),
        "source_profile": checkpoint.get("profile", "legacy"),
        "source_route_sha256": checkpoint.get("route_sha256"),
    }


class Replay:
    """Fixed memory ring; stores 8-bit training images instead of video."""

    def __init__(self, capacity=60_000):
        self.depth = np.empty((capacity, 1, HEIGHT, WIDTH), np.uint8)
        self.state = np.empty((capacity, 9), np.float32)
        self.actions = np.empty((capacity, 3), np.float32)
        self.weight = np.empty(capacity, np.float32)
        self.count = 0

    def __len__(self):
        return min(self.count, len(self.depth))

    def add(self, obs, target, hard_phase):
        i = self.count % len(self.depth)
        self.depth[i, 0] = np.rint(obs.depth * 255).astype(np.uint8)
        self.state[i] = obs.state
        self.actions[i] = target
        nominal = direct_velocity(obs.state[3:6], obs.state[-1])
        residual_size = np.linalg.norm(nominal - target) / obs.state[-1]
        self.weight[i] = 1. + 3.0 * min(1., residual_size) + 1.5 * hard_phase
        self.count += 1

    def batch(self, rng, size, device):
        i = rng.integers(len(self), size=size)
        return (torch.as_tensor(self.depth[i], device=device).float().mul_(1 / 255),
                torch.as_tensor(self.state[i], device=device),
                torch.as_tensor(self.actions[i], device=device),
                torch.as_tensor(self.weight[i], device=device))


@torch.inference_mode()
def infer(model, obs):
    depth = torch.from_numpy(obs.depth)[None, None]
    state = torch.from_numpy(obs.state)[None]
    return model(depth, state)[0].numpy()


def cpu_model(model):
    clone = DepthPolicy().eval()
    clone.load_state_dict(model.state_dict())
    return clone


def make_flight(course, seed, difficulty, sensor_stress):
    if isinstance(course, TransferCourse):
        return course.make_flight(seed, difficulty, sensor_stress)
    return Flight(course, seed, difficulty, sensor_stress)


def instructor(flight):
    if isinstance(flight.course, TransferCourse):
        return flight.course.instructor(flight)
    return teacher(flight)


def evaluate(course, model, cases, stop):
    rows = {actor: [] for actor in ACTORS}
    response_deltas = []
    rejected_cases = []
    for seed, difficulty, sensor_stress in cases:
        if stop():
            return None
        case = dict(seed=seed, difficulty=difficulty, sensor_stress=sensor_stress)
        try:
            first_flight = make_flight(course, seed, difficulty, sensor_stress)
        except InfeasibleScenario as error:
            rejection = dict(**case, geometry_difficulty=error.geometry_difficulty, reason=error.detail)
            rejected_cases.append(rejection)
            for actor in ACTORS:
                rows[actor].append(dict(**case, success=False, collision=False, window_crossed=False,
                                        route_progress=0., sim_seconds=None,
                                        phase_at_end="infeasible_scenario", scenario_rejection=rejection))
            continue
        for index, actor in enumerate(ACTORS):
            flight = first_flight if index == 0 else make_flight(course, seed, difficulty, sensor_stress)
            while not flight.done:
                if stop():
                    return None
                obs = flight.observe(render=actor == "depth_policy")
                if actor == "geometry_teacher":
                    action = instructor(flight)
                elif actor == "blind_waypoint":
                    action = direct_velocity(obs.state[3:6], obs.state[-1])
                else:
                    action = infer(model, obs)
                    if actor == "depth_policy" and flight.steps % 8 == 0:
                        blank = type(obs)(np.zeros_like(obs.depth), obs.state)
                        response_deltas.append(float(np.linalg.norm(action - infer(model, blank))))
                flight.step(action)
            rows[actor].append(dict(**case, **flight.result()))
    means = {}
    for actor, results in rows.items():
        times = [r["sim_seconds"] for r in results if r["success"]]
        durations = [r["sim_seconds"] for r in results if r["sim_seconds"] is not None]
        means[actor] = dict(
            episodes=len(results), successes=len(times),
            evaluated_episodes=len(durations), infeasible_scenarios=len(results)-len(durations),
            collisions=sum(r["collision"] for r in results),
            windows_crossed=sum(r["window_crossed"] for r in results),
            mean_route_progress=statistics.mean(r["route_progress"] for r in results),
            median_success_sim_seconds=statistics.median(times) if times else None,
            median_episode_sim_seconds=statistics.median(durations) if durations else None,
        )
    return {"summary": means, "episodes": rows,
            "infeasible_scenarios": rejected_cases,
            "paired_mean_depth_vs_blank_action_delta_mps": statistics.mean(response_deltas) if response_deltas else None}


def score(evaluation):
    learned = evaluation["summary"]["depth_policy"]
    return (learned["successes"], -learned["collisions"], learned["windows_crossed"],
            learned["mean_route_progress"],
            -(learned["median_success_sim_seconds"] or 1000.))


class Experiment:
    def __init__(self, args, output):
        self.args, self.output = args, output
        self.start = BOOT_TIME
        self.started_at = timestamp()
        self.deadline = self.start + args.max_seconds
        self.signal = None
        self.steps = self.cycles = self.episodes = 0
        self.phase = "initializing"
        self.last_loss = self.latest_evaluation = self.best_validation = self.final_holdout = None
        self.holdout_state = "not_run"
        self.history = deque(maxlen=180)
        self.last_status = self.last_checkpoint = 0.
        self.infeasible_training_rejections = 0
        self.last_infeasible_training_case = None
        self.rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        self.profile = getattr(args, "profile", "legacy")
        if self.profile in TRANSFER_COURSES:
            self.course = TRANSFER_COURSES[self.profile](heading_lookahead_m=args.heading_lookahead)
            transfer_profile = self.course.profile_metadata()
        else:
            self.course = Course(args.camera_forward, args.heading_lookahead)
            transfer_profile = None
        self.heading_sha256 = sha256(self.course.route_path.parents[1] / 'mission/heading.py')
        self.profile_contract = {
            "profile": self.profile, "transfer_profile": transfer_profile,
            "route_sha256": transfer_profile["route_sha256"] if transfer_profile else sha256(self.course.route_path),
            "heading_sha256": self.heading_sha256,
        }
        self.validation_cases, self.holdout_cases = case_banks(args.seed, args.init_checkpoint is not None)
        self.device = torch.device(args.device)
        self.model = DepthPolicy()
        self.initialization = {"mode": "random_weights"}
        if args.init_checkpoint is not None:
            self.initialization = load_initial_weights(self.model, args.init_checkpoint)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=7e-4, weight_decay=1e-4)
        self.replay = Replay()

    def depth_spec(self):
        return {"height": HEIGHT, "width": WIDTH, "horizontal_fov_degrees": HFOV_DEG,
                "max_range_m": MAX_DEPTH, "encoding": "1 - range / max_range",
                "camera_forward_m": self.course.camera_forward_m,
                "assumptions": "forward pinhole ray cast; synthetic metric depth with noise and dropout"}

    def interrupted(self, *, cutoff=None):
        return self.signal is not None or time.monotonic() >= (self.deadline if cutoff is None else cutoff)

    def handle_signal(self, signal_num, _):
        self.signal = signal.Signals(signal_num).name

    def status(self, *, state="running", force=False, error=None):
        now = time.monotonic()
        if not force and now - self.last_status < 5:
            return
        atomic_json(self.output / "status.json", {
            **self.profile_contract,
            "state": state, "phase": self.phase, "started_at": self.started_at, "updated_at": timestamp(),
            "elapsed_seconds": round(now - self.start, 2), "max_seconds": self.args.max_seconds,
            "cycles": self.cycles, "steps": self.steps, "samples": self.replay.count,
            "episodes": self.episodes, "replay_size": len(self.replay), "device": self.device.type,
            "camera_forward_m": self.course.camera_forward_m, "initialization": self.initialization,
            "heading_lookahead_m": self.course.heading_lookahead_m,
            "infeasible_training_rejections": self.infeasible_training_rejections,
            "last_infeasible_training_case": self.last_infeasible_training_case,
            "last_loss": self.last_loss, "latest_evaluation": self.latest_evaluation,
            "best_validation": self.best_validation, "final_holdout": self.final_holdout,
            "holdout_state": self.holdout_state,
            "checkpoints": [name for name in ("latest.pt", "best.pt") if (self.output / name).exists()],
            "signal": self.signal, "error": error,
        })
        self.last_status = now

    def metadata(self):
        root = self.course.route_path.parents[1]
        sources = [
            self.course.route_path, self.course.groups_path, root / "assets/course-v3.zip", root / "mission/heading.py",
            *(self.course.mesh_dir / "meshes" / f"{name}.obj" for name in
              ("c1_chicane_wall", "c2b_chicane_wall", "c2a_chicane_wall_window")),
            *(Path(__file__).with_name(name) for name in ("sim.py", "policy.py", "train.py")),
        ]
        if self.profile in TRANSFER_COURSES:
            sources.extend([Path(__file__).with_name("transfer.py"), root/"mission/airframe.py", X500_SDF])
        if self.profile == DELAY_PROFILE:
            sources.append(Path(__file__).with_name("transfer_delay.py"))
        atomic_json(self.output / "metadata.json", {
            **self.profile_contract,
            "method": METHOD, "seed": self.args.seed, "max_seconds": self.args.max_seconds,
            "device": self.device.type, "torch_version": str(torch.__version__), "threads": self.args.threads,
            "initialization": self.initialization,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters()),
            "route": str(self.course.route_path.relative_to(root)),
            "cad_source_sha256": self.course.groups["source_sha256"],
            "source_sha256": {str(path.relative_to(root)): sha256(path) for path in sources},
            "train_endpoint": "hover over launch through final landing-approach waypoint; no takeoff or touchdown",
            "validation_cases": self.validation_cases, "held_out_cases": self.holdout_cases,
            "case_fields": ["seed", "state_difficulty_geometry_capped_at_0.25"
                            if self.profile in TRANSFER_COURSES else "geometry_and_state_difficulty",
                            "sensor_noise_multiplier"],
            "best_checkpoint_selection": "validation cases only; final held-out cases never select weights",
            "depth": self.depth_spec(),
            "heading_lookahead_m": self.course.heading_lookahead_m,
            "state": ["estimated heading-frame velocity m/s (xyz)", "estimated heading-frame goal vector m (xyz)",
                      "estimated roll and pitch in radians x5", "setpoint speed cap m/s"],
            "action": "three velocity setpoints in yaw-aligned FLU, metres/second; rotate to PX4 NED for transport",
            "sim_step_seconds": DT, "sphere_radius_m": self.course.radius if self.profile == "legacy" else None,
            "limitations": [
                "Approximate CAD obstacle boxes and hollow tube shell; opaque net proxy and box boulders.",
                "No AprilTags, RGB texture, real stereo pipeline, VIO, Gazebo, or PX4 in training.",
                "Estimator noise, bias and wind are simulated; true pose is used only by simulator, teacher, and scorer.",
                "Velocity response is a simple bounded-acceleration model. No thrust, motor or flight controller training.",
                "Completion and collision are scored only in this analytic simulator. No hardware-safety claim.",
            ],
        })

    def checkpoint(self, name):
        atomic_torch(self.output / name, {
            **self.profile_contract,
            "format": 1, "architecture": "DepthPolicy", "model": self.model.state_dict(),
            "camera_forward_m": self.course.camera_forward_m, "depth": self.depth_spec(), "state_size": 9,
            "heading_lookahead_m": self.course.heading_lookahead_m,
            "initialization": self.initialization,
            "selection_split": "validation" if name == "best.pt" else "latest_training_state",
            "steps": self.steps, "samples": self.replay.count, "cycles": self.cycles,
            "elapsed_seconds": time.monotonic() - self.start,
        })
        if name == "latest.pt":
            self.last_checkpoint = time.monotonic()

    def collect(self, cutoff):
        self.phase = "collecting"
        self.status(force=True)
        actor = cpu_model(self.model) if self.steps or self.args.init_checkpoint is not None else None
        for episode in range(6):
            for _ in range(8):
                if self.interrupted(cutoff=cutoff):
                    return
                seed = int(self.rng.integers(1, 1_000_000_000))
                level = .55 + .45 * min(1., self.cycles / 24)
                difficulty = 0. if episode == 0 else float(self.rng.uniform(.30, level))
                stress = 1.
                if episode >= 3 and self.cycles >= 24:
                    # Retain nominal cases while increasing noise and dropout.
                    stress += float(self.rng.uniform()) * min(1., (self.cycles - 24) / 36)
                try:
                    flight = make_flight(self.course, seed, difficulty, stress)
                except InfeasibleScenario as error:
                    self.infeasible_training_rejections += 1
                    self.last_infeasible_training_case = dict(seed=seed, difficulty=difficulty,
                                                             geometry_difficulty=error.geometry_difficulty,
                                                             reason=error.detail)
                    self.status()
                else:
                    break
            else:
                self.status(force=True)
                continue
            if episode in (4, 5):
                # Additional closed-loop experience near the measured aperture
                # without spending every expert query in long straightaways.
                if isinstance(self.course, TransferCourse):
                    flight.start_at_window_approach()
                else:
                    index = self.course.phases.index("WINDOW_APPROACH")
                    flight.index = flight.truth_index = index
                    flight.p = self.course.goals[index - 1].copy()
                    flight.path = [flight.p.copy()]
                    flight.min_clearance = float(flight.scene.clearance(flight.p))
                    delta = self.course.goals[index] - flight.p
                    flight.yaw = float(np.arctan2(delta[1], delta[0]))
            while not flight.done:
                if self.interrupted(cutoff=cutoff):
                    return
                obs = flight.observe()
                label = instructor(flight)
                is_hard = self.course.phases[flight.index] in ("WINDOW_CROSS", "LANDING_APPROACH")
                self.replay.add(obs, label, is_hard)
                # Anneal the expert fraction while retaining regular full
                # demonstrations, then label states encountered by the policy.
                if actor is not None and episode >= 2 and self.rng.random() > max(.10, .55 - .025 * self.cycles):
                    action = infer(actor, obs)
                else:
                    action = label
                if episode > 0:
                    action = action + self.rng.normal(0, .020 * difficulty, size=3)
                flight.step(action)
            self.episodes += 1
            self.status()

    def optimize(self, cutoff):
        self.phase = "optimizing"
        self.status(force=True)
        losses = []
        for group in self.optimizer.param_groups:
            group["lr"] = max(1e-4, 7e-4 / math.sqrt(1 + self.cycles / 32))
        for update in range(280):
            if self.interrupted(cutoff=cutoff):
                break
            depth, state, label, weight = self.replay.batch(self.rng, 128, self.device)
            action = self.model(depth, state)
            # Normalizing by speed cap keeps precise low-speed window commands
            # from being drowned out by nominal high-speed corridor examples.
            normalized = (action - label) / state[:, -1:]
            loss = (F.smooth_l1_loss(normalized, torch.zeros_like(normalized),
                                     reduction="none", beta=.1).mean(dim=1) * weight).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.)
            self.optimizer.step()
            if update % 20 == 0:
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(loss_value):
                    raise FloatingPointError("non-finite imitation loss")
                losses.append(loss_value)
                self.status()
            self.steps += 1
        if losses:
            self.last_loss = statistics.mean(losses)

    def save_metrics(self):
        atomic_json(self.output / "metrics.json", {
            "method": METHOD, "history": list(self.history),
            "best_validation": self.best_validation, "final_holdout": self.final_holdout,
            "holdout_state": self.holdout_state,
        })

    def run(self):
        self.metadata()
        print(f"Training on the marker-free analytic CAD depth simulator ({self.device.type}); "
              f"{METHOD}. Budget: {self.args.max_seconds:g} wall seconds. "
              f"Camera forward offset: {self.course.camera_forward_m:g} m. "
              f"Heading lookahead: {self.course.heading_lookahead_m:g} m. "
              f"Profile: {self.profile}. Initialization: {self.initialization['mode']}.", flush=True)
        self.status(force=True)
        last_validation = self.start - self.args.eval_interval
        # On a five-hour run, reserve a small tail for separately seeded
        # post-selection evaluation. Short diagnostic runs still train first.
        reserve = min(200. if self.profile in TRANSFER_COURSES else 22., self.args.max_seconds * .30)
        training_cutoff = self.deadline - reserve
        while not self.interrupted(cutoff=training_cutoff):
            self.collect(training_cutoff)
            if len(self.replay) < 128 or self.interrupted(cutoff=training_cutoff):
                break
            self.optimize(training_cutoff)
            self.cycles += 1
            now = time.monotonic()
            validate = now - last_validation >= self.args.eval_interval
            if validate or now - self.last_checkpoint >= 80:
                self.checkpoint("latest.pt")
            if validate and not self.interrupted(cutoff=training_cutoff):
                self.phase = "validating"
                self.status(force=True)
                evaluation = evaluate(self.course, cpu_model(self.model), self.validation_cases,
                                      lambda: self.interrupted(cutoff=training_cutoff))
                if evaluation is not None:
                    self.latest_evaluation = evaluation["summary"]
                    evaluated = evaluation["summary"]["depth_policy"]["evaluated_episodes"]
                    if evaluated and (self.best_validation is None
                                      or score(evaluation) > score(self.best_validation["evaluation"])):
                        self.best_validation = {"steps": self.steps, "evaluation": evaluation}
                        self.checkpoint("best.pt")
                    self.history.append({"steps": self.steps, "samples": self.replay.count,
                                         "elapsed_seconds": round(time.monotonic() - self.start, 2),
                                         "loss": self.last_loss, "validation": evaluation["summary"],
                                         "infeasible_scenarios": evaluation["infeasible_scenarios"],
                                         "vision_action_delta_mps": evaluation["paired_mean_depth_vs_blank_action_delta_mps"]})
                    self.save_metrics()
                    learned = evaluation["summary"]["depth_policy"]
                    print(f"step={self.steps} samples={self.replay.count} loss={self.last_loss:.5f} "
                          f"validation={learned['successes']}/{learned['episodes']} "
                          f"collisions={learned['collisions']}", flush=True)
                last_validation = time.monotonic()
        if self.steps:
            self.checkpoint("latest.pt")
        if self.steps and self.best_validation is not None and not self.interrupted():
            self.phase = "held_out_evaluation"
            self.holdout_state = "in_progress"
            self.status(force=True)
            best = DepthPolicy().eval()
            best.load_state_dict(torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)["model"])
            self.final_holdout = evaluate(self.course, best, self.holdout_cases, self.interrupted)
            self.holdout_state = "complete" if self.final_holdout is not None else "interrupted"
            if self.final_holdout:
                print("Held-out summary:", json.dumps(self.final_holdout["summary"]), flush=True)
            else:
                print("Held-out evaluation was interrupted; no partial score reported.", flush=True)
        self.save_metrics()
        self.phase = "finished"
        self.status(state="stopped" if self.signal else "completed", force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new or empty result directory")
    parser.add_argument("--max-seconds", type=float, default=18_000)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--seed", type=int, help="run seed (default 41, or 42 with a warm start)")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="load compatible model weights into a fresh run with a new optimizer and replay")
    parser.add_argument("--profile", choices=("legacy", *TRANSFER_COURSES), default="legacy")
    parser.add_argument("--camera-forward", type=float,
                        help="forward camera offset in metres (legacy: .08, x500 profiles: .34)")
    parser.add_argument("--heading-lookahead", type=float,
                        help="turn toward next leg within this distance (legacy: 0, x500 profiles: 1.8)")
    parser.add_argument("--eval-interval", type=float, default=120, help="seconds between validation rounds")
    args = parser.parse_args()
    if args.camera_forward is None:
        args.camera_forward = CAMERA_FORWARD_M if args.profile in TRANSFER_COURSES else DEFAULT_CAMERA_FORWARD_M
    if args.heading_lookahead is None:
        args.heading_lookahead = DEFAULT_LOOKAHEAD_M if args.profile in TRANSFER_COURSES else 0.
    if args.seed is None:
        args.seed = DEFAULT_SEED + int(args.init_checkpoint is not None)
    if (not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 18_000
            or not np.isfinite(args.eval_interval) or args.eval_interval <= 0
            or not 1 <= args.threads <= 8 or not 0 <= args.seed <= 2**32 - 1):
        parser.error("require 0 < max-seconds <= 18000, positive eval-interval, 1–8 threads and a uint32 seed")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable; use --device cpu")
    if not 0 <= args.camera_forward < 1:
        parser.error("--camera-forward must be in [0, 1) metres")
    if args.profile in TRANSFER_COURSES and args.camera_forward != CAMERA_FORWARD_M:
        parser.error("x500 profiles use the fixed .34 m camera mount")
    if not math.isfinite(args.heading_lookahead) or not (args.heading_lookahead == 0 or args.heading_lookahead > .3):
        parser.error("--heading-lookahead must be zero (disabled) or greater than 0.3 metres")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    experiment = Experiment(args, args.output)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, experiment.handle_signal)
    try:
        experiment.run()
    except Exception as exc:
        experiment.status(state="failed", force=True, error=str(exc))
        raise


if __name__ == "__main__":
    main()
