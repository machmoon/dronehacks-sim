import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from learning.sim import Course, Flight, RAYS, Scene, WIDTH, HEIGHT
from mission.heading import route_yaw


class CameraMountTests(unittest.TestCase):
    def test_forward_offset_moves_camera_along_body_forward(self):
        wall = np.array([[1.5, 0., 1.]])
        orientation = np.eye(3)
        half_extent = np.array([[.05, .8, .8]])
        camera_base = np.array([0., 0., 1.])
        old = Scene(wall, orientation[None], half_extent, {}, .28)
        new = Scene(wall, orientation[None], half_extent, {}, .28, .34)
        center = (HEIGHT // 2, WIDTH // 2)
        delta = old.depth(camera_base, orientation)[center] - new.depth(camera_base, orientation)[center]
        ray_forward = RAYS.reshape(HEIGHT, WIDTH, 3)[center][0]
        self.assertAlmostEqual(delta * ray_forward, .26, places=6)

    def test_course_observation_uses_camera_mount_without_changing_state(self):
        old = Flight(Course(), 17, 0.).observe()
        new = Flight(Course(.34), 17, 0.).observe()
        np.testing.assert_array_equal(old.state, new.state)
        self.assertGreater(np.max(np.abs(old.depth - new.depth)), .04)

    def test_heading_preview_uses_estimated_position_without_changing_local_goal(self):
        old = Flight(Course(.34), 17, 0.)
        new = Flight(Course(.34, 1.2), 17, 0.)
        for phase in ("FIRST_CORRIDOR", "WINDOW_CROSS"):
            with self.subTest(phase=phase):
                index = new.course.phases.index(phase)
                target, following = new.course.goals[index:index + 2]
                for flight in (old, new):
                    flight.index = index
                    flight.p = target - [0., .70, 0.]
                    flight.bias = np.array([.07, .13, 0.])
                    flight.yaw_bias = .1
                np.testing.assert_array_equal(old.observe(render=False).state, new.observe(render=False).state)
                expected = route_yaw(new.p + new.bias, target, following, phase, 1.2) - new.yaw_bias
                self.assertAlmostEqual(new.yaw_command, expected)
                direct = math.atan2(.57, -.07) - old.yaw_bias
                self.assertAlmostEqual(old.yaw_command, direct)
                if phase == "FIRST_CORRIDOR":
                    self.assertGreater(abs(new.yaw_command - old.yaw_command), .3)
                else:
                    self.assertAlmostEqual(new.yaw_command, old.yaw_command)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class WarmStartTests(unittest.TestCase):
    def test_legacy_weights_load_with_exact_source_hash(self):
        import torch

        from learning.policy import DepthPolicy
        from learning.train import load_initial_weights

        original, restored = DepthPolicy(), DepthPolicy()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "best.pt"
            torch.save({"format": 1, "architecture": "DepthPolicy", "model": original.state_dict(),
                        "steps": 5320}, source)
            provenance = load_initial_weights(restored, source)
            self.assertEqual(provenance["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(provenance["mode"], "weights_only_warm_start")
        self.assertEqual(provenance["source_steps"], 5320)
        self.assertEqual(provenance["source_camera_forward_m"], .08)
        for expected, actual in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_seed_42_keeps_validation_and_holdout_disjoint(self):
        from learning.train import HOLDOUT, VALIDATION, case_banks

        validation, holdout = case_banks(42, True)
        sets = [{row[0] for row in bank} for bank in (validation, holdout, VALIDATION, HOLDOUT)]
        self.assertEqual(len(set.union(*sets)), sum(map(len, sets)))
        self.assertGreater(min(sets[0] | sets[1]), 1_000_000_000)
        self.assertEqual((validation, holdout), case_banks(42, True))

    def test_transfer_checkpoint_and_metadata_bind_same_route_and_heading(self):
        import torch
        from learning.train import Experiment

        args = SimpleNamespace(profile='x500-transfer', camera_forward=.34, heading_lookahead=1.8,
                               max_seconds=200., threads=1, seed=44, device='cpu', init_checkpoint=None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            experiment = Experiment(args, output)
            experiment.metadata()
            experiment.checkpoint('latest.pt')
            metadata = json.loads((output/'metadata.json').read_text())
            checkpoint = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        for field in ('profile', 'route_sha256', 'heading_sha256', 'transfer_profile'):
            self.assertEqual(checkpoint[field], metadata[field])
        self.assertEqual(checkpoint['profile'], 'x500-transfer')
        self.assertEqual(checkpoint['transfer_profile']['name'], 'x500-transfer-v1')
        self.assertIsNone(metadata['sphere_radius_m'])

    def test_infeasible_evaluation_case_stays_in_reported_denominator(self):
        from learning.train import ACTORS, evaluate
        from learning.transfer import InfeasibleScenario

        error = InfeasibleScenario(99, .25, {'obstacle': 'blocked'})
        with patch('learning.train.make_flight', side_effect=error) as make_flight:
            result = evaluate(None, None, [(99, .9, 1.7)], lambda: False)
        self.assertEqual(make_flight.call_count, 1)
        self.assertEqual(result['infeasible_scenarios'][0]['seed'], 99)
        self.assertIsNone(result['paired_mean_depth_vs_blank_action_delta_mps'])
        for actor in ACTORS:
            self.assertEqual(result['summary'][actor]['episodes'], 1)
            self.assertEqual(result['summary'][actor]['evaluated_episodes'], 0)
            self.assertFalse(result['episodes'][actor][0]['success'])
