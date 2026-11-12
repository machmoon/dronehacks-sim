import importlib.util
import unittest

import numpy as np

from learning.sim import Course, Flight, teacher


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class DepthPolicyTests(unittest.TestCase):
    def test_learns_visually_conditioned_actions_with_identical_state(self):
        import torch

        from learning.policy import DepthPolicy

        torch.manual_seed(5)
        torch.set_num_threads(2)
        course = Course()
        a = course.phases.index("WINDOW_CROSS")
        rows = []
        # Every input state is identical. Only the randomized aperture changes,
        # so success on new seeds cannot come from local goal or attitude.
        for seed in range(64):
            flight = Flight(course, seed, 1.)
            flight.index = a
            flight.p = course.goals[a - 1].copy()
            flight.yaw = np.arctan2(-course.axes[1, 0], -course.axes[0, 0])
            flight.bias[:] = 0
            flight.wind[:] = 0
            flight.yaw_bias = flight.difficulty = 0.
            obs = flight.observe()
            rows.append((obs.depth, obs.state, teacher(flight)))
        depth, state, label = [torch.tensor(np.stack(col), dtype=torch.float32) for col in zip(*rows)]
        depth = depth[:, None]
        self.assertTrue(torch.all(state == state[:1]))
        model = DepthPolicy()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        for _ in range(120):
            loss = (model(depth[:48], state[:48]) - label[:48]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            seen = model(depth[48:], state[48:])
            blank = model(torch.zeros_like(depth[48:]), state[48:])
            self.assertGreater(torch.linalg.vector_norm(seen - blank, dim=1).mean().item(), .02)
            self.assertLess((seen - label[48:]).square().mean().item(),
                            .6 * (blank - label[48:]).square().mean().item())
            self.assertTrue((torch.linalg.vector_norm(seen, dim=1) <= state[48:, -1] + 1e-6).all())
