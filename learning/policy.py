"""Small marker-free policy: depth image and estimated local motion in, velocity out."""
import torch
from torch import nn

from .sim import HEIGHT, WIDTH


class DepthPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(16 * HEIGHT // 4 * WIDTH // 4, 48), nn.SiLU(),
        )
        self.actor = nn.Sequential(nn.Linear(48 + 9, 72), nn.SiLU(),
                                   nn.Linear(72, 48), nn.SiLU(), nn.Linear(48, 3))
        # Start from a usable waypoint controller while the visual residual learns.
        nn.init.normal_(self.actor[-1].weight, std=.004)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, depth, state):
        goal, cap = state[:, 3:6], state[:, -1:]
        distance = torch.linalg.vector_norm(goal, dim=1, keepdim=True).clamp_min(1e-6)
        direct = (goal / distance) * torch.minimum(torch.minimum(1.75 * distance, cap),
                                                  torch.sqrt(1.25 * distance))
        residual = self.actor(torch.cat([self.vision(depth), state], dim=1))
        action = direct + .85 * cap * torch.tanh(residual)
        norm = torch.linalg.vector_norm(action, dim=1, keepdim=True).clamp_min(1e-6)
        return action * torch.minimum(torch.ones_like(norm), cap / norm)
