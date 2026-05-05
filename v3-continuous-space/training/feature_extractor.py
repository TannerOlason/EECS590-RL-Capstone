"""SB3 feature extractor for ScrollingSquadEnv's Dict observation.

Mirrors the V2 SpatialCnnMlp architecture (multi-channel CNN over a GRID_SIZE
square grid, then concat non-spatial features), adapted to SB3's
BaseFeaturesExtractor API. This extractor is shared by the SAC actor and
the twin critics; SB3 wires them up via `policy_kwargs={"features_extractor_class": ...}`.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE


class ScrollCnnExtractor(BaseFeaturesExtractor):
    """Legacy CNN over the spatial block + raw non-spatial feature concat.

    Kept stable so older checkpoints can still be loaded.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        spatial_shape = observation_space["spatial"].shape
        non_spatial_dim = observation_space["features"].shape[0]

        in_ch = spatial_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *spatial_shape)
            cnn_flat = self.cnn(dummy).shape[1]

        self.cnn_proj = nn.Sequential(nn.Linear(cnn_flat, 256), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(256 + non_spatial_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]
        feats = observations["features"]
        x = self.cnn_proj(self.cnn(spatial))
        return self.head(torch.cat([x, feats], dim=-1))


class SpatialDominantScrollExtractor(BaseFeaturesExtractor):
    """Extractor for new runs that gives spatial channels more capacity.

    The scalar feature branch is compressed before concatenation and the spatial
    branch is scaled, making it harder for the policy to solve the task by
    ignoring the CNN entirely.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        spatial_scale: float = 2.0,
        feature_branch_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        spatial_shape = observation_space["spatial"].shape
        non_spatial_dim = observation_space["features"].shape[0]
        self.spatial_scale = float(spatial_scale)

        in_ch = spatial_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *spatial_shape)
            cnn_flat = self.cnn(dummy).shape[1]

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_flat, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(non_spatial_dim, feature_branch_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_branch_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + feature_branch_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]
        feats = observations["features"]
        x = self.cnn_proj(self.cnn(spatial)) * self.spatial_scale
        f = self.feature_proj(feats)
        return self.head(torch.cat([x, f], dim=-1))


# Sanity-check: GRID_SIZE used implicitly via observation_space.shape; expose for tests.
SPATIAL_GRID_SIZE = GRID_SIZE
