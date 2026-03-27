"""Spatial CNN + MLP hybrid actor model for the High Ground SRPG.

Motivation
----------
The flat 418-dim observation embeds two 13×13 spatial channels (terrain type and
elevation) as the first 338 floats.  A plain MLP must learn to reason about
grid topology from a list of scalars, which is sample-inefficient.  A small
convolutional encoder gives the network translational equivariance over the map,
letting it recognise patrol routes, choke points, and elevation ridges in far
fewer gradient steps than a raw MLP.

Architecture
------------
                         ┌──────────────────────────────────────────────────┐
 obs[..., 0:338]         │  CNN spatial encoder                              │
   reshape → (2, 13, 13) │  Conv(2→16, 3×3, s=1) → ReLU  (16 × 13 × 13)   │
                         │  Conv(16→32, 3×3, s=2) → ReLU  (32 ×  7 ×  7)   │
                         │  Conv(32→32, 3×3, s=2) → ReLU  (32 ×  4 ×  4)   │
                         │  Flatten → 512                                    │
                         └────────────────────┬─────────────────────────────┘
                                              │ concat
 obs[..., 338:418]  (80 non-spatial floats) ──┘
                                              ↓
                                   MLP  [592 → 256 → 256 → output_features]
                                   (via BenchMARL MultiAgentMLP, shared params)

The CNN is shared across all agents (share_params=True in all our experiments).
The critic continues to use MlpConfig([512, 512]) over the 836-dim joint state.

Usage in benchmarl_train.py
----------------------------
    from highground.training.cnn_model import SpatialCnnMlpConfig
    import torch.nn as nn

    actor_config = SpatialCnnMlpConfig(
        cnn_channels=(16, 32, 32),
        mlp_cells=(256, 256),
        activation_class=nn.ReLU,
    )
    # pass as model_config= to Experiment(...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig

from highground.engine.grid import GRID_SIZE
from highground.env.srpg_env import OBS_SIZE

# ── Observation layout constants ──────────────────────────────────────────────
_SPATIAL_END = GRID_SIZE * GRID_SIZE * 2   # 338: end of terrain + elevation block
_NON_SPATIAL  = OBS_SIZE - _SPATIAL_END     #  80: unit / squad / role features
_SPATIAL_CH   = 2                           # terrain channel + elevation channel


class SpatialCnnMlp(Model):
    """Convolutional spatial encoder followed by a shared-parameter MLP trunk.

    Designed for the MAPPO actor (``input_has_agent_dim=True``).  The critic
    should keep using ``MlpConfig`` with a wider MLP over the joint state.

    Args:
        cnn_channels: Output channels for each CNN layer.  The first layer
            preserves spatial resolution (stride 1); subsequent layers use
            stride 2 to downsample.
        mlp_cells: Hidden layer widths for the MLP trunk that follows the CNN.
        activation_class: Activation applied after each CNN and MLP layer.
    """

    def __init__(
        self,
        cnn_channels: Sequence[int],
        mlp_cells: Sequence[int],
        activation_class: Type[nn.Module],
        **kwargs,
    ) -> None:
        super().__init__(
            input_spec        = kwargs.pop("input_spec"),
            output_spec       = kwargs.pop("output_spec"),
            agent_group       = kwargs.pop("agent_group"),
            input_has_agent_dim = kwargs.pop("input_has_agent_dim"),
            n_agents          = kwargs.pop("n_agents"),
            centralised       = kwargs.pop("centralised"),
            share_params      = kwargs.pop("share_params"),
            device            = kwargs.pop("device"),
            action_spec       = kwargs.pop("action_spec"),
            model_index       = kwargs.pop("model_index"),
            is_critic         = kwargs.pop("is_critic"),
        )

        # ── CNN spatial encoder ───────────────────────────────────────────────
        layers: list[nn.Module] = []
        in_ch = _SPATIAL_CH
        for i, out_ch in enumerate(cnn_channels):
            stride = 2 if i > 0 else 1   # first layer: full res; rest: ×0.5
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
            layers.append(activation_class())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers).to(self.device)

        # Compute CNN output flat size once
        with torch.no_grad():
            dummy = torch.zeros(1, _SPATIAL_CH, GRID_SIZE, GRID_SIZE, device=self.device)
            self._cnn_flat: int = int(self.cnn(dummy).flatten(1).shape[1])

        # ── MLP trunk ─────────────────────────────────────────────────────────
        mlp_in = self._cnn_flat + _NON_SPATIAL
        out_features: int = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs  = mlp_in,
                n_agent_outputs = out_features,
                n_agents        = self.n_agents,
                centralised     = self.centralised,
                share_params    = self.share_params,
                device          = self.device,
                num_cells       = list(mlp_cells),
                activation_class = activation_class,
            )
        else:
            # Centralised path (critic): plain MLP(s).  In practice the critic
            # uses MlpConfig, so this branch is a safety fallback.
            n_nets = 1 if self.share_params else self.n_agents
            self.mlp = nn.ModuleList([
                _build_mlp(mlp_in, list(mlp_cells), out_features, activation_class)
                for _ in range(n_nets)
            ])

    # ── BenchMARL interface ───────────────────────────────────────────────────

    def _perform_checks(self) -> None:
        # Run only the base Model checks; skip Mlp's strict shape assertions
        # (which expect 1-D features rather than our flat spatial blob).
        Model._perform_checks(self)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict.get(self.in_keys[0])   # [..., n_agents, 418]

        if self.input_has_agent_dim:
            # obs: [batch, n_agents, 418]  (batch may be multi-dimensional)
            prefix   = obs.shape[:-2]            # e.g. (B,)
            n_agents = obs.shape[-2]
            flat_b   = math.prod(prefix)         # product of batch dims

            # Extract and reshape spatial block → (flat_b * n_agents, 2, 13, 13)
            spatial     = obs[..., :_SPATIAL_END]
            spatial     = spatial.reshape(flat_b * n_agents, _SPATIAL_CH, GRID_SIZE, GRID_SIZE)
            cnn_out     = self.cnn(spatial).flatten(1)          # [flat_b*n_agents, cnn_flat]
            cnn_out     = cnn_out.reshape(*prefix, n_agents, -1) # [*prefix, n_agents, cnn_flat]

            non_spatial = obs[..., _SPATIAL_END:]               # [*prefix, n_agents, 80]
            combined    = torch.cat([cnn_out, non_spatial], dim=-1)

            res = self.mlp(combined)
            if not self.output_has_agent_dim:
                res = res[..., 0, :]
        else:
            # Centralised (fallback) path
            obs_flat = obs.flatten(start_dim=-1)
            if self.share_params:
                res = self.mlp[0](obs_flat)
            else:
                res = torch.stack([net(obs_flat) for net in self.mlp], dim=-2)

        tensordict.set(self.out_key, res)
        return tensordict


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class SpatialCnnMlpConfig(ModelConfig):
    """Dataclass config for :class:`SpatialCnnMlp`.

    Defaults produce the recommended architecture:
      CNN: 2 → 16 → 32 → 32 channels  (512-dim embedding after flatten)
      MLP: 592 → 256 → 256 → output
    """

    cnn_channels:     Sequence[int]       = field(default_factory=lambda: (16, 32, 32))
    mlp_cells:        Sequence[int]       = field(default_factory=lambda: (256, 256))
    activation_class: Type[nn.Module]     = nn.ReLU

    @staticmethod
    def associated_class():
        return SpatialCnnMlp


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_mlp(
    in_features: int,
    hidden: list[int],
    out_features: int,
    activation: Type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)
