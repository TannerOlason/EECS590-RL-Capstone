"""BenchMARLPolicyAdapter: wraps a saved BenchMARL MAPPO policy for QD evaluation.

Exposes the same .predict() interface as MaskablePPO so it can be dropped
into fitness.py and evolve.py without any changes to the QD evaluation logic.

Usage::

    from highground.training.benchmarl_adapter import BenchMARLPolicyAdapter
    model = BenchMARLPolicyAdapter("models/mappo_srpg_policy.pt")
    action, _ = model.predict(obs, deterministic=True, action_masks=mask)

Architecture auto-detection
----------------------------
The adapter inspects the state dict at load time to determine whether it was
saved from a V3 (plain MLP) or V4 (CNN + MLP) training run:

**V3 detection**: no 4D Conv2d weight tensors — the state dict only has 2D
linear-layer weights.  The actor MLP is identified by finding the linear layer
whose ``in_features`` equals OBS_SIZE (418) and rebuilding a ``nn.Sequential``
with Tanh activations (BenchMARL ``MlpConfig`` default).

**V4 detection**: at least one 4D weight tensor whose key contains ``.cnn.``.
The CNN and MLP sub-networks are rebuilt separately from keys matching
``<prefix>.cnn.<N>.weight/bias`` and ``<prefix>.mlp.params.<N>.weight/bias``.
The V4 forward pass is::

    obs[:338]  ──reshape(1,2,13,13)──► CNN ──flatten──► 512-dim ─┐
    obs[338:]  ─────────────────────────────────────────────────► cat(592) ──► MLP ──► logits

CNN strides match ``SpatialCnnMlpConfig``: first conv stride=1, remaining=2.
MLP uses ReLU activations (``SpatialCnnMlpConfig(activation_class=nn.ReLU)``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from highground.env.srpg_env import OBS_SIZE

# ── Spatial observation constants (must match cnn_model.py / srpg_env.py) ─────
_SPATIAL_END = 338   # first 338 features encode two 13×13 grids
_NON_SPATIAL  = 80   # remaining 80 scalar features
_SPATIAL_CH   = 2    # channels (terrain type + elevation)
_SPATIAL_H    = 13
_SPATIAL_W    = 13

# Stride for each successive Conv layer in V4 SpatialCnnMlpConfig
_CNN_STRIDES  = (1, 2, 2)


class BenchMARLPolicyAdapter:
    """Load a BenchMARL-saved policy .pt file and expose a MaskablePPO-style predict() interface.

    Parameters
    ----------
    model_path :
        Path to a ``*_policy.pt`` file saved by ``train_mappo()`` via
        ``torch.save(exp.policy.state_dict(), path)``.
    """

    def __init__(self, model_path: str) -> None:
        self._path = str(model_path)
        self._is_v4, self._cnn, self._mlp = self._load_network(self._path)
        if self._cnn is not None:
            self._cnn.eval()
        self._mlp.eval()

    # ── Architecture reconstruction ───────────────────────────────────────────

    @classmethod
    def _load_network(
        cls, path: str
    ) -> tuple[bool, nn.Sequential | None, nn.Sequential]:
        """Return (is_v4, cnn_or_None, mlp)."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")

        if not isinstance(state, dict):
            state = {k: v for k, v in state.items()} if hasattr(state, "items") else {}

        if cls._is_v4_state_dict(state):
            prefix = cls._find_v4_prefix(state)
            cnn = cls._build_v4_cnn(state, prefix)
            mlp = cls._build_v4_mlp(state, prefix)
            return True, cnn, mlp
        else:
            prefix = cls._find_v3_prefix(state)
            mlp = cls._build_v3_mlp(state, prefix)
            return False, None, mlp

    # ── V4 (CNN + MLP) path ───────────────────────────────────────────────────

    @staticmethod
    def _is_v4_state_dict(state: dict) -> bool:
        """Return True if the state dict contains Conv2d weights (.cnn. keys)."""
        for key, val in state.items():
            if (
                isinstance(val, torch.Tensor)
                and val.ndim == 4
                and ".cnn." in key
            ):
                return True
        return False

    @staticmethod
    def _find_v4_prefix(state: dict) -> str:
        """Return the shared key prefix before the `.cnn.` sub-tree."""
        for key, val in state.items():
            if isinstance(val, torch.Tensor) and val.ndim == 4 and ".cnn." in key:
                prefix = key[: key.index(".cnn.")]
                return prefix
        raise KeyError("V4 state dict has no .cnn. Conv2d weights — unexpected format.")

    @staticmethod
    def _build_v4_cnn(state: dict, prefix: str) -> nn.Sequential:
        """Reconstruct the CNN encoder from ``<prefix>.cnn.<N>.weight/bias`` keys.

        Activations (ReLU) are placed between every conv layer (not after the
        last one since the output feeds directly into the MLP).
        """
        cnn_prefix = f"{prefix}.cnn."
        entries: list[tuple[int, str]] = []
        for key in state:
            if key.startswith(cnn_prefix) and key.endswith(".weight"):
                rest = key[len(cnn_prefix):]  # e.g. "0.weight"
                parts = rest.split(".")
                if len(parts) == 2 and parts[0].isdigit():
                    entries.append((int(parts[0]), key))

        if not entries:
            raise KeyError(
                f"No conv weights found under {cnn_prefix!r}.\n"
                f"Keys (first 20): {list(state.keys())[:20]}"
            )

        entries.sort(key=lambda t: t[0])
        strides = list(_CNN_STRIDES) + [2] * max(0, len(entries) - len(_CNN_STRIDES))

        modules: list[nn.Module] = []
        for layer_num, (_, wkey) in enumerate(entries):
            bkey = wkey[:-len("weight")] + "bias"
            w = state[wkey]   # shape: (out_ch, in_ch, kH, kW)
            b = state[bkey]
            out_ch, in_ch, kH, kW = w.shape
            stride = strides[layer_num]
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kH, kW), stride=stride, padding=1)
            conv.weight = nn.Parameter(w.clone())
            conv.bias   = nn.Parameter(b.clone())
            modules.append(conv)
            modules.append(nn.ReLU())

        # Remove the final ReLU — activation is applied inside the first MLP layer
        if modules and isinstance(modules[-1], nn.ReLU):
            modules.pop()

        modules.append(nn.Flatten())
        return nn.Sequential(*modules)

    @staticmethod
    def _build_v4_mlp(state: dict, prefix: str) -> nn.Sequential:
        """Reconstruct the MLP trunk from ``<prefix>.mlp.params.<N>.weight/bias`` keys.

        Uses ReLU activations between layers (``SpatialCnnMlpConfig`` default).
        """
        mlp_prefix = f"{prefix}.mlp.params."
        entries: list[tuple[int, str]] = []
        for key in state:
            if key.startswith(mlp_prefix) and key.endswith(".weight"):
                rest = key[len(mlp_prefix):]  # e.g. "0.weight"
                parts = rest.split(".")
                if len(parts) == 2 and parts[0].isdigit():
                    entries.append((int(parts[0]), key))

        if not entries:
            raise KeyError(
                f"No MLP weights found under {mlp_prefix!r}.\n"
                f"Keys (first 20): {list(state.keys())[:20]}"
            )

        entries.sort(key=lambda t: t[0])

        modules: list[nn.Module] = []
        for i, (_, wkey) in enumerate(entries):
            bkey = wkey[:-len("weight")] + "bias"
            w = state[wkey]
            b = state[bkey]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=True)
            lin.weight = nn.Parameter(w.clone())
            lin.bias   = nn.Parameter(b.clone())
            modules.append(lin)
            if i < len(entries) - 1:
                modules.append(nn.ReLU())

        return nn.Sequential(*modules)

    # ── V3 (plain MLP) path ───────────────────────────────────────────────────

    @staticmethod
    def _find_v3_prefix(state: dict) -> str:
        """Find the state-dict prefix holding the actor MLP for a V3 checkpoint.

        Locates the 2D weight matrix whose ``in_features`` equals OBS_SIZE.
        """
        candidates: list[str] = []
        for key, val in state.items():
            if not (isinstance(val, torch.Tensor) and val.ndim == 2):
                continue
            if val.shape[1] == OBS_SIZE and key.endswith(".weight"):
                parts = key.rsplit(".", 2)
                if len(parts) == 3 and parts[1].isdigit():
                    candidates.append(parts[0])

        if not candidates:
            raise KeyError(
                f"No actor MLP first layer (in_features={OBS_SIZE}) found in state dict.\n"
                f"Keys (first 20): {list(state.keys())[:20]}"
            )

        candidates.sort(key=len, reverse=True)
        return candidates[0]

    @staticmethod
    def _build_v3_mlp(state: dict, prefix: str) -> nn.Sequential:
        """Reconstruct nn.Sequential from all linear layers under prefix.

        BenchMARL's MlpConfig uses Tanh activations by default.
        """
        entries: list[tuple[int, str]] = []
        for key in state:
            if not (key.startswith(prefix + ".") and key.endswith(".weight")):
                continue
            rest = key[len(prefix) + 1:]
            parts = rest.split(".")
            if len(parts) == 2 and parts[0].isdigit() and parts[1] == "weight":
                entries.append((int(parts[0]), key))

        if not entries:
            raise KeyError(
                f"No linear-layer weights found under prefix {prefix!r}.\n"
                f"Keys (first 20): {list(state.keys())[:20]}"
            )

        entries.sort(key=lambda t: t[0])

        modules: list[nn.Module] = []
        for i, (_, wkey) in enumerate(entries):
            bkey = wkey[:-len("weight")] + "bias"
            w = state[wkey]
            b = state[bkey]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=True)
            lin.weight = nn.Parameter(w.clone())
            lin.bias   = nn.Parameter(b.clone())
            modules.append(lin)
            if i < len(entries) - 1:
                modules.append(nn.Tanh())

        return nn.Sequential(*modules)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, None]:
        """Run the actor and return (action_int, None).

        Signature matches MaskablePPO.predict() for drop-in compatibility.

        Parameters
        ----------
        obs :
            418-dim float32 observation array, or dict with "observation" key.
        deterministic :
            If True (default), take argmax of masked logits.
            If False, sample from the softmax distribution.
        action_masks :
            Boolean array of shape (n_actions,). True = valid action.
            Illegal actions are masked to ``-inf`` before argmax/sampling.
        """
        if isinstance(obs, dict):
            raw = obs.get("observation", next(iter(obs.values())))
        else:
            raw = obs

        x = torch.as_tensor(np.asarray(raw, dtype=np.float32))

        with torch.no_grad():
            if self._is_v4:
                logits = self._forward_v4(x)
            else:
                logits = self._mlp(x)

        if action_masks is not None:
            mask_t = torch.as_tensor(np.asarray(action_masks, dtype=bool))
            logits = logits.masked_fill(~mask_t, float("-inf"))

        if deterministic:
            action = int(logits.argmax().item())
        else:
            probs  = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())

        return action, None

    def predict_logits(self, obs) -> np.ndarray:
        """Return raw logits (12,) before action masking.

        Used by the V6 LLM steering wrapper to obtain an unmasked logit
        vector that can be additively biased before masked softmax.

        Parameters
        ----------
        obs :
            418-dim float32 observation array, or dict with "observation" key.

        Returns
        -------
        np.ndarray
            Shape (12,) float32 raw logit array.
        """
        if isinstance(obs, dict):
            raw = obs.get("observation", next(iter(obs.values())))
        else:
            raw = obs
        x = torch.as_tensor(np.asarray(raw, dtype=np.float32))
        with torch.no_grad():
            if self._is_v4:
                logits = self._forward_v4(x)
            else:
                logits = self._mlp(x)
        return logits.numpy()

    def _forward_v4(self, x: torch.Tensor) -> torch.Tensor:
        """V4 forward: split obs → CNN spatial encoder → concat non-spatial → MLP."""
        spatial     = x[..., :_SPATIAL_END]           # [..., 338]
        non_spatial = x[..., _SPATIAL_END:]            # [..., 80]

        # Reshape to (batch, 2, 13, 13); handle both unbatched and batched obs
        if spatial.ndim == 1:
            sp = spatial.reshape(1, _SPATIAL_CH, _SPATIAL_H, _SPATIAL_W)
            cnn_out = self._cnn(sp).squeeze(0)         # [512]
            combined = torch.cat([cnn_out, non_spatial], dim=-1)   # [592]
        else:
            batch = spatial.shape[0]
            sp = spatial.reshape(batch, _SPATIAL_CH, _SPATIAL_H, _SPATIAL_W)
            cnn_out = self._cnn(sp)                    # [batch, 512]
            combined = torch.cat([cnn_out, non_spatial], dim=-1)   # [batch, 592]

        return self._mlp(combined)

    def __repr__(self) -> str:
        arch = "V4(CNN+MLP)" if self._is_v4 else "V3(MLP)"
        return f"BenchMARLPolicyAdapter({arch}, path={self._path!r})"
