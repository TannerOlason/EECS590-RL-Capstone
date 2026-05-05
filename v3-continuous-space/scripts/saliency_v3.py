"""Perturbation saliency for a trained V3 SAC policy.

The script measures how much deterministic SAC actions change when each
observation channel/feature is perturbed. It writes heatmaps and JSON to the
experiment directory:
    saliency_spatial.png
    saliency_features.png
    saliency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from experiment_utils import resolve_experiment_dir  # noqa: E402
from env.obs_builder import GRID_SIZE, LOCAL_DIRECTIONS, NON_SPATIAL_DIM, SPATIAL_SHAPE  # noqa: E402
from env.scrolling_env import ScrollingSquadEnv  # noqa: E402
from scripts.eval_v3 import _match_checkpoint_obs  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SPATIAL_LABELS = [
    "terrain",
    "elevation",
    "friendly_hp",
    "enemy_fighter",
    "enemy_charger",
    "enemy_ranger",
    "enemy_siege",
    "potion",
]
FEATURE_LABELS = [
    "hp_frac",
    "attack_cd",
    "row_frac",
    "local_x_frac",
    "agent_0",
    "agent_1",
    "agent_2",
    "alive_0",
    "alive_1",
    "alive_2",
    "centroid_x",
    "centroid_row",
    "mean_hp",
    "enemy_count",
    "scroll_norm",
]
for direction_label, _ in LOCAL_DIRECTIONS:
    FEATURE_LABELS.extend([
        f"{direction_label}_can_step",
        f"{direction_label}_enemy",
        f"{direction_label}_potion",
        f"{direction_label}_advances_right",
    ])
FEATURE_LABELS.extend([
    "nearest_enemy_dx",
    "nearest_enemy_dy",
    "nearest_enemy_dist",
    "nearest_enemy_attackable",
])


def _resolve(args) -> tuple[Path, Path]:
    if args.experiment:
        exp_dir = resolve_experiment_dir(REPO_ROOT / "experiments", args.experiment)
        ckpt = Path(args.checkpoint) if args.checkpoint else exp_dir / "best_model.zip"
        if not ckpt.exists():
            ckpt = exp_dir / "sac_final.zip"
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        exp_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
    else:
        raise SystemExit("Pass --experiment <name> OR --checkpoint <path>.")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return ckpt, exp_dir


def _collect_obs(model: SAC, env: ScrollingSquadEnv, episodes: int, max_steps: int, seed: int) -> list[dict]:
    observations = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            observations.append(obs)
            action, _ = model.predict(_match_checkpoint_obs(obs, model), deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    return observations


def _predict(model: SAC, obs: dict) -> np.ndarray:
    action, _ = model.predict(_match_checkpoint_obs(obs, model), deterministic=True)
    return np.asarray(action, dtype=np.float32)


def _perturb_value(value: float, delta: float, low: float, high: float) -> float:
    """Move a scalar within bounds; flip direction when already near high."""
    if value + delta <= high:
        return value + delta
    return max(low, value - delta)


def _saliency(model: SAC, observations: list[dict], sample_limit: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    if len(observations) > sample_limit:
        idxs = rng.choice(len(observations), size=sample_limit, replace=False)
        observations = [observations[int(i)] for i in idxs]

    spatial_sal = np.zeros(SPATIAL_SHAPE, dtype=np.float64)
    feature_sal = np.zeros(NON_SPATIAL_DIM, dtype=np.float64)
    n = max(1, len(observations))

    for obs in observations:
        base = _predict(model, obs)
        spatial = obs["spatial"]
        features = obs["features"]

        for ch in range(spatial.shape[0]):
            for r in range(spatial.shape[1]):
                for c in range(spatial.shape[2]):
                    pert = {"spatial": spatial.copy(), "features": features.copy()}
                    pert["spatial"][ch, r, c] = _perturb_value(
                        float(pert["spatial"][ch, r, c]), delta, 0.0, 1.0
                    )
                    spatial_sal[ch, r, c] += float(np.abs(_predict(model, pert) - base).mean())

        for i in range(features.shape[0]):
            pert = {"spatial": spatial.copy(), "features": features.copy()}
            pert["features"][i] = _perturb_value(float(pert["features"][i]), delta, -1.0, 1.0)
            feature_sal[i] += float(np.abs(_predict(model, pert) - base).mean())

    return spatial_sal / n, feature_sal / n


def _plot_spatial(spatial_sal: np.ndarray, out_path: Path) -> None:
    n_ch = spatial_sal.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    vmax = float(spatial_sal.max()) or 1.0
    for ch, ax in enumerate(axes.flat[:n_ch]):
        im = ax.imshow(spatial_sal[ch], cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(SPATIAL_LABELS[ch] if ch < len(SPATIAL_LABELS) else f"ch{ch}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_features(feature_sal: np.ndarray, out_path: Path) -> None:
    labels = FEATURE_LABELS[:len(feature_sal)]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(feature_sal)), feature_sal, color="#10b981")
    ax.set_xticks(range(len(feature_sal)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean |delta action|")
    ax.set_title("V3 SAC non-spatial feature saliency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sample-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--kill-reward-scale", type=float, default=1.0)
    args = parser.parse_args()

    ckpt, exp_dir = _resolve(args)
    env = ScrollingSquadEnv(
        seed=args.seed,
        progress_reward_scale=args.progress_reward_scale,
        kill_reward_scale=args.kill_reward_scale,
    )
    model = SAC.load(str(ckpt), device="auto")
    observations = _collect_obs(model, env, args.episodes, args.max_steps, args.seed)
    spatial_sal, feature_sal = _saliency(model, observations, args.sample_limit, args.delta)

    spatial_png = exp_dir / "saliency_spatial.png"
    features_png = exp_dir / "saliency_features.png"
    _plot_spatial(spatial_sal, spatial_png)
    _plot_features(feature_sal, features_png)
    payload = {
        "checkpoint": str(ckpt),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "sample_limit": args.sample_limit,
        "delta": args.delta,
        "spatial_channel_saliency": {
            SPATIAL_LABELS[i] if i < len(SPATIAL_LABELS) else f"ch{i}": float(spatial_sal[i].mean())
            for i in range(spatial_sal.shape[0])
        },
        "feature_saliency": {
            FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f"feature_{i}": float(feature_sal[i])
            for i in range(feature_sal.shape[0])
        },
    }
    json_path = exp_dir / "saliency.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[saliency] wrote {spatial_png}")
    print(f"[saliency] wrote {features_png}")
    print(f"[saliency] wrote {json_path}")


if __name__ == "__main__":
    main()
