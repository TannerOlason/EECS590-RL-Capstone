"""Generate a GIF replay from the V2 High Ground environment.

By default this uses the final MAPPO actor checkpoint when it exists, falling
back to random valid actions if the checkpoint is unavailable.

Usage:
    python scripts/replay_v2.py
    python scripts/replay_v2.py --policy random --map central_hill
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-highground")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from matplotlib.animation import FuncAnimation, PillowWriter

from highground.engine.grid import GRID_SIZE
from highground.env.srpg_env import HighGroundEnv
from highground.maps.static_maps import ALL_MAPS
from highground.viz.render_map import render_map


SPATIAL_END = GRID_SIZE * GRID_SIZE * 2
OBS_SIZE = 418
NON_SPATIAL = OBS_SIZE - SPATIAL_END


class MappoActor(nn.Module):
    """Small loader for the final BenchMARL MAPPO actor state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512 + NON_SPATIAL, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 12),
        )

    def forward(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        spatial = x[:, :SPATIAL_END].reshape(1, 2, GRID_SIZE, GRID_SIZE)
        non_spatial = x[:, SPATIAL_END:]
        features = torch.cat([self.cnn(spatial).flatten(1), non_spatial], dim=-1)
        return self.mlp(features).squeeze(0)


def _load_actor(checkpoint: Path, module_idx: int) -> MappoActor:
    raw = torch.load(checkpoint, map_location="cpu")
    actor = MappoActor()
    prefix = f"module.{module_idx}.module.0."
    translated = {}
    for key, value in raw.items():
        if not key.startswith(prefix) or not hasattr(value, "shape"):
            continue
        new_key = key[len(prefix):]
        new_key = new_key.replace("mlp.params.", "mlp.")
        translated[new_key] = value
    actor.load_state_dict(translated, strict=True)
    actor.eval()
    return actor


def _mappo_action(actor_a: MappoActor, actor_b: MappoActor, agent: str, obs: dict) -> int:
    actor = actor_a if agent.startswith("team0") else actor_b
    with torch.no_grad():
        logits = actor(obs["observation"]).clone()
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
    logits[~mask] = -1e9
    return int(torch.argmax(logits).item())


def _snapshot(env: HighGroundEnv, action: int | None) -> dict:
    game = env._game
    return {
        "round": game.round_number,
        "unit": game.current_unit_id,
        "agent": env.agent_selection,
        "action": action,
        "units": [u.copy() for u in game.units],
    }


def record_match(
    map_name: str,
    *,
    policy: str = "mappo",
    checkpoint: Path = Path("models/mappo_phase7_policy.pt"),
    seed: int = 0,
    max_steps: int = 160,
) -> tuple[object, list[dict]]:
    if map_name not in ALL_MAPS:
        names = ", ".join(sorted(ALL_MAPS))
        raise ValueError(f"Unknown map '{map_name}'. Choices: {names}")

    rng = np.random.default_rng(seed)
    grid, spawns_a, spawns_b = ALL_MAPS[map_name]()
    env = HighGroundEnv(grid.copy(), spawns_a, spawns_b, reward_mode="shaped")
    env.reset(seed=seed)
    actor_a = actor_b = None
    used_policy = policy
    if policy == "mappo":
        if checkpoint.exists():
            actor_a = _load_actor(checkpoint, 0)
            actor_b = _load_actor(checkpoint, 1)
        else:
            print(f"[warn] checkpoint not found: {checkpoint}; using random policy")
            used_policy = "random"

    frames = [_snapshot(env, None)]
    for _ in range(max_steps):
        if all(env.terminations.values()):
            break
        obs = env.observe(env.agent_selection)
        valid = np.flatnonzero(obs["action_mask"])
        if len(valid) == 0:
            break
        if used_policy == "mappo" and actor_a is not None and actor_b is not None:
            action = _mappo_action(actor_a, actor_b, env.agent_selection, obs)
        else:
            action = int(rng.choice(valid))
        env.step(action)
        frames.append(_snapshot(env, action))

    return grid, frames


def render_gif(
    grid,
    frames: list[dict],
    output: Path,
    *,
    fps: int = 4,
    dpi: int = 120,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    def update(i: int) -> None:
        ax.clear()
        frame = frames[i]
        action = "start" if frame["action"] is None else str(frame["action"])
        render_map(
            grid,
            units=frame["units"],
            ax=ax,
            title=(
                f"V2 High Ground | frame {i}/{len(frames) - 1} | "
                f"round {frame['round']} | unit {frame['unit']} | action {action}"
            ),
            show_elevation_text=False,
        )

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a V2 High Ground replay GIF.")
    parser.add_argument("--map", default="central_hill", choices=sorted(ALL_MAPS))
    parser.add_argument("--output", default="replays/replay_v2.gif", help="GIF output path")
    parser.add_argument("--policy", choices=("mappo", "random"), default="mappo")
    parser.add_argument("--checkpoint", default="models/mappo_phase7_policy.pt")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    grid, frames = record_match(
        args.map,
        policy=args.policy,
        checkpoint=Path(args.checkpoint),
        seed=args.seed,
        max_steps=args.steps,
    )
    render_gif(grid, frames, Path(args.output), fps=args.fps, dpi=args.dpi)
    print(f"wrote {args.output} ({len(frames)} frames, map={args.map}, policy={args.policy})")


if __name__ == "__main__":
    main()
