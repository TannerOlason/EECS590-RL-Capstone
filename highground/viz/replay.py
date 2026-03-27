"""Record and replay SRPG matches as frame sequences.

Usage:
    from highground.viz.replay import record_match, render_replay_gif

    frames = record_match(grid, spawns_a, spawns_b, model)
    render_replay_gif(frames, "replay.gif")
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from highground.engine.game_state import GameState
from highground.engine.grid import Grid
from highground.engine.units import TEAM_A, Unit
from highground.env.sb3_wrapper import SB3SRPGWrapper
from highground.env.srpg_env import HighGroundEnv
from highground.viz.render_map import render_map


def _snapshot_units(units: list[Unit]) -> list[dict]:
    """Capture unit state as a list of dicts (one snapshot frame)."""
    return [
        {
            "unit_id": u.unit_id,
            "team": u.team,
            "unit_class": u.unit_class,
            "row": u.row,
            "col": u.col,
            "hp": u.hp,
            "max_hp": u.max_hp,
            "alive": u.alive,
            "move_remaining": u.move_remaining,
            "has_attacked": u.has_attacked,
        }
        for u in units
    ]


class Frame:
    """A single frame capturing game state at one point in time."""

    __slots__ = ("round_number", "current_unit_id", "current_team", "units", "action")

    def __init__(
        self,
        round_number: int,
        current_unit_id: int,
        current_team: int,
        units: list[dict],
        action: int | None = None,
    ):
        self.round_number = round_number
        self.current_unit_id = current_unit_id
        self.current_team = current_team
        self.units = units
        self.action = action


def _mask_fn(env: SB3SRPGWrapper) -> np.ndarray:
    return env.action_masks()


def _make_opponent_fn(model: MaskablePPO) -> Callable:
    def opponent_fn(obs: dict[str, np.ndarray]) -> int:
        action, _ = model.predict(
            {"observation": obs["observation"], "action_mask": obs["action_mask"]},
            deterministic=True,
            action_masks=obs["action_mask"],
        )
        return int(action)
    return opponent_fn


def record_match(
    grid: Grid,
    spawns_a: list[tuple[int, int]],
    spawns_b: list[tuple[int, int]],
    model: MaskablePPO,
    *,
    seed: int = 0,
) -> tuple[Grid, list[Frame]]:
    """Play one match and record every step as a Frame.

    Returns:
        (grid, frames) — the grid is needed for rendering.
    """
    aec = HighGroundEnv(grid.copy(), list(spawns_a), list(spawns_b), reward_mode="sparse")
    env = SB3SRPGWrapper(aec, controlled_team=TEAM_A, opponent_fn=_make_opponent_fn(model))
    env = ActionMasker(env, _mask_fn)

    obs, _ = env.reset(seed=seed)
    done = False
    frames: list[Frame] = []

    # Initial frame
    game = aec._game
    frames.append(Frame(
        round_number=game.round_number,
        current_unit_id=game.current_unit_id,
        current_team=game.current_team,
        units=_snapshot_units(game.units),
    ))

    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        action = int(action)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

        frames.append(Frame(
            round_number=game.round_number,
            current_unit_id=game.current_unit_id,
            current_team=game.current_team,
            units=_snapshot_units(game.units),
            action=action,
        ))

    return grid, frames


def _frame_to_mock_units(frame: Frame) -> list:
    """Convert frame unit dicts to simple objects for render_map."""
    class _U:
        pass

    units = []
    for d in frame.units:
        u = _U()
        for k, v in d.items():
            setattr(u, k, v)
        units.append(u)
    return units


def render_replay_gif(
    grid: Grid,
    frames: list[Frame],
    output_path: str = "replay.gif",
    *,
    fps: int = 3,
    dpi: int = 100,
) -> None:
    """Save a replay as an animated GIF.

    Args:
        grid: The map grid.
        frames: Frame sequence from record_match.
        output_path: Path for the output GIF.
        fps: Frames per second.
        dpi: Resolution.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    def update(i: int) -> None:
        ax.clear()
        frame = frames[i]
        units = _frame_to_mock_units(frame)
        render_map(
            grid, units=units, ax=ax,
            title=f"Round {frame.round_number} | Unit {frame.current_unit_id}",
            show_elevation_text=False,
        )

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def render_replay_frames(
    grid: Grid,
    frames: list[Frame],
    output_dir: str = "replay_frames",
    *,
    dpi: int = 100,
    every_n: int = 1,
) -> list[str]:
    """Save individual frame PNGs to a directory.

    Args:
        grid: The map grid.
        frames: Frame sequence from record_match.
        output_dir: Directory for output PNGs.
        dpi: Resolution.
        every_n: Save every Nth frame (for subsampling).

    Returns:
        List of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i in range(0, len(frames), every_n):
        frame = frames[i]
        units = _frame_to_mock_units(frame)
        fig = render_map(
            grid, units=units,
            title=f"Round {frame.round_number} | Unit {frame.current_unit_id}",
        )
        path = os.path.join(output_dir, f"frame_{i:04d}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    return paths
