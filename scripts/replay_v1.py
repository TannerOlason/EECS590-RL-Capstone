"""Generate a GIF for the early V1 routed gridworld prototype.

The original V1 environment under ``src/`` was a small PettingZoo route-following
experiment. This script renders that idea directly so it can be used as a stable
visual artifact without depending on later High Ground engine code.

Usage:
    MPLCONFIGDIR=/tmp/matplotlib-v1 python scripts/replay_v1.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-highground")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class AgentTrack:
    name: str
    color: str
    pos: np.ndarray
    route: list[np.ndarray]
    route_idx: int = 0

    @property
    def target(self) -> np.ndarray:
        return self.route[self.route_idx]

    def step(self) -> None:
        delta = self.target - self.pos
        if np.all(delta == 0):
            self.route_idx = (self.route_idx + 1) % len(self.route)
            delta = self.target - self.pos

        move = np.zeros(2, dtype=int)
        axis = 0 if abs(delta[0]) >= abs(delta[1]) else 1
        if delta[axis] != 0:
            move[axis] = 1 if delta[axis] > 0 else -1
        self.pos = self.pos + move


def _build_tracks() -> list[AgentTrack]:
    route_a = [
        np.array([1, 1]),
        np.array([8, 1]),
        np.array([8, 8]),
        np.array([1, 8]),
    ]
    route_b = [
        np.array([8, 8]),
        np.array([1, 8]),
        np.array([1, 1]),
        np.array([8, 1]),
    ]
    return [
        AgentTrack("agent1", "#2563eb", np.array([0, 0]), route_a),
        AgentTrack("agent2", "#dc2626", np.array([9, 9]), route_b),
    ]


def _snapshot(tracks: list[AgentTrack]) -> list[tuple[str, str, tuple[int, int], tuple[int, int]]]:
    return [
        (
            t.name,
            t.color,
            (int(t.pos[0]), int(t.pos[1])),
            (int(t.target[0]), int(t.target[1])),
        )
        for t in tracks
    ]


def record_frames(steps: int) -> list[list[tuple[str, str, tuple[int, int], tuple[int, int]]]]:
    tracks = _build_tracks()
    frames = [_snapshot(tracks)]
    for _ in range(steps):
        for track in tracks:
            track.step()
        frames.append(_snapshot(tracks))
    return frames


def render_gif(
    frames: list[list[tuple[str, str, tuple[int, int], tuple[int, int]]]],
    output: Path,
    *,
    grid_size: int = 10,
    fps: int = 4,
    dpi: int = 120,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    output.parent.mkdir(parents=True, exist_ok=True)

    route_points = [(1, 1), (8, 1), (8, 8), (1, 8), (1, 1)]

    def update(i: int) -> None:
        ax.clear()
        ax.set_facecolor("#f8fafc")
        ax.set_title(f"V1 routed gridworld | step {i}", fontsize=11)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.tick_params(labelsize=6)
        ax.grid(color="#cbd5e1", linewidth=0.8)

        xs = [p[1] for p in route_points]
        ys = [p[0] for p in route_points]
        ax.plot(xs, ys, "--", color="#64748b", linewidth=1.5, alpha=0.7)

        for row, col in route_points[:-1]:
            ax.scatter(col, row, s=80, marker="x", color="#475569", linewidth=2)

        for name, color, pos, target in frames[i]:
            row, col = pos
            target_row, target_col = target
            ax.scatter(target_col, target_row, s=120, facecolors="none", edgecolors=color, linewidth=2)
            ax.scatter(col, row, s=220, color=color, edgecolors="white", linewidth=2, zorder=5)
            ax.text(col, row, name[-1], ha="center", va="center", color="white", fontweight="bold")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a V1 routed gridworld replay GIF.")
    parser.add_argument("--output", default="replays/replay_v1.gif", help="GIF output path")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    frames = record_frames(args.steps)
    render_gif(frames, Path(args.output), fps=args.fps, dpi=args.dpi)
    print(f"wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
