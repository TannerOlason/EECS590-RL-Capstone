"""Train SARSA briefly and render a greedy-policy replay GIF.

Usage:
    python scripts/replay_sarsa.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-highground")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from classical.nav_env import GRID, SimpleNavigationEnv
from classical.sarsa_lambda import SarsaLambda


def _record(agent: SarsaLambda, env: SimpleNavigationEnv, max_steps: int) -> list[dict]:
    frames = []
    state = env.reset()
    done = False
    for step in range(max_steps + 1):
        frames.append({
            "step": step,
            "agent": env.agent_pos,
            "enemy": env.enemy_pos,
            "goals": list(env.goal_tiles),
            "elevation": env.elevation.copy(),
        })
        if done:
            break
        action = agent.greedy(state)
        state, _, done, _ = env.step(action)
    return frames


def _pad_frames(frames: list[dict], min_frames: int) -> list[dict]:
    if not frames:
        return frames
    while len(frames) < min_frames:
        last = dict(frames[-1])
        last["step"] = frames[-1]["step"]
        frames.append(last)
    return frames


def _rollout_length(agent: SarsaLambda, env: SimpleNavigationEnv, seed: int, max_steps: int) -> tuple[int, str]:
    env.seed = seed
    state = env.reset()
    done = False
    info = {"outcome": "timeout"}
    steps = 0
    while not done and steps < max_steps:
        action = agent.greedy(state)
        state, _, done, info = env.step(action)
        steps += 1
    return steps, info.get("outcome", "timeout")


def _choose_replay_seed(agent: SarsaLambda, env: SimpleNavigationEnv, base_seed: int, max_steps: int) -> int:
    best_seed = base_seed
    best_score = (-1, -1)
    for offset in range(80):
        seed = base_seed + offset
        steps, outcome = _rollout_length(agent, env, seed, max_steps)
        win_bonus = 1 if outcome == "win" else 0
        score = (win_bonus, steps)
        if score > best_score:
            best_score = score
            best_seed = seed
    return best_seed


def _render(frames: list[dict], output: Path, fps: int, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i: int) -> None:
        frame = frames[i]
        ax.clear()
        ax.imshow(frame["elevation"], cmap="YlGnBu", vmin=0, vmax=2, origin="upper")
        ax.set_title(f"SARSA navigation replay | step {frame['step']}", fontsize=11)
        ax.set_xlim(-0.5, GRID - 0.5)
        ax.set_ylim(GRID - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(GRID))
        ax.set_yticks(range(GRID))
        ax.tick_params(labelsize=6)
        ax.grid(color="black", linewidth=0.3, alpha=0.35)

        for row, col in frame["goals"]:
            ax.scatter(col, row, marker="*", s=160, color="#facc15", edgecolors="black", zorder=3)

        ar, ac = frame["agent"]
        er, ec = frame["enemy"]
        ax.scatter(ac, ar, s=220, color="#2563eb", edgecolors="white", linewidth=2, zorder=5)
        ax.text(ac, ar, "A", ha="center", va="center", color="white", fontweight="bold", zorder=6)
        ax.scatter(ec, er, s=220, color="#dc2626", edgecolors="white", linewidth=2, zorder=5)
        ax.text(ec, er, "E", ha="center", va="center", color="white", fontweight="bold", zorder=6)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SARSA navigation replay GIF.")
    parser.add_argument("--output", default="replays/replay_sarsa.gif")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--min-frames", type=int, default=24,
                        help="Hold the final state so short episodes still produce a readable GIF.")
    args = parser.parse_args()

    env = SimpleNavigationEnv(seed=args.seed)
    agent = SarsaLambda(n_actions=env.n_actions)
    agent.train(env, n_episodes=args.episodes, verbose=False)
    replay_seed = _choose_replay_seed(agent, env, args.seed + 1000, args.max_steps)
    env.seed = replay_seed
    frames = _pad_frames(_record(agent, env, args.max_steps), args.min_frames)
    _render(frames, Path(args.output), args.fps, args.dpi)
    print(f"wrote {args.output} ({len(frames)} frames, seed={replay_seed})")


if __name__ == "__main__":
    main()
