"""Train a tiny Q-learning policy for the early V1 routed gridworld.

V1 was a small grid navigation prototype rather than the final SRPG engine.
This script keeps that version easy to demonstrate: one agent learns to move
between four route waypoints on a 10x10 grid.

Usage:
    python scripts/train_v1.py
    python scripts/train_v1.py --episodes 1000 --render-gif replay_v1_policy.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from replay_v1 import render_gif


GRID_SIZE = 10
ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)
ROUTE = [
    np.array([1, 1], dtype=int),
    np.array([8, 1], dtype=int),
    np.array([8, 8], dtype=int),
    np.array([1, 8], dtype=int),
]


def _state(pos: np.ndarray, target_idx: int) -> tuple[int, int, int]:
    return int(pos[0]), int(pos[1]), int(target_idx)


def train(
    *,
    episodes: int,
    max_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    q = np.zeros((GRID_SIZE, GRID_SIZE, len(ROUTE), len(ACTIONS)), dtype=np.float32)
    returns: list[float] = []
    completions = 0

    for ep in range(episodes):
        pos = np.array([0, 0], dtype=int)
        target_idx = 0
        total = 0.0
        eps = max(0.05, epsilon * (0.995 ** ep))

        for _ in range(max_steps):
            s = _state(pos, target_idx)
            if rng.random() < eps:
                action_idx = int(rng.integers(len(ACTIONS)))
            else:
                action_idx = int(np.argmax(q[s]))

            next_pos = np.clip(pos + ACTIONS[action_idx], 0, GRID_SIZE - 1)
            reward = -0.01
            next_target_idx = target_idx
            if np.array_equal(next_pos, ROUTE[target_idx]):
                reward = 1.0
                next_target_idx = (target_idx + 1) % len(ROUTE)
                if next_target_idx == 0:
                    completions += 1

            ns = _state(next_pos, next_target_idx)
            q[s][action_idx] += alpha * (reward + gamma * float(np.max(q[ns])) - q[s][action_idx])
            pos = next_pos
            target_idx = next_target_idx
            total += reward

        returns.append(total)

    return {"q": q, "returns": returns, "completions": completions}


def rollout_frames(q: np.ndarray, *, steps: int) -> list:
    pos = np.array([0, 0], dtype=int)
    target_idx = 0
    frames = []
    for _ in range(steps + 1):
        frames.append([
            (
                "agent1",
                "#2563eb",
                (int(pos[0]), int(pos[1])),
                (int(ROUTE[target_idx][0]), int(ROUTE[target_idx][1])),
            )
        ])
        s = _state(pos, target_idx)
        action_idx = int(np.argmax(q[s]))
        pos = np.clip(pos + ACTIONS[action_idx], 0, GRID_SIZE - 1)
        if np.array_equal(pos, ROUTE[target_idx]):
            target_idx = (target_idx + 1) % len(ROUTE)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the small V1 routed-grid Q learner.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-gif", type=str, default=None,
                        help="Optional GIF path, e.g. replays/train_v1_policy.gif")
    args = parser.parse_args()

    result = train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    returns = result["returns"]
    print(f"V1 Q-learning complete: episodes={args.episodes}")
    print(f"  mean_return_last_50={np.mean(returns[-50:]):+.3f}")
    print(f"  route_laps_completed={result['completions']}")

    if args.render_gif:
        frames = rollout_frames(result["q"], steps=48)
        render_gif(frames, Path(args.render_gif), grid_size=GRID_SIZE, fps=4)
        print(f"  saved GIF: {args.render_gif}")


if __name__ == "__main__":
    main()
