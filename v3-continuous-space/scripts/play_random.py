"""Smoke test: roll out the env with a random policy and (optionally) save a GIF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `_path_shim` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402

import numpy as np  # noqa: E402

from env.scrolling_env import ScrollingSquadEnv  # noqa: E402
from viz.render_scroll import render_frame  # noqa: E402
from viz.replay import save_gif  # noqa: E402


def run(episodes: int, max_steps: int, seed: int, gif_path: str | None) -> None:
    env = ScrollingSquadEnv(seed=seed)
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        frames = []
        ep_reward = 0.0
        steps = 0
        for t in range(max_steps):
            action = rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            steps += 1
            if gif_path and ep == 0 and t % 2 == 0:
                frames.append(render_frame(env))
            if term or trunc:
                break
        print(f"[ep {ep}] steps={steps}  reward={ep_reward:+.2f}  "
              f"scroll={info.get('scroll_offset', 0)}  alive={info.get('squad_alive', 0)}/3")

        if gif_path and ep == 0 and frames:
            save_gif(frames, gif_path, fps=8)
            print(f"  → saved GIF: {gif_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-gif", type=str, default=None)
    args = p.parse_args()
    run(args.episodes, args.max_steps, args.seed, args.render_gif)


if __name__ == "__main__":
    main()
