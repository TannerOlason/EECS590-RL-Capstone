"""Random-policy baseline. Writes the same artefacts that eval_v3.py produces
so it can be compared head-to-head with trained SAC checkpoints.

    experiments/<name>/eval_metrics.json
    experiments/<name>/replay.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402

import numpy as np  # noqa: E402

from experiment_utils import timestamped_experiment_name  # noqa: E402
from env.scrolling_env import ScrollingSquadEnv  # noqa: E402
from viz.render_scroll import render_frame  # noqa: E402
from viz.replay import save_gif  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="baseline_random",
                   help="Experiment base name; writes to "
                        "experiments/<datetime>_<name>/.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    exp_dir = REPO_ROOT / "experiments" / timestamped_experiment_name(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    env = ScrollingSquadEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    per_episode: list[dict] = []
    frames_first_ep: list = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_r = 0.0
        steps = 0
        invalid_move_count = 0
        shielded_invalid_move_count = 0
        invalid_move_penalty_total = 0.0
        whiffed_attack_penalty_total = 0.0
        for t in range(args.max_steps):
            action = rng.uniform(-1, 1, size=(3,)).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            ep_r += reward
            steps += 1
            invalid_move_count += int(info.get("invalid_move_count", 0))
            shielded_invalid_move_count += int(info.get("shielded_invalid_move_count", 0))
            invalid_move_penalty_total += float(info.get("invalid_move_penalty", 0.0))
            whiffed_attack_penalty_total += float(info.get("whiffed_attack_penalty", 0.0))
            if args.render and ep == 0 and t % 2 == 0:
                frames_first_ep.append(render_frame(env))
            if term or trunc:
                break
        ep_kills = sum(1 for e in env.enemies if not e.unit.alive)
        per_episode.append({
            "episode": ep,
            "reward": float(ep_r),
            "steps": steps,
            "scroll_reached": int(info.get("scroll_offset", 0)),
            "squad_alive_at_end": int(info.get("squad_alive", 0)),
            "enemies_killed": int(ep_kills),
            "invalid_move_count": int(invalid_move_count),
            "invalid_move_rate": float(invalid_move_count / max(1, steps)),
            "shielded_invalid_move_count": int(shielded_invalid_move_count),
            "shielded_invalid_move_rate": float(shielded_invalid_move_count / max(1, steps)),
            "invalid_move_penalty_total": float(invalid_move_penalty_total),
            "whiffed_attack_penalty_total": float(whiffed_attack_penalty_total),
            "terminated": bool(term),
            "truncated": bool(trunc),
        })
        print(f"[ep {ep}] reward={ep_r:+.2f}  steps={steps}  "
              f"scroll={per_episode[-1]['scroll_reached']}  "
              f"alive={per_episode[-1]['squad_alive_at_end']}/3  "
              f"kills={per_episode[-1]['enemies_killed']}")

    arr = lambda k: np.array([e[k] for e in per_episode], dtype=float)  # noqa: E731
    summary = {
        "policy": "uniform_random_action",
        "experiment": exp_dir.name,
        "requested_name": args.name,
        "n_episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "mean_reward": float(arr("reward").mean()),
        "std_reward":  float(arr("reward").std()),
        "mean_steps":  float(arr("steps").mean()),
        "mean_scroll": float(arr("scroll_reached").mean()),
        "mean_squad_alive_at_end": float(arr("squad_alive_at_end").mean()),
        "mean_enemies_killed": float(arr("enemies_killed").mean()),
        "mean_invalid_move_count": float(arr("invalid_move_count").mean()),
        "mean_invalid_move_rate": float(arr("invalid_move_rate").mean()),
        "mean_shielded_invalid_move_count": float(arr("shielded_invalid_move_count").mean()),
        "mean_shielded_invalid_move_rate": float(arr("shielded_invalid_move_rate").mean()),
        "mean_invalid_move_penalty_total": float(arr("invalid_move_penalty_total").mean()),
        "mean_whiffed_attack_penalty_total": float(arr("whiffed_attack_penalty_total").mean()),
        "episodes": per_episode,
    }
    (exp_dir / "eval_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[baseline] wrote {exp_dir / 'eval_metrics.json'}")

    if args.render and frames_first_ep:
        save_gif(frames_first_ep, exp_dir / "replay.gif", fps=8)
        print(f"[baseline] wrote {exp_dir / 'replay.gif'}")

    print()
    print(f"Mean episode reward     : {summary['mean_reward']:+.2f}  ± {summary['std_reward']:.2f}")
    print(f"Mean episode length     : {summary['mean_steps']:.1f}")
    print(f"Mean scroll reached     : {summary['mean_scroll']:.1f}")
    print(f"Mean enemies killed     : {summary['mean_enemies_killed']:.2f}")
    print(f"Mean invalid move rate  : {summary['mean_invalid_move_rate']:.2%}")


if __name__ == "__main__":
    main()
