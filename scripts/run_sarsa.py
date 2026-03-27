"""Train and evaluate SARSA(λ) on the navigation subtask.

Usage::

    # Default: SARSA(λ=0.8), 500 episodes
    python scripts/run_sarsa.py

    # Compare SARSA(λ) vs n-step SARSA side-by-side
    python scripts/run_sarsa.py --compare

    # Adjust hyperparameters
    python scripts/run_sarsa.py --episodes 1000 --lambda 0.9 --alpha 0.05

    # Save a learning-curve plot
    python scripts/run_sarsa.py --plot curves.png

This script intentionally stays lightweight: the goal is to demonstrate
the on-policy tabular algorithm and show its conceptual connection to
GAE (the advantage estimator used in MAPPO), not to optimise performance.

Conceptual lineage:  SARSA(λ)  →  GAE  →  MAPPO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SARSA(λ) on the navigation subtask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of training episodes.")
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.8,
                   help="Eligibility trace decay (λ). 0=TD(0), 1=MC.")
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    p.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    p.add_argument("--epsilon", type=float, default=1.0, help="Initial ε.")
    p.add_argument("--n-steps", type=int, default=4,
                   help="n for the SarsaN comparison variant.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--compare", action="store_true",
                   help="Run SARSA(λ), SARSA(n), and TD(0) side-by-side.")
    p.add_argument("--plot", default=None,
                   help="If set, save a learning-curve plot to this path.")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="Greedy evaluation episodes after training.")
    return p.parse_args()


def _evaluate_greedy(agent, env, n_episodes: int = 50, seed_offset: int = 1000) -> dict:
    """Run greedy policy for n_episodes; return summary stats."""
    wins = losses = timeouts = 0
    total_return = 0.0
    for i in range(n_episodes):
        env.seed = seed_offset + i
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = agent.greedy(state)
            state, reward, done, info = env.step(action)
            ep_return += reward
        total_return += ep_return
        outcome = info.get("outcome", "timeout")
        if outcome == "win":
            wins += 1
        elif outcome == "lose":
            losses += 1
        else:
            timeouts += 1
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "mean_return": total_return / n_episodes,
    }


def _smooth(values: list[float], window: int = 20) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def main() -> None:
    args = _parse_args()

    from classical.nav_env import SimpleNavigationEnv
    from classical.sarsa_lambda import SarsaLambda, SarsaN

    env = SimpleNavigationEnv(seed=args.seed)
    print(f"Navigation env — grid: 13×13, state space ≤ {env.state_space_size}, actions: {env.n_actions}")
    print(f"  (row × col × elevation_bucket × enemy_dist_bucket = 13×13×3×3 = {13*13*3*3})")
    print()

    results: dict[str, list[float]] = {}

    if args.compare:
        configs = [
            ("SARSA(λ=0.8)", SarsaLambda(
                n_actions=env.n_actions, lambda_=0.8,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            )),
            ("SARSA(λ=0.0) = TD(0)", SarsaLambda(
                n_actions=env.n_actions, lambda_=0.0,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            )),
            (f"SARSA(n={args.n_steps})", SarsaN(
                n_steps=args.n_steps, n_actions=env.n_actions,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            )),
        ]
    else:
        configs = [
            (f"SARSA(λ={args.lambda_})", SarsaLambda(
                n_actions=env.n_actions, lambda_=args.lambda_,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            )),
        ]

    for name, agent in configs:
        print(f"Training {name} for {args.episodes} episodes ...")
        env.seed = args.seed
        returns = agent.train(env, n_episodes=args.episodes, verbose=True)
        results[name] = returns

        # Greedy evaluation
        stats = _evaluate_greedy(agent, env, n_episodes=args.eval_episodes)
        print(f"  States visited: {agent.n_states_visited}")
        print(f"  Greedy eval ({args.eval_episodes} episodes):")
        print(f"    win_rate={stats['win_rate']:.1%}  "
              f"loss_rate={stats['loss_rate']:.1%}  "
              f"mean_return={stats['mean_return']:+.3f}")
        print()

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, rets in results.items():
            ax.plot(_smooth(rets), label=name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (smoothed, window=20)")
        ax.set_title("SARSA variants — Navigation subtask (High Ground 13×13)")
        ax.legend()
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"Saved plot to {args.plot}")

    # Print the lineage note
    print("─" * 60)
    print("Conceptual lineage:")
    print("  SARSA(λ) → GAE (Generalized Advantage Estimation) → MAPPO")
    print()
    print("  The λ decay here is identical in spirit to the λ in GAE:")
    print("  both weight earlier transitions by λ^k, trading bias (λ=0,")
    print("  fully bootstrapped) against variance (λ=1, Monte-Carlo).")
    print("  MAPPO's advantage: Â_t = Σ_k (γλ)^k δ_{t+k}")
    print("─" * 60)


if __name__ == "__main__":
    main()
