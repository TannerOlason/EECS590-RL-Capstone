"""Run model analysis on a saved MAPPO checkpoint.

Wraps highground/viz/model_analysis.py for standalone use from the rl/ project.

Usage::

    # Analyse the bundled phase-7 checkpoint
    python scripts/run_analysis.py

    # Point at a specific checkpoint
    python scripts/run_analysis.py --checkpoint models/mappo_phase7_policy.pt

    # Save all plots instead of showing interactively
    python scripts/run_analysis.py --save-dir plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse a MAPPO policy checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default=str(_ROOT / "models" / "mappo_phase7_policy.pt"),
        help="Path to the .pt policy checkpoint.",
    )
    p.add_argument(
        "--map",
        default="central_hill",
        help="Static map to run evaluation episodes on.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes for action/value statistics.",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save plots (PNG). If omitted, plots are shown interactively.",
    )
    p.add_argument(
        "--saliency",
        action="store_true",
        default=False,
        help="Run perturbation saliency analysis (slower, ~418 forward passes per state).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        print("Run training first: python scripts/train_mappo.py")
        sys.exit(1)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    print(f"Loading checkpoint: {checkpoint}")
    print(f"Map: {args.map}, episodes: {args.episodes}")

    from highground.viz.model_analysis import ModelAnalyzer

    analyzer = ModelAnalyzer(
        checkpoint_path=str(checkpoint),
        map_name=args.map,
    )

    print("Running evaluation episodes ...")
    analyzer.run_episodes(n_episodes=args.episodes)

    print("Generating plots ...")
    analyzer.plot_action_distribution(save_dir=save_dir)
    analyzer.plot_action_position_heatmap(save_dir=save_dir)
    analyzer.plot_value_heatmap(save_dir=save_dir)

    if args.saliency:
        print("Running perturbation saliency (slow) ...")
        analyzer.plot_perturbation_saliency(save_dir=save_dir)

    if save_dir:
        print(f"Plots saved to {save_dir}/")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
