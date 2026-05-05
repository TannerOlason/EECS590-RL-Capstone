"""Run five V3 reward/curriculum variants sequentially.

Each variant trains with the same curriculum and differs only in:
  - progress_reward_scale: lower means less incentive to rush/scroll
  - kill_reward_scale: higher means more incentive to clear enemies

Outputs go to:
    experiments/<datetime>_<prefix>_p<progress>_k<kill>/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402
from experiment_utils import timestamped_experiment_name  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

VARIANTS: tuple[tuple[str, float, float], ...] = (
    ("balanced", 0.70, 1.50),
    ("combat_lean", 0.50, 2.00),
    ("combat_mid", 0.35, 2.50),
    ("combat_high", 0.25, 3.00),
    ("combat_max", 0.15, 4.00),
)


def _fmt(x: float) -> str:
    return str(x).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="sac_500k_lockclear")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.device == "gpu" else args.device
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-v3")

    for label, progress_scale, kill_scale in VARIANTS:
        base_name = (
            f"{args.prefix}_{label}"
            f"_p{_fmt(progress_scale)}_k{_fmt(kill_scale)}"
        )
        name = timestamped_experiment_name(base_name)
        exp_dir = REPO_ROOT / "experiments" / name
        metrics_path = exp_dir / "eval_metrics.json"
        existing = sorted((REPO_ROOT / "experiments").glob(f"*_{base_name}/eval_metrics.json"))
        if args.skip_existing and (metrics_path.exists() or existing):
            print(f"[sweep] skipping {base_name}: eval_metrics.json exists")
            continue

        print()
        print(
            f"[sweep] training {name} "
            f"(progress_scale={progress_scale}, kill_scale={kill_scale})"
        )
        train_cmd = [
            str(PYTHON),
            "v3-continuous-space/scripts/train_v3.py",
            "--name", base_name,
            "--out-dir", str(exp_dir),
            "--total-timesteps", str(args.total_timesteps),
            "--seed", str(args.seed),
            "--device", device,
            "--progress-reward-scale", str(progress_scale),
            "--kill-reward-scale", str(kill_scale),
        ]
        subprocess.run(train_cmd, cwd=REPO_ROOT, env=env, check=True)

        print(f"[sweep] evaluating {name}")
        eval_cmd = [
            str(PYTHON),
            "v3-continuous-space/scripts/eval_v3.py",
            "--experiment", name,
            "--episodes", str(args.eval_episodes),
            "--max-steps", str(args.max_steps),
            "--render",
            "--progress-reward-scale", str(progress_scale),
            "--kill-reward-scale", str(kill_scale),
        ]
        subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env, check=True)

    print()
    print("[sweep] done")


if __name__ == "__main__":
    if not PYTHON.exists():
        raise SystemExit(f"Expected virtualenv Python at {PYTHON}")
    main()
