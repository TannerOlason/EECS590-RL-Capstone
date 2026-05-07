"""Generate a V3 replay GIF.

By default this searches ``experiments/`` for the newest trained V3 checkpoint,
preferring ``best_model.zip`` over ``sac_final.zip``. If no checkpoint exists,
it falls back to a random-policy rollout.

Examples:
    python scripts/replay_v3.py
    python scripts/replay_v3.py --experiment sac_50k_quick --output replay_v3.gif
    python scripts/replay_v3.py --random
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-highground")


def _run_v3_script(script_name: str, argv: list[str]) -> None:
    repo = Path(__file__).resolve().parents[1]
    target = repo / "v3-continuous-space" / "scripts" / script_name
    sys.argv = [str(target), *argv]
    runpy.run_path(str(target), run_name="__main__")


def _latest_checkpoint(repo: Path) -> tuple[Path, Path] | None:
    experiments = repo / "experiments"
    if not experiments.exists():
        return None

    candidates: list[tuple[float, int, Path, Path]] = []
    for exp_dir in experiments.iterdir():
        if not exp_dir.is_dir():
            continue
        best = exp_dir / "best_model.zip"
        final = exp_dir / "sac_final.zip"
        if best.exists():
            candidates.append((best.stat().st_mtime, 1, best, exp_dir))
        elif final.exists():
            candidates.append((final.stat().st_mtime, 0, final, exp_dir))

    if not candidates:
        return None
    _, _, checkpoint, exp_dir = max(candidates, key=lambda item: (item[0], item[1]))
    return checkpoint, exp_dir


def _run_trained_eval(args: argparse.Namespace, *, checkpoint: Path | None = None) -> None:
    eval_out_dir = Path(args.out_dir) if args.out_dir else Path(args.output).resolve().parent
    argv = [
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--max-steps", str(args.max_steps),
        "--render",
    ]
    if args.experiment:
        argv += ["--experiment", args.experiment]
    if checkpoint is not None:
        argv += ["--checkpoint", str(checkpoint)]
    elif args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    if args.out_dir:
        argv += ["--out-dir", args.out_dir]
    else:
        argv += ["--out-dir", str(eval_out_dir)]

    _run_v3_script("eval_v3.py", argv)

    generated = eval_out_dir / "replay.gif"
    requested = Path(args.output)
    if generated.exists() and generated.resolve() != requested.resolve():
        requested.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, requested)
        print(f"copied trained-policy V3 replay to {requested}")
    else:
        print(f"trained-policy V3 replay written to {generated}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a V3 replay GIF.")
    parser.add_argument("--output", default="replays/replay_v3.gif", help="GIF output path")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Evaluate a trained experiment instead of random play.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate a direct SAC checkpoint instead of random play.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir for trained-policy eval artifacts.")
    parser.add_argument("--random", action="store_true",
                        help="Force a random-policy replay instead of using the latest checkpoint.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.experiment or args.checkpoint:
        _run_trained_eval(args)
        return

    if not args.random:
        latest = _latest_checkpoint(repo)
        if latest is not None:
            checkpoint, exp_dir = latest
            print(f"using latest V3 checkpoint: {checkpoint}")
            print(f"  experiment dir: {exp_dir}")
            _run_trained_eval(args, checkpoint=checkpoint)
            return
        print("[warn] no V3 checkpoint found in experiments/; using random policy")

    _run_v3_script(
        "play_random.py",
        [
            "--episodes", str(args.episodes),
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
            "--render-gif", args.output,
        ],
    )


if __name__ == "__main__":
    main()
