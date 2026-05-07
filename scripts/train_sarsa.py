"""Friendly default entrypoint for the SARSA baseline.

This delegates to ``run_sarsa.py`` and adds a default plot path when the user
does not pass ``--plot``.

Usage:
    python scripts/train_sarsa.py
    python scripts/train_sarsa.py --compare --episodes 1000
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    target = repo / "scripts" / "run_sarsa.py"
    argv = sys.argv[1:]
    if "--plot" not in argv:
        (repo / "replays").mkdir(exist_ok=True)
        argv += ["--plot", str(repo / "replays" / "sarsa_curves.png")]
    sys.argv = [str(target), *argv]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
