"""Train a MAPPO agent on the High Ground SRPG environment.

Thin wrapper around highground/training/benchmarl_train.py for standalone
use from the rl/ project.  All training logic lives in benchmarl_train.py;
this script sets the project root correctly and provides a minimal CLI.

Usage::

    # Default curriculum (7 phases)
    python scripts/train_mappo.py

    # Short smoke-test run (5k timesteps per phase)
    python scripts/train_mappo.py --timesteps 5000

    # Resume from a checkpoint
    python scripts/train_mappo.py --resume models/mappo_phase7_policy.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Delegate entirely to the existing training entrypoint.
if __name__ == "__main__":
    from highground.training.benchmarl_train import main
    main()
