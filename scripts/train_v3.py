"""Default V3 SAC training entrypoint.

Delegates to ``v3-continuous-space/scripts/train_v3.py`` so the real training
logic stays in the V3 module.

Examples:
    python scripts/train_v3.py --help
    python scripts/train_v3.py --name sac_50k_quick --total-timesteps 50000 --quick
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    target = repo / "v3-continuous-space" / "scripts" / "train_v3.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
