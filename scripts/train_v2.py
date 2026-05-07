"""Default V2 training entrypoint.

This is a friendly alias for the MAPPO/BenchMARL trainer. Use small frame
counts for a smoke test; the full curriculum is much slower.

Examples:
    python scripts/train_v2.py --help
    python scripts/train_v2.py --frames 5000 --map flat_open --no-tui
    python scripts/train_v2.py --curriculum --save-path models/mappo_v2
"""

from __future__ import annotations

from highground.training.benchmarl_train import main


if __name__ == "__main__":
    main()
