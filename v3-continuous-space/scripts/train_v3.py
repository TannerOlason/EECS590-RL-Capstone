"""Thin wrapper around training/train_sac.py to keep the entry point with the others."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402

from training.train_sac import main  # noqa: E402


if __name__ == "__main__":
    main()
