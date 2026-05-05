"""Helpers for naming and resolving V3 experiment directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}_")


def timestamped_experiment_name(name: str) -> str:
    """Prefix an experiment name with local run datetime unless already prefixed."""
    if TIMESTAMP_RE.match(name):
        return name
    return f"{datetime.now().strftime(TIMESTAMP_FORMAT)}_{name}"


def resolve_experiment_dir(experiments_root: Path, name: str) -> Path:
    """Resolve exact experiment name, or latest timestamp-prefixed match.

    This lets commands like ``--experiment sac_50k_quick`` keep working after
    new runs are stored as ``YYYYMMDD-HHMMSS_sac_50k_quick``.
    """
    exact = experiments_root / name
    if exact.exists():
        return exact

    matches = sorted(
        p for p in experiments_root.glob(f"*_{name}")
        if p.is_dir() and TIMESTAMP_RE.match(p.name)
    )
    if matches:
        return matches[-1]
    return exact
