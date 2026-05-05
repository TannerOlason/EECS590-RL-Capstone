"""Pytest fixture: ensure v3-continuous-space/ and repo root are on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_REPO_ROOT = _HERE.parent

for p in (_HERE, _REPO_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
