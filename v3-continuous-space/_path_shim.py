"""Adds the repo root to sys.path so V2 modules (highground.*) import cleanly.

Every script/module in v3-continuous-space that needs V2 code does:
    import _path_shim  # noqa: F401
before importing highground.*. The folder name has a hyphen so it isn't
a Python package itself; we run scripts via `python v3-continuous-space/scripts/...`
and let this shim splice the repo root onto sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V3_ROOT = Path(__file__).resolve().parent

for p in (_REPO_ROOT, _V3_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
