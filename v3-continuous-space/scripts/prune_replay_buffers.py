"""Prune saved V3 SAC replay buffers under experiments/.

By default this only prints what it would delete. Pass --delete to remove older
buffers while keeping the newest N buffers and respecting a total size cap.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _buffers(root: Path) -> list[Path]:
    return sorted(root.glob("*/replay_buffer.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--keep-newest", type=int, default=2)
    parser.add_argument("--max-total-mb", type=float, default=4096.0)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    buffers = _buffers(args.experiments_dir)
    total_mb = sum(p.stat().st_size for p in buffers) / (1024 * 1024)
    print(f"[buffers] found {len(buffers)} buffers, total={total_mb:.1f} MB")

    kept = []
    delete_candidates = []
    running_mb = 0.0
    for i, path in enumerate(buffers):
        size_mb = path.stat().st_size / (1024 * 1024)
        must_keep = i < args.keep_newest
        fits_cap = running_mb + size_mb <= args.max_total_mb
        if must_keep or fits_cap:
            kept.append(path)
            running_mb += size_mb
        else:
            delete_candidates.append(path)

    print(f"[buffers] keeping {len(kept)} buffers, projected total={running_mb:.1f} MB")
    for path in delete_candidates:
        size_mb = path.stat().st_size / (1024 * 1024)
        action = "delete" if args.delete else "would delete"
        print(f"[buffers] {action}: {path} ({size_mb:.1f} MB)")
        if args.delete:
            path.unlink()

    if delete_candidates and not args.delete:
        print("[buffers] dry run only; pass --delete to remove files")


if __name__ == "__main__":
    main()
