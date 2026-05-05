"""Summarize and plot V3 experiment diagnostics.

Scans experiments/*/eval_metrics.json and writes:
    experiments/v3_summary.csv
    experiments/v3_summary.md
    experiments/v3_summary.png        (unless --no-plots)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIELDS = [
    "experiment",
    "mean_reward",
    "std_reward",
    "mean_scroll",
    "mean_max_scroll",
    "mean_max_leftmost_world_x",
    "mean_enemies_killed",
    "mean_squad_alive_at_end",
    "mean_right_move_rate",
    "mean_idle_rate",
    "mean_invalid_move_rate",
    "mean_shielded_invalid_move_rate",
    "mean_attack_intent_rate",
    "mean_focus_fire_hits",
    "mean_player_damage_total",
    "mean_visible_enemy_camp_penalty_total",
    "mean_scroll_locked_by_enemy_steps",
    "mean_scroll_locked_by_lagging_player_steps",
    "mean_squad_spread",
    "mean_max_squad_spread",
    "progress_reward_scale",
    "kill_reward_scale",
    "n_episodes",
]


def _load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}


def _rows(experiments_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(experiments_dir.glob("*/eval_metrics.json")):
        exp_dir = metrics_path.parent
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        config = _load_config(exp_dir)
        row = {"experiment": exp_dir.name}
        row.update(config)
        row.update(metrics)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    fields = list(DEFAULT_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields and key != "episodes":
                fields.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else str(value)


def _write_md(rows: list[dict], out_path: Path, top: int) -> None:
    ranked = sorted(rows, key=lambda r: float(r.get("mean_max_leftmost_world_x", r.get("mean_scroll", -1))), reverse=True)
    fields = DEFAULT_FIELDS[:15]
    lines = ["# V3 Experiment Summary", ""]
    lines.append(f"Showing top {min(top, len(ranked))} of {len(rows)} experiments, ranked by max leftmost progress.")
    lines.append("")
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in ranked[:top]:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))


def _plot(rows: list[dict], out_path: Path, top: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(rows, key=lambda r: float(r.get("mean_max_leftmost_world_x", r.get("mean_scroll", -1))), reverse=True)[:top]
    if not ranked:
        return
    names = [r["experiment"] for r in ranked]
    leftmost = [float(r.get("mean_max_leftmost_world_x", 0.0)) for r in ranked]
    kills = [float(r.get("mean_enemies_killed", 0.0)) for r in ranked]
    idle = [float(r.get("mean_idle_rate", 0.0)) for r in ranked]
    invalid = [float(r.get("mean_invalid_move_rate", 0.0)) for r in ranked]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plots = [
        (axes[0, 0], leftmost, "Mean max leftmost x"),
        (axes[0, 1], kills, "Mean enemies killed"),
        (axes[1, 0], idle, "Mean idle rate"),
        (axes[1, 1], invalid, "Mean invalid move rate"),
    ]
    y = list(range(len(names)))
    for ax, values, title in plots:
        ax.barh(y, values, color="#3b82f6")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = _rows(args.experiments_dir)
    if not rows:
        raise SystemExit(f"No eval_metrics.json files under {args.experiments_dir}")
    csv_path = args.experiments_dir / "v3_summary.csv"
    md_path = args.experiments_dir / "v3_summary.md"
    _write_csv(rows, csv_path)
    _write_md(rows, md_path, args.top)
    print(f"[analyze] wrote {csv_path}")
    print(f"[analyze] wrote {md_path}")
    if not args.no_plots:
        png_path = args.experiments_dir / "v3_summary.png"
        _plot(rows, png_path, args.top)
        print(f"[analyze] wrote {png_path}")


if __name__ == "__main__":
    main()
