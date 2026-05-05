"""Matplotlib renderer for ScrollingSquadEnv. Returns RGB ndarray frames."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE, Terrain
from highground.engine.units import UnitClass


_TERRAIN_COLORS = {
    int(Terrain.NORMAL):      "#e8e3d3",   # light tan
    int(Terrain.ROUGH):       "#a89970",   # darker tan
    int(Terrain.UNCROSSABLE): "#3d3d3d",   # dark gray
}

# Single-char glyph per enemy class (visible to humans; the agent gets
# per-class channels in the observation tensor).
_ENEMY_GLYPH = {
    int(UnitClass.FIGHTER): "F",
    int(UnitClass.CHARGER): "C",
    int(UnitClass.RANGER):  "R",
    int(UnitClass.SIEGE):   "S",
}


def _draw_grid(ax, env) -> None:
    g = env.world.grid
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color = _TERRAIN_COLORS[int(g.terrain[r, c])]
            ax.add_patch(Rectangle((c, GRID_SIZE - 1 - r), 1, 1,
                                   facecolor=color, edgecolor="#888", linewidth=0.4))
            elev = int(g.elevation[r, c])
            if elev > 0:
                ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.15, f"{elev}",
                        ha="center", va="bottom", fontsize=6, color="#222", alpha=0.6)


def _draw_unit(ax, row_f: float, world_x_f: float, scroll_offset: int,
               color: str, hp_frac: float, label: str = "") -> None:
    win_col = world_x_f - scroll_offset
    if not (0.0 <= win_col <= GRID_SIZE):
        return
    cx = win_col + 0.5
    cy = (GRID_SIZE - 1 - row_f) + 0.5
    ax.add_patch(Circle((cx, cy), 0.35, facecolor=color, edgecolor="black", linewidth=1.0))
    # HP bar
    bar_w = 0.7
    bar_h = 0.08
    bx = cx - bar_w / 2
    by = cy + 0.4
    ax.add_patch(Rectangle((bx, by), bar_w, bar_h, facecolor="#222", edgecolor="none"))
    ax.add_patch(Rectangle((bx, by), bar_w * max(0.0, min(1.0, hp_frac)),
                           bar_h, facecolor="#3c8", edgecolor="none"))
    if label:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7, color="white")


def render_frame(env) -> np.ndarray:
    """Render the current env state as an RGB ndarray (H, W, 3) uint8."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=80)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    _draw_grid(ax, env)

    # Active player highlight ring.
    active = env.players[env.current_idx]
    if active.unit.alive:
        ac_col = active.world_x_f - env.world.scroll_offset
        ac_row = active.row_f
        if 0.0 <= ac_col <= GRID_SIZE:
            cx = ac_col + 0.5
            cy = (GRID_SIZE - 1 - ac_row) + 0.5
            ax.add_patch(Circle((cx, cy), 0.48, facecolor="none",
                                edgecolor="#ffcc00", linewidth=2.0))

    for i, p in enumerate(env.players):
        if not p.unit.alive:
            continue
        hp = p.unit.hp / max(1, p.unit.max_hp)
        _draw_unit(ax, p.row_f, p.world_x_f, env.world.scroll_offset,
                   color="#1f77b4", hp_frac=hp, label=str(i))

    for e in env.enemies:
        if not e.unit.alive:
            continue
        hp = e.unit.hp / max(1, e.unit.max_hp)
        glyph = _ENEMY_GLYPH.get(int(e.unit.unit_class), "E")
        _draw_unit(ax, e.row_f, e.world_x_f, env.world.scroll_offset,
                   color="#d62728", hp_frac=hp, label=glyph)

    # HP potion sprites: small green plus on top of the tile.
    for pot in env.potions:
        win_col = pot.col_world - env.world.scroll_offset
        if not (0 <= win_col < GRID_SIZE):
            continue
        cx = win_col + 0.5
        cy = (GRID_SIZE - 1 - pot.row) + 0.5
        ax.add_patch(Rectangle((cx - 0.18, cy - 0.06), 0.36, 0.12,
                               facecolor="#3c8", edgecolor="#063", linewidth=0.5))
        ax.add_patch(Rectangle((cx - 0.06, cy - 0.18), 0.12, 0.36,
                               facecolor="#3c8", edgecolor="#063", linewidth=0.5))

    ax.set_title(f"tick={env.macro_tick}  scroll={env.world.scroll_offset}  "
                 f"alive={sum(1 for p in env.players if p.unit.alive)}/3",
                 fontsize=9)

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=80)
    buf.seek(0)
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img
