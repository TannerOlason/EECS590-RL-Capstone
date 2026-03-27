"""Render a Grid as a matplotlib figure with terrain, elevation, and units.

Usage:
    from highground.viz.render_map import render_map
    fig = render_map(grid)
    fig = render_map(grid, units=game_state.units)
    fig.savefig("map.png", dpi=150)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from highground.engine.grid import GRID_SIZE, Grid, Terrain
from highground.engine.units import TEAM_A, TEAM_B, Unit, UnitClass

# ── Color palettes ────────────────────────────────────────────────────

# Terrain base colors (RGBA)
_TERRAIN_COLORS = {
    Terrain.NORMAL: np.array([0.76, 0.87, 0.56, 1.0]),      # light green
    Terrain.ROUGH: np.array([0.72, 0.65, 0.42, 1.0]),        # tan/brown
    Terrain.UNCROSSABLE: np.array([0.35, 0.35, 0.38, 1.0]),  # dark gray
}

# Elevation shading: darken factor per level
_ELEV_DARKEN = {0: 0.0, 1: -0.08, 2: -0.18}

# Team colors
_TEAM_COLORS = {TEAM_A: "#3b82f6", TEAM_B: "#ef4444"}  # blue, red

# Unit class markers
_CLASS_MARKERS = {
    UnitClass.FIGHTER: "s",   # square
    UnitClass.CHARGER: "^",   # triangle
    UnitClass.RANGER: "D",    # diamond
    UnitClass.SIEGE: "p",     # pentagon
}

_CLASS_LABELS = {
    UnitClass.FIGHTER: "F",
    UnitClass.CHARGER: "C",
    UnitClass.RANGER: "R",
    UnitClass.SIEGE: "S",
}


def _terrain_image(grid: Grid) -> np.ndarray:
    """Build an RGBA image (GRID_SIZE, GRID_SIZE, 4) from terrain + elevation."""
    img = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            base = _TERRAIN_COLORS[Terrain(grid.terrain[r, c])].copy()
            elev = int(grid.elevation[r, c])
            factor = _ELEV_DARKEN.get(elev, 0.0)
            base[:3] = np.clip(base[:3] + factor, 0.0, 1.0)
            img[r, c] = base
    return img


def render_map(
    grid: Grid,
    *,
    units: list[Unit] | None = None,
    spawns_a: list[tuple[int, int]] | None = None,
    spawns_b: list[tuple[int, int]] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_elevation_text: bool = True,
) -> plt.Figure:
    """Render a map grid with optional units and spawn markers.

    Args:
        grid: The SRPG Grid to render.
        units: Optional list of units to draw on the map.
        spawns_a: Optional Team A spawn positions.
        spawns_b: Optional Team B spawn positions.
        title: Optional title for the figure.
        ax: Optional existing Axes to draw on.
        show_elevation_text: Show elevation numbers on tiles.

    Returns:
        The matplotlib Figure (useful for saving).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        fig = ax.figure

    # Draw terrain + elevation as background image
    img = _terrain_image(grid)
    ax.imshow(img, origin="upper", extent=(-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5))

    # Grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.3, alpha=0.3)
        ax.axvline(i - 0.5, color="black", linewidth=0.3, alpha=0.3)

    # Elevation text
    if show_elevation_text:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                elev = int(grid.elevation[r, c])
                if elev > 0 and grid.terrain[r, c] != Terrain.UNCROSSABLE:
                    ax.text(c, r, str(elev), ha="center", va="center",
                            fontsize=6, color="white", alpha=0.7, fontweight="bold")

    # Spawn markers
    if spawns_a:
        for r, c in spawns_a:
            ax.plot(c, r, "x", color=_TEAM_COLORS[TEAM_A], markersize=8,
                    markeredgewidth=2, alpha=0.6)
    if spawns_b:
        for r, c in spawns_b:
            ax.plot(c, r, "x", color=_TEAM_COLORS[TEAM_B], markersize=8,
                    markeredgewidth=2, alpha=0.6)

    # Units
    if units:
        for u in units:
            if not u.alive:
                continue
            color = _TEAM_COLORS[u.team]
            marker = _CLASS_MARKERS.get(UnitClass(u.unit_class), "o")
            ax.plot(u.col, u.row, marker, color=color, markersize=14,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            # HP bar below unit
            hp_frac = u.hp / u.max_hp
            bar_w = 0.7
            bar_h = 0.12
            bar_x = u.col - bar_w / 2
            bar_y = u.row + 0.35
            ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h,
                                   facecolor="black", alpha=0.5, zorder=6))
            ax.add_patch(Rectangle((bar_x, bar_y), bar_w * hp_frac, bar_h,
                                   facecolor="#22c55e", alpha=0.9, zorder=7))

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig


def render_map_comparison(
    grids: list[Grid],
    labels: list[str] | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Render multiple maps side by side for comparison."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    if labels is None:
        labels = [f"Map {i}" for i in range(n)]

    for ax, grid, label in zip(axes, grids, labels):
        render_map(grid, ax=ax, title=label, show_elevation_text=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig
