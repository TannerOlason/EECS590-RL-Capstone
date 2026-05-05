"""Sanity tests for the chunk generator + sliding-window grid."""

from __future__ import annotations

import numpy as np

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE, Terrain

from env.chunk_generator import generate_column
from env.infinite_grid import InfiniteGrid


def test_generate_column_shapes():
    rng = np.random.default_rng(0)
    terrain, elev, espawns, pspawns = generate_column(0, rng)
    assert terrain.shape == (GRID_SIZE,)
    assert elev.shape == (GRID_SIZE,)
    assert isinstance(espawns, list)
    assert isinstance(pspawns, list)
    assert terrain.dtype == np.int8
    assert elev.dtype == np.int8


def test_generate_column_no_enemies_in_first_columns():
    rng = np.random.default_rng(0)
    for x in range(0, 7):
        _, _, espawns, _ = generate_column(x, rng)
        assert len(espawns) == 0, f"unexpected enemy spawns at world_x={x}"


def test_potion_spawns_are_rare_but_appear():
    rng = np.random.default_rng(0)
    n_potions = 0
    for x in range(6, 600):
        _, _, _, pspawns = generate_column(x, rng)
        n_potions += len(pspawns)
    # ~3% of 594 columns ≈ 18 potions; allow wide band.
    assert 5 <= n_potions <= 60, f"unexpected potion count: {n_potions}"


def test_infinite_grid_initial_window():
    g = InfiniteGrid(seed=123)
    assert g.scroll_offset == 0
    assert g.grid.terrain.shape == (GRID_SIZE, GRID_SIZE)
    # Should have at least one walkable tile in every column.
    for c in range(GRID_SIZE):
        col = g.grid.terrain[:, c]
        assert (col != Terrain.UNCROSSABLE).any(), f"column {c} fully impassable"


def test_infinite_grid_scroll_shifts_columns():
    g = InfiniteGrid(seed=42)
    pre = g.grid.terrain[:, 5].copy()
    g.scroll_to(3)
    # The column that was at window-col 5 should now appear at window-col 2.
    assert (g.grid.terrain[:, 2] == pre).all()
    assert g.scroll_offset == 3


def test_infinite_grid_full_regen_when_delta_exceeds_window():
    g = InfiniteGrid(seed=7)
    g.scroll_to(GRID_SIZE * 5)
    assert g.scroll_offset == GRID_SIZE * 5
    assert g.grid.terrain.shape == (GRID_SIZE, GRID_SIZE)


def test_world_to_window_translation():
    g = InfiniteGrid(seed=0)
    g.scroll_to(10)
    assert g.world_to_window(10) == 0
    assert g.world_to_window(15) == 5
    assert g.window_to_world(3) == 13


def test_scroll_no_op_returns_pending_then_drains():
    g = InfiniteGrid(seed=99)
    enemies1, potions1 = g.scroll_to(0)
    enemies2, potions2 = g.scroll_to(0)
    assert enemies2 == [] and potions2 == []
    assert isinstance(enemies1, list) and isinstance(potions1, list)
