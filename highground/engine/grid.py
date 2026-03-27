"""Grid and terrain representation for the 13x13 SRPG map."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

GRID_SIZE = 13


class Terrain(IntEnum):
    """Terrain types on the grid."""

    NORMAL = 0
    ROUGH = 1       # +1 movement cost
    UNCROSSABLE = 2  # Cannot be entered


class Grid:
    """A 13x13 SRPG map with terrain types and elevation layers.

    Attributes:
        terrain: (13, 13) array of Terrain values.
        elevation: (13, 13) array of elevation levels (0, 1, or 2).
    """

    def __init__(
        self,
        terrain: np.ndarray | None = None,
        elevation: np.ndarray | None = None,
    ) -> None:
        if terrain is not None:
            assert terrain.shape == (GRID_SIZE, GRID_SIZE)
            self.terrain = terrain.astype(np.int8)
        else:
            self.terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        if elevation is not None:
            assert elevation.shape == (GRID_SIZE, GRID_SIZE)
            self.elevation = elevation.astype(np.int8)
        else:
            self.elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    def is_walkable(self, row: int, col: int) -> bool:
        """Return True if the tile can be entered (not uncrossable, in bounds)."""
        if not self.in_bounds(row, col):
            return False
        return int(self.terrain[row, col]) != Terrain.UNCROSSABLE

    def is_rough(self, row: int, col: int) -> bool:
        return int(self.terrain[row, col]) == Terrain.ROUGH

    def move_cost(self, row: int, col: int) -> int:
        """Movement cost to enter this tile. 1 for normal, 2 for rough."""
        if not self.is_walkable(row, col):
            return 999  # sentinel — should never be used
        return 2 if self.is_rough(row, col) else 1

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE

    def get_elevation(self, row: int, col: int) -> int:
        return int(self.elevation[row, col])

    def copy(self) -> Grid:
        return Grid(self.terrain.copy(), self.elevation.copy())
