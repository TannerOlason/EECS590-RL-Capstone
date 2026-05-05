"""Sliding-window grid that procedurally extends to the right.

Internally holds a V2-compatible 13x13 Grid as the "visible window". The
window is anchored to a `scroll_offset` (the absolute world_x of window
column 0). When `scroll_to(target_offset)` is called with a larger value,
the underlying terrain/elevation arrays are shifted left by the delta and
new columns are filled in from `chunk_generator.generate_column`.

Pending enemy spawns from generated columns are queued in `pending_spawns`
and consumed by the env when scrolling occurs.
"""

from __future__ import annotations

import numpy as np

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE, Grid

from env.chunk_generator import EnemySpawn, PotionSpawn, generate_column


class InfiniteGrid:
    """A Grid wrapper that scrolls right and procedurally generates columns.

    Tile lookups use *world* column indices. Window column = world_col - scroll_offset.
    Tiles whose window column is outside [0, GRID_SIZE) are not addressable; the env
    treats anything with window_col < 0 as "scrolled off-screen left".
    """

    def __init__(self, seed: int = 0, column_config: dict | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._column_config = dict(column_config or {})
        self.scroll_offset: int = 0  # world_x corresponding to window col 0
        self.grid = Grid()
        self.pending_enemy_spawns: list[EnemySpawn] = []
        self.pending_potion_spawns: list[PotionSpawn] = []

        # Pre-fill the entire visible window with generated columns.
        for window_col in range(GRID_SIZE):
            world_x = window_col
            terrain_col, elev_col, espawns, pspawns = generate_column(
                world_x,
                self._rng,
                **self._column_config,
            )
            self.grid.terrain[:, window_col] = terrain_col
            self.grid.elevation[:, window_col] = elev_col
            self.pending_enemy_spawns.extend(espawns)
            self.pending_potion_spawns.extend(pspawns)

    # ── Coordinate translation ────────────────────────────────────────────

    def world_to_window(self, world_col: int) -> int:
        return world_col - self.scroll_offset

    def window_to_world(self, window_col: int) -> int:
        return window_col + self.scroll_offset

    @property
    def world_right_edge(self) -> int:
        """World_x of the rightmost column currently in the window (inclusive)."""
        return self.scroll_offset + GRID_SIZE - 1

    # ── Scrolling ─────────────────────────────────────────────────────────

    def scroll_to(self, new_offset: int) -> tuple[list[EnemySpawn], list[PotionSpawn]]:
        """Slide the window so that window-col 0 corresponds to `new_offset`.

        Generates and returns (enemy_spawns, potion_spawns) for any newly
        exposed columns. Spawns generated during __init__ are returned on
        the first call as well so the env gets initial spawns at reset.
        """
        delta = new_offset - self.scroll_offset

        def _drain() -> tuple[list[EnemySpawn], list[PotionSpawn]]:
            es, ps = self.pending_enemy_spawns, self.pending_potion_spawns
            self.pending_enemy_spawns = []
            self.pending_potion_spawns = []
            return es, ps

        if delta <= 0:
            return _drain()

        # Shift terrain & elevation left by `delta` columns; new columns on right.
        if delta >= GRID_SIZE:
            self.scroll_offset = new_offset
            for window_col in range(GRID_SIZE):
                world_x = self.window_to_world(window_col)
                terrain_col, elev_col, espawns, pspawns = generate_column(
                    world_x,
                    self._rng,
                    **self._column_config,
                )
                self.grid.terrain[:, window_col] = terrain_col
                self.grid.elevation[:, window_col] = elev_col
                self.pending_enemy_spawns.extend(espawns)
                self.pending_potion_spawns.extend(pspawns)
        else:
            self.grid.terrain[:, :-delta] = self.grid.terrain[:, delta:]
            self.grid.elevation[:, :-delta] = self.grid.elevation[:, delta:]
            self.scroll_offset = new_offset
            for k in range(delta):
                window_col = GRID_SIZE - delta + k
                world_x = self.window_to_world(window_col)
                terrain_col, elev_col, espawns, pspawns = generate_column(
                    world_x,
                    self._rng,
                    **self._column_config,
                )
                self.grid.terrain[:, window_col] = terrain_col
                self.grid.elevation[:, window_col] = elev_col
                self.pending_enemy_spawns.extend(espawns)
                self.pending_potion_spawns.extend(pspawns)

        return _drain()

    # ── World-coordinate tile queries (delegate to V2 Grid by translating) ─

    def is_walkable_world(self, row: int, world_col: int) -> bool:
        wcol = self.world_to_window(world_col)
        if not (0 <= wcol < GRID_SIZE):
            return False
        return self.grid.is_walkable(row, wcol)

    def is_in_window(self, row: int, world_col: int) -> bool:
        wcol = self.world_to_window(world_col)
        return self.grid.in_bounds(row, wcol)

    def get_elevation_world(self, row: int, world_col: int) -> int:
        wcol = self.world_to_window(world_col)
        if not (0 <= wcol < GRID_SIZE):
            return 0
        return self.grid.get_elevation(row, wcol)
