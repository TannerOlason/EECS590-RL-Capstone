"""Pre-computed tile index and Dijkstra pathfinding for the LLM steering pipeline.

TileIndex pre-computes walkable and elevation-2 tile sets from a Grid, and
provides a Dijkstra BFS that returns the first-step direction delta toward
the nearest reachable target tile.

Usage::

    from highground.llm.tile_index import TileIndex
    idx = TileIndex(grid)
    delta, cost = idx.dijkstra_first_step((6, 1), idx.elevation2_tiles)
"""

from __future__ import annotations

import heapq

from highground.engine.grid import Grid, GRID_SIZE

# Eight-directional movement deltas: N, S, E, W, NE, NW, SE, SW
ALL_DELTAS = [
    (-1,  0),   # N
    ( 1,  0),   # S
    ( 0,  1),   # E
    ( 0, -1),   # W
    (-1,  1),   # NE
    (-1, -1),   # NW
    ( 1,  1),   # SE
    ( 1, -1),   # SW
]


class TileIndex:
    """Pre-computed spatial indices and BFS navigation for a single grid.

    Parameters
    ----------
    grid:
        The Grid object to index.  Tile sets are computed once at construction.
    """

    def __init__(self, grid: Grid) -> None:
        self._grid = grid

        # Pre-computed frozensets for fast membership tests
        self.elevation2_tiles: frozenset[tuple[int, int]] = frozenset(
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if grid.is_walkable(r, c) and grid.get_elevation(r, c) == 2
        )
        self.walkable_tiles: frozenset[tuple[int, int]] = frozenset(
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if grid.is_walkable(r, c)
        )

    def dijkstra_first_step(
        self,
        start: tuple[int, int],
        targets: frozenset[tuple[int, int]] | set[tuple[int, int]],
    ) -> tuple[tuple[int, int] | None, int]:
        """Find the cheapest path to any target tile and return the first step.

        Parameters
        ----------
        start:
            (row, col) starting position.
        targets:
            Set of (row, col) goal tiles.

        Returns
        -------
        (first_step_delta, cost)
            ``first_step_delta`` is a (dr, dc) tuple from ``ALL_DELTAS`` indicating
            the direction of the first move on the optimal path, or ``None`` if
            already on a target or no path exists.
            ``cost`` is the total Dijkstra path cost; 999999 if unreachable.
        """
        if not targets:
            return None, 999999
        if start in targets:
            return None, 0  # already at goal

        INF = 999999
        dist: dict[tuple[int, int], int] = {start: 0}
        # Heap entries: (cost, row, col, first_step_dr, first_step_dc)
        heap: list[tuple[int, int, int, int, int]] = []

        # Seed the heap with all valid first steps from start
        for dr, dc in ALL_DELTAS:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self._grid.is_walkable(nr, nc):
                cost = self._grid.move_cost(nr, nc)
                heapq.heappush(heap, (cost, nr, nc, dr, dc))

        best_cost = INF
        best_delta: tuple[int, int] | None = None

        while heap:
            cost, r, c, fdr, fdc = heapq.heappop(heap)
            if cost > dist.get((r, c), INF):
                continue
            dist[(r, c)] = cost

            if (r, c) in targets:
                if cost < best_cost:
                    best_cost = cost
                    best_delta = (fdr, fdc)
                break  # Dijkstra guarantees first pop is cheapest

            for dr, dc in ALL_DELTAS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self._grid.is_walkable(nr, nc):
                    new_cost = cost + self._grid.move_cost(nr, nc)
                    if new_cost < dist.get((nr, nc), INF):
                        dist[(nr, nc)] = new_cost
                        heapq.heappush(heap, (new_cost, nr, nc, fdr, fdc))

        return best_delta, best_cost

    def resolve_targets(
        self,
        objective: "ObjectiveType",  # noqa: F821
        unit_pos: tuple[int, int],
        game_state: "GameState",      # noqa: F821
        team_id: int,
    ) -> frozenset[tuple[int, int]]:
        """Translate an ObjectiveType into a set of target tiles for one unit.

        Parameters
        ----------
        objective:
            The high-level goal from :class:`~highground.llm.models.ObjectiveType`.
        unit_pos:
            Actual (row, col) of the unit from ``game_state.units[i].pos``.
        game_state:
            Live game state — used to read enemy/ally positions.
        team_id:
            The acting unit's team.

        Returns
        -------
        frozenset[tuple[int, int]]
            Set of target tiles.  Empty set means no reachable goal; the caller
            should fall back to a default bias.
        """
        from highground.llm.models import ObjectiveType  # local to avoid circular import

        allies  = [u for u in game_state.units if u.team == team_id and u.alive]
        enemies = [u for u in game_state.units if u.team != team_id and u.alive]

        if objective == ObjectiveType.OCCUPY_HIGH_GROUND:
            occupied = {u.pos for u in game_state.units if u.alive}
            free_high = self.elevation2_tiles - occupied
            return free_high if free_high else self.elevation2_tiles

        elif objective == ObjectiveType.FLANK_SOUTH:
            if enemies:
                enemy_centroid_row = sum(e.row for e in enemies) / len(enemies)
                threshold = int(enemy_centroid_row) + 2
            else:
                threshold = 8
            south_tiles = frozenset(
                (r, c) for (r, c) in self.walkable_tiles if r >= threshold
            )
            return south_tiles if south_tiles else self.walkable_tiles

        elif objective == ObjectiveType.FLANK_NORTH:
            if enemies:
                enemy_centroid_row = sum(e.row for e in enemies) / len(enemies)
                threshold = int(enemy_centroid_row) - 2
            else:
                threshold = 4
            north_tiles = frozenset(
                (r, c) for (r, c) in self.walkable_tiles if r <= threshold
            )
            return north_tiles if north_tiles else self.walkable_tiles

        elif objective in (ObjectiveType.RUSH_ENEMY, ObjectiveType.ENGAGE_NEAREST):
            if enemies:
                nearest = min(
                    enemies,
                    key=lambda e: abs(e.row - unit_pos[0]) + abs(e.col - unit_pos[1]),
                )
                return frozenset([(nearest.row, nearest.col)])
            return frozenset()

        elif objective == ObjectiveType.HOLD_POSITION:
            return frozenset([unit_pos])

        elif objective == ObjectiveType.RETREAT:
            if not enemies:
                return self.walkable_tiles
            def _min_enemy_dist(pos: tuple[int, int]) -> int:
                return min(abs(pos[0] - e.row) + abs(pos[1] - e.col) for e in enemies)
            sorted_tiles = sorted(self.walkable_tiles, key=_min_enemy_dist, reverse=True)
            return frozenset(sorted_tiles[:10])

        elif objective == ObjectiveType.SUPPORT_ALLIES:
            if allies:
                cr = sum(u.row for u in allies) / len(allies)
                cc = sum(u.col for u in allies) / len(allies)
                sorted_tiles = sorted(
                    self.walkable_tiles,
                    key=lambda p: abs(p[0] - cr) + abs(p[1] - cc),
                )
                return frozenset(sorted_tiles[:5])
            return self.walkable_tiles

        # Fallback: return all walkable tiles
        return self.walkable_tiles
