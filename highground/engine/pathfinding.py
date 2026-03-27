"""Dijkstra-based pathfinding accounting for terrain costs and occupied tiles."""

from __future__ import annotations

import heapq

from highground.engine.grid import Grid
from highground.engine.units import DIRECTION_DELTAS, Direction


def reachable_tiles(
    grid: Grid,
    start_row: int,
    start_col: int,
    move_points: int,
    occupied: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    """Return all tiles reachable from (start_row, start_col) within move_points.

    Uses Dijkstra's algorithm with terrain-dependent edge costs.
    Units cannot pass through tiles occupied by other units.

    Args:
        grid: The game grid.
        start_row: Starting row.
        start_col: Starting column.
        move_points: Remaining movement points.
        occupied: Set of (row, col) positions occupied by other units.

    Returns:
        Dict mapping (row, col) -> cost to reach that tile.
        The starting tile is included with cost 0.
    """
    if occupied is None:
        occupied = set()

    dist: dict[tuple[int, int], int] = {}
    heap: list[tuple[int, int, int]] = [(0, start_row, start_col)]

    while heap:
        cost, r, c = heapq.heappop(heap)
        if (r, c) in dist:
            continue
        dist[(r, c)] = cost

        for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = r + dr, c + dc
            if not grid.is_walkable(nr, nc):
                continue
            if (nr, nc) in occupied and (nr, nc) != (start_row, start_col):
                continue
            step_cost = grid.move_cost(nr, nc)
            new_cost = cost + step_cost
            if new_cost > move_points:
                continue
            if (nr, nc) not in dist:
                heapq.heappush(heap, (new_cost, nr, nc))

    return dist


def can_step(
    grid: Grid,
    from_row: int,
    from_col: int,
    direction: Direction,
    occupied: set[tuple[int, int]],
) -> tuple[bool, int]:
    """Check if a single step in the given direction is valid.

    Returns:
        (is_valid, cost) — cost is the movement points required for the step.
    """
    dr, dc = DIRECTION_DELTAS[direction]
    nr, nc = from_row + dr, from_col + dc
    if not grid.is_walkable(nr, nc):
        return False, 0
    if (nr, nc) in occupied:
        return False, 0
    return True, grid.move_cost(nr, nc)


def tiles_in_attack_range(
    grid: Grid,
    row: int,
    col: int,
    attack_range: int,
) -> list[tuple[int, int]]:
    """Return all in-bounds tiles within Chebyshev distance of attack_range.

    Chebyshev distance (max(|dr|, |dc|) ≤ range) covers all 8 surrounding
    tiles for range-1 units and a (2r+1)×(2r+1) square for range-r units.
    Does NOT check walkability — attacks fire over obstacles.
    """
    result = []
    for dr in range(-attack_range, attack_range + 1):
        for dc in range(-attack_range, attack_range + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if grid.in_bounds(nr, nc):
                result.append((nr, nc))
    return result
