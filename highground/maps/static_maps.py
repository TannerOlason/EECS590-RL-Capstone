"""Hand-crafted static maps for Phase 1 training and testing.

Each map function returns (terrain, elevation, team_a_spawns, team_b_spawns).
Spawns are lists of 3 (row, col) positions.
"""

from __future__ import annotations

import numpy as np

from highground.engine.grid import GRID_SIZE, Grid, Terrain


def flat_open() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """A flat, open 13x13 map — no obstacles, no elevation. Baseline map."""
    terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    grid = Grid(terrain, elevation)
    team_a = [(6, 1), (5, 0), (7, 0)]
    team_b = [(6, 11), (5, 12), (7, 12)]
    return grid, team_a, team_b


def central_hill() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """A map with a raised hill (elev 2) in the centre, surrounded by elev 1."""
    terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Central hill: 3x3 at elevation 2
    for r in range(5, 8):
        for c in range(5, 8):
            elevation[r, c] = 2

    # Ring at elevation 1
    for r in range(4, 9):
        for c in range(4, 9):
            if elevation[r, c] == 0:
                elevation[r, c] = 1

    grid = Grid(terrain, elevation)
    team_a = [(6, 1), (4, 2), (8, 2)]
    team_b = [(6, 11), (4, 10), (8, 10)]
    return grid, team_a, team_b


def choke_point() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Two open areas connected by a narrow 1-tile-wide corridor."""
    terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Wall across the middle (col 6), except a gap at row 6
    for r in range(GRID_SIZE):
        if r != 6:
            terrain[r, 6] = Terrain.UNCROSSABLE

    grid = Grid(terrain, elevation)
    team_a = [(6, 2), (5, 1), (7, 1)]
    team_b = [(6, 10), (5, 11), (7, 11)]
    return grid, team_a, team_b


def rough_flanks() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Open centre lane with rough terrain on the flanks. Tests movement costs."""
    terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Rough terrain on top and bottom thirds
    for r in range(0, 4):
        for c in range(GRID_SIZE):
            terrain[r, c] = Terrain.ROUGH
    for r in range(9, GRID_SIZE):
        for c in range(GRID_SIZE):
            terrain[r, c] = Terrain.ROUGH

    # Some elevation on the flanks
    for r in range(0, 3):
        for c in range(3, 10):
            elevation[r, c] = 1
    for r in range(10, GRID_SIZE):
        for c in range(3, 10):
            elevation[r, c] = 1

    grid = Grid(terrain, elevation)
    team_a = [(6, 1), (3, 1), (9, 1)]
    team_b = [(6, 11), (3, 11), (9, 11)]
    return grid, team_a, team_b


def asymmetric_heights() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Asymmetric map — Team A starts low, Team B starts on high ground."""
    terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Right half elevated
    for r in range(GRID_SIZE):
        for c in range(7, GRID_SIZE):
            elevation[r, c] = 2
        for c in range(5, 7):
            elevation[r, c] = 1

    # Some obstacles on the slope
    terrain[4, 6] = Terrain.UNCROSSABLE
    terrain[8, 6] = Terrain.UNCROSSABLE

    grid = Grid(terrain, elevation)
    team_a = [(6, 1), (4, 2), (8, 2)]
    team_b = [(6, 11), (4, 10), (8, 10)]
    return grid, team_a, team_b


STATIC_MAPS = {
    "flat_open": flat_open,
    "central_hill": central_hill,
    "choke_point": choke_point,
    "rough_flanks": rough_flanks,
    "asymmetric_heights": asymmetric_heights,
}


# ── Obstacle maps ─────────────────────────────────────────────────────────────
# Each map forces the agent to navigate around non-trivial terrain rather than
# charging east.  Designed so the direct E-path is blocked and at least two
# alternative routes exist so the map is never a dead-end.


def obstacle_square() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Solid 3×3 uncrossable block in the centre.  Units must route north or south."""
    terrain   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # 3×3 impassable block at the centre
    for r in range(5, 8):
        for c in range(5, 8):
            terrain[r, c] = Terrain.UNCROSSABLE

    # Slight elevation ring around the block rewards flanking high ground
    for r in range(4, 9):
        for c in range(4, 9):
            if terrain[r, c] == Terrain.NORMAL:
                elevation[r, c] = 1

    grid = Grid(terrain, elevation)
    team_a = [(6, 1), (4, 1), (8, 1)]
    team_b = [(6, 11), (4, 11), (8, 11)]
    return grid, team_a, team_b


def obstacle_circle() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Circular uncrossable zone (radius 2) in the centre.  Teaches curvilinear routing."""
    terrain   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    cr, cc = 6, 6  # centre
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r - cr) ** 2 + (c - cc) ** 2 <= 4:   # radius 2, 13 cells
                terrain[r, c] = Terrain.UNCROSSABLE

    # Elevation 1 ring just outside the circle rewards high ground around it
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            dist2 = (r - cr) ** 2 + (c - cc) ** 2
            if 4 < dist2 <= 9:
                elevation[r, c] = 1

    grid = Grid(terrain, elevation)
    team_a = [(6, 0), (4, 0), (8, 0)]
    team_b = [(6, 12), (4, 12), (8, 12)]
    return grid, team_a, team_b


def obstacle_hashtag() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Hashtag (#) pattern of walls with four passable gaps.

    Creates 9 interior zones interconnected only through the gaps, requiring
    the agent to choose an entry/exit point rather than marching straight east.

    Layout (rows/cols 3-9):
        col:  3 4 5 6 7 8 9
        row3: . . . . . . .
        row4: . X X . X X .     ← top horizontal bar (gap at col 6)
        row5: . X . . . X .     ← vertical bars only
        row6: . . . . . . .     ← full horizontal gap
        row7: . X . . . X .     ← vertical bars only
        row8: . X X . X X .     ← bottom horizontal bar (gap at col 6)
        row9: . . . . . . .
    """
    terrain   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    def block(r, c):
        terrain[r, c] = Terrain.UNCROSSABLE

    # Top horizontal bar: row 4, cols 4-8 except col 6
    for c in (4, 5, 7, 8):
        block(4, c)
    # Bottom horizontal bar: row 8, cols 4-8 except col 6
    for c in (4, 5, 7, 8):
        block(8, c)
    # Left vertical bar: col 4, rows 4-8 except row 6
    for r in (4, 5, 7, 8):
        block(r, 4)
    # Right vertical bar: col 8, rows 4-8 except row 6
    for r in (4, 5, 7, 8):
        block(r, 8)

    # Interior zone elevated — creates contested high ground
    for r in range(5, 8):
        for c in range(5, 8):
            if terrain[r, c] == Terrain.NORMAL:
                elevation[r, c] = 2

    grid = Grid(terrain, elevation)
    team_a = [(6, 0), (4, 0), (8, 0)]
    team_b = [(6, 12), (4, 12), (8, 12)]
    return grid, team_a, team_b


def two_rooms() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Two open rooms connected by a 3-tile-wide central corridor (rows 5-7, cols 5-7).

    Units from both sides must funnel into the same corridor; width allows
    two-unit-wide engagement rather than the single-file choke_point.
    """
    terrain   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Left wall at col 5 — passable only at rows 5-7
    for r in range(GRID_SIZE):
        if r not in (5, 6, 7):
            terrain[r, 5] = Terrain.UNCROSSABLE

    # Right wall at col 7 — passable only at rows 5-7
    for r in range(GRID_SIZE):
        if r not in (5, 6, 7):
            terrain[r, 7] = Terrain.UNCROSSABLE

    # Elevated corridor interior rewards holding the chokepoint
    for r in (5, 6, 7):
        elevation[r, 6] = 2

    grid = Grid(terrain, elevation)
    team_a = [(6, 2), (4, 2), (8, 2)]
    team_b = [(6, 10), (4, 10), (8, 10)]
    return grid, team_a, team_b


def fortress() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Ring of uncrossable walls with four gates; elevated interior.

    The interior is elevated (level 2) making it strategically valuable.
    Four gates at compass points give multiple approach angles; teams must
    decide whether to split and contest multiple gates or pile into one.

    Ring: rows 3-9, cols 3-9 (boundary).  Gates: N (3,6), S (9,6), W (6,3), E (6,9).
    """
    terrain   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Build the outer ring
    for r in range(3, 10):
        for c in range(3, 10):
            if r in (3, 9) or c in (3, 9):   # boundary
                terrain[r, c] = Terrain.UNCROSSABLE

    # Cut four gates
    terrain[3, 6] = Terrain.NORMAL   # north gate
    terrain[9, 6] = Terrain.NORMAL   # south gate
    terrain[6, 3] = Terrain.NORMAL   # west gate
    terrain[6, 9] = Terrain.NORMAL   # east gate

    # Elevate the interior
    for r in range(4, 9):
        for c in range(4, 9):
            elevation[r, c] = 2

    grid = Grid(terrain, elevation)
    team_a = [(6, 0), (4, 1), (8, 1)]
    team_b = [(6, 12), (4, 11), (8, 11)]
    return grid, team_a, team_b


def twin_gates() -> tuple[Grid, list[tuple[int, int]], list[tuple[int, int]]]:
    """Full-height vertical wall at col 6 with two passage gaps (rows 2 and 10).

    Extends choke_point: two crossing points force teams to decide whether to
    concentrate on one gap or split.  Units not aligned with a gap must navigate
    north or south before they can cross.
    """
    terrain   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    elevation = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Full wall at col 6, open only at rows 2 and 10
    for r in range(GRID_SIZE):
        if r not in (2, 10):
            terrain[r, 6] = Terrain.UNCROSSABLE

    # Slight elevation advantage near each gate
    elevation[2, 5] = elevation[2, 7] = 1
    elevation[10, 5] = elevation[10, 7] = 1

    grid = Grid(terrain, elevation)
    team_a = [(6, 2), (3, 2), (9, 2)]
    team_b = [(6, 10), (3, 10), (9, 10)]
    return grid, team_a, team_b


OBSTACLE_MAPS = {
    "obstacle_square":   obstacle_square,
    "obstacle_circle":   obstacle_circle,
    "obstacle_hashtag":  obstacle_hashtag,
    "two_rooms":         two_rooms,
    "fortress":          fortress,
    "twin_gates":        twin_gates,
}

ALL_MAPS = {**STATIC_MAPS, **OBSTACLE_MAPS}
