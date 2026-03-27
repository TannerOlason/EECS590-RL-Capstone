"""Unit classes and stats for the SRPG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar


class UnitClass(IntEnum):
    """The four unit archetypes."""

    FIGHTER = 0
    CHARGER = 1
    RANGER = 2
    SIEGE = 3


class Direction(IntEnum):
    """Movement directions — cardinal and diagonal."""

    NORTH      = 0  # row - 1
    SOUTH      = 1  # row + 1
    EAST       = 2  # col + 1
    WEST       = 3  # col - 1
    NORTH_EAST = 4  # row - 1, col + 1
    NORTH_WEST = 5  # row - 1, col - 1
    SOUTH_EAST = 6  # row + 1, col + 1
    SOUTH_WEST = 7  # row + 1, col - 1
    NONE       = 8  # stationary


# (delta_row, delta_col) for each direction
DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.NORTH:      (-1,  0),
    Direction.SOUTH:      ( 1,  0),
    Direction.EAST:       ( 0,  1),
    Direction.WEST:       ( 0, -1),
    Direction.NORTH_EAST: (-1,  1),
    Direction.NORTH_WEST: (-1, -1),
    Direction.SOUTH_EAST: ( 1,  1),
    Direction.SOUTH_WEST: ( 1, -1),
}


def opposite_direction(d: Direction) -> Direction:
    """Return the opposite cardinal direction."""
    if d == Direction.NORTH:
        return Direction.SOUTH
    if d == Direction.SOUTH:
        return Direction.NORTH
    if d == Direction.EAST:
        return Direction.WEST
    if d == Direction.WEST:
        return Direction.EAST
    return Direction.NONE


# ── Class stat templates ─────────────────────────────────────────────
#                   HP  ATK  DEF  MOVE  RANGE
CLASS_STATS: dict[UnitClass, dict[str, int]] = {
    UnitClass.FIGHTER: {"hp": 12, "atk": 4, "def": 3, "move": 3, "range": 1},
    UnitClass.CHARGER: {"hp": 10, "atk": 3, "def": 2, "move": 5, "range": 1},
    UnitClass.RANGER:  {"hp":  8, "atk": 3, "def": 1, "move": 3, "range": 3},
    UnitClass.SIEGE:   {"hp":  8, "atk": 5, "def": 1, "move": 2, "range": 4},
}

TEAM_A = 0
TEAM_B = 1


@dataclass
class Unit:
    """A single unit on the battlefield."""

    unit_id: int          # Unique index 0-5
    team: int             # TEAM_A (0) or TEAM_B (1)
    unit_class: UnitClass
    row: int
    col: int

    # Stats (set from CLASS_STATS in __post_init__)
    max_hp: int = 0
    hp: int = 0
    atk: int = 0
    defense: int = 0
    move_range: int = 0
    attack_range: int = 0

    # Turn state (reset each turn)
    move_remaining: int = 0
    has_attacked: bool = False
    alive: bool = True

    # Charger momentum state
    momentum: int = 0
    momentum_dir: Direction = Direction.NONE

    def __post_init__(self) -> None:
        stats = CLASS_STATS[self.unit_class]
        self.max_hp = stats["hp"]
        self.hp = stats["hp"]
        self.atk = stats["atk"]
        self.defense = stats["def"]
        self.move_range = stats["move"]
        self.attack_range = stats["range"]
        self.move_remaining = self.move_range

    def start_turn(self) -> None:
        """Reset per-turn state at the beginning of this unit's turn."""
        self.move_remaining = self.move_range
        self.has_attacked = False

    def take_damage(self, amount: int) -> None:
        self.hp = max(0, self.hp - amount)
        if self.hp == 0:
            self.alive = False

    @property
    def pos(self) -> tuple[int, int]:
        return (self.row, self.col)

    def copy(self) -> Unit:
        """Return a deep copy of this unit."""
        u = Unit(
            unit_id=self.unit_id,
            team=self.team,
            unit_class=self.unit_class,
            row=self.row,
            col=self.col,
        )
        u.max_hp = self.max_hp
        u.hp = self.hp
        u.atk = self.atk
        u.defense = self.defense
        u.move_range = self.move_range
        u.attack_range = self.attack_range
        u.move_remaining = self.move_remaining
        u.has_attacked = self.has_attacked
        u.alive = self.alive
        u.momentum = self.momentum
        u.momentum_dir = self.momentum_dir
        return u
