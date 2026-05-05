"""Scrolling-world unit state wrapping a V2 Unit.

The env keeps two parallel collections of UnitState (player squad + enemies).
Whenever combat / pathfinding helpers from V2 need a Unit with valid (row, col)
window-relative coordinates, call `sync_window_pos(scroll_offset)` first.
Player units move on exact V2 grid tiles; the float fields are retained so
scrolling world-x bookkeeping has one representation shared with enemies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import _path_shim  # noqa: F401
from highground.engine.units import Unit


@dataclass
class UnitState:
    unit: Unit            # V2 Unit (carries HP, atk, def, attack_range, team, class)
    row_f: float          # grid row, stored as float for renderer compatibility
    world_x_f: float      # absolute world column, stored as float

    # AI handle (only used for enemies; None for player units).
    ai: object | None = None

    # Transient display/observation flag for attack availability.
    attack_cd: int = 0

    # ── Convenience accessors ───────────────────────────────────────────
    def row_int(self) -> int:
        return int(round(self.row_f))

    def world_x_int(self) -> int:
        return int(round(self.world_x_f))

    def sync_window_pos(self, scroll_offset: int) -> None:
        """Write rounded window-relative coords into the underlying V2 Unit."""
        self.unit.row = self.row_int()
        self.unit.col = self.world_x_int() - scroll_offset
