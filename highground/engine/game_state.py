"""Core game state machine for the High Ground SRPG.

Turn structure:
    Full-team turns.  All units on Team A act (in index order), then all on
    Team B.  Within a unit's turn the agent issues micro-actions one at a time:
      - MOVE in a cardinal direction (costs movement points)
      - ATTACK an enemy (once per turn)
      - END_TURN to pass
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

from highground.engine.combat import compute_damage
from highground.engine.grid import Grid, Terrain
from highground.engine.pathfinding import can_step, tiles_in_attack_range
from highground.engine.units import (
    DIRECTION_DELTAS,
    Direction,
    Unit,
    UnitClass,
    opposite_direction,
    TEAM_A,
    TEAM_B,
)


# ── Action encoding ──────────────────────────────────────────────────
# Discrete(12): 0-3 = MOVE N/S/E/W,  4-7 = MOVE NE/NW/SE/SW,
#               8-10 = ATTACK enemy 0/1/2,  11 = END_TURN

NUM_ACTIONS = 12
NUM_MOVE_ACTIONS = 8    # indices 0-7
NUM_ATTACK_ACTIONS = 3  # indices 8-10
END_TURN_ACTION = 11

MAX_TURNS = 50  # game draw after this many full rounds


class ActionType(IntEnum):
    MOVE_N  = 0
    MOVE_S  = 1
    MOVE_E  = 2
    MOVE_W  = 3
    MOVE_NE = 4
    MOVE_NW = 5
    MOVE_SE = 6
    MOVE_SW = 7
    ATTACK_0 = 8
    ATTACK_1 = 9
    ATTACK_2 = 10
    END_TURN = 11


MOVE_ACTION_TO_DIR: dict[int, Direction] = {
    ActionType.MOVE_N:  Direction.NORTH,
    ActionType.MOVE_S:  Direction.SOUTH,
    ActionType.MOVE_E:  Direction.EAST,
    ActionType.MOVE_W:  Direction.WEST,
    ActionType.MOVE_NE: Direction.NORTH_EAST,
    ActionType.MOVE_NW: Direction.NORTH_WEST,
    ActionType.MOVE_SE: Direction.SOUTH_EAST,
    ActionType.MOVE_SW: Direction.SOUTH_WEST,
}


class StepResult(NamedTuple):
    """Returned after each micro-action step."""

    done: bool          # Is the game over?
    winner: int | None  # TEAM_A, TEAM_B, or None (draw / ongoing)
    unit_killed: int | None  # unit_id of killed unit, if any


class GameState:
    """Authoritative game state machine.

    Owns the grid, all units, turn tracking, and the valid-action mask.
    """

    def __init__(self, grid: Grid, units: list[Unit]) -> None:
        self.grid = grid
        self.units = list(units)  # indexed by unit_id (0-5)
        assert len(self.units) == 6

        # Turn management
        self.current_team: int = TEAM_A
        self._team_unit_queue: list[int] = []  # unit_ids still to act this team-turn
        self.current_unit_id: int = -1
        self.round_number: int = 0
        self.done: bool = False
        self.winner: int | None = None

        # Kick off the first turn
        self._start_team_turn(TEAM_A)

    # ── Public API ────────────────────────────────────────────────────

    @property
    def current_unit(self) -> Unit:
        return self.units[self.current_unit_id]

    def enemy_units(self, team: int) -> list[Unit]:
        """Return the three enemy units (in order of their local index 0-2)."""
        return [u for u in self.units if u.team != team]

    def allied_units(self, team: int) -> list[Unit]:
        return [u for u in self.units if u.team == team]

    def occupied_positions(self, exclude_id: int | None = None) -> set[tuple[int, int]]:
        """Positions of all living units, optionally excluding one."""
        return {
            u.pos for u in self.units
            if u.alive and u.unit_id != exclude_id
        }

    # ── Action mask ───────────────────────────────────────────────────

    def valid_action_mask(self) -> list[int]:
        """Return a length-12 binary list: 1 = legal, 0 = illegal."""
        mask = [0] * NUM_ACTIONS
        if self.done:
            return mask

        unit = self.current_unit
        occupied = self.occupied_positions(exclude_id=unit.unit_id)

        # MOVE actions (0-3)
        for action_idx, direction in MOVE_ACTION_TO_DIR.items():
            if unit.move_remaining <= 0:
                continue
            valid, cost = can_step(self.grid, unit.row, unit.col, direction, occupied)
            if valid and cost <= unit.move_remaining:
                mask[action_idx] = 1

        # ATTACK actions (4-6)
        if not unit.has_attacked:
            enemies = self.enemy_units(unit.team)
            attack_tiles = set(
                tiles_in_attack_range(self.grid, unit.row, unit.col, unit.attack_range)
            )
            for i, enemy in enumerate(enemies):
                if enemy.alive and enemy.pos in attack_tiles:
                    mask[ActionType.ATTACK_0 + i] = 1

        # END_TURN is always legal
        mask[END_TURN_ACTION] = 1
        return mask

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, action: int) -> StepResult:
        """Execute one micro-action for the current unit.

        Returns a StepResult.
        Raises ValueError on illegal actions.
        """
        if self.done:
            raise RuntimeError("Game is already over.")

        mask = self.valid_action_mask()
        if mask[action] != 1:
            raise ValueError(
                f"Illegal action {action} for unit {self.current_unit_id}. "
                f"Mask: {mask}"
            )

        unit = self.current_unit
        killed: int | None = None

        if action <= ActionType.MOVE_SW:
            killed = self._execute_move(unit, MOVE_ACTION_TO_DIR[action])
        elif action <= ActionType.ATTACK_2:
            enemy_local_idx = action - ActionType.ATTACK_0
            killed = self._execute_attack(unit, enemy_local_idx)
        else:
            # END_TURN
            pass

        # Check win condition
        game_done, winner = self._check_winner()
        if game_done:
            self.done = True
            self.winner = winner
            return StepResult(done=True, winner=winner, unit_killed=killed)

        # If END_TURN or the unit can't do anything else, advance to next unit
        if action == END_TURN_ACTION or self._unit_is_spent(unit):
            self._advance_to_next_unit()

        return StepResult(done=self.done, winner=self.winner, unit_killed=killed)

    # ── Internal: movement ────────────────────────────────────────────

    def _execute_move(self, unit: Unit, direction: Direction) -> int | None:
        dr, dc = DIRECTION_DELTAS[direction]
        new_row, new_col = unit.row + dr, unit.col + dc
        cost = self.grid.move_cost(new_row, new_col)
        old_elev = self.grid.get_elevation(unit.row, unit.col)
        new_elev = self.grid.get_elevation(new_row, new_col)

        unit.row = new_row
        unit.col = new_col
        unit.move_remaining -= cost

        # Update Charger momentum
        if unit.unit_class == UnitClass.CHARGER:
            self._update_charger_momentum(unit, direction, old_elev, new_elev)

        return None

    def _update_charger_momentum(
        self, unit: Unit, direction: Direction, old_elev: int, new_elev: int
    ) -> None:
        """Update momentum for a Charger after moving one tile.

        Rules:
        - +1 per tile moved in the same cardinal direction.
        - +1 extra when going down exactly 1 elevation.
        - Reset when: dropping 2+ elevation, entering rough terrain,
          or changing direction.
        """
        elev_drop = old_elev - new_elev

        # Reset conditions
        if elev_drop >= 2:
            unit.momentum = 0
            unit.momentum_dir = Direction.NONE
            return
        if self.grid.is_rough(unit.row, unit.col):
            unit.momentum = 0
            unit.momentum_dir = Direction.NONE
            return
        if unit.momentum_dir != Direction.NONE and direction != unit.momentum_dir:
            unit.momentum = 0
            unit.momentum_dir = Direction.NONE
            return

        # Accumulate
        unit.momentum_dir = direction
        unit.momentum += 1
        if elev_drop == 1:
            unit.momentum += 1

    # ── Internal: attack ──────────────────────────────────────────────

    def _execute_attack(self, unit: Unit, enemy_local_idx: int) -> int | None:
        enemies = self.enemy_units(unit.team)
        target = enemies[enemy_local_idx]

        damage = compute_damage(unit, target, self.grid, self.units)
        target.take_damage(damage)
        unit.has_attacked = True

        # Reset Charger momentum after attacking
        if unit.unit_class == UnitClass.CHARGER:
            unit.momentum = 0
            unit.momentum_dir = Direction.NONE

        if not target.alive:
            return target.unit_id
        return None

    # ── Internal: turn management ─────────────────────────────────────

    def _unit_is_spent(self, unit: Unit) -> bool:
        """True if the unit has no meaningful actions left."""
        mask = self.valid_action_mask()
        # If only END_TURN is valid, the unit is spent
        return mask == [0, 0, 0, 0, 0, 0, 0, 1]

    def _advance_to_next_unit(self) -> None:
        """Move to the next unit in the team queue, or switch teams."""
        while self._team_unit_queue:
            next_id = self._team_unit_queue.pop(0)
            if self.units[next_id].alive:
                self.current_unit_id = next_id
                self.units[next_id].start_turn()
                return

        # Team turn is over — switch to other team
        next_team = TEAM_B if self.current_team == TEAM_A else TEAM_A
        if next_team == TEAM_A:
            self.round_number += 1
            if self.round_number >= MAX_TURNS:
                self.done = True
                self.winner = None  # draw
                return
        self._start_team_turn(next_team)

    def _start_team_turn(self, team: int) -> None:
        """Begin a new team turn: queue all living units on that team."""
        self.current_team = team
        self._team_unit_queue = [
            u.unit_id for u in self.units if u.team == team and u.alive
        ]
        if not self._team_unit_queue:
            # No living units — the other team wins (should already be caught)
            self.done = True
            self.winner = TEAM_B if team == TEAM_A else TEAM_A
            return
        first_id = self._team_unit_queue.pop(0)
        self.current_unit_id = first_id
        self.units[first_id].start_turn()

    # ── Win condition ─────────────────────────────────────────────────

    def _check_winner(self) -> tuple[bool, int | None]:
        a_alive = any(u.alive for u in self.units if u.team == TEAM_A)
        b_alive = any(u.alive for u in self.units if u.team == TEAM_B)
        if not a_alive and not b_alive:
            return True, None  # mutual destruction draw
        if not a_alive:
            return True, TEAM_B
        if not b_alive:
            return True, TEAM_A
        return False, None

    # ── Utility ───────────────────────────────────────────────────────

    def copy(self) -> GameState:
        """Deep copy the entire game state."""
        new_grid = self.grid.copy()
        new_units = [u.copy() for u in self.units]
        gs = GameState.__new__(GameState)
        gs.grid = new_grid
        gs.units = new_units
        gs.current_team = self.current_team
        gs._team_unit_queue = list(self._team_unit_queue)
        gs.current_unit_id = self.current_unit_id
        gs.round_number = self.round_number
        gs.done = self.done
        gs.winner = self.winner
        return gs
