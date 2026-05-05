"""Scripted enemy AI for the side-scrolling env.

Two simple behaviours:
  - WalkerAI:  closes distance to nearest player; attacks when adjacent.
  - ShooterAI: tries to stand at exactly attack_range from nearest player;
               attacks immediately when in range; otherwise sidesteps.

Both AIs return an `EnemyAction` describing one V2-style tile step plus an
optional attack; the env applies it after walkability and occupancy checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import _path_shim  # noqa: F401
from highground.engine.units import UnitClass


@dataclass
class EnemyAction:
    """One enemy's intended action for the macro tick."""

    drow: int       # -1, 0, or +1  (row delta in tiles)
    dcol_world: int # -1, 0, or +1  (world-col delta in tiles)
    attack_target_id: int | None = None  # player unit_id to attack, if in range


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _chebyshev(r1: int, c1: int, r2: int, c2: int) -> int:
    return max(abs(r1 - r2), abs(c1 - c2))


def _nearest_player(
    enemy_row: int,
    enemy_world_x: int,
    players: list,  # list of UnitState; forward declaration
) -> tuple[int, int, int, int] | None:
    """Return (player_unit_id, p_row, p_world_x, distance) of nearest alive player,
    or None if no players are alive."""
    best = None
    best_d = 10**9
    for p in players:
        if not p.unit.alive:
            continue
        d = _chebyshev(enemy_row, enemy_world_x, p.row_int(), p.world_x_int())
        if d < best_d:
            best_d = d
            best = (p.unit.unit_id, p.row_int(), p.world_x_int(), d)
    return best


class WalkerAI:
    """Melee enemy: walks toward nearest player; attacks if adjacent."""

    def decide(self, enemy_state, players: list) -> EnemyAction:
        e_row = enemy_state.row_int()
        e_wx  = enemy_state.world_x_int()
        target = _nearest_player(e_row, e_wx, players)
        if target is None:
            return EnemyAction(drow=0, dcol_world=-1)  # drift left if no players
        pid, p_row, p_wx, d = target

        if d <= enemy_state.unit.attack_range:
            # In range — attack and don't move.
            return EnemyAction(drow=0, dcol_world=0, attack_target_id=pid)

        return EnemyAction(drow=_sign(p_row - e_row), dcol_world=_sign(p_wx - e_wx))


class ShooterAI:
    """Ranged enemy: keep distance ~= attack_range; attack on sight."""

    def decide(self, enemy_state, players: list) -> EnemyAction:
        e_row = enemy_state.row_int()
        e_wx  = enemy_state.world_x_int()
        target = _nearest_player(e_row, e_wx, players)
        if target is None:
            return EnemyAction(drow=0, dcol_world=-1)
        pid, p_row, p_wx, d = target

        rng = enemy_state.unit.attack_range
        if d <= rng:
            # Always shoot when in range; back away if too close.
            attack = pid
            if d < rng:
                # Step away from player.
                return EnemyAction(
                    drow=-_sign(p_row - e_row),
                    dcol_world=-_sign(p_wx - e_wx),
                    attack_target_id=attack,
                )
            return EnemyAction(drow=0, dcol_world=0, attack_target_id=attack)

        # Out of range — close in.
        return EnemyAction(drow=_sign(p_row - e_row), dcol_world=_sign(p_wx - e_wx))


def make_ai_for(unit_class: UnitClass):
    """Pick an AI implementation based on the enemy's class."""
    if unit_class in (UnitClass.RANGER, UnitClass.SIEGE):
        return ShooterAI()
    return WalkerAI()
