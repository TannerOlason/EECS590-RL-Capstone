"""Combat resolution: damage formula with height advantage and flanking."""

from __future__ import annotations

from highground.engine.grid import Grid
from highground.engine.units import Unit


def compute_damage(
    attacker: Unit,
    defender: Unit,
    grid: Grid,
    all_units: list[Unit],
) -> int:
    """Compute damage dealt from attacker to defender.

    Formula:
        damage = max(1, ATK + height_bonus + flank_bonus + momentum_bonus - DEF)

    Height bonus: +1 if attacker elevation > defender elevation.
    Flank bonus: +1 if any living ally of the attacker (other than the attacker)
                 is adjacent to the defender.
    Momentum bonus: attacker.momentum (only relevant for Chargers).
    """
    base = attacker.atk
    height_bonus = _height_bonus(attacker, defender, grid)
    flank_bonus = _flank_bonus(attacker, defender, all_units)
    momentum_bonus = attacker.momentum  # 0 for non-Chargers
    raw = base + height_bonus + flank_bonus + momentum_bonus - defender.defense
    return max(1, raw)


def _height_bonus(attacker: Unit, defender: Unit, grid: Grid) -> int:
    atk_elev = grid.get_elevation(attacker.row, attacker.col)
    def_elev = grid.get_elevation(defender.row, defender.col)
    if atk_elev > def_elev:
        return 1
    return 0


def _flank_bonus(attacker: Unit, defender: Unit, all_units: list[Unit]) -> int:
    """Flanking: +1 if any living ally of the attacker is adjacent to the defender."""
    for u in all_units:
        if not u.alive:
            continue
        if u.team != attacker.team:
            continue
        if u.unit_id == attacker.unit_id:
            continue
        if _adjacent(u.row, u.col, defender.row, defender.col):
            return 1
    return 0


def _adjacent(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(r1 - r2) + abs(c1 - c2) == 1
