"""Natural-language narrator for the High Ground SRPG game state.

Converts a GameState snapshot into a compact (~150 token) English summary
suitable for injection into an LLM prompt.  The summary is always written
from the perspective of the requested team.

Usage::

    from highground.llm.narrator import narrate
    summary = narrate(game_state, team_id=0)
"""

from __future__ import annotations

from highground.engine.game_state import GameState, MAX_TURNS
from highground.engine.grid import GRID_SIZE
from highground.engine.units import CLASS_STATS, TEAM_A, TEAM_B, UnitClass


def narrate(game_state: GameState, team_id: int) -> str:
    """Return a compact natural-language summary of *game_state* for *team_id*.

    Parameters
    ----------
    game_state:
        The live game state to summarise.
    team_id:
        The team whose perspective to adopt (TEAM_A=0 or TEAM_B=1).

    Returns
    -------
    str
        A multi-line English string of roughly 120–180 tokens.
    """
    grid = game_state.grid
    units = game_state.units
    round_num = game_state.round_number

    allies  = [u for u in units if u.team == team_id and u.alive]
    enemies = [u for u in units if u.team != team_id and u.alive]

    # ── Round header ──────────────────────────────────────────────────────────
    lines: list[str] = [
        f"Round {round_num}/{MAX_TURNS}.",
    ]

    # ── Unit status per side ──────────────────────────────────────────────────
    def _unit_summary(unit_list: list) -> str:
        parts = []
        for u in unit_list:
            class_name = u.unit_class.name.capitalize()
            max_hp = CLASS_STATS[u.unit_class]["hp"]
            parts.append(f"{class_name}({u.hp}/{max_hp}HP @{u.row},{u.col})")
        return ", ".join(parts) if parts else "none"

    lines.append(f"My team ({len(allies)} alive): {_unit_summary(allies)}.")
    lines.append(f"Enemy team ({len(enemies)} alive): {_unit_summary(enemies)}.")

    # ── Acting unit ───────────────────────────────────────────────────────────
    cu = game_state.current_unit
    if cu.team == team_id:
        max_hp_cu = CLASS_STATS[cu.unit_class]["hp"]
        lines.append(
            f"Acting unit: {cu.unit_class.name.capitalize()} at ({cu.row},{cu.col}), "
            f"{cu.hp}/{max_hp_cu} HP, move_remaining={cu.move_remaining}, "
            f"has_attacked={'yes' if cu.has_attacked else 'no'}."
        )
    else:
        lines.append("It is the enemy's turn to act.")

    # ── Enemy proximity ───────────────────────────────────────────────────────
    if allies and enemies:
        total_min_dist = 0.0
        for ally in allies:
            min_dist = min(
                abs(ally.row - e.row) + abs(ally.col - e.col) for e in enemies
            )
            total_min_dist += min_dist
        avg_dist = total_min_dist / len(allies)
        lines.append(f"Avg nearest-enemy distance (Manhattan): {avg_dist:.1f} tiles.")
    elif not enemies:
        lines.append("No enemies remain.")
    else:
        lines.append("No allied units remain.")

    # ── High-ground status ────────────────────────────────────────────────────
    all_alive_positions = {u.pos for u in units if u.alive}
    elevation2_count = 0
    elevation2_occupied_friendly = 0
    elevation2_occupied_enemy = 0

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid.is_walkable(r, c) and grid.get_elevation(r, c) == 2:
                elevation2_count += 1
                pos = (r, c)
                if pos in all_alive_positions:
                    # Determine which team occupies this tile
                    for u in units:
                        if u.alive and u.pos == pos:
                            if u.team == team_id:
                                elevation2_occupied_friendly += 1
                            else:
                                elevation2_occupied_enemy += 1

    unoccupied_high_ground = elevation2_count - elevation2_occupied_friendly - elevation2_occupied_enemy
    lines.append(
        f"Elevation-2 tiles: {elevation2_count} total, "
        f"{elevation2_occupied_friendly} held by my team, "
        f"{elevation2_occupied_enemy} held by enemy, "
        f"{unoccupied_high_ground} unoccupied."
    )

    # ── Central terrain note ──────────────────────────────────────────────────
    obstacle_count = 0
    for r in range(4, 9):
        for c in range(4, 9):
            if not grid.is_walkable(r, c):
                obstacle_count += 1

    if obstacle_count == 0:
        terrain_note = "Central zone (r4-8, c4-8) is open with no obstacles."
    elif obstacle_count <= 4:
        terrain_note = f"Central zone has {obstacle_count} obstacle tile(s) — mostly open."
    else:
        terrain_note = f"Central zone is heavily obstructed ({obstacle_count} obstacle tiles)."
    lines.append(terrain_note)

    return " ".join(lines)
