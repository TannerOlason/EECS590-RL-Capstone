"""Tests for the core game engine: grid, units, pathfinding, combat, game state."""

import numpy as np
import pytest

from highground.engine.grid import GRID_SIZE, Grid, Terrain
from highground.engine.units import (
    DIRECTION_DELTAS,
    TEAM_A,
    TEAM_B,
    Direction,
    Unit,
    UnitClass,
)
from highground.engine.pathfinding import can_step, reachable_tiles, tiles_in_attack_range
from highground.engine.combat import compute_damage
from highground.engine.game_state import (
    ActionType,
    END_TURN_ACTION,
    NUM_ACTIONS,
    GameState,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _flat_grid() -> Grid:
    return Grid()


def _make_units(
    a_pos: list[tuple[int, int]] | None = None,
    b_pos: list[tuple[int, int]] | None = None,
    a_classes: list[UnitClass] | None = None,
    b_classes: list[UnitClass] | None = None,
) -> list[Unit]:
    a_pos = a_pos or [(6, 1), (5, 0), (7, 0)]
    b_pos = b_pos or [(6, 11), (5, 12), (7, 12)]
    a_classes = a_classes or [UnitClass.FIGHTER, UnitClass.CHARGER, UnitClass.RANGER]
    b_classes = b_classes or [UnitClass.FIGHTER, UnitClass.CHARGER, UnitClass.RANGER]
    units = []
    for i, (cls, (r, c)) in enumerate(zip(a_classes, a_pos)):
        units.append(Unit(unit_id=i, team=TEAM_A, unit_class=cls, row=r, col=c))
    for i, (cls, (r, c)) in enumerate(zip(b_classes, b_pos)):
        units.append(Unit(unit_id=i + 3, team=TEAM_B, unit_class=cls, row=r, col=c))
    return units


# ── Grid tests ────────────────────────────────────────────────────────

class TestGrid:
    def test_default_grid_is_all_normal_flat(self):
        g = Grid()
        assert g.terrain.shape == (13, 13)
        assert np.all(g.terrain == 0)
        assert np.all(g.elevation == 0)

    def test_walkability(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        terrain[5, 5] = Terrain.UNCROSSABLE
        terrain[3, 3] = Terrain.ROUGH
        g = Grid(terrain)
        assert g.is_walkable(0, 0)
        assert not g.is_walkable(5, 5)
        assert g.is_walkable(3, 3)
        assert not g.is_walkable(-1, 0)  # out of bounds
        assert not g.is_walkable(13, 0)

    def test_move_cost(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        terrain[3, 3] = Terrain.ROUGH
        g = Grid(terrain)
        assert g.move_cost(0, 0) == 1
        assert g.move_cost(3, 3) == 2


# ── Unit tests ────────────────────────────────────────────────────────

class TestUnit:
    def test_fighter_stats(self):
        u = Unit(0, TEAM_A, UnitClass.FIGHTER, 0, 0)
        assert u.hp == 12
        assert u.atk == 4
        assert u.defense == 3
        assert u.move_range == 3
        assert u.attack_range == 1

    def test_charger_stats(self):
        u = Unit(0, TEAM_A, UnitClass.CHARGER, 0, 0)
        assert u.hp == 10
        assert u.move_range == 5

    def test_take_damage_kills(self):
        u = Unit(0, TEAM_A, UnitClass.RANGER, 0, 0)
        assert u.hp == 8
        u.take_damage(7)
        assert u.hp == 1
        assert u.alive
        u.take_damage(5)
        assert u.hp == 0
        assert not u.alive

    def test_start_turn_resets(self):
        u = Unit(0, TEAM_A, UnitClass.FIGHTER, 0, 0)
        u.move_remaining = 0
        u.has_attacked = True
        u.start_turn()
        assert u.move_remaining == 3
        assert not u.has_attacked


# ── Pathfinding tests ─────────────────────────────────────────────────

class TestPathfinding:
    def test_reachable_flat_open(self):
        g = _flat_grid()
        tiles = reachable_tiles(g, 6, 6, 3)
        # Should include the start tile
        assert (6, 6) in tiles
        assert tiles[(6, 6)] == 0
        # 3 steps away in cardinal direction
        assert (3, 6) in tiles
        assert tiles[(3, 6)] == 3
        # Diagonal: (5,5) is 2 steps
        assert (5, 5) in tiles

    def test_reachable_blocked_by_wall(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        # Wall east of start
        terrain[6, 7] = Terrain.UNCROSSABLE
        g = Grid(terrain)
        tiles = reachable_tiles(g, 6, 6, 2)
        assert (6, 7) not in tiles

    def test_reachable_rough_costs_extra(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        terrain[6, 7] = Terrain.ROUGH
        g = Grid(terrain)
        tiles = reachable_tiles(g, 6, 6, 2)
        # (6,7) costs 2 to enter, so reachable with 2 move
        assert (6, 7) in tiles
        assert tiles[(6, 7)] == 2
        # Can't go further east because 2 + 1 = 3 > 2
        assert (6, 8) not in tiles

    def test_reachable_blocked_by_unit(self):
        g = _flat_grid()
        occupied = {(6, 7)}
        tiles = reachable_tiles(g, 6, 6, 3, occupied)
        assert (6, 7) not in tiles

    def test_can_step(self):
        g = _flat_grid()
        valid, cost = can_step(g, 6, 6, Direction.EAST, set())
        assert valid
        assert cost == 1

    def test_can_step_blocked(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        terrain[6, 7] = Terrain.UNCROSSABLE
        g = Grid(terrain)
        valid, cost = can_step(g, 6, 6, Direction.EAST, set())
        assert not valid

    def test_tiles_in_attack_range(self):
        g = _flat_grid()
        tiles = tiles_in_attack_range(g, 6, 6, 1)
        assert set(tiles) == {(5, 6), (7, 6), (6, 5), (6, 7)}

    def test_tiles_in_attack_range_3(self):
        g = _flat_grid()
        tiles = tiles_in_attack_range(g, 6, 6, 3)
        # Manhattan distance <= 3, excluding self
        for r, c in tiles:
            assert abs(r - 6) + abs(c - 6) <= 3
            assert (r, c) != (6, 6)
        assert len(tiles) > 4


# ── Combat tests ──────────────────────────────────────────────────────

class TestCombat:
    def test_basic_damage(self):
        attacker = Unit(0, TEAM_A, UnitClass.FIGHTER, 6, 5)
        defender = Unit(3, TEAM_B, UnitClass.FIGHTER, 6, 6)
        g = _flat_grid()
        dmg = compute_damage(attacker, defender, g, [attacker, defender])
        # Fighter ATK=4, DEF=3, no bonuses => max(1, 4-3) = 1
        assert dmg == 1

    def test_height_bonus(self):
        attacker = Unit(0, TEAM_A, UnitClass.FIGHTER, 6, 5)
        defender = Unit(3, TEAM_B, UnitClass.FIGHTER, 6, 6)
        elev = np.zeros((13, 13), dtype=np.int8)
        elev[6, 5] = 2  # attacker high
        g = Grid(elevation=elev)
        dmg = compute_damage(attacker, defender, g, [attacker, defender])
        # 4 + 1(height) - 3 = 2
        assert dmg == 2

    def test_flank_bonus(self):
        attacker = Unit(0, TEAM_A, UnitClass.FIGHTER, 6, 5)
        ally = Unit(1, TEAM_A, UnitClass.FIGHTER, 6, 7)  # adjacent to defender
        defender = Unit(3, TEAM_B, UnitClass.FIGHTER, 6, 6)
        g = _flat_grid()
        units = [attacker, ally, defender]
        dmg = compute_damage(attacker, defender, g, units)
        # 4 + 1(flank) - 3 = 2
        assert dmg == 2

    def test_minimum_damage_is_1(self):
        attacker = Unit(0, TEAM_A, UnitClass.RANGER, 6, 3)  # ATK=3
        defender = Unit(3, TEAM_B, UnitClass.FIGHTER, 6, 6)  # DEF=3
        g = _flat_grid()
        dmg = compute_damage(attacker, defender, g, [attacker, defender])
        # 3 - 3 = 0, clamped to 1
        assert dmg == 1

    def test_charger_momentum_damage(self):
        attacker = Unit(0, TEAM_A, UnitClass.CHARGER, 6, 5)
        attacker.momentum = 3
        defender = Unit(3, TEAM_B, UnitClass.FIGHTER, 6, 6)
        g = _flat_grid()
        dmg = compute_damage(attacker, defender, g, [attacker, defender])
        # 3(ATK) + 3(momentum) - 3(DEF) = 3
        assert dmg == 3


# ── GameState tests ───────────────────────────────────────────────────

class TestGameState:
    def test_initial_state(self):
        g = _flat_grid()
        units = _make_units()
        gs = GameState(g, units)
        assert gs.current_team == TEAM_A
        assert gs.current_unit_id == 0
        assert not gs.done

    def test_end_turn_advances(self):
        g = _flat_grid()
        units = _make_units()
        gs = GameState(g, units)
        # Unit 0 ends turn -> Unit 1
        gs.step(END_TURN_ACTION)
        assert gs.current_unit_id == 1

    def test_full_team_rotation(self):
        g = _flat_grid()
        units = _make_units()
        gs = GameState(g, units)
        # All 3 Team A units end turn
        gs.step(END_TURN_ACTION)  # unit 0 -> 1
        gs.step(END_TURN_ACTION)  # unit 1 -> 2
        gs.step(END_TURN_ACTION)  # unit 2 -> Team B unit 3
        assert gs.current_team == TEAM_B
        assert gs.current_unit_id == 3

    def test_move_action(self):
        g = _flat_grid()
        units = _make_units(a_pos=[(6, 6), (0, 0), (0, 1)])
        gs = GameState(g, units)
        # Unit 0 at (6,6), move east
        result = gs.step(ActionType.MOVE_E)
        assert gs.units[0].row == 6
        assert gs.units[0].col == 7
        assert gs.units[0].move_remaining == 2
        assert not result.done

    def test_attack_action(self):
        g = _flat_grid()
        # Place fighter adjacent to enemy
        units = _make_units(
            a_pos=[(6, 5), (0, 0), (0, 1)],
            b_pos=[(6, 6), (12, 12), (12, 11)],
        )
        gs = GameState(g, units)
        initial_hp = gs.units[3].hp
        mask = gs.valid_action_mask()
        # Attack enemy 0 (unit_id 3) should be valid
        assert mask[ActionType.ATTACK_0] == 1
        result = gs.step(ActionType.ATTACK_0)
        assert gs.units[3].hp < initial_hp

    def test_action_mask_no_move_through_wall(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        terrain[6, 7] = Terrain.UNCROSSABLE
        g = Grid(terrain)
        units = _make_units(a_pos=[(6, 6), (0, 0), (0, 1)])
        gs = GameState(g, units)
        mask = gs.valid_action_mask()
        # Can't move east (wall)
        assert mask[ActionType.MOVE_E] == 0
        # Can move in other directions
        assert mask[ActionType.MOVE_N] == 1
        assert mask[ActionType.MOVE_S] == 1
        assert mask[ActionType.MOVE_W] == 1

    def test_action_mask_no_attack_out_of_range(self):
        g = _flat_grid()
        # Enemies far away
        units = _make_units(
            a_pos=[(6, 1), (0, 0), (0, 1)],
            b_pos=[(6, 11), (12, 12), (12, 11)],
        )
        gs = GameState(g, units)
        mask = gs.valid_action_mask()
        # Fighter range=1, enemies at distance 10+
        assert mask[ActionType.ATTACK_0] == 0
        assert mask[ActionType.ATTACK_1] == 0
        assert mask[ActionType.ATTACK_2] == 0

    def test_action_mask_no_move_when_exhausted(self):
        g = _flat_grid()
        units = _make_units(a_pos=[(6, 6), (0, 0), (0, 1)])
        gs = GameState(g, units)
        gs.units[0].move_remaining = 0
        mask = gs.valid_action_mask()
        assert mask[ActionType.MOVE_N] == 0
        assert mask[ActionType.MOVE_S] == 0
        assert mask[ActionType.MOVE_E] == 0
        assert mask[ActionType.MOVE_W] == 0
        assert mask[END_TURN_ACTION] == 1

    def test_action_mask_no_double_attack(self):
        g = _flat_grid()
        units = _make_units(
            a_pos=[(6, 5), (0, 0), (0, 1)],
            b_pos=[(6, 6), (12, 12), (12, 11)],
        )
        gs = GameState(g, units)
        gs.step(ActionType.ATTACK_0)
        mask = gs.valid_action_mask()
        assert mask[ActionType.ATTACK_0] == 0

    def test_illegal_action_raises(self):
        g = _flat_grid()
        units = _make_units()
        gs = GameState(g, units)
        gs.units[0].move_remaining = 0
        with pytest.raises(ValueError):
            gs.step(ActionType.MOVE_N)

    def test_win_condition(self):
        g = _flat_grid()
        units = _make_units(
            a_pos=[(6, 5), (0, 0), (0, 1)],
            b_pos=[(6, 6), (12, 12), (12, 11)],
        )
        gs = GameState(g, units)
        # Kill all enemy units manually
        for u in gs.units:
            if u.team == TEAM_B:
                u.hp = 1
                u.alive = True

        # Attack enemy 0
        gs.units[3].hp = 1
        result = gs.step(ActionType.ATTACK_0)
        if not result.done:
            # Kill remaining enemies
            gs.units[4].hp = 0
            gs.units[4].alive = False
            gs.units[5].hp = 0
            gs.units[5].alive = False
            # The game checks winner after each step, so end turn to trigger check
            gs.step(END_TURN_ACTION)

    def test_charger_momentum_builds(self):
        g = _flat_grid()
        units = _make_units(
            a_pos=[(6, 6), (0, 0), (0, 1)],
            a_classes=[UnitClass.CHARGER, UnitClass.FIGHTER, UnitClass.FIGHTER],
        )
        gs = GameState(g, units)
        # Move east 3 times
        gs.step(ActionType.MOVE_E)
        assert gs.units[0].momentum == 1
        assert gs.units[0].momentum_dir == Direction.EAST
        gs.step(ActionType.MOVE_E)
        assert gs.units[0].momentum == 2
        gs.step(ActionType.MOVE_E)
        assert gs.units[0].momentum == 3

    def test_charger_momentum_resets_on_direction_change(self):
        g = _flat_grid()
        units = _make_units(
            a_pos=[(6, 6), (0, 0), (0, 1)],
            a_classes=[UnitClass.CHARGER, UnitClass.FIGHTER, UnitClass.FIGHTER],
        )
        gs = GameState(g, units)
        gs.step(ActionType.MOVE_E)
        assert gs.units[0].momentum == 1
        gs.step(ActionType.MOVE_N)  # direction change
        assert gs.units[0].momentum == 0

    def test_charger_momentum_resets_on_rough(self):
        terrain = np.zeros((13, 13), dtype=np.int8)
        terrain[6, 8] = Terrain.ROUGH
        g = Grid(terrain)
        units = _make_units(
            a_pos=[(6, 6), (0, 0), (0, 1)],
            a_classes=[UnitClass.CHARGER, UnitClass.FIGHTER, UnitClass.FIGHTER],
        )
        gs = GameState(g, units)
        gs.step(ActionType.MOVE_E)  # to (6,7), momentum=1
        assert gs.units[0].momentum == 1
        gs.step(ActionType.MOVE_E)  # to (6,8) rough -> reset
        assert gs.units[0].momentum == 0

    def test_charger_momentum_bonus_downhill(self):
        elev = np.zeros((13, 13), dtype=np.int8)
        elev[6, 6] = 1  # start elevated
        g = Grid(elevation=elev)
        units = _make_units(
            a_pos=[(6, 6), (0, 0), (0, 1)],
            a_classes=[UnitClass.CHARGER, UnitClass.FIGHTER, UnitClass.FIGHTER],
        )
        gs = GameState(g, units)
        gs.step(ActionType.MOVE_E)  # drop 1 elevation -> +1 base + 1 downhill = 2
        assert gs.units[0].momentum == 2

    def test_max_turns_draw(self):
        g = _flat_grid()
        units = _make_units()
        gs = GameState(g, units)
        gs.round_number = 99
        # Finish Team A's turn
        gs.step(END_TURN_ACTION)
        gs.step(END_TURN_ACTION)
        gs.step(END_TURN_ACTION)
        # Finish Team B's turn — this should trigger round 100 = draw
        gs.step(END_TURN_ACTION)
        gs.step(END_TURN_ACTION)
        gs.step(END_TURN_ACTION)
        assert gs.done
        assert gs.winner is None
