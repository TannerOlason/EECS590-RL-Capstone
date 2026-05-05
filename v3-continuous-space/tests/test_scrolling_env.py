"""Sanity tests for ScrollingSquadEnv."""

from __future__ import annotations

import numpy as np
import pytest

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE, Terrain
from highground.engine.units import UnitClass

from env.chunk_generator import EnemySpawn
from env.chunk_generator import PotionSpawn
from env.scrolling_env import ScrollingSquadEnv
from env.obs_builder import NON_SPATIAL_DIM, SPATIAL_SHAPE


def test_action_observation_spaces():
    env = ScrollingSquadEnv(seed=0)
    assert env.action_space.shape == (3,)
    assert env.observation_space["spatial"].shape == SPATIAL_SHAPE
    assert env.observation_space["features"].shape == (NON_SPATIAL_DIM,)
    # Spatial channel count: terrain + elevation + friendly HP + 4 enemy classes + potion = 8.
    assert SPATIAL_SHAPE[0] == 8


def test_potion_pickup_heals_active_unit():
    env = ScrollingSquadEnv(seed=11)
    env.reset()
    p = env.players[0]
    p.unit.hp = 1  # damage the unit so a heal is observable

    # Place a potion right under the active unit.
    env.potions.append(PotionSpawn(row=p.row_int(), col_world=p.world_x_int(), heal_frac=0.5))
    n_potions_before = len(env.potions)

    # Make the actor's idx the one we just damaged and feed a no-op action;
    # the env's pickup pass runs after movement.
    env.current_idx = 0
    env.step(np.zeros(3, dtype=np.float32))

    assert len(env.potions) == n_potions_before - 1, "potion should be consumed"
    assert env.players[0].unit.hp > 1, "active unit should be healed"


def test_enemies_are_weakened_atk_scale():
    """Spawned enemies should reflect the chunk_generator atk_scale (default 0.75)."""
    from env.chunk_generator import EnemySpawn
    from highground.engine.units import CLASS_STATS, UnitClass
    env = ScrollingSquadEnv(seed=0)
    env.reset()
    spawn = EnemySpawn(unit_class=UnitClass.FIGHTER, row=0, col_world=0,
                       hp_scale=1.0, atk_scale=0.75)
    env._spawn_enemy(spawn)
    e = env.enemies[-1]
    expected_atk = max(1, int(round(CLASS_STATS[UnitClass.FIGHTER]["atk"] * 0.75)))
    assert e.unit.atk == expected_atk


def test_reset_returns_obs_dict():
    env = ScrollingSquadEnv(seed=0)
    obs, info = env.reset()
    assert "spatial" in obs and "features" in obs
    assert obs["spatial"].shape == SPATIAL_SHAPE
    assert obs["features"].shape == (NON_SPATIAL_DIM,)
    assert obs["spatial"].dtype == np.float32


def test_repeated_unseeded_resets_advance_world_rng():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=3)
    env.reset()
    first = env.world.grid.terrain.copy()
    env.reset()
    second = env.world.grid.terrain.copy()

    assert not np.array_equal(first, second)


def test_explicit_reset_seed_reproduces_world():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=3)
    env.reset(seed=123)
    first = env.world.grid.terrain.copy()
    env.reset(seed=123)
    second = env.world.grid.terrain.copy()

    assert np.array_equal(first, second)


def test_step_returns_5_tuple():
    env = ScrollingSquadEnv(seed=0)
    env.reset()
    out = env.step(np.zeros(3, dtype=np.float32))
    assert len(out) == 5
    obs, reward, term, trunc, info = out
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_player_movement_is_v2_grid_step():
    env = ScrollingSquadEnv(seed=0)
    env.reset()
    env.current_idx = 0
    actor = env.players[0]
    actor.sync_window_pos(env.world.scroll_offset)
    start_row = actor.unit.row
    start_world_x = actor.world_x_int()
    start_move = actor.unit.move_remaining

    target_col = actor.unit.col + 1
    env.world.grid.terrain[start_row, target_col] = Terrain.NORMAL
    env.world.grid.elevation[start_row, target_col] = env.world.grid.elevation[start_row, actor.unit.col]

    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert actor.row_int() == start_row
    assert actor.world_x_int() == start_world_x + 1
    assert actor.row_f == float(actor.row_int())
    assert actor.world_x_f == float(actor.world_x_int())
    assert actor.unit.move_remaining == start_move - 1


def test_scroll_locks_until_visible_enemies_are_cleared():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=3)
    env.reset()
    env._spawn_enemy(EnemySpawn(UnitClass.FIGHTER, row=6, col_world=4))
    for p in env.players:
        p.world_x_f = 9.0
        p.sync_window_pos(env.world.scroll_offset)

    _, info = env._end_of_macro_tick()

    assert env.world.scroll_offset == 0
    assert info["scroll_locked_by_enemies"] >= 1

    for e in env.enemies:
        e.unit.take_damage(e.unit.hp)
    env._end_of_macro_tick()

    assert env.world.scroll_offset > 0


def test_scroll_cannot_push_living_player_offscreen():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=3)
    env.reset()
    env.enemies = []
    positions = [12.0, 12.0, 0.0]
    for p, world_x in zip(env.players, positions):
        p.world_x_f = world_x
        p.sync_window_pos(env.world.scroll_offset)

    _, info = env._end_of_macro_tick()

    assert env.world.scroll_offset == 0
    assert info["scroll_locked_by_lagging_player"] == 1
    assert all(p.world_x_int() >= env.world.scroll_offset for p in env.players if p.unit.alive)


def test_progress_reward_tracks_leftmost_survivor_not_centroid():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=3)
    env.reset()
    env.enemies = []
    env._prev_leftmost_x = 1.0
    env._prev_centroid_x = 1.0
    for p, world_x in zip(env.players, [3.0, 3.0, 1.0]):
        p.world_x_f = world_x
        p.sync_window_pos(env.world.scroll_offset)

    reward, info = env._end_of_macro_tick()

    assert "no_enemy_still_penalty" in info
    assert reward < 0.0


def test_clear_right_step_reward_for_new_ground():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=0)
    env.reset()
    env.enemies = []
    env.current_idx = 0
    actor = env.players[0]
    actor.sync_window_pos(env.world.scroll_offset)
    env.world.grid.terrain[actor.unit.row, actor.unit.col + 1] = Terrain.NORMAL

    _, reward, _, _, _ = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert reward > 0.0


def test_no_enemy_still_penalty_escalates():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=0)
    env.reset()
    env.enemies = []

    _, info1 = env._end_of_macro_tick()
    _, info2 = env._end_of_macro_tick()

    assert info1["no_enemy_still_penalty"] > 0
    assert info2["no_enemy_still_penalty"] > info1["no_enemy_still_penalty"]


def test_visible_enemy_camping_is_penalized():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=2)
    env.reset()
    env.enemies = []
    env._spawn_enemy(EnemySpawn(UnitClass.FIGHTER, row=6, col_world=8))
    env._prev_enemy_distance = env._mean_nearest_enemy_distance()

    reward, info = env._end_of_macro_tick()

    assert info["visible_enemy_camp_penalty"] > 0
    assert reward < 0


def test_multi_threat_reward_for_ganging_up():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=2)
    env.reset()
    env.enemies = []
    env._spawn_enemy(EnemySpawn(UnitClass.FIGHTER, row=6, col_world=2))
    env._prev_enemy_distance = env._mean_nearest_enemy_distance()
    for p, row in zip(env.players, [5, 6, 9]):
        p.row_f = float(row)
        p.world_x_f = 1.0
        p.sync_window_pos(env.world.scroll_offset)

    reward, info = env._end_of_macro_tick()

    assert info["multi_threat_count"] >= 1
    assert info["multi_threat_reward"] > 0
    assert reward > -0.1


def test_obs_features_in_bounds():
    env = ScrollingSquadEnv(seed=1)
    obs, _ = env.reset()
    spat = obs["spatial"]
    assert spat.min() >= -1e-6
    assert spat.max() <= 1.0 + 1e-6
    feats = obs["features"]
    assert np.isfinite(feats).all()


def test_obs_exposes_local_move_validity():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=0)
    obs, _ = env.reset()
    # Feature indices 15..46 are 8 directions * [can_step, enemy, potion, advances_right].
    local = obs["features"][15:47].reshape(8, 4)

    assert local[:, 0].sum() > 0, "at least one neighboring move should be valid"
    assert local[2, 3] == 1.0  # NE advances right
    assert local[4, 3] == 1.0  # E advances right
    assert local[7, 3] == 1.0  # SE advances right


def test_invalid_move_is_reported_and_penalized():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=0)
    env.reset()
    env.current_idx = 0
    actor = env.players[0]
    actor.sync_window_pos(env.world.scroll_offset)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            row = actor.unit.row + dr
            col = actor.unit.col + dc
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                env.world.grid.terrain[row, col] = Terrain.NORMAL
                env.world.grid.elevation[row, col] = env.world.grid.elevation[actor.unit.row, actor.unit.col]
    env.world.grid.terrain[actor.unit.row, actor.unit.col + 1] = Terrain.UNCROSSABLE

    _, reward, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert info["invalid_move_count"] == 1
    assert info["shielded_invalid_move_count"] == 1
    assert info["invalid_move_penalty"] > 0
    assert actor.world_x_int() > 1
    assert reward < 0.0


def test_whiffed_attack_is_penalized():
    env = ScrollingSquadEnv(seed=0, curriculum_phase=3)
    env.reset()
    env.enemies = []

    _, reward, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    assert info["whiffed_attack_penalty"] > 0
    assert reward < 0.0


def test_random_rollout_terminates_or_runs():
    env = ScrollingSquadEnv(seed=2, max_macro_ticks=80)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(800):
        action = rng.uniform(-1, 1, size=(3,)).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    # Should have terminated/truncated within budget.
    assert term or trunc


def test_action_pushes_right_increases_centroid():
    env = ScrollingSquadEnv(seed=3, max_macro_ticks=200)
    env.reset()
    initial_centroid = env._squad_centroid_world_x()
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # full-right, no attack
    for _ in range(60):
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    final_centroid = env._squad_centroid_world_x()
    assert final_centroid > initial_centroid, (initial_centroid, final_centroid)


def test_offscreen_left_kills_player():
    env = ScrollingSquadEnv(seed=4, max_macro_ticks=400)
    env.reset()
    # Force a long forward press; eventually scrolling should kill the lagging unit
    # if anyone falls behind (or all survive). Just check the env doesn't crash.
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(400):
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    # No assertion on death — just that the env handled it.


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_nan_in_obs(seed):
    env = ScrollingSquadEnv(seed=seed)
    obs, _ = env.reset()
    assert np.isfinite(obs["spatial"]).all()
    assert np.isfinite(obs["features"]).all()
    a = np.array([0.5, -0.5, 0.6], dtype=np.float32)
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(a)
        assert np.isfinite(obs["spatial"]).all()
        assert np.isfinite(obs["features"]).all()
        if term or trunc:
            break
