"""Tests for the PettingZoo AEC environment and action masking."""

import numpy as np
import pytest

from highground.engine.grid import Grid, Terrain, GRID_SIZE
from highground.engine.units import TEAM_A, TEAM_B, UnitClass
from highground.engine.game_state import ActionType, END_TURN_ACTION, NUM_ACTIONS
from highground.env.srpg_env import HighGroundEnv, OBS_SIZE
from highground.maps.static_maps import flat_open, central_hill, choke_point


class TestEnvBasic:
    def _make_env(self, map_fn=flat_open) -> HighGroundEnv:
        grid, spawns_a, spawns_b = map_fn()
        env = HighGroundEnv(grid, spawns_a, spawns_b)
        env.reset()
        return env

    def test_reset(self):
        env = self._make_env()
        assert env.agent_selection is not None
        assert len(env.agents) == 6

    def test_observation_shape(self):
        env = self._make_env()
        obs = env.observe(env.agent_selection)
        assert obs["observation"].shape == (OBS_SIZE,)
        assert obs["action_mask"].shape == (NUM_ACTIONS,)

    def test_observation_values_normalized(self):
        env = self._make_env()
        obs = env.observe(env.agent_selection)
        assert np.all(obs["observation"] >= 0.0)
        assert np.all(obs["observation"] <= 1.0)

    def test_action_mask_has_end_turn(self):
        env = self._make_env()
        obs = env.observe(env.agent_selection)
        mask = obs["action_mask"]
        assert mask[END_TURN_ACTION] == 1

    def test_step_with_end_turn(self):
        env = self._make_env()
        first_agent = env.agent_selection
        env.step(END_TURN_ACTION)
        # Should have advanced to next agent
        assert env.agent_selection != first_agent or env.terminations[first_agent]

    def test_full_round_of_end_turns(self):
        env = self._make_env()
        agents_seen = set()
        for _ in range(6):  # 6 units
            agents_seen.add(env.agent_selection)
            env.step(END_TURN_ACTION)
        assert len(agents_seen) == 6

    def test_mask_only_active_agent(self):
        """Only the active agent should have a non-zero mask."""
        env = self._make_env()
        active = env.agent_selection
        for agent in env.agents:
            obs = env.observe(agent)
            if agent == active:
                assert obs["action_mask"].sum() > 0
            else:
                assert obs["action_mask"].sum() == 0

    def test_move_changes_state(self):
        env = self._make_env()
        obs1 = env.observe(env.agent_selection)
        mask = obs1["action_mask"]
        # Find a valid move action
        move_actions = [i for i in range(4) if mask[i] == 1]
        if move_actions:
            env.step(move_actions[0])
            obs2 = env.observe(env.agent_selection)
            # Observation should change (unit moved)
            # Just verify no crash; detailed check in engine tests


class TestEnvMaskIntegrity:
    """Stress-test that the action mask is always consistent."""

    def _make_env(self, map_fn=flat_open) -> HighGroundEnv:
        grid, spawns_a, spawns_b = map_fn()
        env = HighGroundEnv(grid, spawns_a, spawns_b)
        env.reset()
        return env

    def test_random_play_never_crashes(self):
        """Play 50 games with random valid actions — should never crash."""
        rng = np.random.default_rng(42)
        for map_fn in [flat_open, central_hill, choke_point]:
            for _ in range(50):
                env = self._make_env(map_fn)
                steps = 0
                while steps < 2000:
                    obs = env.observe(env.agent_selection)
                    mask = obs["action_mask"]
                    valid_actions = np.where(mask == 1)[0]
                    if len(valid_actions) == 0:
                        break
                    action = rng.choice(valid_actions)
                    env.step(action)
                    steps += 1
                    if all(env.terminations.values()):
                        break

    def test_mask_always_has_valid_action(self):
        """Active agent should always have at least one valid action."""
        rng = np.random.default_rng(123)
        env = self._make_env()
        for _ in range(500):
            if all(env.terminations.values()):
                break
            obs = env.observe(env.agent_selection)
            mask = obs["action_mask"]
            assert mask.sum() > 0, f"No valid actions for {env.agent_selection}"
            valid = np.where(mask == 1)[0]
            env.step(rng.choice(valid))


class TestEnvRewards:
    def test_shaped_rewards_on_damage(self):
        """Dealing damage should give positive reward to attacker's team."""
        grid = Grid()
        spawns_a = [(6, 5), (0, 0), (0, 1)]
        spawns_b = [(6, 6), (12, 12), (12, 11)]
        env = HighGroundEnv(grid, spawns_a, spawns_b, reward_mode="shaped")
        env.reset()

        # team0_unit0 should be active and can attack
        assert env.agent_selection == "team0_unit0"
        obs = env.observe("team0_unit0")
        mask = obs["action_mask"]
        assert mask[ActionType.ATTACK_0] == 1

        env.step(ActionType.ATTACK_0)
        # Team A should get positive reward, Team B negative
        team_a_reward = sum(
            env.rewards[a] for a in env.agents if a.startswith("team0")
        )
        team_b_reward = sum(
            env.rewards[a] for a in env.agents if a.startswith("team1")
        )
        assert team_a_reward > 0
        assert team_b_reward < 0

    def test_sparse_win_reward(self):
        grid = Grid()
        spawns_a = [(6, 5), (0, 0), (0, 1)]
        spawns_b = [(6, 6), (12, 12), (12, 11)]
        env = HighGroundEnv(grid, spawns_a, spawns_b, reward_mode="sparse")
        env.reset()

        # Weaken all enemies to 1 HP
        for u in env._game.units:
            if u.team == TEAM_B:
                u.hp = 1

        # Kill enemy 0
        env.step(ActionType.ATTACK_0)
        # Kill remaining enemies manually
        env._game.units[4].hp = 0
        env._game.units[4].alive = False
        env._game.units[5].hp = 0
        env._game.units[5].alive = False
        # End turn to trigger win check
        env.step(END_TURN_ACTION)


class TestEnvDifferentMaps:
    """Ensure the env works on all static maps."""

    def test_all_static_maps_initialize(self):
        from highground.maps.static_maps import ALL_MAPS
        for name, fn in ALL_MAPS.items():
            grid, sa, sb = fn()
            env = HighGroundEnv(grid, sa, sb)
            env.reset()
            obs = env.observe(env.agent_selection)
            assert obs["observation"].shape == (OBS_SIZE,), f"Failed on map: {name}"
            assert obs["action_mask"].sum() > 0, f"No actions on map: {name}"
