"""SB3-compatible single-agent wrapper around the PettingZoo AEC env.

MaskablePPO from sb3-contrib needs a standard Gymnasium env with an
`action_masks()` method.  This wrapper controls ONE team's perspective:
 - When it's the controlled team's turn, it returns observations and
   expects actions from the RL policy.
 - When it's the opponent's turn, the opponent policy is called internally.

Supports two modes:
 - self_play=True:  The same policy acts for both teams (shared weights).
 - self_play=False: An opponent_fn callback provides the opponent's actions.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from highground.engine.game_state import END_TURN_ACTION, NUM_ACTIONS
from highground.engine.grid import Grid
from highground.engine.units import TEAM_A, TEAM_B, UnitClass
from highground.env.srpg_env import HighGroundEnv, OBS_SIZE, _parse_agent_name


class SB3SRPGWrapper(gymnasium.Env):
    """Gymnasium wrapper for MaskablePPO training.

    Args:
        aec_env: The underlying PettingZoo AEC environment.
        controlled_team: Which team (0 or 1) the RL agent controls.
        opponent_fn: Callable(obs_dict) -> action for the opponent.
                     If None and self_play=False, opponent picks random valid actions.
        self_play: If True, the agent's own policy is used for the opponent
                   (the training loop handles this via action_masks).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        aec_env: HighGroundEnv,
        controlled_team: int = TEAM_A,
        opponent_fn: Callable | None = None,
        self_play: bool = False,
    ) -> None:
        super().__init__()
        self.aec = aec_env
        self.controlled_team = controlled_team
        self.opponent_fn = opponent_fn
        self.self_play = self_play
        self._rng = np.random.default_rng()

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(NUM_ACTIONS),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.aec.reset(seed=seed)
        self._skip_opponent_turns()
        return self._get_obs(), {}

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        # Take the controlled agent's action
        self.aec.step(action)
        reward = self._collect_team_reward()
        done = all(self.aec.terminations.values())
        truncated = all(self.aec.truncations.values())

        if not done:
            self._skip_opponent_turns()

        done = all(self.aec.terminations.values())
        truncated = all(self.aec.truncations.values())
        obs = self._get_obs()

        info: dict = {}
        if done or truncated:
            g = self.aec._game
            if g.winner is None:
                info["outcome"] = "draw"
            elif g.winner == self.controlled_team:
                info["outcome"] = "win"
            else:
                info["outcome"] = "loss"
            info["survivors"] = sum(
                1 for u in g.units if u.alive and u.team == self.controlled_team
            )

        return obs, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        """Required by MaskablePPO."""
        obs = self.aec.observe(self.aec.agent_selection)
        return obs["action_mask"]

    # ── Internal ──────────────────────────────────────────────────────

    def _get_obs(self) -> dict[str, np.ndarray]:
        if all(self.aec.terminations.values()):
            return {
                "observation": np.zeros(OBS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(NUM_ACTIONS, dtype=np.int8),
            }
        return self.aec.observe(self.aec.agent_selection)

    def _skip_opponent_turns(self) -> None:
        """Execute opponent's micro-actions until it's the controlled team's turn again."""
        while not all(self.aec.terminations.values()):
            agent = self.aec.agent_selection
            team, _ = _parse_agent_name(agent)
            if team == self.controlled_team:
                break  # It's our turn

            obs = self.aec.observe(agent)
            mask = obs["action_mask"]
            if mask.sum() == 0:
                break

            if self.opponent_fn is not None:
                action = self.opponent_fn(obs)
            else:
                valid = np.where(mask == 1)[0]
                action = self._rng.choice(valid)
            self.aec.step(action)

    def _collect_team_reward(self) -> float:
        """Sum rewards for all agents on the controlled team."""
        total = 0.0
        for a in self.aec.agents:
            team, _ = _parse_agent_name(a)
            if team == self.controlled_team:
                total += self.aec.rewards.get(a, 0.0)
        return total
