"""BenchMARL TaskClass for the High Ground SRPG environment (6c CTDE/MAPPO).

This module provides the glue between HighGroundEnv (PettingZoo AEC) and
BenchMARL's Experiment / MAPPO algorithm.  Key design decisions:

  - Agents are split into two groups: "team_a" (team0_unit*) and "team_b"
    (team1_unit*).  BenchMARL creates one ACTOR shared across both groups
    (share_policy_params=True default) and a SEPARATE CRITIC per group.
    This is critical for a zero-sum game: a shared critic over all 6 agents
    averages +1 and -1 terminal rewards to ~0, killing the advantage signal.
    Separate critics let each team's value function learn independently.

  - The global state is produced by HighGroundEnv.state(), which concatenates
    both teams' observations.  This enables true CTDE: the critic has a
    complete view of the game while the actor only uses local observations.

  - AEC turn structure is preserved: use_mask=True ensures that only the
    active agent's actions count at each step; the others are masked.

  - Map groups ("all", "all_static", "all_obstacle") pick a random map on
    each env factory call.  This makes every batch cover a variety of maps,
    which is important for curriculum phases 2 and 3.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper

from benchmarl.environments import TaskClass

from highground.engine.game_state import MAX_TURNS, NUM_ACTIONS
from highground.env.srpg_env import HighGroundEnv, OBS_SIZE, UNITS_PER_TEAM
from highground.maps.static_maps import ALL_MAPS, STATIC_MAPS, OBSTACLE_MAPS


# ── Constants ─────────────────────────────────────────────────────────────────

# Conservative upper-bound on steps per episode.
# MAX_TURNS rounds × 6 units × up to 5 micro-actions each
_MAX_EPISODE_STEPS = MAX_TURNS * UNITS_PER_TEAM * 2 * 5

_MAP_GROUPS: dict = {
    "all":          ALL_MAPS,
    "all_static":   STATIC_MAPS,
    "all_obstacle": OBSTACLE_MAPS,
}


# ── TaskClass ─────────────────────────────────────────────────────────────────

class HighGroundTaskClass(TaskClass):
    """BenchMARL task wrapping HighGroundEnv for MAPPO/CTDE training.

    Task config keys
    ----------------
    map_name : str
        "flat_open", "central_hill", or any key in ALL_MAPS.
        Also accepts group aliases: "all", "all_static", "all_obstacle".
        Group aliases pick a random map on each env creation (per-episode map
        variety during collection).
    trial_shaping : bool  (default False)
        Enable experimental reward shaping (draw penalty, step cost, proximity).
    win_reward_scale : float  (default 1.0)
        Multiplier on the terminal win/loss reward.
    position_shaping : str | None  (default None)
        PBRS potential: None (off), "center", or "enemy".
    combat_shaping : bool  (default False)
        Enable per-role combat incentive hooks (6b groundwork, currently no-op).
    """

    @staticmethod
    def env_name() -> str:
        return "highground"

    # ── Factory ───────────────────────────────────────────────────────────────

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: str,
    ) -> Callable[[], EnvBase]:
        config = self.config

        def _make() -> EnvBase:
            map_name = config.get("map_name", "flat_open")
            if map_name in _MAP_GROUPS:
                map_fn = random.choice(list(_MAP_GROUPS[map_name].values()))
            else:
                map_fn = ALL_MAPS[map_name]

            grid, spawns_a, spawns_b = map_fn()
            aec = HighGroundEnv(
                grid, spawns_a, spawns_b,
                reward_mode="shaped",
                trial_shaping=config.get("trial_shaping", False),
                win_reward_scale=config.get("win_reward_scale", 1.0),
                position_shaping=config.get("position_shaping", None),
                combat_shaping=config.get("combat_shaping", False),
                pbrs_weight=config.get("pbrs_weight", None),
            )
            # Split agents by team prefix so BenchMARL creates separate critics.
            # Agent IDs are "team0_unit{0,1,2}" and "team1_unit{0,1,2}".
            team_a = [a for a in aec.possible_agents if a.startswith("team0")]
            team_b = [a for a in aec.possible_agents if a.startswith("team1")]
            return PettingZooWrapper(
                env=aec,
                use_mask=True,
                return_state=True,
                group_map={"team_a": team_a, "team_b": team_b},
                categorical_actions=True,
                seed=seed,
                device=device,
            )

        return _make

    # ── Capabilities ──────────────────────────────────────────────────────────

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return _MAX_EPISODE_STEPS

    def has_render(self, env: EnvBase) -> bool:
        return False

    # ── Specs ─────────────────────────────────────────────────────────────────

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        """Actor observation: one 418-dim vector per agent."""
        obs_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_spec = obs_spec[group]
            for key in list(group_spec.keys()):
                if key != "observation":
                    del group_spec[key]
        if "state" in obs_spec.keys():
            del obs_spec["state"]
        return obs_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        """Centralised critic state: 836-dim joint observation (both teams)."""
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        """Per-step valid action mask: [6, 8] booleans (which of 8 actions are legal)."""
        obs_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_spec = obs_spec[group]
            for key in list(group_spec.keys()):
                if key != "action_mask":
                    del group_spec[key]
            if group_spec.is_empty():
                del obs_spec[group]
        if "state" in obs_spec.keys():
            del obs_spec["state"]
        return None if obs_spec.is_empty() else obs_spec
