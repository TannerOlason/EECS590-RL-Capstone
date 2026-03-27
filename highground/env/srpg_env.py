"""PettingZoo AEC environment wrapping the High Ground SRPG engine.

Agent naming:
    team0_unit0, team0_unit1, team0_unit2,
    team1_unit0, team1_unit1, team1_unit2

Each agent acts by issuing micro-actions (MOVE/ATTACK/END_TURN) until its
turn ends, then the next agent in the queue gets control.

Observation is a flat numpy vector so MaskablePPO can consume it directly.
The observation is always from the perspective of the current agent's team:
"my team" channels first, "enemy team" channels second.  This allows a
shared policy to be used for both sides.
"""

from __future__ import annotations

import functools
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from highground.engine.game_state import (
    ActionType,
    END_TURN_ACTION,
    MAX_TURNS,
    NUM_ACTIONS,
    GameState,
)
from highground.engine.grid import GRID_SIZE, Grid, Terrain
from highground.engine.units import (
    TEAM_A,
    TEAM_B,
    Unit,
    UnitClass,
)

# ── Constants ─────────────────────────────────────────────────────────

UNITS_PER_TEAM = 3

# Trial reward-shaping values (used when trial_shaping=True)
_DRAW_PENALTY    = -2.0    # replaces the default 0.0 draw terminal reward
_STEP_COST       = -0.001  # per micro-action survival tax for the acting team
_PROXIMITY_WEIGHT = 0.002  # reward per tile the acting team closes toward enemy

# V2 reward signal constants (Priority 1)
KILL_REWARD           = 1.5    # immediate bonus per enemy unit eliminated
ATTACK_ATTEMPT_BONUS  = 0.05   # reward for choosing any ATTACK action
_HP_SHAPING_SCALE     = 0.05   # HP delta weight (was 0.01; 5× larger attack gradient)
ROUND_SURVIVAL_COST   = -0.02  # charged to all agents at end of each full round
IN_RANGE_BONUS_PER_UNIT = 0.02 # reward per own unit that has an enemy in attack range

# Potential-based reward shaping (PBRS) — Ng, Harada & Russell (1999)
# F = γΦ(s') - Φ(s), applied per step to the acting team.
# Φ options:
#   "center" — Φ = (enemy avg dist to map centre − my avg dist to map centre)
#               normalised by the max possible centre dist.  Rewards occupying
#               the centre relative to the enemy.  Biased toward symmetric maps.
#   "enemy"  — Φ = −(avg min dist from my units to any enemy unit)
#               normalised by max manhattan dist.  Rewards closing distance to
#               the enemy regardless of map topology.  Better for QD maps.
_PBRS_WEIGHT = 0.05         # scales the PBRS delta; ~0.004/step when advancing
_PBRS_GAMMA  = 0.99         # should match MaskablePPO gamma for theoretical guarantee
TOTAL_UNITS = UNITS_PER_TEAM * 2

# Per-role combat shaping constants (activated in V4 Phase 4 via combat_shaping=True).
# Values chosen to nudge role behaviour without overwhelming the combat signal:
#   Vanguard  (slot 0) — Fighter/Charger-class units that should close and melee.
#                        Bonus for executing any attack with attack_range == 1.
#   Flanker   (slot 1) — medium unit; rewarded for lateral/diagonal repositioning.
#                        Bonus for choosing any diagonal move action (NE/NW/SE/SW).
#   Support   (slot 2) — Ranger/Siege-class units that should engage from distance.
#                        Bonus for executing any attack with attack_range >= 3.
_VANGUARD_CLOSE_ATTACK_BONUS = 0.03   # reward per attack by a range-1 unit
_FLANKER_DIAGONAL_BONUS      = 0.01   # reward per diagonal move (actions 4–7)
_SUPPORT_RANGE_ATTACK_BONUS  = 0.02   # reward per attack by a range-3+ unit

# Role assignment: fixed by the unit's local slot within its team (unit_id % UNITS_PER_TEAM).
# Both teams share the same role labels so the shared policy learns symmetric role behaviour.
ROLE_NAMES = ["Vanguard", "Flanker", "Support"]
N_ROLES = len(ROLE_NAMES)

# Observation layout (flat float32 vector):
#   Grid terrain:     13*13 = 169
#   Grid elevation:   13*13 = 169
#   Per-unit features x6: 6 * 11 = 66
#   Current unit index:   1
#   Round fraction:       1
#   Squad features:       9   ← coordination signals (6a)
#     [0] team_centroid_row        avg row of alive allies (perspective-corrected)
#     [1] team_centroid_col        avg col of alive allies (perspective-corrected)
#     [2] enemy_centroid_row       avg row of alive enemies (perspective-corrected)
#     [3] enemy_centroid_col       avg col of alive enemies (perspective-corrected)
#     [4] team_spread              mean abs deviation of allies from their centroid
#     [5] n_allies_alive           alive allied count / UNITS_PER_TEAM
#     [6] n_enemies_alive          alive enemy count  / UNITS_PER_TEAM
#     [7] acting_unit_ally_dist    min Manhattan dist to nearest alive ally (isolation)
#     [8] acting_unit_threat       enemies in attack-range of acting unit / UNITS_PER_TEAM
#   Role token:           3   ← one-hot role for the acting unit (6b)
#     [0] Vanguard (slot 0)        advance and engage
#     [1] Flanker  (slot 1)        off-centre manoeuvre
#     [2] Support  (slot 2)        hold back and fire
#   Total: 418
SQUAD_FEATURES = 9
GRID_CELLS = GRID_SIZE * GRID_SIZE
UNIT_FEATURES = 11  # x, y, hp_frac, team_is_mine, class(4 one-hot), move_frac, attacked, alive
OBS_SIZE = GRID_CELLS * 2 + TOTAL_UNITS * UNIT_FEATURES + 2 + SQUAD_FEATURES + N_ROLES


def _make_agent_name(team: int, local_idx: int) -> str:
    return f"team{team}_unit{local_idx}"


def _parse_agent_name(name: str) -> tuple[int, int]:
    # "team0_unit1" -> (0, 1)
    parts = name.split("_")
    return int(parts[0][-1]), int(parts[1][-1])


class HighGroundEnv(AECEnv):
    """PettingZoo AEC environment for the High Ground SRPG.

    Args:
        grid: The game grid.
        team_a_spawns: 3 (row, col) spawn positions for Team A.
        team_b_spawns: 3 (row, col) spawn positions for Team B.
        team_a_classes: Unit classes for Team A (default: Fighter, Charger, Ranger).
        team_b_classes: Unit classes for Team B (default: Fighter, Charger, Ranger).
        reward_mode: "sparse" (win=+1, lose=-1) or "shaped" (incremental rewards).
    """

    metadata = {"render_modes": ["human"], "name": "highground_v0"}

    def __init__(
        self,
        grid: Grid,
        team_a_spawns: list[tuple[int, int]],
        team_b_spawns: list[tuple[int, int]],
        team_a_classes: list[UnitClass] | None = None,
        team_b_classes: list[UnitClass] | None = None,
        reward_mode: str = "shaped",
        render_mode: str | None = None,
        trial_shaping: bool = False,
        win_reward_scale: float = 1.0,
        position_shaping: str | None = None,
        combat_shaping: bool = False,
        pbrs_weight: float | None = None,
    ) -> None:
        super().__init__()

        self._grid_template = grid
        self._spawns_a = team_a_spawns
        self._spawns_b = team_b_spawns
        self._classes_a = team_a_classes or [
            UnitClass.FIGHTER, UnitClass.CHARGER, UnitClass.RANGER,
        ]
        self._classes_b = team_b_classes or [
            UnitClass.FIGHTER, UnitClass.CHARGER, UnitClass.RANGER,
        ]
        self._reward_mode = reward_mode
        self._trial_shaping = trial_shaping
        self._win_reward_scale = win_reward_scale
        self._position_shaping = position_shaping
        self._combat_shaping = combat_shaping
        self._pbrs_weight = pbrs_weight
        self.render_mode = render_mode

        # Agent list
        self.possible_agents = [
            _make_agent_name(t, i)
            for t in (TEAM_A, TEAM_B)
            for i in range(UNITS_PER_TEAM)
        ]

        self._game: GameState | None = None

    # ── Spaces ────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict({
            "observation": spaces.Box(
                low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
            ),
            "action_mask": spaces.MultiBinary(NUM_ACTIONS),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        units = self._create_units()
        self._game = GameState(self._grid_template.copy(), units)

        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        # Track HP for shaped rewards
        self._prev_hp = {u.unit_id: u.hp for u in self._game.units}

        # Track enemy proximity for trial shaping
        if self._trial_shaping:
            self._prev_min_dist = {
                TEAM_A: self._compute_team_min_enemy_dist(TEAM_A),
                TEAM_B: self._compute_team_min_enemy_dist(TEAM_B),
            }

        # Cache potential values for PBRS
        if self._position_shaping is not None:
            self._prev_potential = {
                TEAM_A: self._position_potential(TEAM_A),
                TEAM_B: self._position_potential(TEAM_B),
            }

        self._update_agent_selection()

    def _create_units(self) -> list[Unit]:
        units = []
        for i, (cls, (r, c)) in enumerate(
            zip(self._classes_a, self._spawns_a)
        ):
            units.append(Unit(unit_id=i, team=TEAM_A, unit_class=cls, row=r, col=c))
        for i, (cls, (r, c)) in enumerate(
            zip(self._classes_b, self._spawns_b)
        ):
            units.append(Unit(unit_id=i + UNITS_PER_TEAM, team=TEAM_B, unit_class=cls, row=r, col=c))
        return units

    # ── Trial shaping helpers ─────────────────────────────────────────

    def _position_potential(self, team: int) -> float:
        """Φ(s) for PBRS.  Normalised to [-1, 1]; higher = better position.

        "center":  my team is closer to the map centre than the enemy is.
                   Assumes strategic value at the centre — works well on
                   symmetric maps, may mislead on asymmetric QD maps.

        "enemy":   my team is close to enemy units (negative of average min
                   Manhattan distance, normalised).  Map-agnostic — any map
                   requires closing distance to win.
        """
        g = self._game
        mine   = [u for u in g.units if u.team == team     and u.alive]
        theirs = [u for u in g.units if u.team != team and u.alive]
        if not mine or not theirs:
            return 0.0

        max_manhattan = (GRID_SIZE - 1) * 2  # 24 for 13×13

        if self._position_shaping == "center":
            centre = (GRID_SIZE - 1) / 2  # 6.0
            max_centre_dist = centre * 2   # 12.0
            my_avg    = sum(abs(u.row - centre) + abs(u.col - centre) for u in mine)   / len(mine)
            their_avg = sum(abs(u.row - centre) + abs(u.col - centre) for u in theirs) / len(theirs)
            # Positive when enemy is farther from centre (we hold the middle)
            return (their_avg - my_avg) / max_centre_dist

        if self._position_shaping == "enemy":
            avg_min = sum(
                min(abs(u.row - e.row) + abs(u.col - e.col) for e in theirs)
                for u in mine
            ) / len(mine)
            # Negative of distance: higher (less negative) = closer to enemy
            return -avg_min / max_manhattan

        return 0.0

    def _compute_team_min_enemy_dist(self, team: int) -> float:
        """Minimum Manhattan distance from any alive friendly unit to any alive enemy."""
        g = self._game
        mine   = [u for u in g.units if u.team == team     and u.alive]
        theirs = [u for u in g.units if u.team != team and u.alive]
        if not mine or not theirs:
            return 0.0
        return float(min(
            abs(u.row - e.row) + abs(u.col - e.col)
            for u in mine for e in theirs
        ))

    # ── Observe ───────────────────────────────────────────────────────

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        if self._game is None:
            return {
                "observation": np.zeros(OBS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(NUM_ACTIONS, dtype=np.int8),
            }

        team, _ = _parse_agent_name(agent)
        obs = self._build_observation(team)
        mask = self._build_mask_for_agent(agent)
        return {"observation": obs, "action_mask": mask}

    def state(self) -> np.ndarray:
        """Global joint state for the centralised critic (6c CTDE).

        Returns both teams' observations concatenated: 2 × OBS_SIZE = 836 floats.
        Team A's observation occupies indices [0:OBS_SIZE], Team B's [OBS_SIZE:2*OBS_SIZE].
        Called by PettingZooWrapper when return_state=True.
        """
        if self._game is None:
            return np.zeros(OBS_SIZE * 2, dtype=np.float32)
        obs_a = self._build_observation(TEAM_A)
        obs_b = self._build_observation(TEAM_B)
        return np.concatenate([obs_a, obs_b]).astype(np.float32)

    def _build_observation(self, perspective_team: int) -> np.ndarray:
        """Build a flat observation vector from the given team's perspective.

        For Team B the column axis of every spatial feature is flipped so that
        "my spawn is at low column values, enemy spawn is at high column values"
        holds for both teams.  This makes the shared policy's spatial intuition
        valid regardless of which side is acting.
        """
        g = self._game
        parts: list[np.ndarray] = []

        # Grid terrain (normalized: 0=normal, 0.5=rough, 1.0=uncrossable)
        terrain = g.grid.terrain.astype(np.float32) / 2.0
        if perspective_team == TEAM_B:
            terrain = terrain[:, ::-1]
        parts.append(terrain.flatten())

        # Grid elevation (normalized: 0, 0.5, 1.0)
        elev = g.grid.elevation.astype(np.float32) / 2.0
        if perspective_team == TEAM_B:
            elev = elev[:, ::-1]
        parts.append(elev.flatten())

        # Unit features: my team first, then enemy team
        my_units = [u for u in g.units if u.team == perspective_team]
        enemy_units = [u for u in g.units if u.team != perspective_team]

        for u in my_units + enemy_units:
            parts.append(self._unit_features(u, perspective_team))

        # Current unit index (which of the 6 units is acting, 0-5 normalized)
        # Map to perspective order: my team [0,1,2], enemy [3,4,5]
        cur = g.current_unit
        if cur.team == perspective_team:
            cur_idx = [u.unit_id for u in my_units].index(cur.unit_id)
        else:
            cur_idx = UNITS_PER_TEAM + [u.unit_id for u in enemy_units].index(cur.unit_id)
        parts.append(np.array([cur_idx / 5.0], dtype=np.float32))

        # Round fraction
        parts.append(np.array([g.round_number / MAX_TURNS], dtype=np.float32))

        # Squad coordination features (6a)
        parts.append(self._squad_features(cur, perspective_team))

        # Role-conditioning token (6b): one-hot for the acting unit's role slot.
        # Role is fixed by local slot within team (unit_id % UNITS_PER_TEAM), so
        # both teams map symmetrically: slot 0 = Vanguard, 1 = Flanker, 2 = Support.
        role_vec = np.zeros(N_ROLES, dtype=np.float32)
        role_vec[cur.unit_id % UNITS_PER_TEAM] = 1.0
        parts.append(role_vec)

        return np.concatenate(parts)

    def _unit_features(self, unit: Unit, perspective_team: int) -> np.ndarray:
        """11 features per unit."""
        if not unit.alive:
            return np.zeros(UNIT_FEATURES, dtype=np.float32)

        f = np.zeros(UNIT_FEATURES, dtype=np.float32)
        f[0] = unit.row / (GRID_SIZE - 1)
        # Mirror column for Team B so the policy always sees "my side = low col"
        col = unit.col if perspective_team == TEAM_A else (GRID_SIZE - 1 - unit.col)
        f[1] = col / (GRID_SIZE - 1)
        f[2] = unit.hp / unit.max_hp
        f[3] = 1.0 if unit.team == perspective_team else 0.0
        # One-hot class encoding (indices 4-7)
        f[4 + int(unit.unit_class)] = 1.0
        f[8] = unit.move_remaining / unit.move_range if unit.move_range > 0 else 0.0
        f[9] = 1.0 if unit.has_attacked else 0.0
        f[10] = 1.0  # alive flag
        return f

    def _squad_features(self, acting_unit: Unit, perspective_team: int) -> np.ndarray:
        """9 team-level coordination features (section 6a of SPATIAL_VS_SQUADDING.md)."""
        g = self._game
        max_manhattan = (GRID_SIZE - 1) * 2  # 24

        mine   = [u for u in g.units if u.team == perspective_team     and u.alive]
        theirs = [u for u in g.units if u.team != perspective_team and u.alive]

        def pcol(unit: Unit) -> float:
            return unit.col if perspective_team == TEAM_A else (GRID_SIZE - 1 - unit.col)

        # Team centroids (perspective-corrected, normalised to [0, 1])
        if mine:
            tc_row = sum(u.row for u in mine)    / len(mine) / (GRID_SIZE - 1)
            tc_col = sum(pcol(u) for u in mine)  / len(mine) / (GRID_SIZE - 1)
        else:
            tc_row = tc_col = 0.0

        if theirs:
            ec_row = sum(u.row for u in theirs)   / len(theirs) / (GRID_SIZE - 1)
            ec_col = sum(pcol(u) for u in theirs) / len(theirs) / (GRID_SIZE - 1)
        else:
            ec_row = ec_col = 0.0

        # Team spread: mean absolute deviation from centroid, normalised
        if len(mine) > 1:
            cr = tc_row * (GRID_SIZE - 1)
            cc = tc_col * (GRID_SIZE - 1)
            spread = sum(abs(u.row - cr) + abs(pcol(u) - cc) for u in mine) / len(mine) / max_manhattan
        else:
            spread = 0.0

        n_allies  = len(mine)   / UNITS_PER_TEAM
        n_enemies = len(theirs) / UNITS_PER_TEAM

        # Acting unit isolation: min Manhattan dist to nearest alive ally (excl. self)
        allies_excl = [u for u in mine if u.unit_id != acting_unit.unit_id]
        if allies_excl:
            ally_dist = min(
                abs(acting_unit.row - u.row) + abs(acting_unit.col - u.col)
                for u in allies_excl
            ) / max_manhattan
        else:
            ally_dist = 1.0  # fully isolated (last unit alive)

        # Threat: fraction of alive enemies within attack range of the acting unit
        threat = sum(
            1 for e in theirs
            if abs(acting_unit.row - e.row) + abs(acting_unit.col - e.col) <= e.attack_range
        ) / UNITS_PER_TEAM

        return np.array(
            [tc_row, tc_col, ec_row, ec_col, spread, n_allies, n_enemies, ally_dist, threat],
            dtype=np.float32,
        )

    def _build_mask_for_agent(self, agent: str) -> np.ndarray:
        """Action mask for a given agent. All zeros if it's not their turn."""
        g = self._game
        team, local_idx = _parse_agent_name(agent)

        # Map local index to unit_id
        unit_id = local_idx if team == TEAM_A else local_idx + UNITS_PER_TEAM

        if g.done or g.current_unit_id != unit_id:
            # Not this agent's turn — only action 7 (END_TURN) if we must
            # provide a valid action; PettingZoo expects us to return the
            # real mask only for the agent_selection.
            return np.zeros(NUM_ACTIONS, dtype=np.int8)

        mask = np.array(g.valid_action_mask(), dtype=np.int8)
        if team == TEAM_B:
            # Flip the E↔W component of every move action so the mask aligns
            # with the column-mirrored observation.  Cardinals: E(2)↔W(3).
            # Diagonals: NE(4)↔NW(5), SE(6)↔SW(7).
            mask[2], mask[3] = int(mask[3]), int(mask[2])
            mask[4], mask[5] = int(mask[5]), int(mask[4])
            mask[6], mask[7] = int(mask[7]), int(mask[6])
        return mask

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, action: int) -> None:
        if (
            self.terminations.get(self.agent_selection, False)
            or self.truncations.get(self.agent_selection, False)
        ):
            self._was_dead_step(action)
            return

        g = self._game
        agent = self.agent_selection
        team, _ = _parse_agent_name(agent)

        # Translate mirrored actions for Team B: flip the E↔W component of
        # every move so the physical direction matches the policy's intent
        # under the column-mirrored observation.
        physical_action = action
        if team == TEAM_B:
            if   action == 2: physical_action = 3   # E  → W
            elif action == 3: physical_action = 2   # W  → E
            elif action == 4: physical_action = 5   # NE → NW
            elif action == 5: physical_action = 4   # NW → NE
            elif action == 6: physical_action = 7   # SE → SW
            elif action == 7: physical_action = 6   # SW → SE

        # Capture acting unit ID before g.step() advances the turn pointer.
        acting_unit_id = g.current_unit_id

        prev_round = g.round_number
        result = g.step(physical_action)
        round_completed = not result.done and g.round_number > prev_round

        # Compute rewards
        self._assign_rewards(result, team, physical_action, round_completed, acting_unit_id)

        # Update terminations / truncations
        if result.done:
            for a in self.agents:
                self.terminations[a] = True

        self._update_agent_selection()

    def _assign_rewards(self, result, acting_team: int, action: int = END_TURN_ACTION, round_completed: bool = False, acting_unit_id: int = 0) -> None:
        """Assign rewards after a step."""
        # Reset rewards each step
        for a in self.agents:
            self.rewards[a] = 0.0

        g = self._game

        if self._reward_mode == "shaped":
            # 1. HP shaping: damage dealt = positive, damage taken = negative
            for u in g.units:
                hp_delta = u.hp - self._prev_hp[u.unit_id]
                if hp_delta != 0:
                    for a in self.agents:
                        a_team, _ = _parse_agent_name(a)
                        if u.team == a_team:
                            self.rewards[a] += hp_delta * _HP_SHAPING_SCALE
                        else:
                            self.rewards[a] -= hp_delta * _HP_SHAPING_SCALE
            self._prev_hp = {u.unit_id: u.hp for u in g.units}

            # 2. Kill reward: immediate bonus when an enemy is eliminated
            if result.unit_killed is not None:
                killed_unit = g.units[result.unit_killed]
                for a in self.agents:
                    a_team, _ = _parse_agent_name(a)
                    sign = +1 if a_team != killed_unit.team else -1
                    self.rewards[a] += sign * KILL_REWARD

            # 3. Attack attempt bonus: reward the choice to attack
            if ActionType.ATTACK_0 <= action <= ActionType.ATTACK_2:
                for a in self.agents:
                    a_team, _ = _parse_agent_name(a)
                    if a_team == acting_team:
                        self.rewards[a] += ATTACK_ATTEMPT_BONUS

            # 4. In-range bonus: reward maintaining attack contact (Priority 3)
            mine   = [u for u in g.units if u.team == acting_team and u.alive]
            theirs = [u for u in g.units if u.team != acting_team and u.alive]
            if mine and theirs:
                units_in_range = sum(
                    1 for u in mine
                    if any(
                        max(abs(u.row - e.row), abs(u.col - e.col)) <= u.attack_range
                        for e in theirs
                    )
                )
                if units_in_range > 0:
                    for a in self.agents:
                        a_team, _ = _parse_agent_name(a)
                        if a_team == acting_team:
                            self.rewards[a] += IN_RANGE_BONUS_PER_UNIT * units_in_range

            # 5. Round survival cost: time pressure per completed round (Priority 2)
            if round_completed:
                for a in self.agents:
                    self.rewards[a] += ROUND_SURVIVAL_COST

        if self._trial_shaping and not result.done:
            # Per-step survival cost: nudges the acting team to end games faster
            for a in self.agents:
                a_team, _ = _parse_agent_name(a)
                if a_team == acting_team:
                    self.rewards[a] += _STEP_COST

            # Proximity reward: reward closing distance to the nearest enemy
            curr_dist = self._compute_team_min_enemy_dist(acting_team)
            dist_closed = self._prev_min_dist[acting_team] - curr_dist
            if dist_closed > 0:
                for a in self.agents:
                    a_team, _ = _parse_agent_name(a)
                    if a_team == acting_team:
                        self.rewards[a] += dist_closed * _PROXIMITY_WEIGHT
            self._prev_min_dist[acting_team] = curr_dist

        if self._position_shaping is not None and not result.done:
            # PBRS — disabled when avg min dist < 3 tiles to prevent saturation
            # (Priority 3: keep PBRS only for the closing phase)
            curr_pot = self._position_potential(acting_team)
            curr_dist_pbrs = self._compute_team_min_enemy_dist(acting_team)
            if curr_dist_pbrs >= 3:
                prev_pot = self._prev_potential[acting_team]
                effective_pbrs_weight = self._pbrs_weight if self._pbrs_weight is not None else _PBRS_WEIGHT
                pbrs = effective_pbrs_weight * (_PBRS_GAMMA * curr_pot - prev_pot)
                for a in self.agents:
                    a_team, _ = _parse_agent_name(a)
                    if a_team == acting_team:
                        self.rewards[a] += pbrs
            # Always update potential to keep the baseline current
            self._prev_potential[acting_team] = curr_pot

        if self._combat_shaping and not result.done:
            # Role is fixed by slot within team: unit_id % UNITS_PER_TEAM.
            # Both teams share the same mapping: 0=Vanguard, 1=Flanker, 2=Support.
            role = acting_unit_id % UNITS_PER_TEAM
            acting_unit = self._game.units[acting_unit_id]
            is_attack = ActionType.ATTACK_0 <= action <= ActionType.ATTACK_2
            bonus = 0.0

            if role == 0 and is_attack and acting_unit.attack_range == 1:
                bonus = _VANGUARD_CLOSE_ATTACK_BONUS
            elif role == 1 and 4 <= action <= 7:
                bonus = _FLANKER_DIAGONAL_BONUS
            elif role == 2 and is_attack and acting_unit.attack_range >= 3:
                bonus = _SUPPORT_RANGE_ATTACK_BONUS

            if bonus != 0.0:
                for a in self.agents:
                    a_team, _ = _parse_agent_name(a)
                    if a_team == acting_team:
                        self.rewards[a] += bonus

        if result.done:
            # Draw penalty always applies (Priority 1: strengthen draw penalty)
            for a in self.agents:
                a_team, _ = _parse_agent_name(a)
                if result.winner is None:
                    self.rewards[a] += _DRAW_PENALTY
                elif result.winner == a_team:
                    self.rewards[a] += self._win_reward_scale
                else:
                    self.rewards[a] -= self._win_reward_scale

    # ── Agent selection ───────────────────────────────────────────────

    def _update_agent_selection(self) -> None:
        """Set agent_selection based on the engine's current_unit_id."""
        g = self._game
        if g.done:
            # Point to the first agent for final observation collection
            self.agent_selection = self.agents[0] if self.agents else self.possible_agents[0]
            return

        uid = g.current_unit_id
        unit = g.units[uid]
        local_idx = uid if unit.team == TEAM_A else uid - UNITS_PER_TEAM
        self.agent_selection = _make_agent_name(unit.team, local_idx)

    # ── Rendering (placeholder) ───────────────────────────────────────

    def render(self) -> None:
        if self.render_mode != "human":
            return
        # TODO: Pygame rendering in Phase 1 stretch goal
        g = self._game
        print(f"\n=== Round {g.round_number} | "
              f"Current: unit {g.current_unit_id} (Team {'A' if g.current_unit == TEAM_A else 'B'}) ===")
        for r in range(GRID_SIZE):
            row_str = ""
            for c in range(GRID_SIZE):
                unit_here = None
                for u in g.units:
                    if u.alive and u.row == r and u.col == c:
                        unit_here = u
                        break
                if unit_here:
                    sym = "FCRG"[int(unit_here.unit_class)]
                    sym = sym.upper() if unit_here.team == TEAM_A else sym.lower()
                    row_str += f" {sym}"
                elif g.grid.terrain[r, c] == Terrain.UNCROSSABLE:
                    row_str += " #"
                elif g.grid.terrain[r, c] == Terrain.ROUGH:
                    row_str += " ~"
                else:
                    row_str += f" {g.grid.elevation[r, c]}"
            print(row_str)

    def close(self) -> None:
        pass
