"""Side-scrolling continuous-action multi-unit env (V3).

A 3-unit player squad moves through an infinite, procedurally generated
strip of V2-style terrain that scrolls right as the squad advances. Simple
scripted enemies spawn ahead and attack on contact / at range. Episode
ends when all 3 player units are dead.

Action space (from SB3's perspective): Box(low=-1, high=1, shape=(3,))
    a[0] = vx        — desired grid-step x direction (+ = right)
    a[1] = vy        — desired grid-step y direction (+ = down)
    a[2] = atk_intent — > 0.5 ⇒ attempt attack on nearest enemy in range

Movement is still V2 tile movement: the continuous `(vx, vy)` vector is
quantized into one of V2's 8 directions, checked with `can_step`, charged the
destination terrain cost, and applied as a one-tile move. Combat also uses V2
attack range, `has_attacked`, Charger momentum reset, and `compute_damage`.

Multi-agent rotation: SB3 sees a single-agent env. Internally the env
rotates `current_idx` over the 3 player units; each gym step() consumes
one unit's action. After all 3 units have stepped (a "macro tick"),
enemies act and the world may scroll.

Reward (per gym step, as described in the V3 plan):
    + 0.05 * Δ centroid_world_x   (clamped >= 0)
    - 0.005                        per micro-step (stagnation)
    + 1.0                          per enemy killed by this micro-action
    + 0.02 * hp dealt this tick
    - 0.02 * hp lost this tick
    - 1.5                          per friendly killed this tick (off-screen or 0 HP)
    - 3.0                          terminal: all friendlies dead
    + 0.01                         per macro tick alive after world_x_centroid > 200
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import _path_shim  # noqa: F401
from highground.engine.combat import compute_damage
from highground.engine.grid import GRID_SIZE
from highground.engine.pathfinding import can_step, tiles_in_attack_range
from highground.engine.units import (
    CLASS_STATS,
    DIRECTION_DELTAS,
    TEAM_A,
    TEAM_B,
    Direction,
    Unit,
    UnitClass,
)

from env.chunk_generator import PotionSpawn
from env.enemy_ai import EnemyAction, make_ai_for
from env.infinite_grid import InfiniteGrid
from env.obs_builder import (
    NON_SPATIAL_DIM,
    N_AGENTS,
    SPATIAL_SHAPE,
    build_obs,
)
from env.unit_state import UnitState


# ── Tunable constants ─────────────────────────────────────────────────────
ATTACK_THRESH   = 0.5   # attack_intent > this triggers attack attempt
MOVE_DEADZONE   = 0.25  # below this, a velocity component is treated as no-op
SCROLL_TRIGGER  = (GRID_SIZE - 1) / 2.0  # centroid window-col triggering scroll
MAX_MACRO_TICKS = 600   # hard cap on episode length


PLAYER_CLASSES = (UnitClass.FIGHTER, UnitClass.RANGER, UnitClass.CHARGER)


V2_STYLE_CURRICULUM_PHASES: tuple[dict[str, Any], ...] = (
    {
        "name": "phase1_scroll_foundation",
        "progress_reward": 0.18,
        "step_cost": 0.002,
        "idle_cost": 0.010,
        "invalid_move_penalty": 0.030,
        "attack_attempt_bonus": 0.000,
        "whiffed_attack_penalty": 0.015,
        "in_range_bonus_per_unit": 0.000,
        "enemy_proximity_weight": 0.000,
        "multi_threat_reward": 0.000,
        "focus_fire_bonus": 0.000,
        "visible_enemy_camp_penalty_base": 0.000,
        "visible_enemy_camp_penalty_cap": 0.000,
        "enemy_movement_enabled": True,
        "damage_scale": 0.020,
        "kill_reward": 1.0,
        "no_enemy_still_penalty_base": 0.010,
        "no_enemy_still_penalty_cap": 0.120,
        "clear_right_step_reward": 0.020,
        "lag_lock_penalty": 0.040,
        "column_config": {
            "enemy_spawn_scale": 0.0,
            "enemy_spawn_min_x": 999_999,
            "terrain_block_scale": 0.0,
            "rough_scale": 0.0,
            "potion_p": 0.04,
        },
    },
    {
        "name": "phase2_weak_contact",
        "progress_reward": 0.14,
        "step_cost": 0.003,
        "idle_cost": 0.008,
        "invalid_move_penalty": 0.035,
        "attack_attempt_bonus": 0.000,
        "whiffed_attack_penalty": 0.020,
        "in_range_bonus_per_unit": 0.000,
        "enemy_proximity_weight": 0.030,
        "multi_threat_reward": 0.030,
        "focus_fire_bonus": 0.080,
        "visible_enemy_camp_penalty_base": 0.020,
        "visible_enemy_camp_penalty_cap": 0.180,
        "enemy_movement_enabled": False,
        "damage_scale": 0.030,
        "kill_reward": 1.25,
        "no_enemy_still_penalty_base": 0.008,
        "no_enemy_still_penalty_cap": 0.100,
        "clear_right_step_reward": 0.015,
        "lag_lock_penalty": 0.035,
        "column_config": {
            "enemy_spawn_scale": 0.45,
            "enemy_spawn_min_x": 10,
            "enemy_hp_scale_mult": 0.50,
            "enemy_atk_scale": 0.50,
            "terrain_block_scale": 0.15,
            "rough_scale": 0.25,
            "potion_p": 0.04,
        },
    },
    {
        "name": "phase3_full_approach",
        "progress_reward": 0.11,
        "step_cost": 0.004,
        "idle_cost": 0.006,
        "invalid_move_penalty": 0.040,
        "attack_attempt_bonus": 0.000,
        "whiffed_attack_penalty": 0.025,
        "in_range_bonus_per_unit": 0.000,
        "enemy_proximity_weight": 0.020,
        "multi_threat_reward": 0.025,
        "focus_fire_bonus": 0.070,
        "visible_enemy_camp_penalty_base": 0.015,
        "visible_enemy_camp_penalty_cap": 0.140,
        "enemy_movement_enabled": True,
        "damage_scale": 0.035,
        "kill_reward": 1.50,
        "no_enemy_still_penalty_base": 0.006,
        "no_enemy_still_penalty_cap": 0.080,
        "clear_right_step_reward": 0.012,
        "lag_lock_penalty": 0.030,
        "column_config": {
            "enemy_spawn_scale": 0.75,
            "enemy_spawn_min_x": 12,
            "enemy_hp_scale_mult": 0.50,
            "enemy_atk_scale": 0.50,
            "terrain_block_scale": 0.45,
            "rough_scale": 0.65,
            "potion_p": 0.035,
        },
    },
    {
        "name": "phase4_full_pressure",
        "progress_reward": 0.09,
        "step_cost": 0.005,
        "idle_cost": 0.005,
        "invalid_move_penalty": 0.045,
        "attack_attempt_bonus": 0.000,
        "whiffed_attack_penalty": 0.030,
        "in_range_bonus_per_unit": 0.000,
        "enemy_proximity_weight": 0.012,
        "multi_threat_reward": 0.018,
        "focus_fire_bonus": 0.050,
        "visible_enemy_camp_penalty_base": 0.010,
        "visible_enemy_camp_penalty_cap": 0.100,
        "enemy_movement_enabled": True,
        "damage_scale": 0.040,
        "kill_reward": 1.50,
        "no_enemy_still_penalty_base": 0.005,
        "no_enemy_still_penalty_cap": 0.060,
        "clear_right_step_reward": 0.010,
        "lag_lock_penalty": 0.025,
        "column_config": {
            "enemy_spawn_scale": 1.0,
            "enemy_spawn_min_x": 8,
            "enemy_hp_scale_mult": 0.50,
            "enemy_atk_scale": 0.50,
            "terrain_block_scale": 1.0,
            "rough_scale": 1.0,
            "potion_p": 0.03,
        },
    },
)


def _make_player_unit(unit_id: int, unit_class: UnitClass) -> Unit:
    return Unit(
        unit_id=unit_id,
        team=TEAM_A,
        unit_class=unit_class,
        row=GRID_SIZE // 2,
        col=0,
    )


def _make_enemy_unit(
    unit_id: int, unit_class: UnitClass, hp_scale: float, atk_scale: float = 1.0
) -> Unit:
    u = Unit(
        unit_id=unit_id,
        team=TEAM_B,
        unit_class=unit_class,
        row=0,
        col=0,
    )
    u.max_hp = max(1, int(round(CLASS_STATS[unit_class]["hp"] * hp_scale)))
    u.hp = u.max_hp
    u.atk = max(1, int(round(CLASS_STATS[unit_class]["atk"] * atk_scale)))
    return u


class ScrollingSquadEnv(gym.Env):
    """Gymnasium env for V3 — continuous control of a 3-unit scrolling squad."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        seed: int | None = None,
        max_macro_ticks: int = MAX_MACRO_TICKS,
        curriculum_phase: int = 3,
        progress_reward_scale: float = 1.0,
        kill_reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "spatial":  spaces.Box(low=0.0, high=1.0, shape=SPATIAL_SHAPE, dtype=np.float32),
            "features": spaces.Box(low=-1.0, high=1.0, shape=(NON_SPATIAL_DIM,), dtype=np.float32),
        })
        self._initial_seed = seed
        self.max_macro_ticks = max_macro_ticks
        self.progress_reward_scale = float(progress_reward_scale)
        self.kill_reward_scale = float(kill_reward_scale)
        self.curriculum_phase = 0
        self._curriculum_config: dict[str, Any] = {}
        self.set_curriculum_phase(curriculum_phase)
        # Set in reset()
        self.world: InfiniteGrid
        self.players: list[UnitState]
        self.enemies: list[UnitState]
        self.potions: list[PotionSpawn]
        self.current_idx: int
        self.macro_tick: int
        self._next_enemy_id: int
        self._steps_in_macro: int
        self._prev_centroid_x: float
        self._prev_leftmost_x: float
        self._prev_enemy_distance: float | None
        self._no_enemy_still_macro_ticks: int
        self._visible_enemy_camp_macro_ticks: int
        self._player_max_world_x: list[int]
        self._has_reset: bool = False

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def set_curriculum_phase(self, phase: int) -> None:
        """Switch V2-inspired reward/generator settings for future resets."""
        phase = int(np.clip(phase, 0, len(V2_STYLE_CURRICULUM_PHASES) - 1))
        self.curriculum_phase = phase
        self._curriculum_config = dict(V2_STYLE_CURRICULUM_PHASES[phase])
        self._curriculum_config["progress_reward"] *= self.progress_reward_scale
        self._curriculum_config["kill_reward"] *= self.kill_reward_scale

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict, dict]:
        if seed is not None:
            super().reset(seed=seed)
            rng_seed = int(seed)
        else:
            if not self._has_reset and self._initial_seed is not None:
                super().reset(seed=self._initial_seed)
            else:
                super().reset(seed=None)
            rng_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))
        self._has_reset = True
        self.world = InfiniteGrid(
            seed=rng_seed,
            column_config=self._curriculum_config.get("column_config", {}),
        )

        # Spawn the player squad in window col 1, vertically centred-ish.
        rows = [GRID_SIZE // 2 - 2, GRID_SIZE // 2, GRID_SIZE // 2 + 2]
        self.players = []
        for i, cls in enumerate(PLAYER_CLASSES):
            unit = _make_player_unit(unit_id=i, unit_class=cls)
            ps = UnitState(unit=unit, row_f=float(rows[i]), world_x_f=1.0)
            ps.unit.start_turn()
            ps.sync_window_pos(self.world.scroll_offset)
            self.players.append(ps)

        # Pre-existing enemies + potions from the initial generated columns.
        self.enemies = []
        self.potions = []
        self._next_enemy_id = 100
        enemy_spawns, potion_spawns = self.world.scroll_to(self.world.scroll_offset)
        for spawn in enemy_spawns:
            self._spawn_enemy(spawn)
        self.potions.extend(potion_spawns)

        self.current_idx = 0
        self.macro_tick = 0
        self._steps_in_macro = 0
        self._prev_centroid_x = self._squad_centroid_world_x()
        self._prev_leftmost_x = self._squad_leftmost_world_x()
        self._prev_enemy_distance = self._mean_nearest_enemy_distance()
        self._no_enemy_still_macro_ticks = 0
        self._visible_enemy_camp_macro_ticks = 0
        self._player_max_world_x = [p.world_x_int() for p in self.players]
        self._dmg_taken_acc = 0
        self._friendly_kills_acc = 0
        self._enemy_invalid_moves_acc = 0
        self._enemy_shielded_moves_acc = 0
        self._player_damage_acc = 0
        self._focus_fire_hits_acc = 0
        self._last_damaged_enemy_id: int | None = None
        self._macro_damaged_enemy_ids: set[int] = set()

        # Skip dead-active-unit slot (no dead at reset, but be defensive).
        self._advance_to_alive_or_done()
        return build_obs(self, self.current_idx), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)

        info: dict[str, Any] = {}
        reward = 0.0

        actor = self.players[self.current_idx]
        # Stagnation cost (per micro-step).
        reward -= float(self._curriculum_config.get("step_cost", 0.005))

        if actor.unit.alive:
            # ── Movement ────────────────────────────────────────────────
            direction = self._continuous_move_to_direction(action[0], action[1])
            moved = False
            attempted_move = direction is not None
            prev_actor_world_x = actor.world_x_int()
            if direction is not None and actor.unit.move_remaining > 0:
                moved, used_direction = self._try_player_move_or_shield(actor, direction)
                if not moved:
                    info["invalid_move_count"] = 1
                elif used_direction != direction:
                    info["invalid_move_count"] = 1
                    info["shielded_invalid_move_count"] = 1
            if attempted_move and "invalid_move_count" in info:
                penalty = float(self._curriculum_config.get("invalid_move_penalty", 0.0))
                reward -= penalty
                if penalty > 0.0:
                    info["invalid_move_penalty"] = penalty
            if moved and actor.world_x_int() > prev_actor_world_x:
                actor_idx = self.current_idx
                if actor.world_x_int() > self._player_max_world_x[actor_idx]:
                    if len(self._visible_alive_enemies()) == 0:
                        reward += float(self._curriculum_config.get("clear_right_step_reward", 0.0))
                    self._player_max_world_x[actor_idx] = actor.world_x_int()
            if not moved:
                reward -= float(self._curriculum_config.get("idle_cost", 0.0))

            # ── HP potion pickup ────────────────────────────────────────
            healed = self._try_potion_pickup(actor)
            if healed > 0:
                # Small positive shaping so the agent learns to detour for potions
                # without making them an over-rewarding crutch.
                reward += 0.1 * (healed / max(1, actor.unit.max_hp))

            # ── Attack ──────────────────────────────────────────────────
            if action[2] > ATTACK_THRESH and not actor.unit.has_attacked:
                killed_count, dmg_dealt, target_id = self._try_player_attack(actor)
                if dmg_dealt > 0:
                    reward += float(self._curriculum_config.get("attack_attempt_bonus", 0.0))
                    reward += float(self._curriculum_config.get("damage_scale", 0.02)) * dmg_dealt
                    self._player_damage_acc += dmg_dealt
                    if target_id is not None:
                        if (
                            target_id == self._last_damaged_enemy_id
                            or target_id in self._macro_damaged_enemy_ids
                        ):
                            bonus = float(self._curriculum_config.get("focus_fire_bonus", 0.0))
                            reward += bonus
                            if bonus > 0.0:
                                info["focus_fire_hit"] = 1
                                info["focus_fire_bonus"] = bonus
                            self._focus_fire_hits_acc += 1
                        self._macro_damaged_enemy_ids.add(target_id)
                        self._last_damaged_enemy_id = target_id
                else:
                    penalty = float(self._curriculum_config.get("whiffed_attack_penalty", 0.0))
                    reward -= penalty
                    if penalty > 0.0:
                        info["whiffed_attack_penalty"] = penalty
                reward += float(self._curriculum_config.get("kill_reward", 1.0)) * killed_count
                if dmg_dealt > 0:
                    actor.attack_cd = 1

        # ── Advance rotation ────────────────────────────────────────────
        self._steps_in_macro += 1
        end_of_macro = self._steps_in_macro >= N_AGENTS
        if end_of_macro:
            macro_reward, macro_info = self._end_of_macro_tick()
            reward += macro_reward
            info.update(macro_info)
            self._steps_in_macro = 0
            self.macro_tick += 1
            for p in self.players:
                if p.unit.alive:
                    p.unit.start_turn()
                    p.attack_cd = 0

        # Choose next active unit.
        self.current_idx = (self.current_idx + 1) % N_AGENTS
        self._advance_to_alive_or_done()

        # ── Termination ─────────────────────────────────────────────────
        all_dead = all(not p.unit.alive for p in self.players)
        terminated = all_dead
        truncated  = self.macro_tick >= self.max_macro_ticks

        if all_dead:
            reward -= 3.0

        obs = build_obs(self, self.current_idx)
        info["macro_tick"] = self.macro_tick
        info["scroll_offset"] = self.world.scroll_offset
        info["squad_alive"] = sum(1 for p in self.players if p.unit.alive)
        info["enemies_alive"] = sum(1 for e in self.enemies if e.unit.alive)
        info["curriculum_phase"] = self.curriculum_phase
        info["curriculum_name"] = self._curriculum_config.get("name", "custom")
        return obs, float(reward), terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────

    def _advance_to_alive_or_done(self) -> None:
        """If the active slot is dead, rotate to the next alive one (no extra cost)."""
        for _ in range(N_AGENTS):
            if self.players[self.current_idx].unit.alive:
                return
            self.current_idx = (self.current_idx + 1) % N_AGENTS
        # All dead — leave current_idx as-is; episode will terminate.

    def _squad_centroid_world_x(self) -> float:
        alive = [p for p in self.players if p.unit.alive]
        if not alive:
            return self._prev_centroid_x if hasattr(self, "_prev_centroid_x") else 0.0
        return float(np.mean([p.world_x_f for p in alive]))

    def _squad_leftmost_world_x(self) -> float:
        alive = [p for p in self.players if p.unit.alive]
        if not alive:
            return self._prev_leftmost_x if hasattr(self, "_prev_leftmost_x") else 0.0
        return float(min(p.world_x_int() for p in alive))

    def _mean_nearest_enemy_distance(self) -> float | None:
        alive_players = [p for p in self.players if p.unit.alive]
        alive_enemies = [e for e in self.enemies if e.unit.alive]
        if not alive_players or not alive_enemies:
            return None
        distances = []
        for p in alive_players:
            nearest = min(
                max(abs(p.row_int() - e.row_int()), abs(p.world_x_int() - e.world_x_int()))
                for e in alive_enemies
            )
            distances.append(nearest)
        return float(np.mean(distances))

    def _players_in_attack_range_count(self) -> int:
        alive_enemies = [e for e in self.enemies if e.unit.alive]
        if not alive_enemies:
            return 0
        count = 0
        for p in self.players:
            if not p.unit.alive:
                continue
            if any(
                max(abs(p.row_int() - e.row_int()), abs(p.world_x_int() - e.world_x_int()))
                <= p.unit.attack_range
                for e in alive_enemies
            ):
                count += 1
        return count

    def _multi_threat_count(self, enemies: list[UnitState] | None = None) -> int:
        """Count enemies currently threatened by 2+ living friendly units."""
        alive_enemies = enemies if enemies is not None else [e for e in self.enemies if e.unit.alive]
        count = 0
        for e in alive_enemies:
            threatened_by = 0
            for p in self.players:
                if not p.unit.alive:
                    continue
                if (
                    max(abs(p.row_int() - e.row_int()), abs(p.world_x_int() - e.world_x_int()))
                    <= p.unit.attack_range
                ):
                    threatened_by += 1
            if threatened_by >= 2:
                count += 1
        return count

    def _visible_alive_enemies(self) -> list[UnitState]:
        visible = []
        for e in self.enemies:
            if not e.unit.alive:
                continue
            wcol = e.world_x_int() - self.world.scroll_offset
            if 0 <= wcol < GRID_SIZE:
                visible.append(e)
        return visible

    def _max_safe_scroll_delta(self) -> int:
        """Largest right-scroll that keeps every living player in the window."""
        alive_players = [p for p in self.players if p.unit.alive]
        if not alive_players:
            return 0
        leftmost_world_x = min(p.world_x_int() for p in alive_players)
        return max(0, leftmost_world_x - self.world.scroll_offset)

    def _tile_occupied_by_other(self, row: int, world_x: int, *, exclude: UnitState) -> bool:
        for u in self.players + self.enemies:
            if u is exclude or not u.unit.alive:
                continue
            if u.row_int() == row and u.world_x_int() == world_x:
                return True
        return False

    def _occupied_window_positions(self, *, exclude: UnitState) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()
        for u in self.players + self.enemies:
            if u is exclude or not u.unit.alive:
                continue
            u.sync_window_pos(self.world.scroll_offset)
            if 0 <= u.unit.col < GRID_SIZE:
                occupied.add((u.unit.row, u.unit.col))
        return occupied

    def _continuous_move_to_direction(self, vx: float, vy: float) -> Direction | None:
        """Map SAC's continuous movement vector onto V2's 8-way grid directions."""
        sx = 0 if abs(float(vx)) < MOVE_DEADZONE else (1 if vx > 0 else -1)
        sy = 0 if abs(float(vy)) < MOVE_DEADZONE else (1 if vy > 0 else -1)
        if sx == 0 and sy == 0:
            return None
        delta_to_direction = {
            (-1, 0): Direction.NORTH,
            (1, 0): Direction.SOUTH,
            (0, 1): Direction.EAST,
            (0, -1): Direction.WEST,
            (-1, 1): Direction.NORTH_EAST,
            (-1, -1): Direction.NORTH_WEST,
            (1, 1): Direction.SOUTH_EAST,
            (1, -1): Direction.SOUTH_WEST,
        }
        return delta_to_direction[(sy, sx)]

    def _try_player_move(self, actor: UnitState, direction: Direction) -> bool:
        """Apply one V2-style grid step for the active player unit."""
        actor.sync_window_pos(self.world.scroll_offset)
        occupied = self._occupied_window_positions(exclude=actor)
        valid, cost = can_step(
            self.world.grid,
            actor.unit.row,
            actor.unit.col,
            direction,
            occupied,
        )
        if not valid or cost > actor.unit.move_remaining:
            return False

        old_elev = self.world.grid.get_elevation(actor.unit.row, actor.unit.col)
        dr, dc = DIRECTION_DELTAS[direction]
        new_row = actor.unit.row + dr
        new_col = actor.unit.col + dc
        new_elev = self.world.grid.get_elevation(new_row, new_col)

        actor.unit.row = new_row
        actor.unit.col = new_col
        actor.unit.move_remaining -= cost
        actor.row_f = float(new_row)
        actor.world_x_f = float(self.world.scroll_offset + new_col)

        if actor.unit.unit_class == UnitClass.CHARGER:
            self._update_charger_momentum(actor.unit, direction, old_elev, new_elev)
        return True

    def _valid_player_directions(self, actor: UnitState) -> list[tuple[Direction, int]]:
        actor.sync_window_pos(self.world.scroll_offset)
        occupied = self._occupied_window_positions(exclude=actor)
        valid_dirs: list[tuple[Direction, int]] = []
        for direction in (
            Direction.NORTH,
            Direction.SOUTH,
            Direction.EAST,
            Direction.WEST,
            Direction.NORTH_EAST,
            Direction.NORTH_WEST,
            Direction.SOUTH_EAST,
            Direction.SOUTH_WEST,
        ):
            valid, cost = can_step(
                self.world.grid,
                actor.unit.row,
                actor.unit.col,
                direction,
                occupied,
            )
            if valid and cost <= actor.unit.move_remaining:
                valid_dirs.append((direction, cost))
        return valid_dirs

    def _try_player_move_or_shield(
        self, actor: UnitState, direction: Direction
    ) -> tuple[bool, Direction]:
        """Try the requested move; if invalid, use the closest valid direction.

        SAC cannot use discrete action masks with this continuous action head, so
        this is a small action shield: invalid intent still gets logged and
        penalized, but it no longer traps the env in visually frozen no-ops.
        """
        if self._try_player_move(actor, direction):
            return True, direction

        desired_dr, desired_dc = DIRECTION_DELTAS[direction]
        valid_dirs = self._valid_player_directions(actor)
        if not valid_dirs:
            return False, direction

        def score(item: tuple[Direction, int]) -> tuple[int, int, int]:
            candidate, cost = item
            dr, dc = DIRECTION_DELTAS[candidate]
            alignment = desired_dr * dr + desired_dc * dc
            rightward = 1 if dc > 0 else 0
            return alignment, rightward, -cost

        fallback = max(valid_dirs, key=score)[0]
        return self._try_player_move(actor, fallback), fallback

    def _update_charger_momentum(
        self, unit: Unit, direction: Direction, old_elev: int, new_elev: int
    ) -> None:
        """Match V2 Charger momentum after a one-tile grid move."""
        elev_drop = old_elev - new_elev
        if elev_drop >= 2 or self.world.grid.is_rough(unit.row, unit.col):
            unit.momentum = 0
            unit.momentum_dir = Direction.NONE
            return
        if unit.momentum_dir != Direction.NONE and direction != unit.momentum_dir:
            unit.momentum = 0
            unit.momentum_dir = Direction.NONE
            return
        unit.momentum_dir = direction
        unit.momentum += 1
        if elev_drop == 1:
            unit.momentum += 1

    def _spawn_enemy(self, spawn) -> None:
        """Spawn a new enemy from a chunk-generator EnemySpawn descriptor."""
        unit = _make_enemy_unit(
            unit_id=self._next_enemy_id,
            unit_class=spawn.unit_class,
            hp_scale=spawn.hp_scale,
            atk_scale=spawn.atk_scale,
        )
        self._next_enemy_id += 1
        es = UnitState(unit=unit, row_f=float(spawn.row), world_x_f=float(spawn.col_world),
                       ai=make_ai_for(spawn.unit_class))
        es.sync_window_pos(self.world.scroll_offset)
        self.enemies.append(es)

    def _try_potion_pickup(self, actor: UnitState) -> int:
        """If the actor's tile holds a potion, consume it and heal. Returns hp restored."""
        if not self.potions:
            return 0
        ar, ax = actor.row_int(), actor.world_x_int()
        for i, pot in enumerate(self.potions):
            if int(round(pot.row)) == ar and int(round(pot.col_world)) == ax:
                heal = max(1, int(round(actor.unit.max_hp * pot.heal_frac)))
                missing = actor.unit.max_hp - actor.unit.hp
                heal = min(heal, missing)
                actor.unit.hp += heal
                self.potions.pop(i)
                return heal
        return 0

    def _all_unit_objs(self) -> list[Unit]:
        """Combined V2-Unit list for combat (flank checks)."""
        return [u.unit for u in (self.players + self.enemies)]

    def _try_player_attack(self, actor: UnitState) -> tuple[int, int, int | None]:
        """Player attempts an attack on the nearest enemy in range."""
        actor.sync_window_pos(self.world.scroll_offset)
        attack_tiles = set(
            tiles_in_attack_range(
                self.world.grid,
                actor.unit.row,
                actor.unit.col,
                actor.unit.attack_range,
            )
        )

        # Find nearest in-window enemy on a V2-legal attack tile.
        best = None
        best_d = 10**9
        for e in self.enemies:
            if not e.unit.alive:
                continue
            e.sync_window_pos(self.world.scroll_offset)
            if not (0 <= e.unit.col < GRID_SIZE):
                continue
            if e.unit.pos not in attack_tiles:
                continue
            d = max(abs(actor.unit.row - e.unit.row), abs(actor.unit.col - e.unit.col))
            if d < best_d:
                best, best_d = e, d
        if best is None:
            return 0, 0, None

        best.sync_window_pos(self.world.scroll_offset)
        dmg = compute_damage(actor.unit, best.unit, self.world.grid, self._all_unit_objs())
        prev_alive = best.unit.alive
        target_id = best.unit.unit_id
        best.unit.take_damage(dmg)
        actor.unit.has_attacked = True
        if actor.unit.unit_class == UnitClass.CHARGER:
            actor.unit.momentum = 0
            actor.unit.momentum_dir = Direction.NONE
        killed = 1 if (prev_alive and not best.unit.alive) else 0
        return killed, dmg, target_id

    def _try_enemy_attack(self, attacker: UnitState, target_pid: int) -> int:
        """Enemy attacks player by id; returns hp dealt."""
        target = self.players[target_pid]
        if not target.unit.alive:
            return 0
        attacker.sync_window_pos(self.world.scroll_offset)
        target.sync_window_pos(self.world.scroll_offset)
        dmg = compute_damage(attacker.unit, target.unit, self.world.grid, self._all_unit_objs())
        target.unit.take_damage(dmg)
        return dmg

    def _end_of_macro_tick(self) -> tuple[float, dict]:
        """Run end-of-macro housekeeping: scroll, enemies act, off-screen kills, scoring."""
        info: dict[str, Any] = {}
        reward = 0.0

        # ── Scrolling: keep the squad centroid near the middle of the window.
        visible_enemies = self._visible_alive_enemies()
        map_clear = len(visible_enemies) == 0
        centroid_world_x = self._squad_centroid_world_x()
        leftmost_world_x = self._squad_leftmost_world_x()
        prev_leftmost = self._prev_leftmost_x
        delta_leftmost = max(0.0, leftmost_world_x - prev_leftmost)
        if map_clear:
            reward += float(self._curriculum_config.get("progress_reward", 0.05)) * delta_leftmost
        self._prev_centroid_x = centroid_world_x
        self._prev_leftmost_x = leftmost_world_x

        # If centroid window-col exceeds the trigger, scroll right by the excess.
        # Lock scrolling while enemies remain visible so the policy cannot outrun
        # combat by escaping into the next strip of terrain.
        centroid_window = centroid_world_x - self.world.scroll_offset
        if map_clear and centroid_window > SCROLL_TRIGGER:
            requested_scroll_delta = int(math.floor(centroid_window - SCROLL_TRIGGER))
            safe_scroll_delta = min(requested_scroll_delta, self._max_safe_scroll_delta())
            if safe_scroll_delta > 0:
                new_offset = self.world.scroll_offset + safe_scroll_delta
                new_enemy_spawns, new_potion_spawns = self.world.scroll_to(new_offset)
                for sp in new_enemy_spawns:
                    self._spawn_enemy(sp)
                self.potions.extend(new_potion_spawns)
                # All units' window positions need re-sync after scroll.
                for u in self.players + self.enemies:
                    u.sync_window_pos(self.world.scroll_offset)
                # Drop any potions that scrolled off the left edge.
                self.potions = [p for p in self.potions if int(round(p.col_world)) >= self.world.scroll_offset]
            elif requested_scroll_delta > 0:
                info["scroll_locked_by_lagging_player"] = 1
                penalty = float(self._curriculum_config.get("lag_lock_penalty", 0.0))
                reward -= penalty
                if penalty > 0:
                    info["lag_lock_penalty"] = penalty
        elif not map_clear and centroid_window > SCROLL_TRIGGER:
            info["scroll_locked_by_enemies"] = len(visible_enemies)

        if map_clear:
            if delta_leftmost <= 0.0:
                self._no_enemy_still_macro_ticks += 1
                penalty = min(
                    float(self._curriculum_config.get("no_enemy_still_penalty_cap", 0.0)),
                    float(self._curriculum_config.get("no_enemy_still_penalty_base", 0.0))
                    * self._no_enemy_still_macro_ticks,
                )
                reward -= penalty
                if penalty > 0:
                    info["no_enemy_still_penalty"] = penalty
            else:
                self._no_enemy_still_macro_ticks = 0
        else:
            self._no_enemy_still_macro_ticks = 0

        in_range_count = self._players_in_attack_range_count()
        if in_range_count > 0:
            reward += (
                float(self._curriculum_config.get("in_range_bonus_per_unit", 0.0))
                * in_range_count
            )
        multi_threat_count = self._multi_threat_count(visible_enemies)
        if multi_threat_count > 0:
            bonus = float(self._curriculum_config.get("multi_threat_reward", 0.0)) * multi_threat_count
            reward += bonus
            if bonus > 0:
                info["multi_threat_count"] = multi_threat_count
                info["multi_threat_reward"] = bonus

        curr_enemy_distance = self._mean_nearest_enemy_distance()
        proximity_weight = float(self._curriculum_config.get("enemy_proximity_weight", 0.0))
        dist_closed = 0.0
        if (
            proximity_weight > 0.0
            and curr_enemy_distance is not None
            and self._prev_enemy_distance is not None
        ):
            dist_closed = self._prev_enemy_distance - curr_enemy_distance
            if dist_closed > 0:
                reward += proximity_weight * dist_closed
                info["approach_reward"] = proximity_weight * dist_closed
                info["enemy_distance_closed"] = dist_closed
        self._prev_enemy_distance = curr_enemy_distance

        player_damage_this_macro = getattr(self, "_player_damage_acc", 0)
        if not map_clear:
            if player_damage_this_macro <= 0 and dist_closed <= 0.0:
                self._visible_enemy_camp_macro_ticks += 1
                penalty = min(
                    float(self._curriculum_config.get("visible_enemy_camp_penalty_cap", 0.0)),
                    float(self._curriculum_config.get("visible_enemy_camp_penalty_base", 0.0))
                    * self._visible_enemy_camp_macro_ticks,
                )
                reward -= penalty
                if penalty > 0:
                    info["visible_enemy_camp_penalty"] = penalty
            else:
                self._visible_enemy_camp_macro_ticks = 0
        else:
            self._visible_enemy_camp_macro_ticks = 0

        # ── Off-screen-left kills: any unit whose world_x < scroll_offset dies.
        for p in self.players:
            if p.unit.alive and p.world_x_int() < self.world.scroll_offset:
                p.unit.take_damage(p.unit.hp)  # finish it
                reward -= 1.5
                info.setdefault("offscreen_player_kills", 0)
                info["offscreen_player_kills"] += 1
        # Enemies that fall off the left edge are removed silently.
        self.enemies = [
            e for e in self.enemies
            if not (e.world_x_int() < self.world.scroll_offset)
        ]

        # ── Enemies act ──────────────────────────────────────────────────
        for e in list(self.enemies):
            if not e.unit.alive:
                continue
            action = e.ai.decide(e, self.players)  # type: ignore[union-attr]
            if not bool(self._curriculum_config.get("enemy_movement_enabled", True)):
                action = EnemyAction(drow=0, dcol_world=0, attack_target_id=action.attack_target_id)
            self._apply_enemy_action(e, action)
            # Track damage dealt to players for shaped reward.

        # Damage taken penalty: compute per-tick HP delta for players.
        # We use a transient cache: store hp_at_macro_start on each player at
        # the start of every macro tick. To keep this method self-contained,
        # we approximate by tracking from prior call. A full implementation
        # would snapshot HP at macro start; here we accept a small bias and
        # apply the penalty in _apply_enemy_action via an in-place tally.
        # (Done inline below: damage_taken_total accumulated on the env.)
        dmg_taken = getattr(self, "_dmg_taken_acc", 0)
        reward -= float(self._curriculum_config.get("damage_scale", 0.02)) * dmg_taken
        if dmg_taken > 0:
            info["damage_taken_macro"] = dmg_taken
        # Friendly killed by enemy hits this macro tick: count newly-dead.
        new_dead = getattr(self, "_friendly_kills_acc", 0)
        if new_dead > 0:
            reward -= 1.5 * new_dead
            info["friendly_kills_macro"] = new_dead
        enemy_invalid = getattr(self, "_enemy_invalid_moves_acc", 0)
        enemy_shielded = getattr(self, "_enemy_shielded_moves_acc", 0)
        if enemy_invalid > 0:
            info["enemy_invalid_move_macro"] = enemy_invalid
        if enemy_shielded > 0:
            info["enemy_shielded_move_macro"] = enemy_shielded
        if player_damage_this_macro > 0:
            info["player_damage_macro"] = player_damage_this_macro
        focus_fire_hits = getattr(self, "_focus_fire_hits_acc", 0)
        if focus_fire_hits > 0:
            info["focus_fire_hits_macro"] = focus_fire_hits
        # Reset accumulators for the next macro tick.
        self._dmg_taken_acc = 0
        self._friendly_kills_acc = 0
        self._enemy_invalid_moves_acc = 0
        self._enemy_shielded_moves_acc = 0
        self._player_damage_acc = 0
        self._focus_fire_hits_acc = 0
        self._macro_damaged_enemy_ids = set()

        # Endurance bonus once we've made it past world_x = 200.
        if centroid_world_x > 200.0 and any(p.unit.alive for p in self.players):
            reward += 0.01

        return reward, info

    def _apply_enemy_action(self, enemy: UnitState, action: EnemyAction) -> None:
        """Apply one enemy's macro action — move + optional attack — with bookkeeping."""
        # Movement
        new_row = enemy.row_int() + action.drow
        new_wx  = enemy.world_x_int() + action.dcol_world
        in_win  = self.world.is_in_window(new_row, new_wx)
        walkable = in_win and self.world.is_walkable_world(new_row, new_wx)
        occupied = self._tile_occupied_by_other(new_row, new_wx, exclude=enemy)
        if walkable and not occupied:
            enemy.row_f = float(new_row)
            enemy.world_x_f = float(new_wx)
            enemy.sync_window_pos(self.world.scroll_offset)
        elif action.drow != 0 or action.dcol_world != 0:
            self._enemy_invalid_moves_acc = getattr(self, "_enemy_invalid_moves_acc", 0) + 1
            fallback = self._enemy_fallback_step(enemy)
            if fallback is not None:
                enemy.row_f = float(fallback[0])
                enemy.world_x_f = float(fallback[1])
                enemy.sync_window_pos(self.world.scroll_offset)
                self._enemy_shielded_moves_acc = getattr(self, "_enemy_shielded_moves_acc", 0) + 1

        # Attack
        if action.attack_target_id is not None:
            tgt_player_idx = next(
                (i for i, p in enumerate(self.players) if p.unit.unit_id == action.attack_target_id),
                None,
            )
            if tgt_player_idx is None:
                return
            prev_alive = self.players[tgt_player_idx].unit.alive
            dmg = self._try_enemy_attack(enemy, tgt_player_idx)
            if dmg > 0:
                self._dmg_taken_acc = getattr(self, "_dmg_taken_acc", 0) + dmg
            if prev_alive and not self.players[tgt_player_idx].unit.alive:
                self._friendly_kills_acc = getattr(self, "_friendly_kills_acc", 0) + 1

    def _enemy_fallback_step(self, enemy: UnitState) -> tuple[int, int] | None:
        """Choose a legal neighboring enemy step that best approaches a player."""
        alive_players = [p for p in self.players if p.unit.alive]
        if not alive_players:
            return None

        def nearest_dist(row: int, world_x: int) -> int:
            return min(
                max(abs(row - p.row_int()), abs(world_x - p.world_x_int()))
                for p in alive_players
            )

        candidates: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                row = enemy.row_int() + dr
                world_x = enemy.world_x_int() + dc
                if not self.world.is_in_window(row, world_x):
                    continue
                if not self.world.is_walkable_world(row, world_x):
                    continue
                if self._tile_occupied_by_other(row, world_x, exclude=enemy):
                    continue
                candidates.append((nearest_dist(row, world_x), row, world_x))
        if not candidates:
            return None
        _, row, world_x = min(candidates)
        return row, world_x

    # ── Render hook (rgb_array) ───────────────────────────────────────────

    def render(self):
        from viz.render_scroll import render_frame
        return render_frame(self)
