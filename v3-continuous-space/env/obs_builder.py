"""Observation construction for ScrollingSquadEnv.

Spatial channels (8 total) so the agent can identify enemies by class
(critical for "focus damage on the dangerous units") and locate HP potions:

    obs["spatial"]:    float32 array of shape (8, H=GRID_SIZE, W=GRID_SIZE)
        ch 0 — terrain / 2.0          (NORMAL=0, ROUGH=0.5, UNCROSSABLE=1.0)
        ch 1 — elevation / 2.0        (0..1)
        ch 2 — friendly HP fraction at each tile
        ch 3 — enemy FIGHTER HP fraction at each tile
        ch 4 — enemy CHARGER HP fraction at each tile
        ch 5 — enemy RANGER  HP fraction at each tile
        ch 6 — enemy SIEGE   HP fraction at each tile
        ch 7 — HP potion presence (1.0 where a potion sits)

    obs["features"]:   float32 vector of shape (NON_SPATIAL_DIM,)
        active unit:      [hp_frac, attack_cd_frac, row_frac, world_x_local_frac]
        agent id one-hot: [id0, id1, id2]
        teammate alive:   [alive0, alive1, alive2]
        squad summary:    [centroid_world_x_frac, centroid_row_frac,
                           mean_hp_frac, n_alive_enemies_frac, scroll_offset_norm]
        local actions:    for each of 8 directions:
                           [can_step, enemy_on_dest, potion_on_dest, advances_right]
        nearest enemy:    [dx_norm, dy_norm, distance_norm, attackable]
"""

from __future__ import annotations

import numpy as np

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE, Terrain
from highground.engine.pathfinding import can_step, tiles_in_attack_range
from highground.engine.units import DIRECTION_DELTAS, TEAM_B, Direction, UnitClass


N_AGENTS = 3
N_LOCAL_DIRECTIONS = 8
N_LOCAL_FEATURES_PER_DIRECTION = 4
N_NEAREST_ENEMY_FEATURES = 4
BASE_NON_SPATIAL_DIM = 4 + N_AGENTS + N_AGENTS + 5  # = 15
NON_SPATIAL_DIM = (
    BASE_NON_SPATIAL_DIM
    + N_LOCAL_DIRECTIONS * N_LOCAL_FEATURES_PER_DIRECTION
    + N_NEAREST_ENEMY_FEATURES
)  # = 51

LOCAL_DIRECTIONS: tuple[tuple[str, Direction], ...] = (
    ("NW", Direction.NORTH_WEST),
    ("N", Direction.NORTH),
    ("NE", Direction.NORTH_EAST),
    ("W", Direction.WEST),
    ("E", Direction.EAST),
    ("SW", Direction.SOUTH_WEST),
    ("S", Direction.SOUTH),
    ("SE", Direction.SOUTH_EAST),
)

# 4 enemy-class HP channels (FIGHTER, CHARGER, RANGER, SIEGE) keep the same
# class index ordering as UnitClass so ch_index = 3 + int(unit_class).
N_ENEMY_CLASS_CHANNELS = 4
SPATIAL_SHAPE = (3 + N_ENEMY_CLASS_CHANNELS + 1, GRID_SIZE, GRID_SIZE)  # = 8


def _occupied_window_positions(env, *, exclude) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for u in env.players + env.enemies:
        if u is exclude or not u.unit.alive:
            continue
        u.sync_window_pos(env.world.scroll_offset)
        if 0 <= u.unit.col < GRID_SIZE:
            occupied.add((u.unit.row, u.unit.col))
    return occupied


def _unit_at_world(env, row: int, world_x: int, *, team: int | None = None) -> bool:
    for u in env.players + env.enemies:
        if not u.unit.alive:
            continue
        if team is not None and u.unit.team != team:
            continue
        if u.row_int() == row and u.world_x_int() == world_x:
            return True
    return False


def _potion_at_world(env, row: int, world_x: int) -> bool:
    return any(
        int(round(p.row)) == row and int(round(p.col_world)) == world_x
        for p in env.potions
    )


def build_obs(env, active_idx: int) -> dict:
    """Build the observation dict for the player unit at index `active_idx`."""
    H = W = GRID_SIZE
    scroll = env.world.scroll_offset

    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)

    # Terrain & elevation channels (already in window-coords).
    spatial[0] = env.world.grid.terrain.astype(np.float32) / float(Terrain.UNCROSSABLE)
    spatial[1] = env.world.grid.elevation.astype(np.float32) / 2.0

    # Friendly HP channel.
    for p in env.players:
        if not p.unit.alive:
            continue
        wcol = p.world_x_int() - scroll
        row  = p.row_int()
        if 0 <= wcol < W and 0 <= row < H:
            hp_frac = p.unit.hp / max(1, p.unit.max_hp)
            # If multiple friendlies share a tile, take max.
            spatial[2, row, wcol] = max(spatial[2, row, wcol], hp_frac)

    # Enemy HP channels — one per UnitClass so the agent can identify type.
    for e in env.enemies:
        if not e.unit.alive:
            continue
        wcol = e.world_x_int() - scroll
        row  = e.row_int()
        if 0 <= wcol < W and 0 <= row < H:
            hp_frac = e.unit.hp / max(1, e.unit.max_hp)
            ch = 3 + int(e.unit.unit_class)   # FIGHTER=0 → ch3, ... SIEGE=3 → ch6
            spatial[ch, row, wcol] = max(spatial[ch, row, wcol], hp_frac)

    # HP potion channel.
    potion_ch = 3 + N_ENEMY_CLASS_CHANNELS
    for p in env.potions:
        wcol = int(round(p.col_world)) - scroll
        row  = int(round(p.row))
        if 0 <= wcol < W and 0 <= row < H:
            spatial[potion_ch, row, wcol] = 1.0

    # ── Non-spatial features ──────────────────────────────────────────────
    feats = np.zeros(NON_SPATIAL_DIM, dtype=np.float32)

    me = env.players[active_idx]
    # If the active unit is dead we still need a defined obs; use zeros for
    # most fields and let the agent-id one-hot identify which slot is acting.
    if me.unit.alive:
        wcol_local = (me.world_x_int() - scroll) / float(W)
        feats[0] = me.unit.hp / max(1, me.unit.max_hp)
        feats[1] = me.attack_cd / 3.0          # cd capped at ~3 ticks for normalisation
        feats[2] = me.row_int() / float(H - 1)
        feats[3] = float(np.clip(wcol_local, 0.0, 1.0))

    # Agent id one-hot.
    feats[4 + active_idx] = 1.0

    # Teammate alive flags (in fixed slot order).
    for i in range(N_AGENTS):
        feats[4 + N_AGENTS + i] = 1.0 if env.players[i].unit.alive else 0.0

    # Squad summary.
    alive_players = [p for p in env.players if p.unit.alive]
    if alive_players:
        cx = np.mean([(p.world_x_int() - scroll) for p in alive_players]) / float(W)
        cy = np.mean([p.row_int() for p in alive_players]) / float(H - 1)
        mh = np.mean([p.unit.hp / max(1, p.unit.max_hp) for p in alive_players])
    else:
        cx = cy = mh = 0.0
    n_alive_enemies = sum(1 for e in env.enemies if e.unit.alive)
    base = 4 + N_AGENTS + N_AGENTS
    feats[base + 0] = float(np.clip(cx, 0.0, 1.0))
    feats[base + 1] = float(cy)
    feats[base + 2] = float(mh)
    feats[base + 3] = float(min(n_alive_enemies, 8) / 8.0)
    feats[base + 4] = float(np.tanh(scroll / 100.0))  # how far we've scrolled

    # Local action-validity features. These give SAC an immediate tactical
    # readout for the 8 movement choices, making failed moves diagnosable and
    # learnable even when the spatial CNN is under-used.
    local_base = BASE_NON_SPATIAL_DIM
    if me.unit.alive:
        me.sync_window_pos(scroll)
        occupied = _occupied_window_positions(env, exclude=me)
        for dir_idx, (_, direction) in enumerate(LOCAL_DIRECTIONS):
            dr, dc = DIRECTION_DELTAS[direction]
            dest_row = me.unit.row + dr
            dest_col = me.unit.col + dc
            dest_world_x = scroll + dest_col
            valid, cost = can_step(env.world.grid, me.unit.row, me.unit.col, direction, occupied)
            can_move = valid and cost <= me.unit.move_remaining
            out = local_base + dir_idx * N_LOCAL_FEATURES_PER_DIRECTION
            feats[out + 0] = 1.0 if can_move else 0.0
            feats[out + 1] = 1.0 if _unit_at_world(env, dest_row, dest_world_x, team=TEAM_B) else 0.0
            feats[out + 2] = 1.0 if _potion_at_world(env, dest_row, dest_world_x) else 0.0
            feats[out + 3] = 1.0 if dest_world_x > me.world_x_int() else 0.0

    nearest_base = local_base + N_LOCAL_DIRECTIONS * N_LOCAL_FEATURES_PER_DIRECTION
    visible_enemies = []
    if me.unit.alive:
        for e in env.enemies:
            if not e.unit.alive:
                continue
            e_col = e.world_x_int() - scroll
            if 0 <= e_col < GRID_SIZE:
                visible_enemies.append(e)
    if me.unit.alive and visible_enemies:
        nearest = min(
            visible_enemies,
            key=lambda e: max(abs(e.row_int() - me.row_int()), abs(e.world_x_int() - me.world_x_int())),
        )
        dx = nearest.world_x_int() - me.world_x_int()
        dy = nearest.row_int() - me.row_int()
        dist = max(abs(dx), abs(dy))
        attack_tiles = set(
            tiles_in_attack_range(env.world.grid, me.unit.row, me.unit.col, me.unit.attack_range)
        )
        nearest.sync_window_pos(scroll)
        feats[nearest_base + 0] = float(np.clip(dx / float(GRID_SIZE), -1.0, 1.0))
        feats[nearest_base + 1] = float(np.clip(dy / float(GRID_SIZE), -1.0, 1.0))
        feats[nearest_base + 2] = float(np.clip(dist / float(GRID_SIZE), 0.0, 1.0))
        feats[nearest_base + 3] = 1.0 if nearest.unit.pos in attack_tiles else 0.0

    return {"spatial": spatial, "features": feats}
