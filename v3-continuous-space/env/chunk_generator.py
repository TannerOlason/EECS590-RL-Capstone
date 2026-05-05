"""Procedural column-chunk generation for the infinite scrolling world.

Each generated chunk is a single column (one tile wide, GRID_SIZE tall)
of (terrain, elevation) values plus zero or more enemy spawn descriptors.
Difficulty (enemy density, enemy class, enemy HP scaling) increases with
the absolute world_x of the generated column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import _path_shim  # noqa: F401
from highground.engine.grid import GRID_SIZE, Terrain
from highground.engine.units import UnitClass


@dataclass
class EnemySpawn:
    """Description of an enemy to spawn at a given world position."""

    unit_class: UnitClass
    row: int
    col_world: int
    hp_scale: float = 1.0  # multiplier on max_hp
    atk_scale: float = 0.75  # multiplier on atk — enemies are deliberately weaker


@dataclass
class PotionSpawn:
    """An HP potion sitting on a tile until walked over by a friendly unit."""

    row: int
    col_world: int
    heal_frac: float = 0.25  # restores this fraction of the picker's max_hp


def _difficulty(world_x: int) -> float:
    """Smoothly ramp difficulty from 0 → 1 over the first ~150 columns."""
    return float(np.tanh(max(0, world_x) / 75.0))


def generate_column(
    world_x: int,
    rng: np.random.Generator,
    *,
    enemy_spawn_scale: float = 1.0,
    enemy_spawn_min_x: int = 8,
    enemy_hp_scale_mult: float = 1.0,
    enemy_atk_scale: float = 0.75,
    terrain_block_scale: float = 1.0,
    rough_scale: float = 1.0,
    potion_p: float = 0.03,
) -> tuple[
    np.ndarray, np.ndarray, list[EnemySpawn], list[PotionSpawn]
]:
    """Generate one column of the world.

    Args:
        world_x: Absolute world column index (>= 0).
        rng: Numpy random generator for reproducibility.

    Returns:
        terrain_col: (GRID_SIZE,) int8 of Terrain values.
        elevation_col: (GRID_SIZE,) int8 of elevation levels (0..2).
        enemy_spawns: list of EnemySpawn for any enemies placed in this column.
        potion_spawns: list of PotionSpawn for any HP pickups in this column.
    """
    diff = _difficulty(world_x)

    # Terrain: mostly NORMAL with sparse ROUGH and rare UNCROSSABLE pillars.
    terrain = np.zeros(GRID_SIZE, dtype=np.int8)
    for r in range(GRID_SIZE):
        u = rng.random()
        if u < (0.05 + 0.05 * diff) * terrain_block_scale:
            terrain[r] = Terrain.UNCROSSABLE
        elif u < 0.05 * terrain_block_scale + 0.15 * rough_scale:
            terrain[r] = Terrain.ROUGH
        else:
            terrain[r] = Terrain.NORMAL

    # Never block the entire column — guarantee at least one walkable tile.
    if (terrain == Terrain.UNCROSSABLE).all():
        terrain[GRID_SIZE // 2] = Terrain.NORMAL

    # Elevation: low-frequency noise centred at 0; occasional ridges of 1-2.
    elevation = np.zeros(GRID_SIZE, dtype=np.int8)
    if rng.random() < 0.25:
        ridge_row = int(rng.integers(0, GRID_SIZE))
        ridge_height = 1 + int(rng.random() < 0.4)  # 1 or 2
        # Narrow ridge: ±1 tile of the centre.
        for r in (ridge_row - 1, ridge_row, ridge_row + 1):
            if 0 <= r < GRID_SIZE:
                elevation[r] = max(int(elevation[r]), ridge_height)

    # Enemy spawns: probability scales with difficulty. Avoid spawning in
    # the very-first columns so the squad has breathing room.
    enemy_spawns: list[EnemySpawn] = []
    spawn_threshold = enemy_spawn_min_x
    if world_x >= spawn_threshold:
        # Sparser than V3.0: 2.5 % at start → ~12.5 % at high diff (was 4 %→24 %).
        spawn_p = (0.025 + 0.10 * diff) * enemy_spawn_scale
        if rng.random() < spawn_p:
            walkable_rows = [r for r in range(GRID_SIZE) if terrain[r] != Terrain.UNCROSSABLE]
            row = int(rng.choice(walkable_rows))

            # Class mix: early game mostly Fighters; later, Rangers and Chargers appear.
            if diff < 0.3:
                cls = UnitClass.FIGHTER
            else:
                weights = np.array([0.5, 0.2, 0.25, 0.05])  # FIGHTER, CHARGER, RANGER, SIEGE
                cls = UnitClass(int(rng.choice(4, p=weights)))

            enemy_spawns.append(EnemySpawn(
                unit_class=cls,
                row=row,
                col_world=world_x,
                hp_scale=(1.0 + 0.25 * diff) * enemy_hp_scale_mult,
                atk_scale=enemy_atk_scale,
            ))

    # HP potion spawns: rare, modest heal. Skip the very-first columns.
    potion_spawns: list[PotionSpawn] = []
    if world_x >= 6:
        if rng.random() < potion_p:
            walkable_rows = [r for r in range(GRID_SIZE) if terrain[r] != Terrain.UNCROSSABLE]
            row = int(rng.choice(walkable_rows))
            potion_spawns.append(PotionSpawn(
                row=row,
                col_world=world_x,
                heal_frac=0.25,
            ))

    return terrain, elevation, enemy_spawns, potion_spawns
