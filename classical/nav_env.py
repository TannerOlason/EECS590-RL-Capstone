"""SimpleNavigationEnv — a minimal tabular environment for classical RL.

A single unit must reach the highest-elevation tile on a reduced grid
before an enemy scout reaches it.  The observation is discretised into
a small state tuple so that Q-tables and eligibility-trace tables are
tractable (~2 000 states).

State tuple
-----------
    (row, col, elevation_bucket, enemy_dist_bucket)

    row / col         : 0 .. GRID-1   (default 13)
    elevation_bucket  : 0 (flat) | 1 (mid) | 2 (high)
    enemy_dist_bucket : 0 (close, ≤2) | 1 (medium, ≤5) | 2 (far, >5)

Actions (4-connected movement + stay)
--------------------------------------
    0 = NORTH, 1 = SOUTH, 2 = EAST, 3 = WEST, 4 = STAY

Reward
------
    +1.0   on reaching a goal tile (elevation == 2)
    -0.01  per step (living cost)
    -0.5   if the enemy scout reaches the goal first (episode ends)

Episode ends when
-----------------
    - The agent reaches any elevation-2 tile  (win)
    - The enemy scout reaches any elevation-2 tile first  (lose)
    - MAX_STEPS reached                                    (timeout)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

GRID = 13
MAX_STEPS = 100
N_ACTIONS = 5  # N, S, E, W, STAY

_DR = (-1, 1, 0, 0, 0)  # row deltas
_DC = (0, 0, 1, -1, 0)  # col deltas


class NavState(NamedTuple):
    """Discrete state used as a table key."""
    row: int
    col: int
    elev_bucket: int      # 0-2
    enemy_dist_bucket: int  # 0-2


def _elev_bucket(e: int) -> int:
    if e >= 2:
        return 2
    if e >= 1:
        return 1
    return 0


def _dist_bucket(d: float) -> int:
    if d <= 2:
        return 0
    if d <= 5:
        return 1
    return 2


def _chebyshev(r1: int, c1: int, r2: int, c2: int) -> int:
    return max(abs(r1 - r2), abs(c1 - c2))


def _make_elevation_grid(rng: random.Random) -> np.ndarray:
    """Generate a simple elevation grid with a few elevated regions.

    One elevation-2 tile is always placed at the center (6, 6) so that the
    race between agent and enemy is competitive.  Additional random peaks
    are scattered for variety.
    """
    elev = np.zeros((GRID, GRID), dtype=np.int8)
    # scatter elevation-1 patches
    for _ in range(12):
        r, c = rng.randint(1, GRID - 2), rng.randint(1, GRID - 2)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if 0 <= r + dr < GRID and 0 <= c + dc < GRID:
                    elev[r + dr, c + dc] = max(elev[r + dr, c + dc], 1)
    # Fixed central peak: the main race objective
    mid = GRID // 2
    elev[mid, mid] = 2
    # Additional random elevation-2 peaks for variety
    for _ in range(3):
        r, c = rng.randint(2, GRID - 3), rng.randint(2, GRID - 3)
        elev[r, c] = 2
    return elev


@dataclass
class SimpleNavigationEnv:
    """Minimal single-agent navigation environment for tabular RL.

    Usage::

        env = SimpleNavigationEnv(seed=42)
        state = env.reset()
        done = False
        while not done:
            action = policy[state]   # table lookup
            state, reward, done, info = env.step(action)
    """

    seed: int = 0
    max_steps: int = MAX_STEPS

    # Set after reset()
    elevation: np.ndarray = field(default_factory=lambda: np.zeros((GRID, GRID), dtype=np.int8))
    agent_pos: tuple[int, int] = (0, 0)
    enemy_pos: tuple[int, int] = (GRID - 1, GRID - 1)
    goal_tiles: list[tuple[int, int]] = field(default_factory=list)
    _step_count: int = 0
    _episode: int = 0        # auto-incremented to vary start positions each episode
    _rng: random.Random = field(default_factory=random.Random)

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> NavState:
        # Map is fixed per-seed so the state space is stable across episodes.
        map_rng = random.Random(self.seed)
        self.elevation = _make_elevation_grid(map_rng)
        self.goal_tiles = [
            (r, c)
            for r in range(GRID)
            for c in range(GRID)
            if self.elevation[r, c] == 2
        ]
        # Start positions vary each episode so the Q-table gets broad coverage.
        # Agent spawns in the left half, enemy in the right half.
        start_rng = random.Random(self.seed + self._episode * 97)
        self.agent_pos = (
            start_rng.randint(0, GRID - 1),
            start_rng.randint(0, GRID // 2 - 1),
        )
        self.enemy_pos = (
            start_rng.randint(0, GRID - 1),
            start_rng.randint(GRID // 2, GRID - 1),
        )
        self._step_count = 0
        self._episode += 1
        return self._observe()

    def step(self, action: int) -> tuple[NavState, float, bool, dict]:
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"
        self._step_count += 1

        # Move agent
        ar, ac = self.agent_pos
        nr = ar + _DR[action]
        nc = ac + _DC[action]
        if 0 <= nr < GRID and 0 <= nc < GRID:
            self.agent_pos = (nr, nc)

        # Move enemy scout toward nearest goal tile (greedy)
        self.enemy_pos = self._enemy_step()

        # Check terminal conditions
        agent_at_goal = self.agent_pos in self.goal_tiles
        enemy_at_goal = self.enemy_pos in self.goal_tiles
        timeout = self._step_count >= self.max_steps

        if agent_at_goal:
            return self._observe(), 1.0, True, {"outcome": "win"}
        if enemy_at_goal:
            return self._observe(), -0.5, True, {"outcome": "lose"}
        if timeout:
            return self._observe(), -0.1, True, {"outcome": "timeout"}

        return self._observe(), -0.01, False, {}

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def state_space_size(self) -> int:
        """Upper bound on discrete state space size."""
        return GRID * GRID * 3 * 3  # row × col × elev_bucket × dist_bucket

    # ── Internals ────────────────────────────────────────────────────────────

    def _observe(self) -> NavState:
        ar, ac = self.agent_pos
        er, ec = self.enemy_pos
        elev = int(self.elevation[ar, ac])
        dist = _chebyshev(ar, ac, er, ec)
        return NavState(
            row=ar,
            col=ac,
            elev_bucket=_elev_bucket(elev),
            enemy_dist_bucket=_dist_bucket(dist),
        )

    def _enemy_step(self) -> tuple[int, int]:
        """Greedy scout: move one cardinal step toward the nearest goal tile.

        Cardinal-only movement keeps the scout at roughly the same speed as
        the agent, making the race competitive enough to teach the agent to
        prioritise high-ground tiles.
        """
        if not self.goal_tiles:
            return self.enemy_pos
        er, ec = self.enemy_pos
        # Pick nearest goal by Manhattan distance (matches cardinal movement)
        target = min(
            self.goal_tiles,
            key=lambda g: abs(g[0] - er) + abs(g[1] - ec),
        )
        tr, tc = target
        # Choose the axis with the larger gap to reduce first (breaks ties)
        row_gap = abs(tr - er)
        col_gap = abs(tc - ec)
        if row_gap >= col_gap and row_gap > 0:
            dr = 1 if tr > er else -1
            dc = 0
        elif col_gap > 0:
            dr = 0
            dc = 1 if tc > ec else -1
        else:
            return self.enemy_pos  # already at goal
        nr, nc = er + dr, ec + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            return (nr, nc)
        return self.enemy_pos
