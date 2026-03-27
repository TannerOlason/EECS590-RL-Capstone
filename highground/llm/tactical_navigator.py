"""Tactical navigator: converts a StrategicIntent into per-action logit biases.

The TacticalNavigator runs BFS (via TileIndex) to find the optimal first
direction toward the current objective's target tiles, then constructs a
12-element additive bias vector that can be added to the policy logits before
masked softmax.

Usage::

    navigator = TacticalNavigator(tile_index, max_bias=2.0)
    bias = navigator.compute_bias(intent, unit_pos, game_state, team_id)
    steered_logits = base_logits + bias
"""

from __future__ import annotations

import numpy as np

from highground.llm.models import ObjectiveType, StrategicIntent
from highground.llm.tile_index import TileIndex

# Map (dr, dc) → action index (0-7 are move actions)
DELTA_TO_ACTION: dict[tuple[int, int], int] = {
    (-1,  0): 0,   # MOVE_N
    ( 1,  0): 1,   # MOVE_S
    ( 0,  1): 2,   # MOVE_E
    ( 0, -1): 3,   # MOVE_W
    (-1,  1): 4,   # MOVE_NE
    (-1, -1): 5,   # MOVE_NW
    ( 1,  1): 6,   # MOVE_SE
    ( 1, -1): 7,   # MOVE_SW
}
ACTION_TO_DELTA: dict[int, tuple[int, int]] = {v: k for k, v in DELTA_TO_ACTION.items()}

# Soft-bias neighbors: adjacent directional actions to the primary direction
# (e.g., if moving NE, also softly bias N and E)
ADJACENT_DIRS: dict[tuple[int, int], list[tuple[int, int]]] = {
    (-1,  0): [(-1,  1), (-1, -1)],   # N  → NE, NW
    ( 1,  0): [( 1,  1), ( 1, -1)],   # S  → SE, SW
    ( 0,  1): [(-1,  1), ( 1,  1)],   # E  → NE, SE
    ( 0, -1): [(-1, -1), ( 1, -1)],   # W  → NW, SW
    (-1,  1): [(-1,  0), ( 0,  1)],   # NE → N, E
    (-1, -1): [(-1,  0), ( 0, -1)],   # NW → N, W
    ( 1,  1): [( 1,  0), ( 0,  1)],   # SE → S, E
    ( 1, -1): [( 1,  0), ( 0, -1)],   # SW → S, W
}


class TacticalNavigator:
    """Converts a StrategicIntent into a 12-element additive logit bias.

    Parameters
    ----------
    tile_index:
        Pre-built TileIndex for the current map.
    max_bias:
        Maximum absolute bias value added to the primary action's logit.
        Adjacent directions receive ``max_bias * 0.5``.
    """

    def __init__(self, tile_index: TileIndex, max_bias: float = 2.0) -> None:
        self._tile_index = tile_index
        self.max_bias = float(max_bias)

    def compute_bias(
        self,
        intent: StrategicIntent,
        unit_pos: tuple[int, int],
        game_state: "GameState",  # noqa: F821
        team_id: int,
    ) -> np.ndarray:
        """Compute the 12-element additive logit bias for the acting unit.

        The bias vector encodes three signals:
        1. **Directional movement bias** — points toward the nearest target tile
           via Dijkstra BFS, with a soft spread to adjacent directions.
        2. **Hold penalty** — if the unit is already on target and hold_after=True,
           movement actions are penalised.
        3. **Attack urgency** — scales up attack action logits by intent.attack_urgency.

        Parameters
        ----------
        intent:
            Strategic intent from the LLM (or hardcoded for tactical mode).
        unit_pos:
            Actual (row, col) from game_state — NOT the obs perspective values.
        game_state:
            Live game state used to resolve target tiles.
        team_id:
            The acting unit's team.

        Returns
        -------
        np.ndarray
            Shape (12,) float32 additive bias.
        """
        bias = np.zeros(12, dtype=np.float32)

        targets = self._tile_index.resolve_targets(
            intent.objective, unit_pos, game_state, team_id
        )

        if not targets:
            # No targets resolved — fall back to pure attack urgency
            bias[8:11] += intent.attack_urgency * self.max_bias
            return bias

        on_target = unit_pos in targets

        if on_target and intent.hold_after:
            # Penalise all movement to encourage staying put
            bias[:8] -= 1.5
        elif not on_target:
            # BFS to find cheapest first step toward any target
            delta, _dist = self._tile_index.dijkstra_first_step(unit_pos, targets)
            if delta is not None and delta in DELTA_TO_ACTION:
                primary_action = DELTA_TO_ACTION[delta]
                bias[primary_action] += self.max_bias
                # Soft bias on adjacent directions
                for adj_delta in ADJACENT_DIRS.get(delta, []):
                    if adj_delta in DELTA_TO_ACTION:
                        bias[DELTA_TO_ACTION[adj_delta]] += self.max_bias * 0.5

        # Attack urgency always applies (additive on top of movement bias)
        bias[8:11] += intent.attack_urgency * self.max_bias

        return bias
