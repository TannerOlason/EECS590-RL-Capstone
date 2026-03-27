"""LLM steering wrapper for the BenchMARL policy adapter.

Wraps a BenchMARLPolicyAdapter with optional LLM-based logit steering.
Supports four operating modes that can be compared side-by-side in the
smoke test viewer.

Modes
-----
"base"
    No steering. Identical to base policy. Useful as a control condition.
"semantic"
    Attack urgency and hold penalty only; no directional BFS. Tests whether
    LLM-provided urgency signals alone change behaviour.
"tactical"
    BFS directional bias from a fixed hardcoded objective; no LLM query.
    Tests whether navigation steering alone is useful.
"both"
    Full pipeline: LLM → StrategicIntent → TacticalNavigator → per-unit bias.

Usage::

    wrapper = LLMSteeringWrapper(base_policy, tile_index, mode="both",
                                  strategy_manager=mgr)
    action, _ = wrapper.predict(obs, deterministic=True,
                                 action_masks=mask, game_state=gs, team_id=0)
"""

from __future__ import annotations

import numpy as np

from highground.llm.models import ObjectiveType, StrategicIntent, StrategyPhase, StepRecord
from highground.llm.narrator import narrate
from highground.llm.tile_index import TileIndex
from highground.llm.tactical_navigator import TacticalNavigator, DELTA_TO_ACTION, ACTION_TO_DELTA
from highground.llm.strategy_manager import StrategyManager
from highground.training.benchmarl_adapter import BenchMARLPolicyAdapter

ACTION_NAMES = [
    "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
    "MOVE_NE", "MOVE_NW", "MOVE_SE", "MOVE_SW",
    "ATTACK_0", "ATTACK_1", "ATTACK_2", "END_TURN",
]


def _masked_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary action mask and return softmax probabilities.

    Illegal actions (mask==0) receive -1e9 before softmax to effectively
    zero out their probability.

    Parameters
    ----------
    logits:
        (12,) raw logit array.
    mask:
        (12,) binary array; 1 = legal, 0 = illegal.

    Returns
    -------
    np.ndarray
        (12,) probability array summing to ~1.0.
    """
    masked = logits.copy().astype(np.float64)
    masked[mask == 0] = -1e9
    shifted = masked - masked.max()
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)


def _choose(probs: np.ndarray, deterministic: bool) -> int:
    """Select an action from a probability distribution.

    Parameters
    ----------
    probs:
        (12,) probability array.
    deterministic:
        If True, take argmax. If False, sample from the distribution.

    Returns
    -------
    int
        Selected action index (0–11).
    """
    if deterministic:
        return int(np.argmax(probs))
    # Renormalise to guard against floating-point drift
    p = probs.astype(np.float64)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


class LLMSteeringWrapper:
    """Wraps BenchMARLPolicyAdapter with optional LLM-based logit steering.

    Parameters
    ----------
    base_policy:
        The underlying BenchMARL policy to steer.
    tile_index:
        Pre-built TileIndex for the current map.
    mode:
        One of "base", "semantic", "tactical", "both".
    max_bias:
        Maximum additive logit bias magnitude (TacticalNavigator parameter).
    strategy_manager:
        Required for "semantic" and "both" modes.  Provides LLM intents.
    fixed_objective:
        Used in "tactical" mode (and "semantic"/"both" when --no-llm is set)
        as a hardcoded objective.
    fixed_attack_urgency:
        Attack urgency used alongside fixed_objective in "tactical" mode.
    """

    def __init__(
        self,
        base_policy: BenchMARLPolicyAdapter,
        tile_index: TileIndex,
        mode: str = "both",
        max_bias: float = 2.0,
        strategy_manager: StrategyManager | None = None,
        fixed_objective: ObjectiveType = ObjectiveType.RUSH_ENEMY,
        fixed_attack_urgency: float = 0.5,
    ) -> None:
        assert mode in ("base", "semantic", "tactical", "both"), (
            f"Invalid mode {mode!r}. Must be one of: base, semantic, tactical, both."
        )
        self._policy = base_policy
        self._tile_index = tile_index
        self._mode = mode
        self._max_bias = max_bias
        self._strategy_manager = strategy_manager
        self._fixed_objective = fixed_objective
        self._fixed_attack_urgency = fixed_attack_urgency
        self._navigator = TacticalNavigator(tile_index, max_bias=max_bias)

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(
        self,
        obs,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
        game_state=None,
        team_id: int = 0,
    ) -> tuple[int, None]:
        """Drop-in replacement for BenchMARLPolicyAdapter.predict().

        Parameters
        ----------
        obs:
            418-dim float32 observation array or dict.
        deterministic:
            Argmax if True, sample if False.
        action_masks:
            (12,) binary mask from game_state.valid_action_mask().
        game_state:
            Required for "tactical", "semantic", "both" modes.
        team_id:
            Acting team index.

        Returns
        -------
        tuple[int, None]
            (action, None) matching MaskablePPO signature.
        """
        raw_obs = obs.get("observation", next(iter(obs.values()))) if isinstance(obs, dict) else obs
        mask = np.asarray(action_masks, dtype=np.float32) if action_masks is not None else np.ones(12, dtype=np.float32)

        logits = self._policy.predict_logits(raw_obs)

        bias = self._compute_bias(raw_obs, mask, game_state, team_id)
        steered_logits = logits + bias
        probs = _masked_probs(steered_logits, mask)
        action = _choose(probs, deterministic)
        return action, None

    def get_step_record(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        game_state,
        team_id: int,
        semantic_manager: StrategyManager | None = None,
        tactical_objective: ObjectiveType = ObjectiveType.RUSH_ENEMY,
        tactical_attack_urgency: float = 0.5,
        both_manager: StrategyManager | None = None,
        deterministic: bool = True,
    ) -> StepRecord:
        """Compute all four variant actions and return a StepRecord.

        This method computes base/semantic/tactical/both biases and actions
        without actually advancing the game state.  Used by the smoke test
        viewer to record the full decision context at each step.

        Parameters
        ----------
        obs:
            (418,) float32 observation for the current acting unit.
        action_mask:
            (12,) binary mask.
        game_state:
            Live game state (read-only here).
        team_id:
            Acting team.
        semantic_manager:
            StrategyManager for the "semantic" variant (LLM urgency only).
        tactical_objective:
            Fixed objective for the "tactical" variant (no LLM).
        tactical_attack_urgency:
            Attack urgency for the "tactical" variant.
        both_manager:
            StrategyManager for the "both" variant (full LLM + BFS).
        deterministic:
            Action selection mode.

        Returns
        -------
        StepRecord
        """
        mask = np.asarray(action_mask, dtype=np.float32)
        logits_base = self._policy.predict_logits(obs)
        round_num = game_state.round_number
        cu = game_state.current_unit
        unit_pos = (cu.row, cu.col)

        # ── Semantic bias (urgency only, no BFS) ─────────────────────────────
        intent_sem: StrategicIntent | None = None
        if semantic_manager is not None:
            summary = narrate(game_state, team_id)
            intent_sem = semantic_manager.get_intent(summary, round_num)
            bias_sem = self._semantic_only_bias(intent_sem, mask)
        else:
            bias_sem = np.zeros(12, dtype=np.float32)

        # ── Tactical bias (BFS only, fixed objective) ─────────────────────────
        intent_tac = StrategicIntent(
            objective=tactical_objective,
            attack_urgency=tactical_attack_urgency,
            hold_after=False,
            phase_complete=False,
            reasoning="[tactical: fixed objective]",
        )
        bias_tac = self._navigator.compute_bias(intent_tac, unit_pos, game_state, team_id)

        # ── Both bias (LLM + BFS) ─────────────────────────────────────────────
        intent_both: StrategicIntent | None = None
        if both_manager is not None:
            summary_both = narrate(game_state, team_id)
            intent_both = both_manager.get_intent(summary_both, round_num)
            bias_both = self._navigator.compute_bias(intent_both, unit_pos, game_state, team_id)
        else:
            # Fallback to tactical bias when no LLM manager provided
            intent_both = intent_tac
            bias_both = bias_tac.copy()

        # ── Compute probabilities and actions for all variants ────────────────
        probs_base = _masked_probs(logits_base, mask)
        probs_sem  = _masked_probs(logits_base + bias_sem, mask)
        probs_tac  = _masked_probs(logits_base + bias_tac, mask)
        probs_both = _masked_probs(logits_base + bias_both, mask)

        action_base = _choose(probs_base, deterministic)
        action_sem  = _choose(probs_sem, deterministic)
        action_tac  = _choose(probs_tac, deterministic)
        action_both = _choose(probs_both, deterministic)

        # ── BFS debug info ────────────────────────────────────────────────────
        active_intent = intent_both if intent_both is not None else intent_tac
        targets = self._tile_index.resolve_targets(
            active_intent.objective, unit_pos, game_state, team_id
        )
        bfs_target: tuple[int, int] | None = None
        bfs_direction: str | None = None
        if targets:
            delta, _cost = self._tile_index.dijkstra_first_step(unit_pos, targets)
            if delta is not None:
                action_idx = DELTA_TO_ACTION.get(delta)
                if action_idx is not None:
                    bfs_direction = ACTION_NAMES[action_idx]
            # Nearest target tile for display
            if unit_pos not in targets:
                bfs_target = min(
                    targets,
                    key=lambda t: abs(t[0] - unit_pos[0]) + abs(t[1] - unit_pos[1]),
                )

        # ── Map snapshot ──────────────────────────────────────────────────────
        map_snapshot = _build_map_snapshot(game_state, team_id)

        from highground.engine.units import CLASS_STATS
        max_hp = CLASS_STATS[cu.unit_class]["hp"]

        units_snapshot = [
            {
                "id":     u.unit_id,
                "team":   u.team,
                "class":  u.unit_class,
                "hp":     u.hp if u.alive else 0,
                "max_hp": CLASS_STATS[u.unit_class]["hp"],
                "alive":  u.alive,
                "row":    u.row,
                "col":    u.col,
            }
            for u in game_state.units
        ]

        return StepRecord(
            round_num=round_num,
            unit_id=f"team{team_id}_unit{cu.unit_id}",
            unit_class=cu.unit_class.name,
            unit_pos=unit_pos,
            unit_hp=float(cu.hp),
            unit_max_hp=max_hp,
            obs=obs.copy(),
            action_mask=mask.copy(),
            logits_base=logits_base.copy(),
            bias_sem=bias_sem.copy(),
            bias_tac=bias_tac.copy(),
            bias_both=bias_both.copy(),
            probs_base=probs_base,
            probs_sem=probs_sem,
            probs_tac=probs_tac,
            probs_both=probs_both,
            action_base=action_base,
            action_sem=action_sem,
            action_tac=action_tac,
            action_both=action_both,
            intent=intent_both if intent_both is not None else intent_sem,
            bfs_target=bfs_target,
            bfs_direction=bfs_direction,
            map_snapshot=map_snapshot,
            units_snapshot=units_snapshot,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_bias(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        game_state,
        team_id: int,
    ) -> np.ndarray:
        """Compute additive bias according to self._mode."""
        if self._mode == "base" or game_state is None:
            return np.zeros(12, dtype=np.float32)

        cu = game_state.current_unit
        unit_pos = (cu.row, cu.col)
        round_num = game_state.round_number

        if self._mode == "semantic":
            intent = self._get_intent_or_fallback(obs, game_state, team_id, round_num)
            return self._semantic_only_bias(intent, mask)

        elif self._mode == "tactical":
            intent = StrategicIntent(
                objective=self._fixed_objective,
                attack_urgency=self._fixed_attack_urgency,
                hold_after=False,
                phase_complete=False,
                reasoning="[tactical: fixed objective]",
            )
            return self._navigator.compute_bias(intent, unit_pos, game_state, team_id)

        elif self._mode == "both":
            intent = self._get_intent_or_fallback(obs, game_state, team_id, round_num)
            return self._navigator.compute_bias(intent, unit_pos, game_state, team_id)

        return np.zeros(12, dtype=np.float32)

    def _get_intent_or_fallback(
        self,
        obs: np.ndarray,
        game_state,
        team_id: int,
        round_num: int,
    ) -> StrategicIntent:
        """Query the strategy manager or return a fallback if unavailable."""
        if self._strategy_manager is not None:
            summary = narrate(game_state, team_id)
            return self._strategy_manager.get_intent(summary, round_num)
        # No manager — return a fixed fallback
        return StrategicIntent(
            objective=self._fixed_objective,
            attack_urgency=self._fixed_attack_urgency,
            hold_after=False,
            phase_complete=False,
            reasoning="[fallback: no strategy manager]",
        )

    def _semantic_only_bias(self, intent: StrategicIntent, mask: np.ndarray) -> np.ndarray:
        """Semantic mode: apply attack urgency and hold penalty, but no BFS direction."""
        bias = np.zeros(12, dtype=np.float32)
        # Attack urgency
        bias[8:11] += intent.attack_urgency * self._max_bias
        # Hold penalty (penalise movement when hold_after is set)
        if intent.hold_after:
            bias[:8] -= 1.5
        return bias


def _build_map_snapshot(game_state, team_id: int) -> list[list[str]]:
    """Build a 13x13 ASCII character map for the smoke test viewer.

    Character legend:
    - '.' = normal walkable (elevation 0)
    - 'h' = elevation 1 walkable
    - 'H' = elevation 2 walkable
    - '#' = uncrossable obstacle
    - 'r' = rough normal terrain
    - 'R' = rough elevation 1
    - Unit overlays:
        '*' = current acting unit (overrides team char)
        'A' = team 0 alive unit
        'a' = team 0 unit (has_attacked or hp < 50%)
        'B' = team 1 alive unit
        'b' = team 1 unit (has_attacked or hp < 50%)

    Parameters
    ----------
    game_state:
        Live game state.
    team_id:
        Perspective team (affects 'A'/'B' labels).

    Returns
    -------
    list[list[str]]
        13×13 nested list of single-character strings.
    """
    from highground.engine.grid import GRID_SIZE, Terrain

    grid = game_state.grid
    snapshot: list[list[str]] = []

    for r in range(GRID_SIZE):
        row: list[str] = []
        for c in range(GRID_SIZE):
            terrain = int(grid.terrain[r, c])
            elev = grid.get_elevation(r, c)

            if terrain == int(Terrain.UNCROSSABLE):
                row.append('#')
            elif terrain == int(Terrain.ROUGH):
                row.append('R' if elev >= 1 else 'r')
            else:
                # Normal terrain
                if elev == 2:
                    row.append('H')
                elif elev == 1:
                    row.append('h')
                else:
                    row.append('.')
        snapshot.append(row)

    # Overlay units
    current_id = game_state.current_unit_id
    for u in game_state.units:
        if not u.alive:
            continue
        r, c = u.row, u.col
        if u.unit_id == current_id:
            snapshot[r][c] = '*'
        elif u.team == 0:
            low_hp = u.hp < u.max_hp * 0.5
            snapshot[r][c] = 'a' if (u.has_attacked or low_hp) else 'A'
        else:
            low_hp = u.hp < u.max_hp * 0.5
            snapshot[r][c] = 'b' if (u.has_attacked or low_hp) else 'B'

    return snapshot
