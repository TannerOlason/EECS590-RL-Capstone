"""Data models for the V6 LLM controller.

Defines enums, Pydantic models, and dataclasses used across the LLM steering
pipeline: objectives, strategic intents, phase descriptors, and per-step
recording structures for the smoke test viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class ObjectiveType(str, Enum):
    """High-level tactical objectives that the LLM can assign to a team."""

    OCCUPY_HIGH_GROUND = "OCCUPY_HIGH_GROUND"
    FLANK_SOUTH        = "FLANK_SOUTH"
    FLANK_NORTH        = "FLANK_NORTH"
    RUSH_ENEMY         = "RUSH_ENEMY"
    HOLD_POSITION      = "HOLD_POSITION"
    RETREAT            = "RETREAT"
    ENGAGE_NEAREST     = "ENGAGE_NEAREST"
    SUPPORT_ALLIES     = "SUPPORT_ALLIES"


class StrategicIntent(BaseModel):
    """Structured LLM output representing a tactical directive for one game round.

    Attributes
    ----------
    objective:
        One of the ObjectiveType enum values specifying the primary goal.
    attack_urgency:
        Float in [0, 1]. 0 = focus on positioning, never attack unless free.
        1 = attack as aggressively as possible each step.
    hold_after:
        If True, once a unit reaches the target tile it should stay there
        rather than continuing to advance.
    phase_complete:
        When True, the strategy manager will advance to the next phase on the
        *following* round (current round finishes with this intent).
    reasoning:
        Short free-text explanation. Used for logging and the smoke test viewer.
    """

    objective: ObjectiveType
    attack_urgency: float = Field(ge=0.0, le=1.0)
    hold_after: bool
    phase_complete: bool
    reasoning: str


class StrategyPhase(BaseModel):
    """One phase in a multi-phase strategy programme.

    Attributes
    ----------
    prompt:
        The tactical directive injected into the LLM user message for this phase.
    max_rounds:
        Optional round limit. If the LLM has not set phase_complete by this
        round, the manager will advance automatically. None = no auto-advance.
    """

    prompt: str
    max_rounds: int | None = None


@dataclass
class StepRecord:
    """Complete snapshot of one micro-action decision for all four steering variants.

    Used by the smoke test viewer to compare base policy vs. semantic steering
    vs. tactical steering vs. both combined, step by step.
    """

    # ── Game context ──────────────────────────────────────────────────────────
    round_num:     int
    unit_id:       str                # e.g. "team0_unit2"
    unit_class:    str                # e.g. "FIGHTER"
    unit_pos:      tuple[int, int]    # (row, col) from game_state (not obs)
    unit_hp:       float
    unit_max_hp:   int

    # ── Raw tensors ───────────────────────────────────────────────────────────
    obs:           np.ndarray         # (418,) raw observation vector
    action_mask:   np.ndarray         # (12,) binary mask (1 = legal)
    logits_base:   np.ndarray         # (12,) raw policy logits before masking
    bias_sem:      np.ndarray         # (12,) semantic-only additive bias
    bias_tac:      np.ndarray         # (12,) tactical-only additive bias
    bias_both:     np.ndarray         # (12,) combined (semantic + tactical) bias

    # ── Probabilities after masking + softmax ─────────────────────────────────
    probs_base:    np.ndarray         # (12,) with no steering
    probs_sem:     np.ndarray         # (12,) with semantic bias applied
    probs_tac:     np.ndarray         # (12,) with tactical bias applied
    probs_both:    np.ndarray         # (12,) with both biases applied

    # ── Chosen actions ────────────────────────────────────────────────────────
    action_base:   int
    action_sem:    int
    action_tac:    int
    action_both:   int

    # ── LLM / navigator state ─────────────────────────────────────────────────
    intent:        StrategicIntent | None   # LLM intent for this round (may be None for base)
    bfs_target:    tuple[int, int] | None   # resolved BFS target tile
    bfs_direction: str | None               # e.g. "MOVE_NE" or None

    # ── ASCII map snapshot ────────────────────────────────────────────────────
    map_snapshot:  list[list[str]]          # 13×13 character grid for display

    # ── Full unit roster for HP bars ──────────────────────────────────────────
    # Each dict: {id, team, class, hp, max_hp, alive, row, col}
    units_snapshot: list[dict] = field(default_factory=list)
