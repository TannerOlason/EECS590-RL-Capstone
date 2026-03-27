"""LLM-backed strategy manager for the High Ground SRPG.

Manages a phase-based strategy state machine, querying a local Ollama LLM
(via the instructor + openai libraries) to produce structured StrategicIntent
objects.  Results are cached per round to avoid redundant LLM calls.

Usage::

    phases = [
        StrategyPhase(prompt="Seize the high ground early.", max_rounds=5),
        StrategyPhase(prompt="Rush the weakened enemy."),
    ]
    mgr = StrategyManager(team_id=0, phases=phases)
    intent = mgr.get_intent(narrate(game_state, 0), round_num=3)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

from highground.llm.models import ObjectiveType, StrategicIntent, StrategyPhase

log = logging.getLogger(__name__)

# Load system prompt from the bundled text file
_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system_base.txt"

try:
    _SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    _SYSTEM_PROMPT = (
        "You are a tactics advisor for a grid strategy game. "
        "Output valid JSON matching the StrategicIntent schema."
    )


def _fallback_intent(objective: ObjectiveType) -> StrategicIntent:
    """Return a safe fallback StrategicIntent when the LLM is unavailable."""
    return StrategicIntent(
        objective=objective,
        attack_urgency=0.5,
        hold_after=False,
        phase_complete=False,
        reasoning="[fallback: LLM unavailable]",
    )


class StrategyManager:
    """Phase-driven LLM strategy controller.

    Parameters
    ----------
    team_id:
        Which team this manager controls (TEAM_A=0 or TEAM_B=1).
    phases:
        Ordered list of StrategyPhase objects.  The manager advances through
        them as each phase is completed by the LLM.
    model:
        Ollama model name (e.g. "llama3.1:8b", "mistral:7b").
    base_url:
        Base URL for the Ollama OpenAI-compatible API endpoint.
    fallback_objective:
        ObjectiveType to use when the LLM fails or is unavailable.
    """

    def __init__(
        self,
        team_id: int,
        phases: list[StrategyPhase],
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        fallback_objective: ObjectiveType = ObjectiveType.RUSH_ENEMY,
    ) -> None:
        self._team_id = team_id
        self._phases = phases if phases else [
            StrategyPhase(prompt="Rush the enemy and attack aggressively.")
        ]
        self._model = model
        self._base_url = base_url
        self._fallback_objective = fallback_objective

        self._phase_idx: int = 0
        self._pending_advance: bool = False  # advance on *next* call

        # Cache: (team_id, round_num) → StrategicIntent
        self._cache: dict[tuple[int, int], StrategicIntent] = {}

        # Lazy-initialised instructor client
        self._client: Optional[object] = None
        self._client_error: bool = False  # True = already failed to connect, skip

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def phase_index(self) -> int:
        """Index of the current active phase."""
        return self._phase_idx

    @property
    def current_phase_prompt(self) -> str:
        """The text directive of the current active phase."""
        return self._phases[self._phase_idx].prompt

    def get_intent(self, summary: str, round_num: int) -> StrategicIntent:
        """Return a StrategicIntent for the current round.

        Results are cached per (team_id, round_num).  If the LLM returns
        ``phase_complete=True``, the next call (on a later round) will
        automatically advance to the next phase.

        Parameters
        ----------
        summary:
            Natural-language game state from :func:`~highground.llm.narrator.narrate`.
        round_num:
            Current game round number (used as cache key).

        Returns
        -------
        StrategicIntent
            Structured intent from the LLM, or a safe fallback on any error.
        """
        # Apply pending phase advance from a previous round
        if self._pending_advance:
            self._advance_phase()
            self._pending_advance = False

        cache_key = (self._team_id, round_num)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check max_rounds auto-advance
        current_phase = self._phases[self._phase_idx]
        if current_phase.max_rounds is not None and round_num >= current_phase.max_rounds:
            log.info(
                "StrategyManager team=%d: auto-advancing past phase %d (max_rounds=%d reached).",
                self._team_id,
                self._phase_idx,
                current_phase.max_rounds,
            )
            self._advance_phase()
            current_phase = self._phases[self._phase_idx]

        intent = self._query_llm(summary, current_phase.prompt)
        self._cache[cache_key] = intent

        # Schedule phase advance — takes effect on the *next* round's call
        if intent.phase_complete:
            self._pending_advance = True

        return intent

    # ── Internal ──────────────────────────────────────────────────────────────

    def _advance_phase(self) -> None:
        """Move to the next phase, clamping at the last phase."""
        if self._phase_idx < len(self._phases) - 1:
            self._phase_idx += 1
            log.info(
                "StrategyManager team=%d: advanced to phase %d: %s",
                self._team_id,
                self._phase_idx,
                self._phases[self._phase_idx].prompt[:60],
            )

    def _get_client(self) -> Optional[object]:
        """Lazy-initialise the instructor/openai client.  Returns None on failure."""
        if self._client_error:
            return None
        if self._client is not None:
            return self._client

        try:
            import instructor
            from openai import OpenAI

            raw_client = OpenAI(base_url=self._base_url, api_key="ollama")
            self._client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
            log.debug("StrategyManager: instructor client initialised (model=%s).", self._model)
        except ImportError as exc:
            log.warning(
                "StrategyManager: instructor or openai not installed (%s). "
                "Using fallback objective.",
                exc,
            )
            self._client_error = True
            return None
        except Exception as exc:
            log.warning(
                "StrategyManager: failed to initialise LLM client (%s). "
                "Using fallback objective.",
                exc,
            )
            self._client_error = True
            return None

        return self._client

    def _query_llm(self, summary: str, phase_prompt: str) -> StrategicIntent:
        """Send a query to the LLM and parse the structured response.

        On any exception (connection error, validation error, timeout),
        logs a warning and returns the fallback intent.
        """
        client = self._get_client()
        if client is None:
            return _fallback_intent(self._fallback_objective)

        user_message = (
            f"Phase directive: {phase_prompt}\n\n"
            f"Current game state:\n{summary}\n\n"
            "Output a JSON object conforming to the StrategicIntent schema."
        )

        try:
            intent: StrategicIntent = client.chat.completions.create(
                model=self._model,
                response_model=StrategicIntent,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
            )
            log.debug(
                "StrategyManager team=%d: LLM returned objective=%s urgency=%.2f reasoning=%s",
                self._team_id,
                intent.objective,
                intent.attack_urgency,
                intent.reasoning[:80],
            )
            return intent

        except ConnectionError as exc:
            log.warning(
                "StrategyManager team=%d: Ollama connection error (%s). Using fallback.",
                self._team_id,
                exc,
            )
        except Exception as exc:
            log.warning(
                "StrategyManager team=%d: LLM query failed (%s: %s). Using fallback.",
                self._team_id,
                type(exc).__name__,
                exc,
            )

        return _fallback_intent(self._fallback_objective)
