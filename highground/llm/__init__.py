"""V6 LLM controller for the High Ground SRPG.

This package provides LLM-driven logit steering on top of a trained
BenchMARL policy.  The pipeline is:

1. :class:`~highground.llm.narrator.StateNarrator` converts GameState
   to a compact natural-language summary.
2. :class:`~highground.llm.strategy_manager.StrategyManager` queries an
   Ollama LLM (via instructor + openai) to produce a
   :class:`~highground.llm.models.StrategicIntent`.
3. :class:`~highground.llm.tile_index.TileIndex` resolves the intent's
   objective to a set of target tiles and runs Dijkstra BFS for first-step
   navigation.
4. :class:`~highground.llm.tactical_navigator.TacticalNavigator` converts
   the BFS result into a 12-element additive logit bias.
5. :class:`~highground.llm.wrapper.LLMSteeringWrapper` applies the bias to
   the policy logits and returns the steered action.

Public exports
--------------
Models:
    ObjectiveType, StrategicIntent, StrategyPhase, StepRecord

Navigation:
    TileIndex, TacticalNavigator

Strategy:
    StrategyManager

Narrator:
    narrate  (module-level function from narrator.py)

Wrapper:
    LLMSteeringWrapper
"""

from __future__ import annotations

from highground.llm.models import (
    ObjectiveType,
    StrategicIntent,
    StrategyPhase,
    StepRecord,
)
from highground.llm.narrator import narrate as StateNarrator  # exported as StateNarrator alias
from highground.llm.tile_index import TileIndex
from highground.llm.tactical_navigator import TacticalNavigator
from highground.llm.strategy_manager import StrategyManager
from highground.llm.wrapper import LLMSteeringWrapper

__all__ = [
    "ObjectiveType",
    "StrategicIntent",
    "StrategyPhase",
    "StepRecord",
    "StateNarrator",
    "TileIndex",
    "TacticalNavigator",
    "StrategyManager",
    "LLMSteeringWrapper",
]
