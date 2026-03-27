"""MAPPO training via BenchMARL for the High Ground SRPG (6c CTDE).

Replaces the MaskablePPO single-agent training pipeline.  Key differences:

  - All 6 units train simultaneously in one MAPPO experiment.
  - Critic receives the joint global state (836 dims) for CTDE.
  - Curriculum is multi-phase: policy weights carry over, frame counters reset.
  - Rich TUI shows three panels: MAPPO metrics | live battle replay | episode stats.

Usage (standalone):
    python -m highground.training.benchmarl_train
    python -m highground.training.benchmarl_train --frames 2500000 --curriculum

Entry point for run_pipeline.py:
    from highground.training.benchmarl_train import train_mappo
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TaskProgressColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Callback, Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

from highground.training.cnn_model import SpatialCnnMlpConfig

from highground.engine.game_state import MAX_TURNS, NUM_ACTIONS
from highground.engine.grid import GRID_SIZE, Terrain
from highground.engine.units import TEAM_A, TEAM_B, UnitClass
from highground.env.srpg_env import OBS_SIZE, UNITS_PER_TEAM
from highground.training.benchmarl_task import HighGroundTaskClass


# ── Palette & constants ───────────────────────────────────────────────────────

_SPARK   = " ▁▂▃▄▅▆▇█"
_SPIN    = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BORDER  = "bright_blue"
_CA      = "bright_cyan"    # Team A colour
_CB      = "bright_red"     # Team B colour

# Offsets into the 418-dim observation vector
_TERRAIN_END  = GRID_SIZE * GRID_SIZE * 2   # 338 — end of terrain+elevation block
_UNIT_SIZE    = 11                           # features per unit

# Class → max HP (matches HighGroundEnv unit stats)
_CLASS_MAX_HP = {0: 12, 1: 10, 2: 8, 3: 12}

_TERRAIN_CELL: dict[tuple[int, int], tuple[str, str]] = {
    (int(Terrain.NORMAL),      0): ("··", "dim green"),
    (int(Terrain.NORMAL),      1): ("·▪", "green"),
    (int(Terrain.NORMAL),      2): ("·▲", "bold bright_green"),
    (int(Terrain.ROUGH),       0): ("≈·", "yellow"),
    (int(Terrain.ROUGH),       1): ("≈▪", "bright_yellow"),
    (int(Terrain.ROUGH),       2): ("≈▲", "bold bright_yellow"),
    (int(Terrain.UNCROSSABLE), 0): ("██", "on grey23"),
    (int(Terrain.UNCROSSABLE), 1): ("██", "on grey23"),
    (int(Terrain.UNCROSSABLE), 2): ("██", "on grey23"),
}


# ── Replay frame decoding ─────────────────────────────────────────────────────

class _Frame:
    """Lightweight snapshot of one game step for the live replay."""
    __slots__ = ("step", "active_id", "round_num", "units")

    def __init__(self, step: int, active_id: int, round_num: int, units: list) -> None:
        self.step      = step
        self.active_id = active_id   # agent index 0-5 that is acting
        self.round_num = round_num
        self.units     = units       # list of dicts: id/row/col/team/hp/max_hp/alive/class


# ── Early-advancement support ─────────────────────────────────────────────────

class _PhaseComplete(Exception):
    """Raised by AutoAdvanceCallback to exit exp.run() and advance the curriculum."""
    def __init__(self, frames: int, reason: str) -> None:
        super().__init__(reason)
        self.frames = frames
        self.reason = reason


class AutoAdvanceCallback(Callback):
    """Monitors training signals and raises _PhaseComplete when the policy plateaus.

    Advancement triggers (either condition fires after ``min_frames``):
      1. Return plateau  — last ``plateau_window`` evaluation returns span < ``return_threshold``.
      2. Critic-loss plateau — relative variation of the last 200 ``loss_critic`` values < ``loss_rel_threshold``.
    """

    def __init__(
        self,
        min_frames: int,
        plateau_window: int = 6,
        return_threshold: float = 0.3,
        loss_rel_threshold: float = 0.10,
    ) -> None:
        super().__init__()
        self._min_frames         = min_frames
        self._window             = plateau_window
        self._ret_thresh         = return_threshold
        self._loss_thresh        = loss_rel_threshold

        self._return_evals: deque = deque(maxlen=plateau_window)
        self._loss_hist:    deque = deque(maxlen=200)
        self._act_counts          = [0] * NUM_ACTIONS
        self._act_total:    int   = 0

        self.frames_at_advance:  int = 0
        self.advance_reason:     str = ""

    # ── Metric collection ─────────────────────────────────────────────────────

    def on_batch_collected(self, batch) -> None:
        try:
            actions = batch["team_a"]["action"].flatten().tolist()
            for a in actions:
                idx = int(a)
                if 0 <= idx < NUM_ACTIONS:
                    self._act_counts[idx] += 1
                    self._act_total += 1
        except Exception:
            pass

    def on_train_end(self, training_td, group: str) -> None:
        try:
            if "loss_critic" in training_td.keys():
                self._loss_hist.append(float(training_td["loss_critic"].mean()))
        except Exception:
            pass

    def on_evaluation_end(self, rollouts: list) -> None:
        if not rollouts:
            return
        try:
            rewards = rollouts[0].get(("next", "team_a", "reward"), None)
            if rewards is not None:
                self._return_evals.append(float(rewards.sum()))
        except Exception:
            pass
        self._check()

    # ── Advancement logic ─────────────────────────────────────────────────────

    def _check(self) -> None:
        if self.experiment is None:
            return
        cur = self.experiment.total_frames
        if cur < self._min_frames:
            return

        # Condition 1: evaluation return plateau
        if len(self._return_evals) >= self._window:
            recent = list(self._return_evals)
            span = max(recent) - min(recent)
            if span < self._ret_thresh:
                self._fire(
                    cur,
                    f"return plateau (span={span:.3f} < {self._ret_thresh} "
                    f"over {self._window} evals)",
                )

        # Condition 2: critic-loss plateau (backup when evals are sparse)
        if len(self._loss_hist) >= 200:
            recent = list(self._loss_hist)
            hi = max(recent)
            if hi > 0:
                rel = (hi - min(recent)) / hi
                if rel < self._loss_thresh:
                    self._fire(
                        cur,
                        f"critic-loss plateau (rel_var={rel:.4f} < {self._loss_thresh})",
                    )

    def _fire(self, frames: int, reason: str) -> None:
        self.frames_at_advance = frames
        self.advance_reason    = reason
        raise _PhaseComplete(frames, reason)

    @property
    def attack_pct(self) -> float:
        if self._act_total == 0:
            return 0.0
        return sum(self._act_counts[8:11]) / self._act_total


def _grid_from_obs(obs0: np.ndarray):
    """Reconstruct terrain and elevation grids from the first 338 dims of a 418-dim obs.

    Encoding (see srpg_env._build_observation):
        terrain[r,c] stored as terrain_int / 2.0  → 0=NORMAL, 0.5=ROUGH, 1.0=UNCROSSABLE
        elev[r,c]    stored as elev_int    / 2.0  → 0, 0.5, 1.0
    Team A perspective means columns are NOT flipped, so this always matches the
    true board layout.
    """
    n = GRID_SIZE * GRID_SIZE
    terrain_arr = (obs0[:n] * 2.0 + 0.5).astype(np.int8).reshape(GRID_SIZE, GRID_SIZE)
    elev_arr    = (obs0[n: 2 * n] * 2.0 + 0.5).astype(np.int8).reshape(GRID_SIZE, GRID_SIZE)

    class _GridProxy:
        __slots__ = ("terrain", "elevation")
        def __init__(self, t, e):
            self.terrain   = t
            self.elevation = e

    return _GridProxy(terrain_arr, elev_arr)


def _decode_frame(step_td, step_idx: int) -> Optional[_Frame]:
    """Decode one rollout step into a _Frame for rendering.

    Uses agent 0's observation (Team A's perspective).  Since perspective_team
    is TEAM_A, column coordinates are absolute (no flip).
    """
    try:
        # obs shape: [3, 418] (team_a group only); take agent 0's view (no col-flip)
        obs0 = step_td["team_a"]["observation"]["observation"][0].cpu().float().numpy()

        units: list[dict[str, Any]] = []
        for i in range(UNITS_PER_TEAM * 2):
            f = obs0[_TERRAIN_END + i * _UNIT_SIZE: _TERRAIN_END + (i + 1) * _UNIT_SIZE]
            alive   = float(f[10]) > 0.5
            row     = round(float(f[0]) * (GRID_SIZE - 1))
            col     = round(float(f[1]) * (GRID_SIZE - 1))
            # Team is determined by position in the obs vector (indices 0–2 = Team A,
            # 3–5 = Team B), NOT by f[3] (team_is_mine), which is zeroed out for
            # dead units and would misclassify a dead Team A unit as Team B.
            team    = TEAM_A if i < UNITS_PER_TEAM else TEAM_B
            cls_idx = int(np.argmax(f[4:8]))
            hp_frac = float(f[2])
            max_hp  = _CLASS_MAX_HP.get(cls_idx, 10)
            units.append({
                "id":     i,
                "row":    row,
                "col":    col,
                "team":   team,
                "alive":  alive,
                "hp":     round(hp_frac * max_hp),
                "max_hp": max_hp,
                "class":  list(UnitClass)[cls_idx],
            })

        # Active agent: whichever group has mask=True; team_b agents are offset by 3
        mask_a = step_td["team_a"]["mask"]   # [3] bool
        mask_b = step_td["team_b"]["mask"]   # [3] bool
        if mask_a.any():
            active_id = int(mask_a.long().argmax().item())
        else:
            active_id = 3 + int(mask_b.long().argmax().item())

        # Round number from obs scalar at index _TERRAIN_END - 1 (round fraction)
        round_num = round(float(obs0[_TERRAIN_END - 1]) * MAX_TURNS)

        return _Frame(step=step_idx, active_id=active_id,
                      round_num=round_num, units=units)
    except Exception:
        return None


# ── Mini-map renderer ─────────────────────────────────────────────────────────

def _render_mini_map(frame: _Frame, grid) -> Text:
    """13×13 ASCII board with units overlaid (matches QDDisplay style)."""
    unit_at: dict[tuple[int, int], dict] = {
        (u["row"], u["col"]): u for u in frame.units if u["alive"]
    }
    lines = Text()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pos = (row, col)
            if grid is not None:
                t_val = int(grid.terrain[row, col])
                e_val = int(grid.elevation[row, col])
            else:
                t_val, e_val = int(Terrain.NORMAL), 0

            if t_val == int(Terrain.UNCROSSABLE):
                lines.append("██", style="on grey23")
            elif pos in unit_at:
                u = unit_at[pos]
                ltr   = "A" if u["team"] == TEAM_A else "B"
                label = f"{ltr}{u['id'] % 3}"
                base  = _CA if u["team"] == TEAM_A else _CB
                style = (f"bold reverse {base}"
                         if u["id"] == frame.active_id else f"bold {base}")
                lines.append(label, style=style)
            else:
                chars, style = _TERRAIN_CELL.get((t_val, e_val), ("??", "white"))
                lines.append(chars, style=style)
        lines.append("\n")
    return lines


def _hp_bar(hp: int, max_hp: int, width: int = 6, team: int = TEAM_A) -> Text:
    colour = _CA if team == TEAM_A else _CB
    filled = max(0, min(width, round(hp / max_hp * width) if max_hp else 0))
    t = Text()
    t.append("█" * filled, style=colour)
    t.append("░" * (width - filled), style="dim")
    t.append(f" {hp:>2}/{max_hp}", style="dim white")
    return t


# ── Rich TUI Callback ─────────────────────────────────────────────────────────

class RichMAPPOCallback(Callback):
    """BenchMARL Callback that drives a three-panel Rich TUI during MAPPO training.

    Left   — MAPPO METRICS: policy/value/entropy loss curves + action dist.
    Middle — LIVE BATTLE REPLAY: animated ASCII board from evaluation rollouts.
    Right  — EPISODE STATS: return trend, episode lengths from eval.

    The replay panel is updated at every evaluation interval; a background
    ticker advances the replay animation at ~4 fps independent of the training
    iteration rate.
    """

    _HIST_LEN   = 60
    _OUTCOME_W  = 200

    def __init__(
        self,
        total_frames: int,
        phase_label: str = "",
        phase_purpose: str = "",
        phase_behavior: str = "",
        phase_quirks: str = "",
    ) -> None:
        super().__init__()
        self._total  = total_frames
        self._label  = phase_label
        self._purpose  = phase_purpose
        self._behavior = phase_behavior
        self._quirks   = phase_quirks
        self._t0     = time.time()
        self._tick   = 0

        # Training metric histories (filled from on_train_end)
        self._policy_loss: deque = deque(maxlen=self._HIST_LEN)
        self._value_loss:  deque = deque(maxlen=self._HIST_LEN)
        self._return_hist: deque = deque(maxlen=self._HIST_LEN)
        self._last: dict[str, float] = {}

        # Action distribution (from on_batch_collected)
        self._act_counts = [0] * NUM_ACTIONS
        self._act_total  = 0

        # Episode stats (from on_evaluation_end rollouts)
        self._ep_lens:  list[int]   = []
        self._outcomes: list[str]   = []   # "win" / "draw" / "loss"

        # Replay state
        self._frames:      list[_Frame] = []
        self._replay_grid  = None
        self._replay_idx   = 0

        # Rich progress bar
        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bright_cyan"),
            TextColumn("[bold bright_white]Training MAPPO (6c CTDE)[/]"),
            BarColumn(bar_width=None, style="bright_blue",
                      complete_style="bright_cyan", finished_style="bright_green"),
            TaskProgressColumn(style="bright_yellow"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA[/]"),
            TimeRemainingColumn(),
            expand=True,
        )
        self._task_id = self._progress.add_task("train", total=total_frames)
        self._live = Live(self._build(), refresh_per_second=4,
                          screen=False, vertical_overflow="visible")

        self._ticker_running = False
        self._ticker_thread: Optional[threading.Thread] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "RichMAPPOCallback":
        self._live.__enter__()
        self._ticker_running = True
        self._ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._ticker_thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._ticker_running = False
        if self._ticker_thread:
            self._ticker_thread.join(timeout=1.0)
        self._live.__exit__(*args)

    def __getstate__(self) -> dict:
        """Return a pickle-safe snapshot for BenchMARL's experiment-config persistence.

        BenchMARL calls ``pickle.dump(self.callbacks, f)`` in ``_setup_name()``
        to record the experiment configuration.  Rich ``Live``/``Progress``
        objects and ``threading`` primitives cannot be pickled, but the saved
        file is never loaded back into the running training loop, so it is safe
        to replace them with ``None`` in the serialised state.
        """
        state = self.__dict__.copy()
        state["_progress"]       = None
        state["_task_id"]        = None
        state["_live"]           = None
        state["_ticker_thread"]  = None
        state["_ticker_running"] = False
        return state

    def _ticker_loop(self) -> None:
        while self._ticker_running:
            time.sleep(0.25)
            self._tick += 1
            if self._frames:
                self._replay_idx = (self._replay_idx + 1) % len(self._frames)
            try:
                self._live.update(self._build())
            except Exception:
                pass

    # ── BenchMARL hooks ───────────────────────────────────────────────────────

    def on_batch_collected(self, batch) -> None:
        if self.experiment is None:
            return
        self._progress.update(self._task_id,
                               completed=self.experiment.total_frames)
        try:
            actions = batch["team_a"]["action"].flatten().tolist()
            for a in actions:
                idx = int(a)
                if 0 <= idx < NUM_ACTIONS:
                    self._act_counts[idx] += 1
                    self._act_total += 1
        except Exception:
            pass

    def on_train_end(self, training_td, group: str) -> None:
        # MAPPO folds entropy into loss_objective and deletes the entropy key,
        # so we only track the two keys that are actually present.
        try:
            if "loss_objective" in training_td.keys():
                v = float(training_td["loss_objective"].mean())
                self._policy_loss.append(v)
                self._last["policy_loss"] = v
            if "loss_critic" in training_td.keys():
                v = float(training_td["loss_critic"].mean())
                self._value_loss.append(v)
                self._last["value_loss"] = v
        except Exception:
            pass

    def on_evaluation_end(self, rollouts: list) -> None:
        if not rollouts:
            return

        # Decode replay frames from first eval episode
        rollout = rollouts[0]
        frames = []
        for t in range(rollout.shape[0]):
            frame = _decode_frame(rollout[t], t)
            if frame is not None:
                frames.append(frame)
        if frames:
            self._frames    = frames
            self._replay_idx = 0

        # Decode the terrain grid from the first frame's observation.
        # This is more reliable than test_env._env._grid_template, which may have
        # been reset to a randomly-chosen map (including flat_open) between
        # evaluation episodes, or may be inaccessible through the TransformedEnv proxy.
        try:
            obs0 = rollout[0]["team_a"]["observation"]["observation"][0].cpu().float().numpy()
            self._replay_grid = _grid_from_obs(obs0)
        except Exception:
            # Fallback: try to pull grid reference from the test environment
            try:
                self._replay_grid = self.experiment.test_env._env._grid_template
            except Exception:
                pass

        # Episode length from rollout shapes
        try:
            for r in rollouts:
                ep_len = r.shape[0]
                self._ep_lens.append(ep_len)
                if len(self._ep_lens) > self._OUTCOME_W:
                    self._ep_lens.pop(0)
        except Exception:
            pass

        # Episode return — reward lives in rollout["next"]["team_a"]["reward"]
        try:
            rewards = rollouts[0].get(("next", "team_a", "reward"), None)
            if rewards is not None:
                ep_ret = float(rewards.sum())
                self._return_hist.append(ep_ret)
                self._last["return"] = ep_ret
        except Exception:
            pass

        self._live.update(self._build())

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def _sparkline(self, vals, width: int = 22) -> str:
        if not vals:
            return "─" * width
        vs = list(vals)[-width:]
        lo, hi = min(vs), max(vs)
        return "".join(
            _SPARK[0 if hi == lo else int((v - lo) / (hi - lo) * 8)]
            for v in vs
        ).rjust(width, "─")

    def _pct_bar(self, frac: float, width: int = 12) -> str:
        filled = max(0, min(width, round(frac * width)))
        return "█" * filled + "░" * (width - filled)

    # ── Panel builders ────────────────────────────────────────────────────────

    def _metrics_panel(self) -> Panel:
        spinner = _SPIN[self._tick % len(_SPIN)]
        elapsed = time.time() - self._t0
        m = self._last

        t = Text()
        t.append(f"  {spinner}  ", style="bright_cyan")
        t.append(self._label or "MAPPO", style="bold bright_white")
        t.append(f"   {elapsed:.0f}s\n\n", style="dim white")
        if self._purpose:
            t.append(f"  Purpose  ", style="dim white")
            t.append(f"{self._purpose}\n", style="white")
        if self._behavior:
            t.append(f"  Expect   ", style="dim white")
            t.append(f"{self._behavior}\n", style="bright_cyan")
        if self._quirks:
            t.append(f"  Note     ", style="dim white")
            t.append(f"{self._quirks}\n", style="yellow")
        if self._purpose or self._behavior or self._quirks:
            t.append("\n")

        def row(label: str, key: str, fmt: str, colour: str, hist) -> None:
            val = m.get(key)
            t.append(f"  {label:<18}", style="dim white")
            if val is not None:
                t.append(f"{val:{fmt}}", style=colour)
            else:
                t.append("  —  ", style="dim")
            t.append("  " + self._sparkline(hist) + "\n")

        row("mean_return",  "return",      "+.3f", "bright_green", self._return_hist)
        t.append("\n")
        row("policy_loss",  "policy_loss", ".5f",  "yellow",       self._policy_loss)
        row("value_loss",   "value_loss",  ".4f",  "yellow",       self._value_loss)
        t.append("\n")

        if self._act_total > 0:
            ac  = self._act_counts
            tot = self._act_total
            t.append("  ACTION DIST\n", style="bold dim white")
            t.append(f"  {'move':<12}", style="dim white")
            t.append(f"{sum(ac[:8])/tot:5.1%}\n", style="bright_blue")
            t.append(f"  {'attack':<12}", style="dim white")
            t.append(f"{sum(ac[8:11])/tot:5.1%}\n", style="bright_red")
            t.append(f"  {'end_turn':<12}", style="dim white")
            t.append(f"{ac[11]/tot:5.1%}\n", style="bright_yellow")

        return Panel(t, title="[bold]MAPPO METRICS[/]",
                     style=_BORDER, box=box.ROUNDED, padding=(0, 1))

    def _replay_panel(self) -> Panel:
        if not self._frames:
            spinner = _SPIN[self._tick % len(_SPIN)]
            ph = Text(justify="center")
            ph.append(f"\n\n\n  {spinner}  Waiting for first evaluation…\n",
                      style="dim")
            return Panel(ph, title="[bold]LIVE BATTLE REPLAY[/]",
                         style=_BORDER, box=box.ROUNDED)

        frame = self._frames[self._replay_idx]
        total = len(self._frames)

        outer = Table.grid(padding=(0, 1))
        outer.add_column("map",  no_wrap=True)
        outer.add_column("info", no_wrap=False)

        map_text = _render_mini_map(frame, self._replay_grid)

        right = Text()
        right.append(f" round {frame.round_num}  ·  "
                     f"frame {self._replay_idx + 1}/{total}\n", style="dim white")

        # Replay progress bar
        bar_w = 28
        bar_f = round(self._replay_idx / max(1, total - 1) * bar_w)
        right.append(" ▶ ", style="bright_green")
        right.append("━" * bar_f, style="bright_cyan")
        right.append("╸", style="bright_white")
        right.append("─" * max(0, bar_w - bar_f), style="dim")
        right.append("\n\n")

        right.append(" UNITS\n", style="bold dim white")
        _CLS = {UnitClass.FIGHTER: "Fighter", UnitClass.CHARGER: "Charger",
                UnitClass.RANGER:  "Ranger",  UnitClass.SIEGE:   "Siege"}
        team_a = [u for u in frame.units if u["team"] == TEAM_A]
        team_b = [u for u in frame.units if u["team"] == TEAM_B]

        for ua, ub in zip(team_a, team_b):
            a_act = ua["id"] == frame.active_id
            b_act = ub["id"] == frame.active_id
            a_sty = f"bold {_CA}" if a_act else (_CA if ua["alive"] else "dim strike")
            b_sty = f"bold {_CB}" if b_act else (_CB if ub["alive"] else "dim strike")

            right.append("▶ " if a_act else "  ",
                         style="bright_green" if a_act else "")
            right.append(f"A{ua['id']%3}({_CLS.get(ua['class'], '?')[0]}) ",
                         style=a_sty)
            if ua["alive"]:
                right.append_text(_hp_bar(ua["hp"], ua["max_hp"], team=TEAM_A))
            else:
                right.append("  [KO]  ", style="dim red")
            right.append("  ")
            right.append("▶ " if b_act else "  ",
                         style="bright_green" if b_act else "")
            right.append(f"B{ub['id']%3}({_CLS.get(ub['class'], '?')[0]}) ",
                         style=b_sty)
            if ub["alive"]:
                right.append_text(_hp_bar(ub["hp"], ub["max_hp"], team=TEAM_B))
            else:
                right.append("  [KO]  ", style="dim red")
            right.append("\n")

        outer.add_row(map_text, right)
        return Panel(outer, title="[bold]LIVE BATTLE REPLAY[/]",
                     style=_BORDER, box=box.ROUNDED, padding=(0, 1))

    def _episode_panel(self) -> Panel:
        t = Text()

        if self._ep_lens:
            recent = self._ep_lens[-50:]
            lo, hi = int(min(recent)), int(max(recent))
            sd = float(np.std(recent))
            t.append(f"  {'ep_len range':<14}", style="dim white")
            t.append(f"{lo} – {hi}  ", style="bright_cyan")
            t.append(f"σ={sd:.0f}\n\n", style="dim")

        if self._return_hist:
            vals = list(self._return_hist)
            t.append(f"  {'mean return':<14}", style="dim white")
            t.append(f"{vals[-1]:+.3f}\n", style="bright_green")
            t.append("  ")
            t.append(self._sparkline(vals), style="bright_green")
            t.append("\n\n")

        # Evaluation count
        n_evals = len(self._return_hist)
        t.append(f"  evaluations:   {n_evals}\n", style="dim white")

        if self._act_total > 0:
            _ACT_LABELS = ["N","S","E","W","NE","NW","SE","SW","atk0","atk1","atk2","end"]
            t.append("\n  CUMULATIVE ACTIONS\n", style="bold dim white")
            for i, label in enumerate(_ACT_LABELS[:NUM_ACTIONS]):
                pct = self._act_counts[i] / self._act_total
                t.append(f"  {label:<6}", style="dim white")
                style = ("bright_red"    if 8 <= i <= 10
                         else "bright_yellow" if i == 11
                         else "bright_blue")
                t.append(f"{pct:5.1%}\n", style=style)

        return Panel(t, title="[bold]EPISODE STATS[/]",
                     style="cyan", box=box.ROUNDED, padding=(0, 1))

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="panels"),
        )
        layout["progress"].update(
            Panel(self._progress, style=_BORDER, box=box.DOUBLE_EDGE, padding=(0, 1))
        )
        layout["panels"].split_row(
            Layout(name="metrics", ratio=30),
            Layout(name="replay",  ratio=42),
            Layout(name="episode", ratio=28),
        )
        layout["panels"]["metrics"].update(self._metrics_panel())
        layout["panels"]["replay"].update(self._replay_panel())
        layout["panels"]["episode"].update(self._episode_panel())
        return layout


# ── V3 Drip-Feed Curriculum ───────────────────────────────────────────────────
# Five phases that each introduce exactly ONE new training dimension.
# See docs/DRIP-FEED.md for the research rationale.

V3_CURRICULUM_PHASES: List[Dict[str, Any]] = [
    {
        "map_name":          "flat_open",
        "n_frames":          400_000,
        "min_frames":        200_000,
        "trial_shaping":     False,
        "position_shaping":  None,
        "pbrs_weight":       None,
        "entropy_coef":      0.05,
        "n_minibatch_iters": 20,
        "label":    "Phase 1 — Combat Foundation",
        "purpose":  "Learn kill, HP, and attack rewards on open terrain. No approach shaping.",
        "behavior": "Attack % should rise above 20 %. Games should end decisively before the 50-round cap. Wide return variance is healthy — it means wins and losses are both occurring.",
        "quirks":   "Units may cluster ('wolfpack') before attacking. One team may dominate briefly before the shared policy converges. Ignore draw-like returns in the first 50k frames.",
    },
    {
        "map_name":          "flat_open",
        "n_frames":          200_000,
        "min_frames":        100_000,
        "trial_shaping":     False,
        "position_shaping":  "enemy",
        "pbrs_weight":       0.025,
        "entropy_coef":      0.04,
        "n_minibatch_iters": 15,
        "label":    "Phase 2a — Approach Shaping (half-weight PBRS)",
        "purpose":  "Introduce PBRS 'enemy' approach reward at half weight on the same flat map. Isolates PBRS from map complexity.",
        "behavior": "Units should advance more directly toward enemies. Approach paths tighten. Maps are still flat so wall-routing is not required yet.",
        "quirks":   "Brief return dip (~50–100k frames) is expected as the critic re-fits to the new PBRS signal. If attack % drops below Phase 1's final value, PBRS is temporarily overriding the kill signal — this self-corrects.",
    },
    {
        "map_name":          "all_static",
        "n_frames":          300_000,
        "min_frames":        150_000,
        "trial_shaping":     False,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.03,
        "n_minibatch_iters": 12,
        "label":    "Phase 2b — Terrain Transfer (full PBRS)",
        "purpose":  "Generalise to 5 static maps with full-weight PBRS. Approach behaviour is already established before map variety is introduced.",
        "behavior": "Units navigate elevation gradients and choke points. Episode lengths may rise temporarily. Some replays will show clear flanking attempts around hills.",
        "quirks":   "The replay panel may still show flat terrain — flat_open is 1-of-5 static maps. choke_point and asymmetric_heights show '\u2588\u2588' wall blocks; these are not rendering bugs.",
    },
    {
        "map_name":          "all_static",
        "n_frames":          300_000,
        "min_frames":        150_000,
        "trial_shaping":     True,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.03,
        "n_minibatch_iters": 10,
        "label":    "Phase 2c — Time Pressure (trial shaping)",
        "purpose":  "Add step cost (−0.001/action) and proximity reward on static maps. Discourages stalling and drawn-out games.",
        "behavior": "Episode lengths shorten. Units engage sooner. Draw rate drops. Return values are mechanically ~0.4 lower than Phase 2b due to the step cost — this is expected.",
        "quirks":   "Aggressive rushing may appear for 100–150k frames before the policy balances speed with positioning. Do not mistake early over-aggression for a training failure.",
    },
    {
        "map_name":          "all",
        "n_frames":          1_500_000,
        "min_frames":        750_000,
        "trial_shaping":     True,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.01,
        "n_minibatch_iters": 10,
        "label":    "Phase 3 — Full Complexity (all maps)",
        "purpose":  "Fine-tune on all maps including obstacle maps (two_rooms, fortress, twin_gates). Extended budget allows Team A/B parity to recover after Phase 2c shock.",
        "behavior": "Obstacle maps with '\u2588\u2588' blocks appear in replays. Units route around walls, push through doorways, and contest chokepoints. Expect Team A asymmetry for the first 200–300k frames before parity returns.",
        "quirks":   "Fortress and twin_gates create bottlenecks that can produce 'camping' near doorways. Units may exploit unexpected wall shortcuts that look odd but are mechanically valid. Low entropy means less visual variety — the policy is exploiting rather than exploring.",
    },
]


# ── V4 Curriculum ─────────────────────────────────────────────────────────────
# Seven phases adding two new stages to V3:
#   Phase 2d — dedicated obstacle-map warm-up before the full mixed phase.
#   Phase 4  — role differentiation: Vanguard/Flanker/Support bonuses activated.
# Total budget: ~3.7 M max frames (vs 2.9 M for V3).

V4_CURRICULUM_PHASES: List[Dict[str, Any]] = [
    {
        "map_name":          "flat_open",
        "n_frames":          400_000,
        "min_frames":        200_000,
        "trial_shaping":     False,
        "position_shaping":  None,
        "pbrs_weight":       None,
        "entropy_coef":      0.05,
        "n_minibatch_iters": 20,
        "combat_shaping":    False,
        "label":    "Phase 1 — Combat Foundation",
        "purpose":  "Learn kill, HP, and attack rewards on open terrain. No approach shaping.",
        "behavior": "Attack % should rise above 20 %. Games end decisively before the 50-round cap.",
        "quirks":   "Units may cluster before attacking. Wide return variance is healthy.",
    },
    {
        "map_name":          "flat_open",
        "n_frames":          200_000,
        "min_frames":        100_000,
        "trial_shaping":     False,
        "position_shaping":  "enemy",
        "pbrs_weight":       0.025,
        "entropy_coef":      0.04,
        "n_minibatch_iters": 15,
        "combat_shaping":    False,
        "label":    "Phase 2a — Approach Shaping (half-weight PBRS)",
        "purpose":  "Introduce PBRS 'enemy' approach reward at half weight on the same flat map.",
        "behavior": "Units advance more directly toward enemies. Approach paths tighten.",
        "quirks":   "Brief return dip (~50–100k frames) as the critic re-fits to the PBRS signal.",
    },
    {
        "map_name":          "all_static",
        "n_frames":          300_000,
        "min_frames":        150_000,
        "trial_shaping":     False,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.03,
        "n_minibatch_iters": 12,
        "combat_shaping":    False,
        "label":    "Phase 2b — Terrain Transfer (full PBRS)",
        "purpose":  "Generalise to 5 static maps with full-weight PBRS.",
        "behavior": "Units navigate elevation gradients and choke points.",
        "quirks":   "Episode lengths may rise temporarily while the policy adapts to static map variety.",
    },
    {
        "map_name":          "all_static",
        "n_frames":          300_000,
        "min_frames":        150_000,
        "trial_shaping":     True,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.03,
        "n_minibatch_iters": 10,
        "combat_shaping":    False,
        "label":    "Phase 2c — Time Pressure (trial shaping)",
        "purpose":  "Add step cost and proximity reward on static maps. Discourages stalling.",
        "behavior": "Episode lengths shorten. Draw rate drops. Returns are ~0.4 lower due to step cost.",
        "quirks":   "Aggressive rushing for 100–150k frames before the policy balances speed with positioning.",
    },
    {
        "map_name":          "all_obstacle",
        "n_frames":          300_000,
        "min_frames":        150_000,
        "trial_shaping":     True,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.03,
        "n_minibatch_iters": 10,
        "combat_shaping":    False,
        "label":    "Phase 2d — Obstacle Navigation",
        "purpose":  "Specialise on two_rooms / fortress / twin_gates before mixing with static maps. "
                    "In V3 the jump from static to all-maps left obstacle routing undertrained.",
        "behavior": "Units learn to route through doorways and contest chokepoints. "
                    "Expect longer episodes initially as the policy discovers wall-routing paths.",
        "quirks":   "two_rooms' single doorway creates tight congestion; units may initially camp "
                    "on each side. This resolves as attack bonuses pull them through the gap.",
    },
    {
        "map_name":          "all",
        "n_frames":          2_000_000,
        "min_frames":        1_000_000,
        "trial_shaping":     True,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.02,
        "n_minibatch_iters": 10,
        "combat_shaping":    False,
        "label":    "Phase 3 — Full Complexity (all maps)",
        "purpose":  "Fine-tune on all maps including obstacle maps. Extended budget and higher entropy "
                    "(0.02 vs 0.01 in V3) gives the CNN front-end room to refine spatial features.",
        "behavior": "Obstacle maps appear in replays. Units route around walls, push through doorways, "
                    "and contest chokepoints. Expect Team A asymmetry for the first 200–300k frames.",
        "quirks":   "Low entropy means less visual variety; the policy is exploiting spatial patterns "
                    "learned by the CNN rather than exploring. This is expected and desirable.",
    },
    {
        "map_name":          "all",
        "n_frames":          500_000,
        "min_frames":        250_000,
        "trial_shaping":     True,
        "position_shaping":  "enemy",
        "pbrs_weight":       None,
        "entropy_coef":      0.005,
        "n_minibatch_iters": 8,
        "combat_shaping":    True,
        "label":    "Phase 4 — Role Differentiation",
        "purpose":  "Activate Vanguard / Flanker / Support role bonuses to produce behavioural "
                    "specialisation without retraining from scratch. Very low entropy locks in "
                    "the patterns from Phase 3 while the role signal reshapes fine-grained behaviour.",
        "behavior": "Slot-0 units (Fighter/Charger) should engage at close range more frequently. "
                    "Slot-1 units should show more diagonal repositioning. "
                    "Slot-2 units (Ranger/Siege) should maintain distance and fire from max range.",
        "quirks":   "Role bonuses are small (0.01–0.03) relative to kill rewards (1.5), so changes "
                    "are subtle. Monitor the action-dist panel for diagonal % rising on team_a_unit1.",
    },
]


# ── Training function ─────────────────────────────────────────────────────────

def train_mappo(
    phase_map_names: list[str] | None = None,
    phase_frames: list[int] | None = None,
    save_path: str = "models/mappo_srpg",
    use_rich: bool = True,
    trial_shaping: bool = False,
    win_reward_scale: float = 1.0,
    position_shaping: Optional[str] = None,
    combat_shaping: bool = False,
    seed: int = 42,
    entropy_coef_by_phase: Optional[List[float]] = None,
    minibatch_iters_by_phase: Optional[List[int]] = None,
    trial_shaping_by_phase: Optional[List[bool]] = None,
    position_shaping_by_phase: Optional[List[Optional[str]]] = None,
    pbrs_weight_by_phase: Optional[List[Optional[float]]] = None,
    advance_min_frames_by_phase: Optional[List[int]] = None,
    phase_configs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Run a multi-phase curriculum with BenchMARL MAPPO.

    Preferred call (V3 drip-feed curriculum):
        train_mappo(phase_configs=V3_CURRICULUM_PHASES, save_path="models/v3")

    Legacy parametric call (V2 compatibility):
        train_mappo(phase_map_names=[...], phase_frames=[...], ...)

    Each phase exits early if AutoAdvanceCallback detects a return or
    critic-loss plateau.  Savings are logged to stdout.

    Args:
        phase_configs:    List of phase config dicts (each with map_name, n_frames,
                          min_frames, trial_shaping, position_shaping, pbrs_weight,
                          entropy_coef, n_minibatch_iters, label, purpose, behavior,
                          quirks).  When provided, all *_by_phase args are ignored.
        phase_map_names:  Legacy: map name per phase.
        phase_frames:     Legacy: frame budget per phase.
        save_path:        Base path for model outputs.
        use_rich:         Show the three-panel Rich TUI.
        trial_shaping:    Legacy: enable trial shaping (all phases).
        win_reward_scale: Terminal win/loss reward multiplier.
        position_shaping: Legacy: PBRS potential type.
        combat_shaping:   Enable per-role combat incentive hooks.
        seed:             Random seed.
        entropy_coef_by_phase:       Legacy per-phase entropy coefficients.
        minibatch_iters_by_phase:    Legacy per-phase minibatch iteration counts.
        trial_shaping_by_phase:      Legacy per-phase trial shaping flags.
        position_shaping_by_phase:   Legacy per-phase PBRS types.
        pbrs_weight_by_phase:        Legacy per-phase PBRS weight overrides.
        advance_min_frames_by_phase: Legacy per-phase minimum frames before auto-advance.
    """
    # ── Resolve phase list ────────────────────────────────────────────────────
    if phase_configs is not None:
        resolved = phase_configs
    else:
        assert phase_map_names and phase_frames and len(phase_map_names) == len(phase_frames), \
            "phase_map_names and phase_frames must be provided and equal length when phase_configs is None"
        n = len(phase_map_names)
        resolved = [
            {
                "map_name":          phase_map_names[i],
                "n_frames":          phase_frames[i],
                "min_frames":        (advance_min_frames_by_phase[i]
                                      if advance_min_frames_by_phase else phase_frames[i] // 2),
                "trial_shaping":     (trial_shaping_by_phase[i]
                                      if trial_shaping_by_phase is not None else trial_shaping),
                "position_shaping":  (position_shaping_by_phase[i]
                                      if position_shaping_by_phase is not None else position_shaping),
                "pbrs_weight":       (pbrs_weight_by_phase[i]
                                      if pbrs_weight_by_phase is not None else None),
                "entropy_coef":      (entropy_coef_by_phase[i]
                                      if entropy_coef_by_phase is not None else 0.01),
                "n_minibatch_iters": (minibatch_iters_by_phase[i]
                                      if minibatch_iters_by_phase is not None else 10),
                "label":    f"Phase {i + 1}/{n} — {phase_map_names[i]}",
                "purpose":  "",
                "behavior": "",
                "quirks":   "",
            }
            for i in range(n)
        ]

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    prev_policy_state: Optional[dict] = None
    total_frames_used = 0

    for phase_idx, phase_cfg in enumerate(resolved):
        map_name             = phase_cfg["map_name"]
        n_frames             = phase_cfg["n_frames"]
        min_frames           = phase_cfg.get("min_frames", n_frames // 2)
        phase_trial_shaping  = phase_cfg.get("trial_shaping", False)
        phase_pos_shaping    = phase_cfg.get("position_shaping", None)
        phase_pbrs_weight    = phase_cfg.get("pbrs_weight", None)
        phase_entropy        = phase_cfg.get("entropy_coef", 0.01)
        phase_mb_iters       = phase_cfg.get("n_minibatch_iters", 10)
        phase_combat_shaping = phase_cfg.get("combat_shaping", combat_shaping)
        phase_label          = phase_cfg.get("label", f"Phase {phase_idx + 1}/{len(resolved)}")
        phase_purpose        = phase_cfg.get("purpose", "")
        phase_behavior       = phase_cfg.get("behavior", "")
        phase_quirks         = phase_cfg.get("quirks", "")

        phase_num  = phase_idx + 1
        phase_save = str(save_dir / f"mappo_phase{phase_num}")

        # ── Task ──────────────────────────────────────────────────────────────
        task = HighGroundTaskClass(
            name=f"HIGHGROUND_P{phase_num}",
            config={
                "map_name":         map_name,
                "trial_shaping":    phase_trial_shaping,
                "win_reward_scale": win_reward_scale,
                "position_shaping": phase_pos_shaping,
                "combat_shaping":   phase_combat_shaping,
                "pbrs_weight":      phase_pbrs_weight,
            },
        )

        # ── Algorithm config ──────────────────────────────────────────────────
        algorithm_config = MappoConfig.get_from_yaml()
        algorithm_config.entropy_coef       = phase_entropy
        algorithm_config.clip_epsilon       = 0.2
        algorithm_config.lmbda              = 0.95
        algorithm_config.critic_coef        = 1.0
        algorithm_config.share_param_critic = True

        # ── Model configs ─────────────────────────────────────────────────────
        # Actor: CNN spatial encoder (2×13×13 → 512) + MLP trunk (256×256).
        # The CNN gives the policy translational equivariance over terrain and
        # elevation, replacing the first 338 floats-as-scalars MLP approach.
        model_config = SpatialCnnMlpConfig(
            cnn_channels=(16, 32, 32),
            mlp_cells=(256, 256),
            activation_class=torch.nn.ReLU,
        )

        # Critic: wider plain MLP over the 836-dim joint state (both teams).
        # No CNN here — the global state is not a single spatial perspective,
        # it is two agents' obs concatenated, so CNN spatial priors do not apply.
        critic_model_config = MlpConfig.get_from_yaml()
        critic_model_config.num_cells = [512, 512]

        # ── Experiment config ─────────────────────────────────────────────────
        exp_config = ExperimentConfig.get_from_yaml()
        _train_device = "cuda" if torch.cuda.is_available() else "cpu"
        exp_config.sampling_device = "cpu"       # env is Python/NumPy — must be CPU
        exp_config.train_device    = _train_device
        exp_config.buffer_device   = "cpu"       # on-policy buffer is small; avoids GPU pressure

        exp_config.share_policy_params       = True
        exp_config.prefer_continuous_actions = False

        exp_config.gamma          = 0.99
        exp_config.lr             = 3e-4
        exp_config.adam_eps       = 1e-5
        exp_config.clip_grad_norm = True
        exp_config.clip_grad_val  = 0.5

        exp_config.max_n_frames = n_frames

        exp_config.on_policy_collected_frames_per_batch = 2048 * 6
        exp_config.on_policy_n_envs_per_worker          = 1
        exp_config.on_policy_n_minibatch_iters          = phase_mb_iters
        exp_config.on_policy_minibatch_size             = 64

        exp_config.evaluation                       = True
        exp_config.evaluation_interval              = 2048 * 6 * 4
        exp_config.evaluation_episodes              = 10
        exp_config.evaluation_deterministic_actions = False
        exp_config.render                           = False

        exp_config.loggers     = []
        exp_config.create_json = True

        exp_config.save_folder = str(save_dir / f"benchmarl_phase{phase_num}")
        os.makedirs(exp_config.save_folder, exist_ok=True)
        exp_config.checkpoint_at_end    = False
        exp_config.checkpoint_interval  = 0

        # ── Callbacks ─────────────────────────────────────────────────────────
        auto_cb  = AutoAdvanceCallback(min_frames=min_frames)
        callbacks: list[Callback] = []
        rich_cb: Optional[RichMAPPOCallback] = None
        if use_rich:
            rich_cb = RichMAPPOCallback(
                total_frames=n_frames,
                phase_label=phase_label,
                phase_purpose=phase_purpose,
                phase_behavior=phase_behavior,
                phase_quirks=phase_quirks,
            )
            callbacks.append(rich_cb)
        callbacks.append(auto_cb)   # always last so display updates before advance check

        # ── Build experiment ──────────────────────────────────────────────────
        exp = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=exp_config,
            callbacks=callbacks,
        )

        # Warm-start from previous phase (curriculum transfer)
        if prev_policy_state is not None:
            try:
                exp.policy.load_state_dict(prev_policy_state)
                if not use_rich:
                    print(f"  Phase {phase_num}: loaded weights from phase {phase_idx}")
            except Exception as e:
                print(f"[warn] Phase {phase_num}: could not load prior policy: {e}")

        # ── Run ───────────────────────────────────────────────────────────────
        frames_used = n_frames
        try:
            if use_rich and rich_cb is not None:
                with rich_cb:
                    exp.run()
            else:
                exp.run()
        except _PhaseComplete as e:
            frames_used = e.frames
            savings     = n_frames - frames_used
            msg = (
                f"Phase {phase_num} advanced early at {frames_used:,} / {n_frames:,} frames "
                f"(+{savings:,} saved) — {e.reason}"
            )
            if not use_rich:
                print(f"  [auto-advance] {msg}")
            else:
                from rich.console import Console as _Con
                _Con().print(f"\n  [bold bright_green]▶ Auto-advance:[/] {msg}\n")

        total_frames_used += frames_used

        # Save phase checkpoint
        torch.save(exp.policy.state_dict(), phase_save + "_policy.pt")
        prev_policy_state = exp.policy.state_dict()
        if not use_rich:
            print(f"Phase {phase_num} complete → {phase_save}_policy.pt")

    # Final save
    if prev_policy_state is not None:
        torch.save(prev_policy_state, save_path + "_policy.pt")
        if not use_rich:
            print(f"Final policy → {save_path}_policy.pt  "
                  f"(total frames used: {total_frames_used:,})")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train MAPPO (BenchMARL) for High Ground SRPG")
    p.add_argument("--frames", type=int, default=500_000,
                   help="Frames per phase (default 500k; use with --curriculum for 3-phase)")
    p.add_argument("--map", type=str, default="flat_open",
                   help="Map for single-phase training")
    p.add_argument("--curriculum", action="store_true",
                   help="Run full V4 curriculum (7 phases, ~3.7 M max frames)")
    p.add_argument("--v3", action="store_true",
                   help="Use legacy V3 curriculum (5 phases) instead of V4")
    p.add_argument("--curriculum-frames", type=int, nargs=3,
                   default=[400_000, 200_000, 800_000],
                   metavar=("P1", "P2", "P3"),
                   help="Legacy: per-phase frame budgets (ignored with --curriculum)")
    p.add_argument("--save-path", type=str, default="models/mappo_srpg")
    p.add_argument("--no-tui", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.curriculum:
        phases = V3_CURRICULUM_PHASES if args.v3 else V4_CURRICULUM_PHASES
        train_mappo(
            phase_configs=phases,
            save_path=args.save_path,
            use_rich=not args.no_tui,
            seed=args.seed,
        )
    else:
        train_mappo(
            phase_map_names=[args.map],
            phase_frames=[args.frames],
            save_path=args.save_path,
            use_rich=not args.no_tui,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
