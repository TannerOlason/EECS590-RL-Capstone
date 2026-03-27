"""MaskablePPO training script for the High Ground SRPG.

Usage:
    python -m highground.training.train [--timesteps N] [--map MAP_NAME] [--self-play]

Supports:
    - Single policy vs random opponent (default)
    - Self-play mode (shared policy controls both teams)
    - Training on specific or all static maps
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from highground.engine.units import TEAM_A, TEAM_B, UnitClass
from highground.env.sb3_wrapper import SB3SRPGWrapper
from highground.env.srpg_env import HighGroundEnv
from highground.maps.static_maps import ALL_MAPS, STATIC_MAPS, OBSTACLE_MAPS


class MetricsCallback(BaseCallback):
    """Saves training metrics to a JSON sidecar file for post-hoc plotting.

    Always attaches to training; runs alongside RichTrainingCallback or SB3
    default output.  Writes to *save_path* at training end.
    """

    _TRAIN_KEYS = (
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
    )

    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._save_path = save_path
        self._data: dict[str, list] = {
            "timesteps":             [],
            "ep_rew_mean":           [],
            "ep_len_mean":           [],
            "policy_gradient_loss":  [],
            "value_loss":            [],
            "entropy_loss":          [],
            "approx_kl":             [],
        }

    def _on_rollout_end(self) -> None:
        self._data["timesteps"].append(self.num_timesteps)

        ep_buf = getattr(self.model, "ep_info_buffer", None)
        if ep_buf and len(ep_buf) > 0:
            self._data["ep_rew_mean"].append(float(np.mean([ep["r"] for ep in ep_buf])))
            self._data["ep_len_mean"].append(float(np.mean([ep["l"] for ep in ep_buf])))
        else:
            self._data["ep_rew_mean"].append(None)
            self._data["ep_len_mean"].append(None)

        nvp = getattr(self.model.logger, "name_to_value", {})
        for k in self._TRAIN_KEYS:
            short = k.split("/")[1]
            self._data[short].append(float(nvp[k]) if k in nvp else None)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        with open(self._save_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)


class RichTrainingCallback(BaseCallback):
    """SB3 callback that drives a Rich live display during MaskablePPO training.

    Two-panel layout:
      Left  — TRAINING METRICS: loss curves with sparklines for all 6 metrics.
      Right — EPISODE STATS: rolling win/draw/loss rates, ep-length range,
              survivors at game end, and action-type breakdown.
    """

    _SPARK = " ▁▂▃▄▅▆▇█"
    _SPIN  = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _WIN_WINDOW = 100   # rolling window size for outcome / survivor stats
    _LEN_WINDOW = 100   # rolling window size for per-episode lengths

    def __init__(self, total_timesteps: int, map_name: str = "", verbose: int = 0) -> None:
        super().__init__(verbose)
        self._total    = total_timesteps
        self._map_name = map_name
        self._t0       = time.time()
        self._rollout  = 0
        self._tick     = 0

        # Per-rollout sparkline histories
        self._rew_hist:  list[float] = []
        self._len_hist:  list[float] = []
        self._pg_hist:   list[float] = []
        self._vl_hist:   list[float] = []
        self._ent_hist:  list[float] = []
        self._kl_hist:   list[float] = []
        self._last_metrics: dict[str, float] = {}

        # Per-episode outcome rolling buffers (populated in _on_step via info dict)
        self._outcome_buf:  list[str]   = []   # "win" / "draw" / "loss"
        self._survivor_buf: list[float] = []   # friendly units alive at episode end
        self._ep_lens_raw:  list[float] = []   # individual episode step counts

        # Cumulative action distribution
        self._act_counts = [0] * 8
        self._act_total  = 0

        # Rich objects
        from rich.live import Live
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                   SpinnerColumn, TaskProgressColumn,
                                   TextColumn, TimeElapsedColumn, TimeRemainingColumn)
        from rich.panel import Panel

        self._Panel = Panel
        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bright_cyan"),
            TextColumn("[bold bright_white]Training MaskablePPO[/]"),
            BarColumn(bar_width=None, style="bright_blue",
                      complete_style="bright_cyan", finished_style="bright_green"),
            TaskProgressColumn(style="bright_yellow"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA[/]"),
            TimeRemainingColumn(),
            expand=True,
        )
        self._task = self._progress.add_task("train", total=total_timesteps)
        self._live = Live(self._build(), refresh_per_second=4,
                          screen=False, vertical_overflow="visible")

    # ── Rendering helpers ─────────────────────────────────────────────

    def _sparkline(self, vals: list[float], width: int = 20) -> str:
        if not vals:
            return "─" * width
        vs = vals[-width:]
        lo, hi = min(vs), max(vs)
        return "".join(
            self._SPARK[0 if hi == lo else int((v - lo) / (hi - lo) * 8)]
            for v in vs
        ).rjust(width, "─")

    def _pct_bar(self, frac: float, width: int = 12) -> str:
        filled = max(0, min(width, round(frac * width)))
        return "█" * filled + "░" * (width - filled)

    def _build_metrics_text(self):
        from rich.text import Text
        spinner  = self._SPIN[self._tick % len(self._SPIN)]
        elapsed  = time.time() - self._t0
        m        = self._last_metrics

        t = Text()
        t.append(f"  {spinner}  ", style="bright_cyan")
        t.append(self._map_name or "flat_open", style="bold bright_white")
        t.append(f"   rollout {self._rollout}   elapsed {elapsed:.0f}s\n\n",
                 style="dim white")

        def row(label: str, key: str, fmt: str, color: str, hist: list[float]) -> None:
            val = m.get(key)
            t.append(f"  {label:<16}", style="dim white")
            if val is not None:
                t.append(f"{val:{fmt}}", style=color)
            else:
                t.append("  —  ", style="dim")
            t.append("  " + self._sparkline(hist) + "\n")

        row("ep_rew_mean",  "rollout/ep_rew_mean",         "+.3f", "bright_green",  self._rew_hist)
        row("ep_len_mean",  "rollout/ep_len_mean",          ".1f",  "bright_cyan",   self._len_hist)
        t.append("\n")
        row("policy_loss",  "train/policy_gradient_loss",  ".5f",  "yellow",         self._pg_hist)
        row("value_loss",   "train/value_loss",             ".4f",  "yellow",         self._vl_hist)
        row("entropy_loss", "train/entropy_loss",           ".4f",  "dim yellow",     self._ent_hist)
        row("approx_kl",    "train/approx_kl",              ".5f",  "dim white",      self._kl_hist)
        return t

    def _build_episode_text(self):
        from rich.text import Text
        t = Text()

        # ── Outcome rates ─────────────────────────────────────────────
        n = len(self._outcome_buf)
        if n > 0:
            wins   = self._outcome_buf.count("win")   / n
            draws  = self._outcome_buf.count("draw")  / n
            losses = self._outcome_buf.count("loss")  / n
            t.append(f"  {'wins':<10}", style="dim white")
            t.append(f"{wins:5.1%}  {self._pct_bar(wins)}\n",   style="bright_green")
            t.append(f"  {'draws':<10}", style="dim white")
            t.append(f"{draws:5.1%}  {self._pct_bar(draws)}\n", style="bright_yellow")
            t.append(f"  {'losses':<10}", style="dim white")
            t.append(f"{losses:5.1%}  {self._pct_bar(losses)}\n", style="bright_red")
            t.append(f"  last {n} eps\n\n", style="dim")
        else:
            t.append("  no episodes yet\n\n", style="dim")

        # ── Episode length range ──────────────────────────────────────
        if self._ep_lens_raw:
            recent = self._ep_lens_raw[-50:]
            lo, hi = int(min(recent)), int(max(recent))
            sd = float(np.std(recent))
            t.append(f"  {'ep_len range':<10}", style="dim white")
            t.append(f"{lo} – {hi}  ", style="bright_cyan")
            t.append(f"σ={sd:.0f}\n", style="dim")

        # ── Survivors ────────────────────────────────────────────────
        if self._survivor_buf:
            avg_sv = float(np.mean(self._survivor_buf[-50:]))
            t.append(f"  {'survivors':<10}", style="dim white")
            t.append(f"{avg_sv:.1f} / 3\n", style="bright_magenta")

        t.append("\n")

        # ── Action breakdown ──────────────────────────────────────────
        if self._act_total > 0:
            ac   = self._act_counts
            tot  = self._act_total
            move    = sum(ac[:6]) / tot
            attack  = ac[6]       / tot
            end_t   = ac[7]       / tot
            t.append(f"  {'move':<10}", style="dim white")
            t.append(f"{move:5.1%}\n",    style="bright_blue")
            t.append(f"  {'attack':<10}", style="dim white")
            t.append(f"{attack:5.1%}\n",  style="bright_red")
            t.append(f"  {'end_turn':<10}", style="dim white")
            t.append(f"{end_t:5.1%}\n",  style="bright_yellow")
        return t

    def _build(self):
        from rich.console import Group
        from rich.columns import Columns
        from rich import box
        return Group(
            self._Panel(self._progress, style="bright_blue",
                        box=box.DOUBLE_EDGE, padding=(0, 1)),
            Columns([
                self._Panel(self._build_metrics_text(),
                            title="[bold]TRAINING METRICS[/]",
                            style="bright_blue", box=box.ROUNDED, padding=(0, 1)),
                self._Panel(self._build_episode_text(),
                            title="[bold]EPISODE STATS[/]",
                            style="cyan", box=box.ROUNDED, padding=(0, 1)),
            ], equal=True),
        )

    # ── SB3 hooks ─────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        self._live.__enter__()

    def _on_step(self) -> bool:
        # Accumulate action distribution from each policy step
        for act in np.atleast_1d(self.locals.get("actions", [])):
            self._act_counts[int(act)] += 1
            self._act_total += 1

        # Capture per-episode outcomes, lengths, and survivors from info dicts
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info:
                self._ep_lens_raw.append(ep_info["l"])
                if len(self._ep_lens_raw) > self._LEN_WINDOW:
                    self._ep_lens_raw.pop(0)

            outcome = info.get("outcome")
            if outcome:
                self._outcome_buf.append(outcome)
                if len(self._outcome_buf) > self._WIN_WINDOW:
                    self._outcome_buf.pop(0)
                sv = info.get("survivors")
                if sv is not None:
                    self._survivor_buf.append(float(sv))
                    if len(self._survivor_buf) > self._WIN_WINDOW:
                        self._survivor_buf.pop(0)
        return True

    def _on_rollout_end(self) -> None:
        self._rollout += 1
        self._tick += 1
        self._progress.update(self._task, completed=self.num_timesteps)

        ep_buf = getattr(self.model, "ep_info_buffer", None)
        if ep_buf and len(ep_buf) > 0:
            rew    = float(np.mean([ep["r"] for ep in ep_buf]))
            length = float(np.mean([ep["l"] for ep in ep_buf]))
            self._rew_hist.append(rew)
            self._len_hist.append(length)
            self._last_metrics["rollout/ep_rew_mean"] = rew
            self._last_metrics["rollout/ep_len_mean"] = length

        nvp = getattr(self.model.logger, "name_to_value", {})
        for k, hist in (
            ("train/policy_gradient_loss", self._pg_hist),
            ("train/value_loss",           self._vl_hist),
            ("train/entropy_loss",         self._ent_hist),
            ("train/approx_kl",            self._kl_hist),
        ):
            if k in nvp:
                v = float(nvp[k])
                hist.append(v)
                self._last_metrics[k] = v

        self._live.update(self._build())

    def _on_training_end(self) -> None:
        self._progress.update(self._task, completed=self._total)
        self._live.update(self._build())
        self._live.__exit__(None, None, None)


class RoundRobinMapEnv(gymnasium.Wrapper):
    """Cycles through a list of maps on each reset (round-robin).

    The first episode uses maps[0].  Each subsequent reset() advances to the
    next map in order, wrapping back to maps[0] after the last one.
    """

    def __init__(
        self,
        maps: list[tuple],
        controlled_team: int,
        self_play: bool,
        trial_shaping: bool,
        win_reward_scale: float = 1.0,
        position_shaping: str | None = None,
        combat_shaping: bool = False,
    ) -> None:
        grid, sa, sb = maps[0]
        aec = HighGroundEnv(grid, sa, sb, reward_mode="shaped",
                            trial_shaping=trial_shaping,
                            win_reward_scale=win_reward_scale,
                            position_shaping=position_shaping,
                            combat_shaping=combat_shaping)
        inner = SB3SRPGWrapper(aec, controlled_team=controlled_team, self_play=self_play)
        super().__init__(inner)
        # Cycle starts at maps[1] so the next reset() moves to the second map
        self._cycle = itertools.cycle(maps)
        next(self._cycle)  # skip maps[0] — already loaded in __init__

    def reset(self, **kwargs):
        grid, sa, sb = next(self._cycle)
        aec = self.env.aec
        aec._grid_template = grid
        aec._spawns_a = list(sa)
        aec._spawns_b = list(sb)
        return self.env.reset(**kwargs)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def make_env(
    map_name: str = "flat_open",
    controlled_team: int = TEAM_A,
    self_play: bool = False,
    trial_shaping: bool = False,
    win_reward_scale: float = 1.0,
    position_shaping: str | None = None,
    combat_shaping: bool = False,
) -> gymnasium.Env:
    """Create a training environment.

    If map_name is "all", returns a RoundRobinMapEnv that cycles through every
    static map in order on each episode reset.
    """
    env_kwargs = dict(
        reward_mode="shaped",
        trial_shaping=trial_shaping,
        win_reward_scale=win_reward_scale,
        position_shaping=position_shaping,
        combat_shaping=combat_shaping,
    )
    _MAP_GROUPS = {
        "all":          ALL_MAPS,
        "all_static":   STATIC_MAPS,
        "all_obstacle": OBSTACLE_MAPS,
    }
    if map_name in _MAP_GROUPS:
        maps = [fn() for fn in _MAP_GROUPS[map_name].values()]
        return RoundRobinMapEnv(maps, controlled_team, self_play, trial_shaping,
                                win_reward_scale=win_reward_scale,
                                position_shaping=position_shaping,
                                combat_shaping=combat_shaping)
    map_fn = ALL_MAPS[map_name]
    grid, spawns_a, spawns_b = map_fn()
    aec = HighGroundEnv(grid, spawns_a, spawns_b, **env_kwargs)
    return SB3SRPGWrapper(aec, controlled_team=controlled_team, self_play=self_play)


def mask_fn(env: SB3SRPGWrapper) -> np.ndarray:
    """Action mask function for sb3-contrib ActionMasker wrapper."""
    return env.action_masks()


def train(
    timesteps: int = 100_000,
    map_name: str = "flat_open",
    self_play: bool = False,
    save_path: str = "models/maskable_ppo_srpg",
    use_rich: bool = False,
    trial_shaping: bool = False,
    win_reward_scale: float = 1.0,
    position_shaping: str | None = None,
    combat_shaping: bool = False,
) -> MaskablePPO:
    """Train a MaskablePPO agent.

    Args:
        use_rich: If True, suppress SB3 stdout and drive a Rich TUI instead.
        trial_shaping: If True, enable experimental reward shaping (draw penalty,
            per-step survival cost, territorial proximity reward).
        win_reward_scale: Multiplier on the terminal win/loss reward (default 1.0).
            Higher values (e.g. 5.0) make the win signal dominate over shaping.
        position_shaping: PBRS potential type — None (off), "center", or "enemy".
        combat_shaping: If True, enable per-role combat incentives (6b groundwork).
            Currently a no-op until scale constants are tuned.
    """
    env = make_env(map_name=map_name, self_play=self_play, trial_shaping=trial_shaping,
                   win_reward_scale=win_reward_scale, position_shaping=position_shaping,
                   combat_shaping=combat_shaping)
    env = ActionMasker(env, mask_fn)

    # MetricsCallback always runs so model_analysis.py can plot training curves
    callbacks: list[BaseCallback] = [MetricsCallback(save_path + "_metrics.json")]
    verbose = 1
    if use_rich:
        callbacks.append(RichTrainingCallback(total_timesteps=timesteps, map_name=map_name))
        verbose = 0   # silence SB3's own table so it doesn't fight the TUI

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=timesteps, callback=callbacks)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    if not use_rich:
        print(f"Model saved to {save_path}")
    return model


def train_curriculum(
    phase_map_names: list[str],
    phase_timesteps: list[int],
    save_path: str,
    self_play: bool = False,
    use_rich: bool = False,
    trial_shaping: bool = False,
    win_reward_scale: float = 1.0,
    position_shaping: str | None = None,
    combat_shaping: bool = False,
) -> MaskablePPO:
    """Multi-phase curriculum: train → save → load → new map → train → repeat.

    Each phase loads the previous phase's weights and continues training on a
    new (broader) map set.  Phase checkpoints are saved as
    ``{save_path}_phase{n}.zip`` in addition to the final ``{save_path}.zip``.
    """
    assert len(phase_map_names) == len(phase_timesteps), \
        "phase_map_names and phase_timesteps must have the same length"

    model: MaskablePPO | None = None

    for phase_idx, (phase_map, phase_ts) in enumerate(zip(phase_map_names, phase_timesteps)):
        env = make_env(
            map_name=phase_map,
            self_play=self_play,
            trial_shaping=trial_shaping,
            win_reward_scale=win_reward_scale,
            position_shaping=position_shaping,
            combat_shaping=combat_shaping,
        )
        env = ActionMasker(env, mask_fn)

        phase_label = f"{phase_map} [phase {phase_idx + 1}/{len(phase_map_names)}]"
        metrics_path = f"{save_path}_phase{phase_idx + 1}_metrics.json"
        callbacks: list[BaseCallback] = [MetricsCallback(metrics_path)]
        verbose = 1

        if use_rich:
            callbacks.append(RichTrainingCallback(
                total_timesteps=phase_ts,
                map_name=phase_label,
            ))
            verbose = 0

        if model is None:
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                verbose=verbose,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./tb_logs/",
            )
        else:
            prev_phase_path = f"{save_path}_phase{phase_idx}"
            model = MaskablePPO.load(prev_phase_path, env=env)
            model.verbose = verbose

        model.learn(
            total_timesteps=phase_ts,
            callback=callbacks,
            reset_num_timesteps=True,
        )

        phase_save = f"{save_path}_phase{phase_idx + 1}"
        os.makedirs(os.path.dirname(phase_save) or ".", exist_ok=True)
        model.save(phase_save)
        if not use_rich:
            print(f"Phase {phase_idx + 1} checkpoint → {phase_save}.zip")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    return model


def evaluate(
    model_path: str,
    map_name: str = "flat_open",
    n_games: int = 100,
) -> dict[str, float]:
    """Evaluate a trained model against a random opponent."""
    model = MaskablePPO.load(model_path)
    wins, losses, draws = 0, 0, 0

    for i in range(n_games):
        env = make_env(map_name=map_name)
        env = ActionMasker(env, mask_fn)
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks())
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

        game = env.unwrapped.aec._game
        if game.winner == TEAM_A:
            wins += 1
        elif game.winner == TEAM_B:
            losses += 1
        else:
            draws += 1

    results = {
        "wins": wins / n_games,
        "losses": losses / n_games,
        "draws": draws / n_games,
    }
    print(f"Evaluation ({n_games} games on '{map_name}'): {results}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO for High Ground SRPG")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--map", type=str, default="flat_open",
                        choices=list(ALL_MAPS.keys()) + ["all", "all_static", "all_obstacle"])
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--save-path", type=str, default="models/maskable_ppo_srpg")
    parser.add_argument("--evaluate", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--eval-games", type=int, default=100)
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, map_name=args.map, n_games=args.eval_games)
    else:
        train(
            timesteps=args.timesteps,
            map_name=args.map,
            self_play=args.self_play,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    main()
