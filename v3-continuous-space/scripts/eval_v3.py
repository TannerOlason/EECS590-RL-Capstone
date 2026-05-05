"""Evaluate a trained SAC checkpoint on the V3 scrolling env.

Writes a metrics JSON and a replay GIF into the experiment directory:
    experiments/<name>/eval_metrics.json
    experiments/<name>/replay.gif        (if --render-gif)

Pass either --checkpoint <path> + --out-dir <dir>, or --experiment <name>
to load experiments/<name>/best_model.zip and write the eval artefacts back
into the same folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _path_shim  # noqa: F401,E402

import numpy as np  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from experiment_utils import resolve_experiment_dir  # noqa: E402
from env.scrolling_env import ScrollingSquadEnv  # noqa: E402
from viz.render_scroll import render_frame  # noqa: E402
from viz.replay import save_gif  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
MOVE_LABELS = {
    (-1, -1): "NW",
    (0, -1): "N",
    (1, -1): "NE",
    (-1, 0): "W",
    (0, 0): "IDLE",
    (1, 0): "E",
    (-1, 1): "SW",
    (0, 1): "S",
    (1, 1): "SE",
}


def _resolve_paths(args) -> tuple[str, Path]:
    if args.experiment:
        exp_dir = resolve_experiment_dir(REPO_ROOT / "experiments", args.experiment)
        if args.checkpoint:
            ckpt = args.checkpoint
        else:
            best_model = exp_dir / "best_model.zip"
            final_model = exp_dir / "sac_final.zip"
            ckpt = str(best_model if best_model.exists() else final_model)
    else:
        if not args.checkpoint:
            raise SystemExit("Pass --experiment <name> OR --checkpoint <path>.")
        ckpt = args.checkpoint
        exp_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    exp_dir.mkdir(parents=True, exist_ok=True)
    return ckpt, exp_dir


def _match_checkpoint_obs(obs: dict, model: SAC) -> dict:
    """Adapt current env observations to older checkpoint observation shapes."""
    model_spaces = model.observation_space.spaces
    adapted = dict(obs)
    for key, space in model_spaces.items():
        expected = space.shape
        current = adapted[key]
        if current.shape == expected:
            continue
        if len(current.shape) == len(expected) and current.shape[1:] == expected[1:]:
            if current.shape[0] >= expected[0]:
                adapted[key] = current[:expected[0]]
            else:
                padded = np.zeros(expected, dtype=current.dtype)
                padded[:current.shape[0]] = current
                adapted[key] = padded
            continue
        if len(current.shape) == len(expected) and current.shape[0] >= expected[0]:
            adapted[key] = current[:expected[0]]
            continue
        raise ValueError(
            f"Cannot adapt observation key {key!r} from {current.shape} "
            f"to checkpoint shape {expected}."
        )
    return adapted


def _action_label(action: np.ndarray) -> str:
    sx = 0 if abs(float(action[0])) < 0.25 else (1 if action[0] > 0 else -1)
    sy = 0 if abs(float(action[1])) < 0.25 else (1 if action[1] > 0 else -1)
    return MOVE_LABELS[(sx, sy)]


def _squad_spread(env: ScrollingSquadEnv) -> float:
    alive = [p for p in env.players if p.unit.alive]
    if len(alive) <= 1:
        return 0.0
    xs = np.array([p.world_x_int() for p in alive], dtype=float)
    ys = np.array([p.row_int() for p in alive], dtype=float)
    return float((xs.max() - xs.min()) + (ys.max() - ys.min()))


def _leftmost_x(env: ScrollingSquadEnv) -> int:
    alive = [p for p in env.players if p.unit.alive]
    return int(min((p.world_x_int() for p in alive), default=0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default=None,
                   help="Experiment name; loads experiments/<name>/best_model.zip "
                        "or the latest experiments/<datetime>_<name>/ match.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Direct path to a .zip checkpoint (overrides --experiment).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write eval_metrics.json + replay.gif "
                        "(defaults to the experiment dir or the checkpoint's parent).")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render", "--render-gif", dest="render", action="store_true",
                   help="Save a GIF of the first episode (replay.gif).")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--progress-reward-scale", type=float, default=1.0,
                   help="Reward scale used during training; recorded in metrics.")
    p.add_argument("--kill-reward-scale", type=float, default=1.0,
                   help="Kill reward scale used during training; recorded in metrics.")
    args = p.parse_args()

    ckpt, exp_dir = _resolve_paths(args)
    print(f"[eval] checkpoint = {ckpt}")
    print(f"[eval] writing to = {exp_dir}")

    env = ScrollingSquadEnv(
        seed=args.seed,
        progress_reward_scale=args.progress_reward_scale,
        kill_reward_scale=args.kill_reward_scale,
    )
    model = SAC.load(ckpt, device="auto")
    checkpoint_obs_shapes = {
        key: tuple(space.shape)
        for key, space in model.observation_space.spaces.items()
    }
    print(f"[eval] checkpoint obs shapes = {checkpoint_obs_shapes}")

    per_episode: list[dict] = []
    frames_first_ep: list = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_r = 0.0
        steps = 0
        move_counts = {label: 0 for label in MOVE_LABELS.values()}
        attack_intent_count = 0
        lock_enemy_count = 0
        lock_lag_count = 0
        invalid_move_count = 0
        shielded_invalid_move_count = 0
        enemy_invalid_move_count = 0
        enemy_shielded_move_count = 0
        no_enemy_still_penalty_total = 0.0
        lag_lock_penalty_total = 0.0
        invalid_move_penalty_total = 0.0
        whiffed_attack_penalty_total = 0.0
        approach_reward_total = 0.0
        enemy_distance_closed_total = 0.0
        visible_enemy_camp_penalty_total = 0.0
        multi_threat_reward_total = 0.0
        focus_fire_bonus_total = 0.0
        focus_fire_hits = 0
        player_damage_total = 0
        offscreen_kills = 0
        damage_taken_total = 0
        max_scroll = 0
        max_leftmost = _leftmost_x(env)
        spread_samples: list[float] = []
        for t in range(args.max_steps):
            action, _ = model.predict(_match_checkpoint_obs(obs, model), deterministic=True)
            move_counts[_action_label(action)] += 1
            if float(action[2]) > 0.5:
                attack_intent_count += 1
            obs, reward, term, trunc, info = env.step(action)
            ep_r += reward
            steps += 1
            lock_enemy_count += int("scroll_locked_by_enemies" in info)
            lock_lag_count += int("scroll_locked_by_lagging_player" in info)
            invalid_move_count += int(info.get("invalid_move_count", 0))
            shielded_invalid_move_count += int(info.get("shielded_invalid_move_count", 0))
            enemy_invalid_move_count += int(info.get("enemy_invalid_move_macro", 0))
            enemy_shielded_move_count += int(info.get("enemy_shielded_move_macro", 0))
            no_enemy_still_penalty_total += float(info.get("no_enemy_still_penalty", 0.0))
            lag_lock_penalty_total += float(info.get("lag_lock_penalty", 0.0))
            invalid_move_penalty_total += float(info.get("invalid_move_penalty", 0.0))
            whiffed_attack_penalty_total += float(info.get("whiffed_attack_penalty", 0.0))
            approach_reward_total += float(info.get("approach_reward", 0.0))
            enemy_distance_closed_total += float(info.get("enemy_distance_closed", 0.0))
            visible_enemy_camp_penalty_total += float(info.get("visible_enemy_camp_penalty", 0.0))
            multi_threat_reward_total += float(info.get("multi_threat_reward", 0.0))
            focus_fire_bonus_total += float(info.get("focus_fire_bonus", 0.0))
            focus_fire_hits += int(info.get("focus_fire_hit", 0))
            player_damage_total += int(info.get("player_damage_macro", 0))
            offscreen_kills += int(info.get("offscreen_player_kills", 0))
            damage_taken_total += int(info.get("damage_taken_macro", 0))
            max_scroll = max(max_scroll, int(info.get("scroll_offset", 0)))
            max_leftmost = max(max_leftmost, _leftmost_x(env))
            spread_samples.append(_squad_spread(env))
            if args.render and ep == 0 and t % 2 == 0:
                frames_first_ep.append(render_frame(env))
            if term or trunc:
                break
        ep_kills = sum(1 for e in env.enemies if not e.unit.alive)
        rightish = move_counts["E"] + move_counts["NE"] + move_counts["SE"]
        leftish = move_counts["W"] + move_counts["NW"] + move_counts["SW"]
        ep_record = {
            "episode": ep,
            "reward": float(ep_r),
            "steps": steps,
            "scroll_reached": int(info.get("scroll_offset", 0)),
            "max_scroll_reached": int(max_scroll),
            "max_leftmost_world_x": int(max_leftmost),
            "squad_alive_at_end": int(info.get("squad_alive", 0)),
            "enemies_killed": int(ep_kills),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "move_counts": move_counts,
            "right_move_rate": float(rightish / max(1, steps)),
            "left_move_rate": float(leftish / max(1, steps)),
            "idle_rate": float(move_counts["IDLE"] / max(1, steps)),
            "invalid_move_count": int(invalid_move_count),
            "invalid_move_rate": float(invalid_move_count / max(1, steps)),
            "shielded_invalid_move_count": int(shielded_invalid_move_count),
            "shielded_invalid_move_rate": float(shielded_invalid_move_count / max(1, steps)),
            "enemy_invalid_move_count": int(enemy_invalid_move_count),
            "enemy_shielded_move_count": int(enemy_shielded_move_count),
            "attack_intent_rate": float(attack_intent_count / max(1, steps)),
            "scroll_locked_by_enemy_steps": int(lock_enemy_count),
            "scroll_locked_by_lagging_player_steps": int(lock_lag_count),
            "no_enemy_still_penalty_total": float(no_enemy_still_penalty_total),
            "lag_lock_penalty_total": float(lag_lock_penalty_total),
            "invalid_move_penalty_total": float(invalid_move_penalty_total),
            "whiffed_attack_penalty_total": float(whiffed_attack_penalty_total),
            "approach_reward_total": float(approach_reward_total),
            "enemy_distance_closed_total": float(enemy_distance_closed_total),
            "visible_enemy_camp_penalty_total": float(visible_enemy_camp_penalty_total),
            "multi_threat_reward_total": float(multi_threat_reward_total),
            "focus_fire_bonus_total": float(focus_fire_bonus_total),
            "focus_fire_hits": int(focus_fire_hits),
            "player_damage_total": int(player_damage_total),
            "offscreen_player_kills": int(offscreen_kills),
            "damage_taken_total": int(damage_taken_total),
            "mean_squad_spread": float(np.mean(spread_samples)) if spread_samples else 0.0,
            "max_squad_spread": float(np.max(spread_samples)) if spread_samples else 0.0,
        }
        per_episode.append(ep_record)
        print(f"[ep {ep}] reward={ep_r:+.2f}  steps={steps}  "
              f"scroll={ep_record['scroll_reached']}  "
              f"alive={ep_record['squad_alive_at_end']}/3  "
              f"kills={ep_record['enemies_killed']}")

    # Aggregate.
    arr = lambda k: np.array([e[k] for e in per_episode], dtype=float)  # noqa: E731
    summary = {
        "checkpoint": str(ckpt),
        "n_episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "progress_reward_scale": args.progress_reward_scale,
        "kill_reward_scale": args.kill_reward_scale,
        "mean_reward": float(arr("reward").mean()),
        "std_reward":  float(arr("reward").std()),
        "mean_steps":  float(arr("steps").mean()),
        "mean_scroll": float(arr("scroll_reached").mean()),
        "mean_max_scroll": float(arr("max_scroll_reached").mean()),
        "mean_max_leftmost_world_x": float(arr("max_leftmost_world_x").mean()),
        "mean_squad_alive_at_end": float(arr("squad_alive_at_end").mean()),
        "mean_enemies_killed": float(arr("enemies_killed").mean()),
        "mean_right_move_rate": float(arr("right_move_rate").mean()),
        "mean_idle_rate": float(arr("idle_rate").mean()),
        "mean_invalid_move_count": float(arr("invalid_move_count").mean()),
        "mean_invalid_move_rate": float(arr("invalid_move_rate").mean()),
        "mean_shielded_invalid_move_count": float(arr("shielded_invalid_move_count").mean()),
        "mean_shielded_invalid_move_rate": float(arr("shielded_invalid_move_rate").mean()),
        "mean_enemy_invalid_move_count": float(arr("enemy_invalid_move_count").mean()),
        "mean_enemy_shielded_move_count": float(arr("enemy_shielded_move_count").mean()),
        "mean_attack_intent_rate": float(arr("attack_intent_rate").mean()),
        "mean_scroll_locked_by_enemy_steps": float(arr("scroll_locked_by_enemy_steps").mean()),
        "mean_scroll_locked_by_lagging_player_steps": float(arr("scroll_locked_by_lagging_player_steps").mean()),
        "mean_no_enemy_still_penalty_total": float(arr("no_enemy_still_penalty_total").mean()),
        "mean_lag_lock_penalty_total": float(arr("lag_lock_penalty_total").mean()),
        "mean_invalid_move_penalty_total": float(arr("invalid_move_penalty_total").mean()),
        "mean_whiffed_attack_penalty_total": float(arr("whiffed_attack_penalty_total").mean()),
        "mean_approach_reward_total": float(arr("approach_reward_total").mean()),
        "mean_enemy_distance_closed_total": float(arr("enemy_distance_closed_total").mean()),
        "mean_visible_enemy_camp_penalty_total": float(arr("visible_enemy_camp_penalty_total").mean()),
        "mean_multi_threat_reward_total": float(arr("multi_threat_reward_total").mean()),
        "mean_focus_fire_bonus_total": float(arr("focus_fire_bonus_total").mean()),
        "mean_focus_fire_hits": float(arr("focus_fire_hits").mean()),
        "mean_player_damage_total": float(arr("player_damage_total").mean()),
        "mean_offscreen_player_kills": float(arr("offscreen_player_kills").mean()),
        "mean_damage_taken_total": float(arr("damage_taken_total").mean()),
        "mean_squad_spread": float(arr("mean_squad_spread").mean()),
        "mean_max_squad_spread": float(arr("max_squad_spread").mean()),
        "episodes": per_episode,
    }
    metrics_path = exp_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval] wrote {metrics_path}")

    if args.render and frames_first_ep:
        gif_path = exp_dir / "replay.gif"
        save_gif(frames_first_ep, gif_path, fps=8)
        print(f"[eval] wrote {gif_path}")

    print()
    print(f"Mean episode reward     : {summary['mean_reward']:+.2f}  ± {summary['std_reward']:.2f}")
    print(f"Mean episode length     : {summary['mean_steps']:.1f}")
    print(f"Mean scroll reached     : {summary['mean_scroll']:.1f}")
    print(f"Mean max leftmost x     : {summary['mean_max_leftmost_world_x']:.1f}")
    print(f"Mean enemies killed     : {summary['mean_enemies_killed']:.2f}")
    print(f"Mean right move rate    : {summary['mean_right_move_rate']:.2%}")
    print(f"Mean idle rate          : {summary['mean_idle_rate']:.2%}")
    print(f"Mean invalid move rate  : {summary['mean_invalid_move_rate']:.2%}")
    print(f"Mean lag-lock steps     : {summary['mean_scroll_locked_by_lagging_player_steps']:.1f}")


if __name__ == "__main__":
    main()
