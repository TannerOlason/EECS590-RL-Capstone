"""SAC training entry point for the V3 scrolling-squad env (stable-baselines3)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _path_shim  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from experiment_utils import resolve_experiment_dir, timestamped_experiment_name
from env.scrolling_env import V2_STYLE_CURRICULUM_PHASES, ScrollingSquadEnv
from training.feature_extractor import SpatialDominantScrollExtractor


REPO_ROOT = Path(__file__).resolve().parents[2]


class V3CurriculumCallback(BaseCallback):
    """Drip-feed V3 difficulty using V2-style reward/curriculum phases."""

    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self._total_timesteps = max(1, total_timesteps)
        self._phase_starts = [0.0, 0.25, 0.55, 0.80]
        self._current_phase = -1

    def _on_step(self) -> bool:
        frac = self.num_timesteps / self._total_timesteps
        phase = 0
        for i, start in enumerate(self._phase_starts):
            if frac >= start:
                phase = i
        phase = min(phase, len(V2_STYLE_CURRICULUM_PHASES) - 1)
        if phase != self._current_phase:
            self.training_env.env_method("set_curriculum_phase", phase)
            self._current_phase = phase
            name = V2_STYLE_CURRICULUM_PHASES[phase]["name"]
            print(f"[curriculum] phase {phase}: {name}")
        return True


def make_env(
    seed: int,
    curriculum_phase: int,
    progress_reward_scale: float,
    kill_reward_scale: float,
):
    def _thunk():
        env = ScrollingSquadEnv(
            seed=seed,
            curriculum_phase=curriculum_phase,
            progress_reward_scale=progress_reward_scale,
            kill_reward_scale=kill_reward_scale,
        )
        env = Monitor(env)
        return env
    return _thunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment base name. Outputs go to "
                             "experiments/<datetime>_<name>/. Defaults to "
                             "sac_<timesteps>_<seed>.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override the experiment directory. If unset, uses "
                             "experiments/<datetime>_<name>/.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a SAC .zip checkpoint to continue training.")
    parser.add_argument("--resume-experiment", type=str, default=None,
                        help="Experiment name to resume. Uses best_model.zip if present, "
                             "otherwise sac_final.zip. Base names resolve to the latest "
                             "timestamped experiment.")
    parser.add_argument("--resume-final", action="store_true",
                        help="With --resume-experiment, prefer sac_final.zip over best_model.zip.")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller buffer / fewer warmup steps for sanity training")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable the V2-style drip-feed curriculum.")
    parser.add_argument("--curriculum-phase", type=int, default=3,
                        help="Fixed phase when --no-curriculum is used; eval always uses phase 3.")
    parser.add_argument("--progress-reward-scale", type=float, default=1.0,
                        help="Multiplier for every curriculum phase's forward-progress reward.")
    parser.add_argument("--kill-reward-scale", type=float, default=1.0,
                        help="Multiplier for every curriculum phase's enemy kill reward.")
    parser.add_argument("--save-replay-buffer", action="store_true",
                        help="Save SAC replay buffer to replay_buffer.pkl after training.")
    parser.add_argument("--replay-buffer-max-mb", type=float, default=1024.0,
                        help="Assert saved replay buffer stays under this size.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    if args.device == "gpu":
        args.device = "cuda"

    resume_path = None
    resume_exp_dir = None
    if args.resume_experiment:
        resume_exp_dir = resolve_experiment_dir(REPO_ROOT / "experiments", args.resume_experiment)
        preferred = resume_exp_dir / ("sac_final.zip" if args.resume_final else "best_model.zip")
        fallback = resume_exp_dir / ("best_model.zip" if args.resume_final else "sac_final.zip")
        resume_path = preferred if preferred.exists() else fallback
        if not resume_path.exists():
            raise SystemExit(f"No checkpoint found in {resume_exp_dir}")
    elif args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise SystemExit(f"--resume-from checkpoint does not exist: {resume_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        requested_name = args.name or out_dir.name
    elif resume_exp_dir is not None:
        out_dir = resume_exp_dir
        requested_name = args.name or out_dir.name
    elif resume_path is not None:
        out_dir = resume_path.parent
        requested_name = args.name or out_dir.name
    else:
        requested_name = args.name or f"sac_{args.total_timesteps}_{args.seed}"
        name = timestamped_experiment_name(requested_name)
        out_dir = REPO_ROOT / "experiments" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # TensorBoard is optional — skip silently if not installed.
    try:
        import tensorboard  # noqa: F401
        tb_dir: str | None = str(out_dir / "tb")
    except ImportError:
        tb_dir = None
        print("[train_sac] tensorboard not installed — skipping TB logging.")

    train_phase = args.curriculum_phase if args.no_curriculum else 0
    train_env = DummyVecEnv([make_env(
        args.seed,
        train_phase,
        args.progress_reward_scale,
        args.kill_reward_scale,
    )])
    eval_env = DummyVecEnv([make_env(
        args.seed + 1000,
        3,
        args.progress_reward_scale,
        args.kill_reward_scale,
    )])

    learning_starts = 1_000 if args.quick else 5_000
    buffer_size = 50_000 if args.quick else 200_000

    policy_kwargs = dict(
        features_extractor_class=SpatialDominantScrollExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            spatial_scale=2.0,
            feature_branch_dim=64,
        ),
        net_arch=[256, 256],
    )

    if resume_path is not None:
        print(f"[train_sac] resuming from {resume_path}")
        print(f"[train_sac] adding {args.total_timesteps} timesteps")
        model = SAC.load(
            str(resume_path),
            env=train_env,
            tensorboard_log=tb_dir,
            device=args.device,
            seed=args.seed,
            verbose=1,
        )
    else:
        model = SAC(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir,
            device=args.device,
            seed=args.seed,
            verbose=1,
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir / "eval"),
        eval_freq=10_000 if not args.quick else 2_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=str(ckpt_dir),
        name_prefix="sac",
    )

    callbacks = [eval_cb, ckpt_cb]
    if not args.no_curriculum:
        curriculum_total = (
            int(model.num_timesteps) + args.total_timesteps
            if resume_path is not None
            else args.total_timesteps
        )
        callbacks.append(V3CurriculumCallback(curriculum_total))

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=resume_path is None,
    )

    final_path = out_dir / "sac_final.zip"
    model.save(str(final_path))

    replay_buffer_path = None
    replay_buffer_size_mb = None
    if args.save_replay_buffer:
        replay_buffer_path = out_dir / "replay_buffer.pkl"
        model.save_replay_buffer(str(replay_buffer_path))
        replay_buffer_size_mb = replay_buffer_path.stat().st_size / (1024 * 1024)
        if replay_buffer_size_mb > args.replay_buffer_max_mb:
            raise RuntimeError(
                f"Replay buffer is {replay_buffer_size_mb:.1f} MB, above "
                f"--replay-buffer-max-mb={args.replay_buffer_max_mb:.1f}."
            )

    # Persist the run config alongside the model for later traceability.
    import json
    (out_dir / "config.json").write_text(json.dumps({
        "experiment": out_dir.name,
        "requested_name": requested_name,
        "total_timesteps": args.total_timesteps,
        "resumed_from": str(resume_path) if resume_path else None,
        "resumed_existing_timesteps": int(model.num_timesteps - args.total_timesteps) if resume_path else 0,
        "model_num_timesteps_after_run": int(model.num_timesteps),
        "seed": args.seed,
        "quick": args.quick,
        "curriculum": not args.no_curriculum,
        "fixed_curriculum_phase": args.curriculum_phase if args.no_curriculum else None,
        "progress_reward_scale": args.progress_reward_scale,
        "kill_reward_scale": args.kill_reward_scale,
        "save_replay_buffer": args.save_replay_buffer,
        "replay_buffer_path": str(replay_buffer_path) if replay_buffer_path else None,
        "replay_buffer_size_mb": replay_buffer_size_mb,
        "replay_buffer_max_mb": args.replay_buffer_max_mb,
        "curriculum_phases": [
            {
                "index": i,
                "name": phase["name"],
                "progress_reward": phase["progress_reward"],
                "step_cost": phase["step_cost"],
                "idle_cost": phase["idle_cost"],
                "invalid_move_penalty": phase["invalid_move_penalty"],
                "attack_attempt_bonus": phase["attack_attempt_bonus"],
                "whiffed_attack_penalty": phase["whiffed_attack_penalty"],
                "in_range_bonus_per_unit": phase["in_range_bonus_per_unit"],
                "enemy_proximity_weight": phase["enemy_proximity_weight"],
                "multi_threat_reward": phase["multi_threat_reward"],
                "focus_fire_bonus": phase["focus_fire_bonus"],
                "visible_enemy_camp_penalty_base": phase["visible_enemy_camp_penalty_base"],
                "visible_enemy_camp_penalty_cap": phase["visible_enemy_camp_penalty_cap"],
                "enemy_movement_enabled": phase["enemy_movement_enabled"],
                "damage_scale": phase["damage_scale"],
                "kill_reward": phase["kill_reward"],
                "no_enemy_still_penalty_base": phase["no_enemy_still_penalty_base"],
                "no_enemy_still_penalty_cap": phase["no_enemy_still_penalty_cap"],
                "clear_right_step_reward": phase["clear_right_step_reward"],
                "lag_lock_penalty": phase["lag_lock_penalty"],
                "column_config": phase["column_config"],
            }
            for i, phase in enumerate(V2_STYLE_CURRICULUM_PHASES)
        ],
        "device": args.device,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "policy_kwargs": {
            "features_extractor_class": "SpatialDominantScrollExtractor",
            "net_arch": [256, 256],
            "features_dim": 256,
            "spatial_scale": 2.0,
            "feature_branch_dim": 64,
        },
    }, indent=2))

    print(f"Saved final model → {final_path}")
    print(f"Best model (per EvalCallback) → {out_dir / 'best_model.zip'}")
    print(f"Experiment dir → {out_dir}")


if __name__ == "__main__":
    main()
