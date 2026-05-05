# V3 — Continuous-Action Side-Scrolling Survival

V3 of the High Ground RL methods study. Where V1 was tabular nav, and V2 was discrete-action multi-agent MAPPO on a fixed 13×13 grid, V3 keeps V2's tile movement/combat rules but changes the policy interface and world container: **continuous SAC action interface** + **grid-snapped V2 movement/combat** + **non-stationary world** (infinite, procedurally generated, scrolling left → right).

## Problem statement

A 3-unit player squad spawns on the left side of a 13×13 visible window over an infinite procedurally generated terrain strip. The window scrolls to the right when the squad's centroid advances. New columns of terrain are generated on the right edge as we scroll, sometimes carrying enemy spawns that walk or shoot at the squad. A unit dies if its HP drops to 0 *or* if it falls off the left edge (the squad must keep moving). The episode ends when all 3 player units are dead.

| | V2 (MAPPO) | **V3 (SAC)** |
|---|---|---|
| Action space | `Discrete(12)` | **`Box([-1,-1,-1], [1,1,1])`** = `(vx, vy, attack_intent)` |
| State | discrete tile positions, fixed 13×13 map | **discrete tile positions in an infinite scrolling world** |
| Episode | turn-based, 50 turns max | **continuous-time micro-ticks, 600 macro ticks max** |
| Algorithm | MAPPO (on-policy, CTDE) via BenchMARL | **SAC** (off-policy, max-entropy) via stable-baselines3 |
| Multi-agent | 3 actors, shared params, per-team critic | **3 actors, shared params + single critic** (parameter-sharing reduction) |
| Reward | shaped, terminal win/loss | shaped, terminal squad-wipe |

## Why SAC?

The action space is now continuous, so DQN / REINFORCE / A2C / MAPPO are off the table. Among continuous-control candidates:

| Algorithm | Decision |
|---|---|
| **SAC** | ✓ Chosen. Off-policy with a replay buffer (sample-efficient on a procedural env where each episode is high-variance), and the max-entropy term gives intrinsic exploration without manual schedules. Robust to the non-stationary dynamics introduced by the other 2 squad members updating during training. |
| DDPG | Deterministic policy + no entropy term ⇒ brittle to hyperparameters and prone to local optima in procedural maps. SAC strictly dominates here. |
| TD3 | Twin critics fix overestimation but it still lacks the entropy bonus. SAC also uses twin critics under the hood, plus the temperature-tuned exploration. |
| PPO-continuous | On-policy, throws away rollouts after each update. We already did on-policy in V2 (MAPPO); SAC gives the **off-policy contrast** the rubric asks for. |
| TRPO | Same on-policy issue + heavier implementation than PPO with no benefit over SAC for this problem. |

**Pedagogical pairing:** V2 = on-policy / discrete / multi-agent, V3 = off-policy / continuous / shared-parameter multi-agent. Clean axis flip across the two versions.

## Architecture

```
                Squad obs (Dict)
                 ┌──────────────────┐
                 │ spatial:  (8,13,13)│   ← terrain | elevation | friendly HP
                 │                    │     | enemy_FIGHTER HP | enemy_CHARGER HP
                 │                    │     | enemy_RANGER HP  | enemy_SIEGE HP
                 │                    │     | HP potion presence
                 │ features: (15,)    │   ← active unit, agent id, alive flags, squad summary
                 └──────────┬─────────┘
                            ↓
              ScrollCnnExtractor (mirrors V2 SpatialCnnMlp)
                Conv 8→16 (s=1) → ReLU
                Conv 16→32 (s=2) → ReLU
                Conv 32→32 (s=2) → ReLU
                Flatten → Linear 256 → ReLU
                Concat features → Linear features_dim → ReLU
                            ↓
                  SAC actor + twin critics
                  (shared extractor; net_arch=[256,256])
```

The 3 squad units share the same actor/critic networks. Each Gym `step()` advances **one** unit; after 3 micro-steps (a "macro tick") enemies act and the world may scroll. The `agent_id` one-hot in the observation tells the shared policy which unit it is acting for. The SAC action is continuous, but `(vx, vy)` is quantized into one of V2's 8 grid directions; the resulting move is checked with V2 `can_step`, pays terrain movement cost, and updates Charger momentum just like V2. Attack intent triggers a V2 range-checked attack using `compute_damage` and `has_attacked`.

## V2-style curriculum

V3 training uses a V2-inspired drip-feed curriculum by default:

| Phase | Purpose | Main knobs |
|---|---|---|
| 1 | Scroll foundation | no enemies, easier terrain, strong forward-progress reward |
| 2 | Weak contact | sparse weak enemies, attack attempt + in-range bonuses |
| 3 | Approach transfer | more enemies/terrain, V2-style damage/kill/contact shaping |
| 4 | Full pressure | full generator difficulty with shaped combat and movement pressure |

This mirrors the V2 MAPPO curriculum idea: learn one new pressure at a time
instead of asking the policy to discover movement, combat, and survival in the
full procedural environment from frame 1.

Scrolling is locked while any enemy is alive in the visible 13x13 window, and
it is also capped so no living squad member can be pushed off-screen by camera
movement. Progress reward is based on the **leftmost living squad member**, not
the centroid, so the policy must keep survivors moving as a formation before it
can advance into newly generated columns. When there are no visible enemies,
new rightward ground earns a small dense reward, standing still incurs an
escalating penalty, and letting a lagging unit block scrolling is penalised.

The V2 `models/mappo_phase7_policy.pt` checkpoint is **not** a compatible
weight initialization for V3. It was trained with MAPPO/BenchMARL, a discrete
action head, and a different observation interface; V3 uses SB3 SAC with a
continuous action policy and different actor/critic state. What transfers is
the design: reward shaping, curriculum staging, grid mechanics, and evaluation
metrics.

## Reward shaping

Tuned conservatively, mirroring V2's PBRS/curriculum approach:

| Signal | Value |
|---|---|
| Forward progress | phase-dependent `+0.09..0.18 · Δ leftmost_survivor_world_x` |
| Step cost | phase-dependent `-0.002..-0.005` per micro-step |
| Failed/no movement | phase-dependent `-0.005..-0.010` |
| Clear-map right step | phase-dependent bonus for reaching new rightward ground |
| Lag-lock | phase-dependent penalty when a living unit blocks safe scrolling |
| Attack attempt | phase-dependent `+0.00..+0.05` |
| In-range contact | phase-dependent `+0.00..+0.03` per friendly in range |
| Enemy proximity | phase-dependent bonus for closing distance to enemies |
| Enemy kill | `+1.0..+1.5` |
| Damage dealt/taken | phase-dependent HP delta shaping |
| HP potion picked up | `+0.1 · (heal / max_hp)` |
| Friendly killed | `−1.5` per death |
| All friendlies dead | `−3.0` + episode end |
| Endurance bonus | `+0.01` per macro tick once `centroid_world_x > 200` |

`train_v3.py` exposes `--progress-reward-scale` and `--kill-reward-scale`.
For an overnight comparison across five settings:

```bash
.venv/bin/python v3-continuous-space/scripts/run_reward_sweep.py \
    --prefix sac_500k_lockclear \
    --total-timesteps 500000 \
    --device auto \
    --eval-episodes 10
```

The sweep trains and evaluates:

| Variant | Progress scale | Kill scale |
|---|---:|---:|
| `balanced` | 0.70 | 1.50 |
| `combat_lean` | 0.50 | 2.00 |
| `combat_mid` | 0.35 | 2.50 |
| `combat_high` | 0.25 | 3.00 |
| `combat_max` | 0.15 | 4.00 |

## Enemies and items

**Enemies are deliberately weak and sparse** so the env is learnable in a reasonable amount of training:

- Spawn probability per generated column: `0.025 + 0.10·diff` (≈ 2.5 % early, ≈ 12.5 % late)
- HP scaling: `1.0 + 0.25·diff` (no enemy starts with more than 1.25× base class HP)
- ATK scaling: `0.75×` base, so the strongest late-game Siege does about `5·0.75 ≈ 4` damage per hit instead of 5
- Class mix: pure Fighters early; Chargers/Rangers/Siege phase in once `diff > 0.3`
- Behaviour: `WalkerAI` (Fighters/Chargers — close + melee) or `ShooterAI` (Rangers/Siege — keep distance + ranged)

The agent can **identify enemies by class** through the per-class spatial channels (chs 3-6). This is essential for "focus damage on the dangerous units" — the agent learns to prioritise high-ATK Sieges or to flank Rangers before they can keep distance.

**HP potions** appear at ≈ 1 per 33 columns. A potion is consumed when any friendly unit's tile matches its position; the unit is healed by `0.25 · max_hp`, capped at the missing amount. The shaped pickup reward is small (`+0.1 · heal_frac`) — enough that the agent learns to detour for them, but not so generous that potions become a crutch over learning to dodge and focus-fire.

### Why these knobs?

Without potions, the agent is brittle to early unlucky hits — one bad opening burst can leave a unit dead-walking for the rest of the episode with no recovery path. Without per-class enemy identification, the policy has to infer threat type from HP fraction alone, which is a much weaker signal than knowing *what* a thing is. Together these changes preserve the pressure to dodge and prioritise targets while making the curriculum tractable in fewer million timesteps.

## Reused V2 code (imports, no duplication)

- `highground.engine.grid.Grid`, `Terrain`, `GRID_SIZE` — visible window storage
- `highground.engine.units.Unit`, `UnitClass`, `CLASS_STATS`, `TEAM_A/B` — unit stats and HP
- `highground.engine.combat.compute_damage` — height/flank/momentum bonuses still apply
- `highground.engine.pathfinding` — used implicitly by enemy AI (Chebyshev distance + step validity)

The CNN architecture in `training/feature_extractor.py` *mirrors* `highground/training/cnn_model.py` (4-channel version) but is rewritten against `BaseFeaturesExtractor` because the V2 model extends BenchMARL's `Model` base class.

## Layout

```
v3-continuous-space/
├── _path_shim.py             # adds repo root to sys.path
├── env/
│   ├── chunk_generator.py    # procedural column generation
│   ├── infinite_grid.py      # sliding-window grid wrapper
│   ├── enemy_ai.py           # WalkerAI + ShooterAI
│   ├── unit_state.py         # scrolling-world wrapper around V2 Unit
│   ├── obs_builder.py        # Dict observation construction
│   └── scrolling_env.py      # Gymnasium env (Box action, multi-unit rotation)
├── training/
│   ├── feature_extractor.py  # SB3 BaseFeaturesExtractor (CNN+MLP)
│   └── train_sac.py          # SAC training loop with eval + checkpointing
├── viz/
│   ├── render_scroll.py      # rgb_array renderer (matplotlib)
│   └── replay.py             # GIF export
├── scripts/
│   ├── play_random.py        # smoke test
│   ├── train_v3.py           # train SAC
│   ├── eval_v3.py            # eval checkpoint + dump metrics/GIF
│   └── run_baseline_random.py # random baseline metrics/GIF
├── tests/                    # pytest suite (17 tests)
└── models/                   # legacy local SB3 checkpoints
```

Experiment artifacts are stored at repo root under
`experiments/<YYYYMMDD-HHMMSS_name>/`.
Each run can contain `best_model.zip`, `sac_final.zip`, `config.json`,
`eval_metrics.json`, `replay.gif`, TensorBoard logs, and intermediate
checkpoints.

## How to run

Set up a venv and install the V3 deps:

```bash
python -m venv .venv
.venv/bin/pip install -e .                              # base deps from pyproject.toml
.venv/bin/pip install tqdm rich tensorboard             # SB3 progress bar + TB logging
```

Then:

```bash
# 1) Tests
.venv/bin/python -m pytest v3-continuous-space/tests/ -v

# 2) Random-policy baseline experiment
MPLCONFIGDIR=/tmp/matplotlib-v3 .venv/bin/python \
    v3-continuous-space/scripts/run_baseline_random.py \
    --name baseline_random --episodes 10 --seed 42 --max-steps 1500 --render

# 3) Evaluate the bundled 3k SAC smoke checkpoint
MPLCONFIGDIR=/tmp/matplotlib-v3 .venv/bin/python \
    v3-continuous-space/scripts/eval_v3.py \
    --experiment sac_3k_smoke --episodes 10 --seed 42 --max-steps 1500 --render

# 4) Quick SAC training sanity (~5–10 min on CPU)
.venv/bin/python v3-continuous-space/scripts/train_v3.py \
    --name sac_50k_quick --total-timesteps 50000 --quick --device cpu

# 5) Full training (~hours on GPU)
.venv/bin/python v3-continuous-space/scripts/train_v3.py \
    --name sac_1m --total-timesteps 1000000

# 6) Continue a promising run with more timesteps
.venv/bin/python v3-continuous-space/scripts/train_v3.py \
    --resume-experiment sac_1m --total-timesteps 500000 --device gpu

# 7) Evaluate a trained checkpoint + GIF
.venv/bin/python v3-continuous-space/scripts/eval_v3.py \
    --experiment sac_50k_quick --episodes 10 --render
```

Training writes to `experiments/<YYYYMMDD-HHMMSS_name>/` by default, so folders
sort in run order. The printed `Experiment dir` line is the exact folder to use
for follow-up commands. Evaluation with `--experiment <name>` first looks for an
exact folder and then falls back to the latest timestamp-prefixed match, so both
`--experiment 20260504-231122_sac_50k_quick` and
`--experiment sac_50k_quick` work. It loads `best_model.zip` and writes
`eval_metrics.json` plus `replay.gif` back into the same folder. The legacy 3k
checkpoint in `experiments/sac_3k_smoke/` was trained before the current
8-channel observation; evaluation adapts it to the older 4-channel input for
checkpoint compatibility.

Curriculum is enabled by default in `train_v3.py`. To train on the full V3
environment from the first frame, pass `--no-curriculum --curriculum-phase 3`.
The curriculum now includes a combat-initiative ramp: early combat uses weak,
stationary enemies so the squad must approach; later phases restore enemy
movement while preserving smaller rewards for closing distance, multi-unit
threat on the same target, and focus-fire. Visible-enemy camping is penalized
when the squad neither closes distance nor deals damage.

To resume training, use `--resume-experiment <name>` or
`--resume-from <checkpoint.zip>`. In resume mode, `--total-timesteps` means
additional timesteps. `--resume-experiment` prefers `best_model.zip`; add
`--resume-final` to continue from `sac_final.zip` instead.

## Diagnostics

V3 eval writes behavior diagnostics into `eval_metrics.json`, including action
rates, idle rate, invalid move count/rate/penalty, attack intent rate,
leftmost progress, squad spread, scroll-lock counts, lag-lock counts,
stillness penalties, approach reward, enemy-distance closing, multi-threat
reward, focus-fire hits, visible-enemy camping penalties, damage taken, kills,
and survival. Invalid player movement is action-shielded during training/eval:
the requested invalid move is still counted and penalized, but the env tries the
closest valid neighboring step so episodes do not collapse into long frozen
no-op loops.

Summarize all experiment folders:

```bash
MPLCONFIGDIR=/tmp/matplotlib-v3 .venv/bin/python \
    v3-continuous-space/scripts/analyze_v3.py --top 20
```

This writes `experiments/v3_summary.csv`, `experiments/v3_summary.md`, and
`experiments/v3_summary.png`.

Run perturbation saliency for a trained SAC policy:

```bash
MPLCONFIGDIR=/tmp/matplotlib-v3 .venv/bin/python \
    v3-continuous-space/scripts/saliency_v3.py \
    --experiment sac_1000k_high_leftmost \
    --episodes 2 --max-steps 120 --sample-limit 40
```

This writes `saliency_spatial.png`, `saliency_features.png`, and
`saliency.json` into the experiment folder.

The non-spatial feature saliency includes local tactical features for all 8
neighboring movement directions: whether the step is valid, whether the
destination holds an enemy or potion, and whether it advances rightward. It
also includes nearest-visible-enemy relative position, distance, and attackable
flags. These features were added specifically to diagnose and penalize policies
that repeatedly attempt blocked moves.

Fresh training uses `SpatialDominantScrollExtractor`, which keeps the spatial
CNN branch wider and compresses scalar features before concatenation. The legacy
`ScrollCnnExtractor` remains available so older checkpoints can still load.

To save SAC replay buffers for later inspection:

```bash
.venv/bin/python v3-continuous-space/scripts/train_v3.py \
    --name sac_debug_buffer --total-timesteps 100000 \
    --save-replay-buffer --replay-buffer-max-mb 1024
```

Replay buffers are not saved by default. To prune saved buffers:

```bash
.venv/bin/python v3-continuous-space/scripts/prune_replay_buffers.py \
    --keep-newest 2 --max-total-mb 4096 --delete
```

## Notes on partial observability

The observation is intentionally **window-local** — the agent cannot see terrain or enemies outside the visible 13×13 strip. This makes the env technically a POMDP (the agent doesn't know what's in the next chunk to the right until it scrolls there). We did *not* use a recurrent policy here to keep V3 tightly scoped; an LSTM/GRU SAC variant is a natural V3.5 follow-on.
