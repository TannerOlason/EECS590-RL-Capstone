# High Ground SRPG — RL Methods Study

A self-contained RL project extracted from the QD map-evolution pipeline.
The environment is a 3v3 turn-based tactics game on a 13×13 procedurally
generated grid.  This repository is an RL methods study, covering the
progression from classical tabular methods to modern multi-agent deep RL.

## Algorithm Lineage

```
SARSA(λ)  →  PPO  →  MAPPO (CTDE)
   ↑               ↑
eligibility    Generalized Advantage Estimation (GAE)
 traces         uses the same λ-decay weighting
```

The λ in SARSA(λ) and the λ in GAE are the same idea: weight earlier
transitions by λ^k, interpolating between fully bootstrapped (λ=0, TD(0))
and full Monte-Carlo returns (λ=1).  This connection is made explicit in
`scripts/run_sarsa.py`.

## Environment

| Property | Value |
|----------|-------|
| Grid | 13×13 procedural terrain |
| Teams | 2 teams × 3 units (Fighter, Ranger, Charger) |
| Actions | 12 per unit (8 moves + 3 attacks + end turn) |
| Observation | 418-dimensional per unit (local + terrain features) |
| Global state | 836-dimensional (for centralised critic) |
| Episode length | ≤ 50 turns |

## Quick Start

```bash
pip install -e .

# Classical RL: SARSA(λ) on navigation subtask (runs in seconds)
python scripts/run_sarsa.py --compare --plot plots/sarsa_curves.png

# Analyse the bundled MAPPO checkpoint
python scripts/run_analysis.py --episodes 20

# Train MAPPO from scratch (requires BenchMARL)
python scripts/train_mappo.py
```

## Project Layout

```
rl/
├── classical/
│   ├── nav_env.py          # Simplified 13×13 navigation subtask
│   └── sarsa_lambda.py     # SARSA(λ) + SarsaN (n-step variant)
├── highground/
│   ├── engine/             # Game engine (Grid, Units, GameState, combat)
│   ├── env/                # Gymnasium + PettingZoo wrappers
│   ├── training/           # MAPPO via BenchMARL (benchmarl_train.py)
│   │   └── cnn_model.py    # CNN+MLP spatial architecture
│   ├── maps/               # Static map presets
│   ├── metrics/            # Terrain exploitation metrics
│   ├── viz/
│   │   ├── render_map.py   # Matplotlib grid renderer
│   │   ├── replay.py       # Frame-by-frame GIF export
│   │   └── model_analysis.py  # Training curves, action dist, saliency
│   └── llm/                # [Optional] LLM steering extension
├── models/
│   └── mappo_phase7_policy.pt   # Best trained checkpoint (curriculum phase 7)
├── scripts/
│   ├── run_sarsa.py        # Train + evaluate SARSA variants
│   ├── train_mappo.py      # Train MAPPO (delegates to benchmarl_train.py)
│   ├── run_analysis.py     # Model analysis and saliency plots
│   └── smoke_test_viewer.py  # [Optional] LLM steering TUI demo
└── tests/
    ├── test_engine.py
    └── test_env.py
```

## Algorithm Justifications

### Why MAPPO and not the others?

| Algorithm | Decision |
|-----------|----------|
| **MAPPO** | ✓ Chosen. Multi-agent PPO with CTDE: per-team centralised critic during training, each unit acts on its own local observation. Handles joint coordination. |
| REINFORCE | PPO is a variance-reduced, trust-region-constrained superset. REINFORCE without the clip objective would diverge on the multi-phase curriculum. |
| Vanilla Actor-Critic | MAPPO with CTDE is actor-critic with a better critic architecture. Single-agent A2C is a regression. |
| DQN | Discrete Q-learning doesn't extend naturally to shared-policy multi-agent. MAPPO is strictly better here. |
| DDPG / TD3 | Continuous-action algorithms. The action space here is discrete (12 actions). Not applicable. |
| SAC | Also continuous-action. Max-entropy framework is interesting but requires discretisation and a replay buffer. On-policy PPO is more appropriate for curriculum training. |
| TRPO | PPO was designed as the practical replacement for TRPO. The clip objective gives the same trust-region guarantee without the Fisher-vector products. |

### Why SARSA(λ) for the classical baseline?

SARSA is on-policy, exactly like PPO.  The eligibility trace (λ) is the
direct conceptual ancestor of GAE.  One implementation covers both the
trace-based and n-cutoff (SarsaN) variants.  Q-learning is off-policy
and would break the on-policy narrative.

## MAPPO Architecture

```
Observation (418-dim)
    ↓
[Spatial channels]  [Unit features]
 13×13×2 terrain     80-dim vector
    ↓                    ↓
 Conv 8×(3×3)       passthrough
 Conv 16×(3×3)
 Conv 32×(3×3)
    ↓ flatten
    └──────────────────┘
          ↓
     Linear(256)
     ReLU
     Linear(128)
     ReLU
        ↓
  Action logits (12)    Value head
```

Actor and critic are trained with:
- Clipped PPO objective (ε = 0.2)
- GAE advantage estimation (λ = 0.95)
- Entropy regularisation (0.05 → 0.02 → 0.01 across curriculum phases)
- Action masking (invalid moves/attacks zeroed before softmax)

## Optional: LLM Steering

With `pip install -e ".[llm]"`, the `highground/llm/` extension provides
LLM-based logit steering of the trained policy.  Run the interactive viewer:

```bash
python scripts/smoke_test_viewer.py
```

This shows per-action probability bars across four steering variants
(baseline, aggressive, defensive, terrain-aware), making it a manual
saliency viewer for the policy's action distribution.
