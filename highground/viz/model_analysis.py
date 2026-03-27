"""Post-training model analysis for the High Ground SRPG agent.

Generates PNGs in out_dir covering:
  - Training curves (from JSON sidecar saved alongside the model)
  - Win / loss / draw rate and episode-length distribution
  - Action frequency (overall and per unit class)
  - Where each action is taken on the map (action × position heatmaps)
  - State-value heatmap: critic V(s) averaged over the grid
  - Perturbation saliency: which features most shift V(s)
  - Policy / value MLP weight matrices
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_log = logging.getLogger(__name__)

# ── Constants matching srpg_env.py ───────────────────────────────────────────
GRID_SIZE    = 13
OBS_SIZE     = 418          # terrain(169) + elevation(169) + units(66) + 2 scalars + squad(9) + role(3)
TERRAIN_SL   = slice(0,   169)
ELEV_SL      = slice(169, 338)
UNIT_SL      = slice(338, 404)
SQUAD_SL     = slice(406, 415)
ROLE_SL      = slice(415, 418)
N_ACTIONS    = 8
ACTION_SHORT = ["N", "S", "E", "W", "NE", "SW", "ATK", "END"]
ACTION_FULL  = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
                "MOVE_NE", "MOVE_SW", "ATTACK", "END_TURN"]
CLASS_NAMES  = {0: "Fighter", 1: "Charger", 2: "Ranger", 3: "Siege"}
UNIT_FEAT_NAMES = [
    "row", "col", "hp_frac", "is_mine",
    "cls_Fighter", "cls_Charger", "cls_Ranger", "cls_Siege",
    "move_remain", "has_attacked", "alive",
]
SQUAD_FEAT_NAMES = [
    "team_centroid_row", "team_centroid_col",
    "enemy_centroid_row", "enemy_centroid_col",
    "team_spread",
    "n_allies_alive", "n_enemies_alive",
    "acting_ally_dist", "acting_threat",
]
ROLE_FEAT_NAMES = ["role:Vanguard", "role:Flanker", "role:Support"]

_DARK_BG  = "#1e1e2e"
_PANEL_BG = "#2d2d3f"
_MAP_BG   = "#111111"
_PALETTE  = ["#4fc3f7","#ff5252","#69f0ae","#ffd740",
              "#ea80fc","#ff6e40","#40c4ff","#ccff90"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dark_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")

def _map_ax(ax):
    ax.set_facecolor(_MAP_BG)
    ax.set_xticks([])
    ax.set_yticks([])

def _savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log.info("Saved %s", path)


# ── Rollout collection ────────────────────────────────────────────────────────

def _collect_rollouts(model, map_name: str, seed: int, n_episodes: int) -> dict:
    from sb3_contrib.common.wrappers import ActionMasker
    from highground.training.train import make_env, mask_fn
    from highground.engine.units import TEAM_A, TEAM_B

    rng = np.random.default_rng(seed)

    data: dict[str, Any] = {
        "wins": 0, "losses": 0, "draws": 0,
        "ep_lengths": [], "ep_rewards": [],
        "obs": [], "masks": [],
        "actions": [], "values": [],
        "unit_rows": [], "unit_cols": [], "unit_classes": [],
    }

    for ep in range(n_episodes):
        env_inner = make_env(map_name=map_name)
        env = ActionMasker(env_inner, mask_fn)
        obs, _ = env.reset()
        ep_reward, ep_len, done = 0.0, 0, False

        while not done:
            # --- unit info before action ---
            try:
                game = env.unwrapped.aec._game
                u    = game.current_unit
                ur, uc, ucls = u.row, u.col, int(u.unit_class)
            except Exception:
                ur, uc, ucls = -1, -1, -1

            # --- critic value estimate ---
            try:
                obs_t, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    val = float(model.policy.predict_values(obs_t).cpu().numpy().flat[0])
            except Exception:
                val = float("nan")

            action_mask = env.action_masks()
            action, _   = model.predict(obs, action_masks=action_mask, deterministic=False)
            action      = int(action)   # model.predict returns np.ndarray; game needs a plain int

            data["obs"].append(obs["observation"].copy())
            data["masks"].append(action_mask.copy())
            data["actions"].append(int(action))
            data["values"].append(val)
            data["unit_rows"].append(ur)
            data["unit_cols"].append(uc)
            data["unit_classes"].append(ucls)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len    += 1
            done = terminated or truncated

        try:
            winner = env.unwrapped.aec._game.winner
            if winner == TEAM_A:
                data["wins"] += 1
            elif winner == TEAM_B:
                data["losses"] += 1
            else:
                data["draws"] += 1
        except Exception:
            data["draws"] += 1

        data["ep_lengths"].append(ep_len)
        data["ep_rewards"].append(ep_reward)
        env.close()

    data["obs"]          = np.array(data["obs"],          dtype=np.float32)
    data["masks"]        = np.array(data["masks"],        dtype=np.float32)
    data["actions"]      = np.array(data["actions"],      dtype=np.int32)
    data["values"]       = np.array(data["values"],       dtype=np.float32)
    data["unit_rows"]    = np.array(data["unit_rows"],    dtype=np.int32)
    data["unit_cols"]    = np.array(data["unit_cols"],    dtype=np.int32)
    data["unit_classes"] = np.array(data["unit_classes"], dtype=np.int32)
    return data


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_training_curves(metrics_json: str, out_dir: str) -> None:
    with open(metrics_json) as f:
        raw = json.load(f)

    metrics = [
        ("ep_rew_mean",          "Episode Reward Mean",    "#4fc3f7"),
        ("ep_len_mean",          "Episode Length Mean",    "#ff9800"),
        ("policy_gradient_loss", "Policy Gradient Loss",   "#ef5350"),
        ("value_loss",           "Value Loss",             "#ab47bc"),
        ("entropy_loss",         "Entropy Loss",           "#26c6da"),
        ("approx_kl",            "Approx KL",              "#9ccc65"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor(_DARK_BG)

    ts = raw.get("timesteps", [])
    for ax, (key, label, color) in zip(axes.flat, metrics):
        _dark_ax(ax)
        pairs = [(t, v) for t, v in zip(ts, raw.get(key, [])) if v is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.7)
            if len(ys) >= 7:
                k = max(5, len(ys) // 20)
                rm = np.convolve(ys, np.ones(k) / k, mode="valid")
                ax.plot(xs[k - 1:], rm, color="white", linewidth=1.2)
        ax.set_title(label, color="white", fontsize=9)
        ax.set_xlabel("Timesteps", color="gray", fontsize=7)

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "training_curves.png"))


def _plot_outcomes(data: dict, out_dir: str) -> None:
    n = data["wins"] + data["losses"] + data["draws"]
    if n == 0:
        return
    lengths = data["ep_lengths"]
    rewards = data["ep_rewards"]

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle("Episode Summary", color="white", fontweight="bold")
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # --- Win / loss / draw bar ---
    ax0 = fig.add_subplot(gs[0])
    _dark_ax(ax0)
    labels  = ["Win", "Loss", "Draw"]
    counts  = [data["wins"], data["losses"], data["draws"]]
    colors  = ["#69f0ae", "#ff5252", "#ffd740"]
    bars    = ax0.bar(labels, [c / n * 100 for c in counts],
                      color=colors, edgecolor="#444", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"{cnt/n*100:.1f}%\n(n={cnt})",
                 ha="center", va="bottom", color="white", fontsize=9)
    ax0.set_ylim(0, 115)
    ax0.set_ylabel("%", color="gray")
    ax0.set_title(f"Outcome  (n={n})", color="white")

    # --- Episode length histogram ---
    ax1 = fig.add_subplot(gs[1])
    _dark_ax(ax1)
    ax1.hist(lengths, bins=20, color="#4fc3f7", edgecolor="#2d2d3f", alpha=0.85)
    ax1.axvline(np.mean(lengths), color="#ffd740", linestyle="--",
                label=f"μ={np.mean(lengths):.0f}")
    ax1.axvline(np.median(lengths), color="#69f0ae", linestyle=":",
                label=f"med={np.median(lengths):.0f}")
    ax1.set_xlabel("Steps", color="gray")
    ax1.set_ylabel("Count", color="gray")
    ax1.set_title("Episode Length", color="white")
    ax1.legend(facecolor=_PANEL_BG, labelcolor="white", edgecolor="#555")

    # --- Reward distribution ---
    ax2 = fig.add_subplot(gs[2])
    _dark_ax(ax2)
    ax2.hist(rewards, bins=20, color="#ea80fc", edgecolor="#2d2d3f", alpha=0.85)
    ax2.axvline(np.mean(rewards), color="#ffd740", linestyle="--",
                label=f"μ={np.mean(rewards):.2f}")
    ax2.set_xlabel("Total reward", color="gray")
    ax2.set_ylabel("Count", color="gray")
    ax2.set_title("Episode Reward", color="white")
    ax2.legend(facecolor=_PANEL_BG, labelcolor="white", edgecolor="#555")

    _savefig(fig, os.path.join(out_dir, "episode_summary.png"))


def _plot_action_distribution(data: dict, out_dir: str) -> None:
    actions = data["actions"]
    classes = data["unit_classes"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle("Action Distribution", color="white", fontweight="bold")

    # Overall frequency
    ax0 = axes[0]
    _dark_ax(ax0)
    counts = np.bincount(actions, minlength=N_ACTIONS)
    pcts   = counts / counts.sum() * 100
    bars   = ax0.bar(ACTION_FULL, pcts, color=_PALETTE, edgecolor="#1e1e2e")
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{pct:.1f}%", ha="center", va="bottom", color="white", fontsize=8)
    ax0.set_ylabel("%", color="gray")
    ax0.set_title("Overall Action Frequency", color="white")
    ax0.tick_params(axis="x", rotation=35)

    # Per unit-class heatmap
    ax1 = axes[1]
    ax1.set_facecolor(_MAP_BG)
    class_ids = sorted(CLASS_NAMES)
    matrix    = np.zeros((len(class_ids), N_ACTIONS))
    for ci, cls in enumerate(class_ids):
        mask = classes == cls
        if mask.any():
            c = np.bincount(actions[mask], minlength=N_ACTIONS).astype(float)
            matrix[ci] = c / (c.sum() + 1e-9)

    im = ax1.imshow(matrix, aspect="auto", cmap="plasma", vmin=0, vmax=matrix.max() or 1)
    ax1.set_xticks(range(N_ACTIONS))
    ax1.set_xticklabels(ACTION_FULL, rotation=35, color="gray", fontsize=8)
    ax1.set_yticks(range(len(class_ids)))
    ax1.set_yticklabels([CLASS_NAMES[c] for c in class_ids], color="gray")
    ax1.set_title("Action Frequency by Unit Class", color="white")

    # annotate cells
    for ci in range(len(class_ids)):
        for ai in range(N_ACTIONS):
            v = matrix[ci, ai]
            if v > 0.01:
                ax1.text(ai, ci, f"{v:.2f}", ha="center", va="center",
                         color="white" if v < matrix.max() * 0.7 else "black", fontsize=7)

    plt.colorbar(im, ax=ax1, label="Fraction")
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "action_distribution.png"))


def _plot_action_position_heatmaps(data: dict, out_dir: str) -> None:
    """8-panel figure: where on the map each action type is taken."""
    rows    = data["unit_rows"]
    cols    = data["unit_cols"]
    actions = data["actions"]
    valid   = (rows >= 0) & (cols >= 0)
    rows, cols, actions = rows[valid], cols[valid], actions[valid]

    # Total visit count per cell (denominator for conditional probability)
    total = np.zeros((GRID_SIZE, GRID_SIZE))
    for r, c in zip(rows, cols):
        total[r, c] += 1

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle("Action × Position Heatmaps  (P(action | unit at cell))",
                 color="white", fontweight="bold")

    for i, (ax, name) in enumerate(zip(axes.flat, ACTION_FULL)):
        _map_ax(ax)
        hmap = np.zeros((GRID_SIZE, GRID_SIZE))
        sel  = actions == i
        if sel.any():
            for r, c in zip(rows[sel], cols[sel]):
                hmap[r, c] += 1
            hmap = np.where(total > 0, hmap / (total + 1e-9), 0)
        im = ax.imshow(hmap, cmap="inferno", aspect="equal", vmin=0, vmax=hmap.max() or 1)
        ax.set_title(name, color="white", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "action_position_heatmaps.png"))


def _plot_value_heatmap(data: dict, out_dir: str) -> None:
    rows   = data["unit_rows"]
    cols   = data["unit_cols"]
    values = data["values"]
    valid  = (rows >= 0) & (cols >= 0) & np.isfinite(values)
    rows, cols, values = rows[valid], cols[valid], values[valid]

    grid_sum = np.zeros((GRID_SIZE, GRID_SIZE))
    grid_cnt = np.zeros((GRID_SIZE, GRID_SIZE))
    for r, c, v in zip(rows, cols, values):
        grid_sum[r, c] += v
        grid_cnt[r, c] += 1
    grid_avg = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan)

    # Also split by team (first half of obs units = team A controlled)
    # Use classes data for per-class value
    classes = data["unit_classes"][valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle("Critic V(s) by Unit Position", color="white", fontweight="bold")

    for ax in axes:
        _map_ax(ax)

    # Mean value
    p5, p95 = np.nanpercentile(grid_avg, 5), np.nanpercentile(grid_avg, 95)
    im0 = axes[0].imshow(grid_avg, cmap="RdYlGn", aspect="equal", vmin=p5, vmax=p95)
    axes[0].set_title("Mean V(s) per Cell", color="white")
    plt.colorbar(im0, ax=axes[0], label="V(s)")

    # Visit frequency
    im1 = axes[1].imshow(grid_cnt, cmap="plasma", aspect="equal")
    axes[1].set_title("Visit Count", color="white")
    plt.colorbar(im1, ax=axes[1], label="Count")

    # Std of values (uncertainty in V(s))
    grid_sq  = np.zeros_like(grid_sum)
    for r, c, v in zip(rows, cols, values):
        grid_sq[r, c] += v ** 2
    grid_std = np.where(grid_cnt > 1,
                        np.sqrt(np.maximum(grid_sq / grid_cnt - (grid_avg ** 2), 0)),
                        np.nan)
    im2 = axes[2].imshow(grid_std, cmap="magma", aspect="equal")
    axes[2].set_title("Std(V(s)) — Value Uncertainty", color="white")
    plt.colorbar(im2, ax=axes[2], label="σ V(s)")

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "value_heatmap.png"))


def _plot_saliency(model, data: dict, out_dir: str) -> None:
    """Perturbation-based saliency: |ΔV(s)| when each feature is perturbed by +1 std."""
    obs_all  = data["obs"]
    mask_all = data["masks"]
    if len(obs_all) == 0:
        return

    n_sample = min(300, len(obs_all))
    idx  = np.random.default_rng(0).choice(len(obs_all), n_sample, replace=False)
    obs  = obs_all[idx]
    masks = mask_all[idx]

    std = obs.std(axis=0) + 1e-6

    obs_t  = torch.tensor(obs,   dtype=torch.float32, device=model.device)
    mask_t = torch.tensor(masks, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        base_vals = model.policy.predict_values(
            {"observation": obs_t, "action_mask": mask_t}
        ).cpu().numpy().flatten()

    _log.info("Computing saliency over %d features × %d observations…", OBS_SIZE, n_sample)
    saliency = np.zeros(OBS_SIZE)
    for i in range(OBS_SIZE):
        pert      = obs.copy()
        pert[:, i] += std[i]
        pert_t    = torch.tensor(pert, dtype=torch.float32, device=model.device)
        with torch.no_grad():
            pert_vals = model.policy.predict_values(
                {"observation": pert_t, "action_mask": mask_t}
            ).cpu().numpy().flatten()
        saliency[i] = np.abs(pert_vals - base_vals).mean()

    # ── Figure 1: spatial saliency maps ──────────────────────────────────────
    terrain_sal = saliency[TERRAIN_SL].reshape(GRID_SIZE, GRID_SIZE)
    elev_sal    = saliency[ELEV_SL].reshape(GRID_SIZE, GRID_SIZE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle("Spatial Feature Saliency  (perturbation |ΔV(s)|)",
                 color="white", fontweight="bold")

    for ax in axes:
        _map_ax(ax)

    im0 = axes[0].imshow(terrain_sal, cmap="hot", aspect="equal")
    axes[0].set_title("Terrain Features", color="white")
    plt.colorbar(im0, ax=axes[0], label="|ΔV|")

    im1 = axes[1].imshow(elev_sal, cmap="hot", aspect="equal")
    axes[1].set_title("Elevation Features", color="white")
    plt.colorbar(im1, ax=axes[1], label="|ΔV|")

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "saliency_spatial.png"))

    # ── Figure 2: feature-index bar chart + unit feature breakdown ────────────
    unit_sal   = saliency[UNIT_SL].reshape(6, 11)          # (6 units, 11 features)
    per_feat   = unit_sal.mean(axis=0)                      # average across 6 units
    per_unit   = unit_sal.mean(axis=1)                      # average across 11 features
    squad_sal  = saliency[SQUAD_SL]                         # 9 squad coordination features
    role_sal   = saliency[ROLE_SL]                          # 3 role-conditioning features

    fig2, axes2 = plt.subplots(1, 5, figsize=(30, 5))
    fig2.patch.set_facecolor(_DARK_BG)
    fig2.suptitle("Feature Saliency Breakdown", color="white", fontweight="bold")

    # All 418 features
    ax = axes2[0]
    _dark_ax(ax)
    ax.bar(range(OBS_SIZE), saliency, width=1, color="#4fc3f7", alpha=0.8)
    ax.axvspan(0,   169, alpha=0.12, color="#ffd740", label="terrain")
    ax.axvspan(169, 338, alpha=0.12, color="#69f0ae", label="elevation")
    ax.axvspan(338, 404, alpha=0.12, color="#ea80fc", label="unit feats")
    ax.axvspan(406, 415, alpha=0.20, color="#ff6e40", label="squad feats")
    ax.axvspan(415, 418, alpha=0.30, color="#69f0ae", label="role token")
    ax.set_xlim(0, OBS_SIZE)
    ax.set_title("All 418 Features", color="white")
    ax.set_xlabel("Feature index", color="gray")
    ax.set_ylabel("|ΔV|", color="gray")
    ax.legend(facecolor=_PANEL_BG, labelcolor="white", edgecolor="#555", fontsize=8)

    # Per unit-feature position (11 features, averaged across 6 units)
    ax = axes2[1]
    _dark_ax(ax)
    ax.bar(UNIT_FEAT_NAMES, per_feat, color="#ea80fc", edgecolor="#1e1e2e")
    ax.set_title("Saliency per Unit Feature\n(mean across 6 units)", color="white")
    ax.set_xlabel("Feature", color="gray")
    ax.set_ylabel("|ΔV|", color="gray")
    ax.tick_params(axis="x", rotation=40)

    # Per unit slot (6 units, averaged across 11 features)
    ax = axes2[2]
    _dark_ax(ax)
    unit_labels = [f"Unit {i}" for i in range(6)]
    ax.bar(unit_labels, per_unit, color=_PALETTE[:6], edgecolor="#1e1e2e")
    ax.set_title("Saliency per Unit Slot\n(mean across 11 features)", color="white")
    ax.set_xlabel("Unit slot", color="gray")
    ax.set_ylabel("|ΔV|", color="gray")

    # Squad coordination features (6a)
    ax = axes2[3]
    _dark_ax(ax)
    ax.bar(SQUAD_FEAT_NAMES, squad_sal, color="#ff6e40", edgecolor="#1e1e2e")
    ax.set_title("Squad Feature Saliency (6a)", color="white")
    ax.set_xlabel("Feature", color="gray")
    ax.set_ylabel("|ΔV|", color="gray")
    ax.tick_params(axis="x", rotation=40)

    # Role-conditioning token (6b)
    ax = axes2[4]
    _dark_ax(ax)
    ax.bar(ROLE_FEAT_NAMES, role_sal, color="#69f0ae", edgecolor="#1e1e2e")
    ax.set_title("Role Token Saliency (6b)", color="white")
    ax.set_xlabel("Role", color="gray")
    ax.set_ylabel("|ΔV|", color="gray")
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    _savefig(fig2, os.path.join(out_dir, "saliency_features.png"))


def _plot_network_weights(model, out_dir: str) -> None:
    """Visualize weight matrices of the policy and value MLPs."""
    try:
        policy_net = model.policy.mlp_extractor.policy_net
        value_net  = model.policy.mlp_extractor.value_net
    except AttributeError:
        _log.warning("mlp_extractor not accessible; skipping weight visualization")
        return

    def _layers(net, prefix):
        out = []
        for name, module in net.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.detach().cpu().numpy()
                out.append((f"{prefix}/{name or 'L'}", w))
        return out

    layers = (
        _layers(policy_net, "policy")
        + _layers(value_net, "value")
    )
    try:
        layers.append(("action_head",
                        model.policy.action_net.weight.detach().cpu().numpy()))
        layers.append(("value_head",
                        model.policy.value_net.weight.detach().cpu().numpy()))
    except Exception:
        pass

    if not layers:
        _log.warning("No weight matrices found; skipping")
        return

    n    = len(layers)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes_grid = plt.subplots(rows, cols,
                                  figsize=(6 * cols, 4 * rows),
                                  squeeze=False)
    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle("Network Weight Matrices", color="white", fontweight="bold")

    axes_flat = list(axes_grid.flat)
    for (name, w), ax in zip(layers, axes_flat):
        _map_ax(ax)
        vmax = np.abs(w).mean() + 2 * np.abs(w).std()
        im   = ax.imshow(w, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name}  {w.shape}", color="white", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "network_weights.png"))


def _plot_value_by_class(data: dict, out_dir: str) -> None:
    """Box-plots of V(s) per unit class."""
    values  = data["values"]
    classes = data["unit_classes"]
    valid   = np.isfinite(values) & (classes >= 0)
    values, classes = values[valid], classes[valid]

    class_ids = sorted(CLASS_NAMES)
    grouped   = [values[classes == c] for c in class_ids]
    labels    = [CLASS_NAMES[c] for c in class_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(_DARK_BG)
    _dark_ax(ax)

    bp = ax.boxplot(grouped, labels=labels, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        plt.setp(bp[element], color="#888")

    ax.set_ylabel("V(s)", color="gray")
    ax.set_title("Critic Value by Unit Class", color="white", fontweight="bold")
    _savefig(fig, os.path.join(out_dir, "value_by_class.png"))


# ── Top-level entry point ─────────────────────────────────────────────────────

def analyze_model(
    model_path: str,
    out_dir: str,
    metrics_json: str | None = None,
    map_name: str = "central_hill",
    seed: int = 42,
    n_eval_episodes: int = 50,
) -> None:
    """Run all analyses and save PNGs to *out_dir*."""
    from sb3_contrib import MaskablePPO

    os.makedirs(out_dir, exist_ok=True)

    _log.info("Loading model from %s", model_path)
    model = MaskablePPO.load(model_path)

    # Resolve "all" to a concrete map for rollout collection
    analysis_map = "central_hill" if map_name == "all" else map_name

    # --- Training curves (needs metrics sidecar from MetricsCallback) ---
    if metrics_json and os.path.exists(metrics_json):
        _log.info("Plotting training curves from %s", metrics_json)
        _plot_training_curves(metrics_json, out_dir)
    else:
        _log.info("No metrics JSON at %s — skipping training curves", metrics_json)

    # --- Rollout collection ---
    _log.info("Collecting %d rollout episodes on map=%s", n_eval_episodes, analysis_map)
    data = _collect_rollouts(model, analysis_map, seed, n_eval_episodes)

    # --- Static / cheap plots ---
    _log.info("Plotting episode summary")
    _plot_outcomes(data, out_dir)

    _log.info("Plotting action distribution")
    _plot_action_distribution(data, out_dir)

    _log.info("Plotting action × position heatmaps")
    _plot_action_position_heatmaps(data, out_dir)

    _log.info("Plotting value heatmap")
    _plot_value_heatmap(data, out_dir)

    _log.info("Plotting value by unit class")
    _plot_value_by_class(data, out_dir)

    # --- Saliency (406 forward passes — moderate cost) ---
    _log.info("Computing perturbation saliency (%d passes × %d obs)…", OBS_SIZE, min(300, len(data["obs"])))
    _plot_saliency(model, data, out_dir)

    # --- Network weights ---
    _log.info("Plotting network weights")
    _plot_network_weights(model, out_dir)

    _log.info("Model analysis complete → %s/", out_dir)
