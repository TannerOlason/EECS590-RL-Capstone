"""Interactive smoke test viewer for the V6 LLM steering pipeline.

Runs a full game using all four steering variants (base / +semantic / +tactical
/ +both), records a StepRecord at each micro-action, then lets you navigate
the replay interactively in the terminal.

Usage::

    python scripts/smoke_test_viewer.py \\
        --model-path models/mappo_policy.pt \\
        --map central_hill \\
        --model llama3.1:8b \\
        --seed 42

Navigation keys:
    → / l       Next step
    ← / h       Previous step
    ]           Jump to first step of next round
    [           Jump to first step of current or previous round
    q           Quit
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive smoke test viewer for the V6 LLM steering pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map",
        default="central_hill",
        choices=["central_hill", "flat_open", "choke_point", "asymmetric_heights", "fortress"],
        help="Map to use for the smoke test.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model name for the LLM strategy manager.",
    )
    parser.add_argument(
        "--strategy",
        default="Seize the high ground, then rush the weakened enemy.",
        help="Free-text strategy prompt injected as the first phase directive.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the BenchMARL policy .pt file.",
    )
    parser.add_argument(
        "--max-bias",
        type=float,
        default=2.0,
        help="Maximum logit bias magnitude for the tactical navigator.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Skip LLM queries. The +semantic and +both variants will use "
            "a hardcoded OCCUPY_HIGH_GROUND objective instead."
        ),
    )
    return parser.parse_args()


def _load_map(map_name: str):
    """Load a static map by name and return (grid, spawns_a, spawns_b)."""
    from highground.maps.static_maps import STATIC_MAPS, OBSTACLE_MAPS
    all_maps = {**STATIC_MAPS, **OBSTACLE_MAPS}
    if map_name not in all_maps:
        raise ValueError(f"Unknown map: {map_name!r}. Available: {list(all_maps)}")
    return all_maps[map_name]()


def _build_game_state(grid, spawns_a, spawns_b):
    """Build a fresh GameState from a grid and spawn positions."""
    from highground.engine.game_state import GameState
    from highground.engine.units import Unit, UnitClass, TEAM_A, TEAM_B

    unit_classes = [UnitClass.FIGHTER, UnitClass.CHARGER, UnitClass.RANGER]
    units = []
    for i, (r, c) in enumerate(spawns_a):
        units.append(Unit(unit_id=i, team=TEAM_A, unit_class=unit_classes[i], row=r, col=c))
    for i, (r, c) in enumerate(spawns_b):
        units.append(Unit(unit_id=i + 3, team=TEAM_B, unit_class=unit_classes[i], row=r, col=c))
    return GameState(grid, units)


def _build_obs(game_state, team_id: int) -> tuple:
    """Build observation and action mask from game state.

    Returns (obs_array, mask_array) using the env's observation builder.
    Falls back to zero obs if env not importable.
    """
    try:
        from highground.env.srpg_env import SRPGEnv
        # We use the env's _build_observation method by instantiating a minimal env
        # but it's easier to import from the env module directly
        from highground.env.srpg_env import _build_observation  # type: ignore
    except ImportError:
        pass

    # Build observation directly from game_state following the obs layout
    import numpy as np
    from highground.env.srpg_env import OBS_SIZE
    from highground.engine.grid import GRID_SIZE
    from highground.engine.units import TEAM_A, TEAM_B, UnitClass

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    grid = game_state.grid

    # Terrain grid (169 floats)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            obs[r * GRID_SIZE + c] = float(grid.terrain[r, c]) * 2.0 / 2.0
    # Elevation grid (169 floats)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            obs[169 + r * GRID_SIZE + c] = float(grid.elevation[r, c]) * 2.0 / 2.0

    # Per-unit features (6 units × 11 features)
    my_units     = [u for u in game_state.units if u.team == team_id]
    enemy_units  = [u for u in game_state.units if u.team != team_id]
    all_obs_units = my_units + enemy_units  # my team first

    flip_col = (team_id == TEAM_B)

    for i, u in enumerate(all_obs_units):
        base = 338 + i * 11
        obs[base + 0] = u.row / 12.0
        obs[base + 1] = (12.0 - u.col) / 12.0 if flip_col else u.col / 12.0
        obs[base + 2] = u.hp / u.max_hp if u.alive else 0.0
        obs[base + 3] = 1.0 if u.team == team_id else 0.0
        # One-hot class (4 classes)
        class_idx = int(u.unit_class)
        obs[base + 4 + class_idx] = 1.0
        obs[base + 8] = u.move_remaining / u.move_range if u.alive else 0.0
        obs[base + 9] = 1.0 if u.has_attacked else 0.0
        obs[base + 10] = 1.0 if u.alive else 0.0

    # Current unit index
    obs[404] = float(game_state.current_unit_id) / 5.0
    # Round fraction
    from highground.engine.game_state import MAX_TURNS
    obs[405] = float(game_state.round_number) / float(MAX_TURNS)

    # Squad features (simplified)
    alive_allies  = [u for u in my_units if u.alive]
    alive_enemies = [u for u in enemy_units if u.alive]
    from highground.engine.units import TEAM_A as _TA
    UNITS_PER_TEAM = 3
    if alive_allies:
        cr = sum(u.row for u in alive_allies) / len(alive_allies)
        cc = sum(u.col for u in alive_allies) / len(alive_allies)
    else:
        cr = cc = 6.0
    if alive_enemies:
        er = sum(u.row for u in alive_enemies) / len(alive_enemies)
        ec = sum(u.col for u in alive_enemies) / len(alive_enemies)
    else:
        er = ec = 6.0

    obs[406] = cr / 12.0
    obs[407] = cc / 12.0
    obs[408] = er / 12.0
    obs[409] = ec / 12.0
    spread = 0.0
    if alive_allies:
        spread = sum(abs(u.row - cr) + abs(u.col - cc) for u in alive_allies) / len(alive_allies)
    obs[410] = spread / 12.0
    obs[411] = len(alive_allies) / UNITS_PER_TEAM
    obs[412] = len(alive_enemies) / UNITS_PER_TEAM
    # min ally dist
    if alive_allies and alive_enemies:
        min_dist = min(
            abs(a.row - e.row) + abs(a.col - e.col)
            for a in alive_allies for e in alive_enemies
        )
        obs[413] = min_dist / 24.0
    # threat ratio
    obs[414] = len(alive_enemies) / max(1, len(alive_allies)) / 3.0
    # Role token (slot-based: 0=Vanguard, 1=Flanker, 2=Support)
    cu = game_state.current_unit
    role_slot = cu.unit_id % UNITS_PER_TEAM
    obs[415 + role_slot] = 1.0

    # Action mask
    mask = np.array(game_state.valid_action_mask(), dtype=np.float32)
    return obs, mask


def _run_game(
    game_state,
    wrapper,
    base_policy,
    tile_index,
    team_id: int,
    semantic_manager,
    tactical_objective,
    tactical_attack_urgency: float,
    both_manager,
    deterministic: bool = True,
) -> list:
    """Run a full game and collect StepRecords.

    Team 0 uses the wrapper (with all 4 variants recorded).
    Team 1 uses the base policy only.

    Returns a list of StepRecord objects.
    """
    from highground.engine.units import TEAM_A, TEAM_B

    records = []
    step_count = 0
    max_steps = 10000  # safety limit

    while not game_state.done and step_count < max_steps:
        cu = game_state.current_unit
        obs, mask = _build_obs(game_state, cu.team)

        if cu.team == team_id:
            # Record full step for team 0
            record = wrapper.get_step_record(
                obs=obs,
                action_mask=mask,
                game_state=game_state,
                team_id=cu.team,
                semantic_manager=semantic_manager,
                tactical_objective=tactical_objective,
                tactical_attack_urgency=tactical_attack_urgency,
                both_manager=both_manager,
                deterministic=deterministic,
            )
            records.append(record)
            # Execute the "both" action for actual game progression
            action = record.action_both
        else:
            # Team 1: base policy only
            action, _ = base_policy.predict(obs, deterministic=True, action_masks=mask)

        # Validate and step
        valid_mask = game_state.valid_action_mask()
        if valid_mask[action] == 0:
            # Fall back to END_TURN if action is somehow invalid
            action = 11
        game_state.step(action)
        step_count += 1

    return records


def _try_import_readchar():
    """Try to import readchar; return (module_or_None, available_bool)."""
    try:
        import readchar
        return readchar, True
    except ImportError:
        return None, False


# ── Terminal rendering ────────────────────────────────────────────────────────

# Palette — mirrors highground/qd/display.py exactly
_CA     = "bright_cyan"       # Team A
_CB     = "bright_red"        # Team B
_BORDER = "bright_blue"
_METRIC = "bright_yellow"
_POS    = "bold bright_green"
_NEG    = "bold bright_red"
_ZERO   = "dim white"

_HP_FULL  = "█"
_HP_EMPTY = "░"

# Terrain cells — identical to display.py's _TERRAIN_CELL
_TERRAIN_CELL = {
    (0, 0): ("··", "dim green"),
    (0, 1): ("·▪", "green"),
    (0, 2): ("·▲", "bold bright_green"),
    (1, 0): ("≈·", "yellow"),
    (1, 1): ("≈▪", "bright_yellow"),
    (1, 2): ("≈▲", "bold bright_yellow"),
    (2, 0): ("██", "on grey23"),
    (2, 1): ("██", "on grey23"),
    (2, 2): ("██", "on grey23"),
}


def _hp_bar(hp: int, max_hp: int, width: int = 8, team: int = 0) -> "Text":
    """HP bar matching display.py's _hp_bar exactly."""
    from rich.text import Text
    colour = _CA if team == 0 else _CB
    filled = round(hp / max_hp * width) if max_hp else 0
    filled = max(0, min(width, filled))
    t = Text()
    t.append(_HP_FULL * filled, style=colour)
    t.append(_HP_EMPTY * (width - filled), style="dim")
    t.append(f" {hp:>2}/{max_hp}", style="dim white")
    return t


def _prob_cell(p: float, chosen: bool, masked: bool, bar_width: int = 8) -> "Text":
    """Coloured probability bar for the action table."""
    from rich.text import Text
    filled = max(0, min(bar_width, int(round(p * bar_width))))
    if masked:
        colour = "dim"
    elif p > 0.40:
        colour = _POS
    elif p > 0.18:
        colour = _METRIC
    elif p > 0.07:
        colour = "yellow"
    else:
        colour = "dim"
    t = Text()
    t.append("▶ " if chosen else "  ", style=_POS if chosen else "")
    t.append("▓" * filled,           style=colour)
    t.append("░" * (bar_width - filled), style="dim")
    t.append(f" {p:.2f}",            style=colour)
    return t


def _render_frame(
    records: list,
    step_idx: int,
    grid,
    console,
) -> None:
    """Render one viewer frame using the QD display palette."""
    from rich import box as rbox
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from highground.engine.grid import GRID_SIZE, Terrain
    from highground.engine.units import TEAM_A, TEAM_B
    from highground.llm.wrapper import ACTION_NAMES

    rec   = records[step_idx]
    total = len(records)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = Text(justify="center")
    hdr.append("⚔  HIGH GROUND  ", style="bold bright_white")
    hdr.append("◆  ", style=_METRIC)
    hdr.append("LLM STEERING SMOKE TEST", style=f"bold {_CA}")
    hdr.append("  ◆  ", style=_METRIC)
    hdr.append(f"Step {step_idx + 1}/{total}  Round {rec.round_num}", style="bright_white")
    hdr.append("  ·  ", style="dim")
    hdr.append("[←/→] step  [[] round  [q] quit", style="dim")
    header_panel = Panel(hdr, style=_BORDER, box=rbox.DOUBLE_EDGE, padding=(0, 1))

    # ── Map (Rich Text, two chars per tile matching display.py) ───────────────
    unit_at = {(u["row"], u["col"]): u for u in rec.units_snapshot if u["alive"]}
    current_uid = next(
        (u["id"] for u in rec.units_snapshot
         if u["row"] == rec.unit_pos[0] and u["col"] == rec.unit_pos[1]),
        -1,
    )

    map_text = Text()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            t = int(grid.terrain[row, col])
            e = int(grid.elevation[row, col])
            pos = (row, col)
            if t == int(Terrain.UNCROSSABLE):
                map_text.append("██", style="on grey23")
            elif pos in unit_at:
                u = unit_at[pos]
                ltr   = "A" if u["team"] == TEAM_A else "B"
                label = f"{ltr}{u['id'] % 3}"
                base  = _CA if u["team"] == TEAM_A else _CB
                is_active = u["id"] == current_uid
                style = f"bold reverse {base}" if is_active else f"bold {base}"
                map_text.append(label, style=style)
            else:
                chars, style = _TERRAIN_CELL.get((t, e), ("??", "white"))
                map_text.append(chars, style=style)
        map_text.append("\n")

    # ── Unit roster with HP bars ───────────────────────────────────────────────
    roster = Text()
    roster.append(" UNITS\n", style="bold dim white")
    team_a = sorted([u for u in rec.units_snapshot if u["team"] == TEAM_A], key=lambda u: u["id"])
    team_b = sorted([u for u in rec.units_snapshot if u["team"] == TEAM_B], key=lambda u: u["id"])
    cls_abbr = {0: "F", 1: "C", 2: "R", 3: "S"}

    for ua, ub in zip(team_a, team_b):
        a_act   = ua["id"] == current_uid
        b_act   = ub["id"] == current_uid
        a_alive = ua["alive"]
        b_alive = ub["alive"]
        a_style = f"bold {_CA}" if a_act else (_CA if a_alive else "dim strike")
        b_style = f"bold {_CB}" if b_act else (_CB if b_alive else "dim strike")
        ca = cls_abbr.get(int(ua["class"]), "?")
        cb = cls_abbr.get(int(ub["class"]), "?")

        roster.append("▶ " if a_act else "  ", style=_POS if a_act else "")
        roster.append(f"A{ua['id'] % 3}({ca}) ", style=a_style)
        if a_alive:
            roster.append_text(_hp_bar(ua["hp"], ua["max_hp"], width=6, team=TEAM_A))
        else:
            roster.append("  [KO]  ", style="dim red")

        roster.append("   ")

        roster.append("▶ " if b_act else "  ", style=_POS if b_act else "")
        roster.append(f"B{ub['id'] % 3}({cb}) ", style=b_style)
        if b_alive:
            roster.append_text(_hp_bar(ub["hp"], ub["max_hp"], width=6, team=TEAM_B))
        else:
            roster.append("  [KO]  ", style="dim red")
        roster.append("\n")

    # ── LLM intent + BFS + action log ─────────────────────────────────────────
    info = Text()

    info.append("─" * 40 + "\n", style=f"dim {_BORDER}")
    info.append(" LLM INTENT\n", style="bold dim white")
    if rec.intent is not None:
        info.append("  Objective  ", style="dim white")
        info.append(f"{rec.intent.objective.value}\n", style=_METRIC)
        info.append("  Urgency    ", style="dim white")
        urg = rec.intent.attack_urgency
        urg_col = _POS if urg > 0.6 else ("yellow" if urg > 0.3 else "dim")
        info.append(f"{urg:.2f}\n", style=urg_col)
        info.append("  Hold after ", style="dim white")
        info.append(f"{'yes' if rec.intent.hold_after else 'no'}\n", style=_CA)
        info.append("  Phase done ", style="dim white")
        pc_col = "bold bright_magenta" if rec.intent.phase_complete else _ZERO
        info.append(f"{'YES ✦' if rec.intent.phase_complete else 'no'}\n", style=pc_col)
        info.append("  Reasoning  ", style="dim white")
        reason = rec.intent.reasoning
        # Wrap at ~38 chars
        words, line, lines = reason.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 38:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        info.append(lines[0] + "\n", style="italic dim white")
        for extra in lines[1:]:
            info.append("             " + extra + "\n", style="italic dim white")
    else:
        info.append("  (disabled — run without --no-llm)\n", style="dim")

    info.append("─" * 40 + "\n", style=f"dim {_BORDER}")
    info.append(" TACTICAL NAV\n", style="bold dim white")
    if rec.bfs_target:
        info.append("  Target tile ", style="dim white")
        info.append(f"{rec.bfs_target}\n", style=_CA)
        info.append("  First step  ", style="dim white")
        info.append(f"{rec.bfs_direction or '—'}\n", style=_METRIC)
    else:
        info.append("  On target / no target\n", style="dim")

    info.append("─" * 40 + "\n", style=f"dim {_BORDER}")
    info.append(" RECENT ACTIONS (team 0)\n", style="bold dim white")
    start = max(0, step_idx - 4)
    for i in range(start, step_idx + 1):
        r2     = records[i]
        is_cur = (i == step_idx)
        bn     = ACTION_NAMES[r2.action_base]
        tn     = ACTION_NAMES[r2.action_both]
        agree  = (bn == tn)
        row_style = "bright_white" if is_cur else "dim white"
        info.append("▶ " if is_cur else "  ", style=_POS if is_cur else "")
        info.append(f"R{r2.round_num:>2} u{r2.unit_id[-1]}  ", style=row_style)
        info.append(f"base={bn:<8} ", style=(_CA if not agree else _ZERO))
        info.append(f"→{tn}\n",       style=(_METRIC if not agree else "dim green"))

    # ── Left panel assembly ───────────────────────────────────────────────────
    left_grid = Table.grid(padding=(0, 1))
    left_grid.add_column(no_wrap=True)
    left_grid.add_column(no_wrap=False)
    left_grid.add_row(map_text, Group(roster, info))
    left_panel = Panel(
        left_grid,
        title="[bold]BATTLE MAP[/]",
        style=_BORDER,
        box=rbox.ROUNDED,
        padding=(0, 1),
    )

    # ── Probability table ─────────────────────────────────────────────────────
    prob_tbl = Table(
        box=rbox.SIMPLE,
        show_header=True,
        header_style="bold dim white",
        padding=(0, 1),
    )
    prob_tbl.add_column("Action",    style="white",    width=10, no_wrap=True)
    prob_tbl.add_column("BASE",      no_wrap=True)
    prob_tbl.add_column("+SEMANTIC", no_wrap=True)
    prob_tbl.add_column("+TACTICAL", no_wrap=True)
    prob_tbl.add_column("+BOTH",     no_wrap=True)
    prob_tbl.add_column("",          width=3, no_wrap=True)   # mask tag

    for i, name in enumerate(ACTION_NAMES):
        masked  = int(rec.action_mask[i]) == 0
        name_t  = Text(name, style="dim" if masked else "white")
        mask_t  = Text("[M]" if masked else "   ", style="dim")
        prob_tbl.add_row(
            name_t,
            _prob_cell(float(rec.probs_base[i]), i == rec.action_base, masked),
            _prob_cell(float(rec.probs_sem[i]),  i == rec.action_sem,  masked),
            _prob_cell(float(rec.probs_tac[i]),  i == rec.action_tac,  masked),
            _prob_cell(float(rec.probs_both[i]), i == rec.action_both, masked),
            mask_t,
        )

    right_panel = Panel(
        prob_tbl,
        title="[bold]ACTION PROBABILITIES[/]",
        style=_BORDER,
        box=rbox.ROUNDED,
        padding=(0, 1),
    )

    # ── Render ────────────────────────────────────────────────────────────────
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_column()
    body.add_row(left_panel, right_panel)

    console.clear()
    console.print(header_panel)
    console.print(body)


def _interactive_viewer(records: list, grid) -> None:
    """Run the interactive terminal viewer over the recorded steps."""
    readchar_mod, has_readchar = _try_import_readchar()

    from rich.console import Console
    console = Console()

    if not has_readchar:
        print(
            "\nNote: 'readchar' is not installed. Install it with:\n"
            "    pip install readchar\n"
            "Falling back to press-Enter navigation.\n"
        )

    step_idx = 0
    total = len(records)

    if total == 0:
        print("No steps recorded. Game may have ended immediately.")
        return

    while True:
        _render_frame(records, step_idx, grid, console)

        if has_readchar:
            key = readchar_mod.readkey()
            # Right arrow / 'l' = next step
            if key in (readchar_mod.key.RIGHT, 'l'):
                step_idx = min(step_idx + 1, total - 1)
            # Left arrow / 'h' = prev step
            elif key in (readchar_mod.key.LEFT, 'h'):
                step_idx = max(step_idx - 1, 0)
            # ']' = next round
            elif key == ']':
                cur_round = records[step_idx].round_num
                for i in range(step_idx + 1, total):
                    if records[i].round_num > cur_round:
                        step_idx = i
                        break
                else:
                    step_idx = total - 1
            # '[' = prev round
            elif key == '[':
                cur_round = records[step_idx].round_num
                # Find first step of current round
                first_this_round = step_idx
                for i in range(step_idx - 1, -1, -1):
                    if records[i].round_num == cur_round:
                        first_this_round = i
                    else:
                        break
                if first_this_round < step_idx:
                    step_idx = first_this_round
                else:
                    # Go to first step of previous round
                    for i in range(step_idx - 1, -1, -1):
                        if records[i].round_num < cur_round:
                            # Find first step of that round
                            prev_round = records[i].round_num
                            for j in range(i, -1, -1):
                                if records[j].round_num < prev_round:
                                    step_idx = j + 1
                                    break
                            else:
                                step_idx = 0
                            break
            # 'q' = quit
            elif key in ('q', 'Q', readchar_mod.key.CTRL_C):
                break
        else:
            # Press-Enter fallback
            print(f"\n[Step {step_idx + 1}/{total}] Enter=next, p=prev, q=quit: ", end="", flush=True)
            cmd = input().strip().lower()
            if cmd == '' or cmd == 'n':
                step_idx = min(step_idx + 1, total - 1)
            elif cmd == 'p':
                step_idx = max(step_idx - 1, 0)
            elif cmd == 'q':
                break

    print("\nViewer exited.")


def main() -> None:
    """Entry point for the smoke test viewer."""
    import numpy as np
    import random

    args = _parse_args()

    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading map: {args.map} ...")
    grid, spawns_a, spawns_b = _load_map(args.map)

    print(f"Loading policy: {args.model_path} ...")
    from highground.training.benchmarl_adapter import BenchMARLPolicyAdapter
    base_policy = BenchMARLPolicyAdapter(args.model_path)
    print(f"  Loaded: {base_policy}")

    from highground.llm.tile_index import TileIndex
    from highground.llm.wrapper import LLMSteeringWrapper
    from highground.llm.strategy_manager import StrategyManager
    from highground.llm.models import ObjectiveType, StrategyPhase

    tile_index = TileIndex(grid)

    # Build strategy managers
    semantic_manager = None
    both_manager = None
    tactical_objective = ObjectiveType.OCCUPY_HIGH_GROUND
    tactical_attack_urgency = 0.5

    if not args.no_llm:
        phases = [StrategyPhase(prompt=args.strategy)]
        semantic_manager = StrategyManager(
            team_id=0,
            phases=phases,
            model=args.model,
        )
        both_manager = StrategyManager(
            team_id=0,
            phases=phases,
            model=args.model,
        )
        print(f"Strategy managers built (model={args.model}).")
    else:
        print("--no-llm: using fixed objective OCCUPY_HIGH_GROUND for +semantic and +both.")

    # Build wrapper (mode="both" is used only for get_step_record, not predict)
    wrapper = LLMSteeringWrapper(
        base_policy=base_policy,
        tile_index=tile_index,
        mode="both",
        max_bias=args.max_bias,
        strategy_manager=both_manager,
        fixed_objective=tactical_objective,
        fixed_attack_urgency=tactical_attack_urgency,
    )

    print("Building game state and running game ...")
    game_state = _build_game_state(grid, spawns_a, spawns_b)

    records = _run_game(
        game_state=game_state,
        wrapper=wrapper,
        base_policy=base_policy,
        tile_index=tile_index,
        team_id=0,
        semantic_manager=semantic_manager,
        tactical_objective=tactical_objective,
        tactical_attack_urgency=tactical_attack_urgency,
        both_manager=both_manager,
        deterministic=True,
    )

    winner_label = "Team 0" if game_state.winner == 0 else (
        "Team 1" if game_state.winner == 1 else "Draw"
    )
    print(f"\nGame over: {winner_label}  |  Rounds played: {game_state.round_number}  |  Steps recorded: {len(records)}")
    print("\nEntering interactive viewer ...\n")

    _interactive_viewer(records, grid)


if __name__ == "__main__":
    main()
