# Technical Challenges

## Initial setup death spiral

The original project trained directly on actively changing, procedurally generated maps, and it never really learned the fundamentals it needed to take advantage of the depth of the maps. The agents tended to just rush the other side.

Non-mirrored self-play caused both teams to gravitate towards the right side as that's where Team A tended to win at.

Turtle strategies before encouraging movement by making every move and end turn slightly negative.

Turtle strategies still happening until combat shaping with giving less negative reward near enemy.

Masked PPO abandoned, MAPPO adopted. Very slow to train.

Poor, turtle formation, but now as a group.

Sharp changed in initial curriculum shocked policy too much, sometimes never recovered during a phase.

Sparser, drip-fed curriculum much more time costly. Still learning when phase changes, and even at the end. Hard to decide when it has absorbed enough.

Abandoned as exploring agent for QD map project as too time costly in favor of leaner, dumber map terrain exploiters.

LLM steering overrides logits at runtime. Is there even anything the RL agent learned that helps this setup? RL as fallback, still turtling. Opponent team learns that attacking high ground is difficult, so they hide in the corner.

## V3 — continuous action / scrolling world

`Grid` was hardcoded to 13×13 with assertions, so the V3 sliding window keeps that fixed shape and rolls columns in-place; chunk generation writes the rightmost column every scroll step. The first attempt at "wider visible window" required forking the Grid class, which would have duplicated the V2 engine.

Continuous (vx, vy) on a tile-based world needed careful coupling: stored sub-tile float positions per unit and only synced rounded `(row, col)` back into the V2 `Unit` object before calls into combat/flank logic that read those fields.

SAC under shared-policy parameter sharing means the "other agents" change between transitions in the replay buffer. Including the agent-id one-hot in the obs was necessary so the off-policy replay didn't blur the three roles together.

V3 diagnostics gap: early SAC runs produced plausible GIFs but not enough
behavioral evidence to explain stalls. Added eval-time action rates, idle rate,
attack intent rate, leftmost progress, squad spread, scroll-lock counts,
lag-lock counts, saliency, experiment summaries, and optional replay-buffer
saving/pruning so failures can be diagnosed without manually watching every
GIF.

Reward shaping for forward progress had to trigger on the **squad centroid**, not per-unit, otherwise the policy could exploit "send the fastest unit ahead, leave the others behind" and rack up progress reward right up until the slow units fell off the left edge.

Off-screen-left death is the primary "keep moving" pressure. Without it the squad learns to camp behind the first ridge and farm enemies indefinitely (a continuous-action analogue of V2's turtling problem).
