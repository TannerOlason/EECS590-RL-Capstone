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
