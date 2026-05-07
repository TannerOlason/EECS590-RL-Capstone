# EECS590 - Final - V3

We'll start with an introduction to the original purpose of this project and why 
it veered so off-base from the original premise. Initially, I wanted this project 
to be an evaluation agent for a procedural map generator for an strategy role 
playing game (SRPG). The game space was originally 11x11, bumped up to 13x13 later.
Game tiles can be one of three types, regular, rough, or impassable. They can also 
be one of three heights, 0, 1, or 2. Two teams are meant to battle with three class
units each. Each class has their own way of taking advantage of the terrain. The 
goal of this project was to evaluate maps for the depth of gameplay a map's terrain
could provide in optimal play settings. Evaluation scores of the map would be 
separate from the RL agents' reward scores.

I started with V1, a simple pettingzoo grid map navigator agent. It worked okay.
Initial intentions were to have two agents reach a rotating goal, then introduce
obstacles, but I don't think I managed to make the obstacles work very well in 
the game engine.

I moved on to a few different iterations of MARL implementations for 
actually evaluating the usefulness of maps.

The first attempt of V2 was doomed to fail from the start. I set it up to adjust the procedural
map generator's settings after some amount of evaluations. This resulted in an obvious
death spiral of behavioral gaming at both ends. Bad idea. A lot of maps just had a 
central area. It didn't matter at all that I was using BenchMARL. Optimal play wouldn't
help the useless setup. 

The second attempt of V2 was a MAPPO implementation that eventually worked well enough
after many, many passes of training. I had staggered self-play, meaning old checkpoints 
versus new checkpoints to ensure that self-play wouldn't fall into some weird behavioral 
corner. Here is where I learned some pretty rudimentary things such 
as the effects of not mirroring agent map interpretations during self-play, curriculum 
for basic encounters (1v1s, placing spawns closer, navigation, static maps) could speed things along, 
and the basic cost of wall time when it comes to setting experiments up. It was very 
frustrating. I masked logits onto the RL agent to better evaluate specific parts 
of maps in the game's context in order to fulfill the role I wanted it to in the 
context of the map generator project. The mappo_phase7_policy.pt was the final 
model used for this. It tended to not approach enemies on advantageous terrain, and
since I was already controlling the agents using a strategic intent navigation layer,
I removed it from the overall map generator evaluation project.

This version lives in the highground/ directory, split into the game engine at 
engine/, the wrapping environment layer at env/srpg_env.py, and the the training 
layer using BenchMARL at training/benchmarl_train.py. Our game engine defined 
the state of the game, helped with finding legal actions for the RL agent, and 
kept things loosely coupled. The wrapping environment allowed us to switch 
things over quicker. Our training script allowed for passing specific parameters
in each experiment. Experiments were labeled by date and dumped in experiments/

The MAPPO setup worked well because the problem was discrete, turn-based, and had a
multi-agent approach. Our critic handled coordination and credit assignment, while 
PPO gave stable on-policy updates. Stability was the main problem we had previously, 
and this paradigm helped keep things on track even with some logical flaws that 
still hadn't been picked up yet.

V2 also wanted a classical method attached, so I had a SARSA example thrown together.

The final version, V3, was an attempt at having an agent play in a continuous space, 
one where the enemies were weaker, the map stretched on forever, and there were 
ways to heal units that lost health points in battle. The 13x13 map would continue on 
to the right when a player's units were not at risk of being left behind by the map 
renderer and all enemy units were defeated. I moved the project to Soft Actor Critic 
via stable-baselines3 on a single-agent Gym wrapper with internal unit rotation
because it was relatively simple to implement, then began finding behavioral bugs 
and fixing them. This change from on-policy, discrete, fixed-map, and multi-agent 
to off-policy, continuous action, scrolling-world SAC setup went smoother than 
previous jumps of RL methodology. The curriculum from V2 was useful here, and the
diagnoses of our replay.gif showed some pretty obvious behavior patterns.

## CNN in V2 and V3

A CNN was found useful in this very spatially-dependent map-based setup. V2 had 
13x13 tiles with various tiles types, unit location, elevation, and enemies. Having 
a simple vector representation of this could still allow our RL agents to learn, but
having the map represented as a CNN model, the patterns are easier for the agents to 
learn from, requiring less overhead to put together spatial relations. With the engine
for V3 having a similar 13x13 grid to read from, it made sense to bring this feature
over.

## Replays

The replays/ folder contains some gifs of the top models for each version.

## Final Remarks

While I had high hopes for this project as a map generator evalution tool, it came 
up short, and taught me a bunch about reinforcement learning. Reward structuring 
and observation design can go a long ways to make any RL agent perform well enough.
The tools and algorithms chosen allowed me to go quickly from the agents simply 
not learning to the agents simply learning shortcuts and loopholes. Overall, an
interesting experience, and a topic I'd like to study further.

## As aside on the use of LLMs/coding agents

The single dumbest thing I did while working on this project was using these while 
attempting to learn. They're very confident, like all your ideas, believe any hunch 
you have, have no concept of the frustration of time wasted, and constantly misinterpret
what you write. They're a tool, and a useful one, but they're useful for iterating 
fast. Which is not a good quality for actually learning. I regret not choosing a 
much simpler task, though I thought it was simple enough at the beginning of the 
semester, to work on this project so I wouldn't have face the 3 hour walls of waiting
for an agent to train to see how it messed up this time. I suppose that's part of the 
learning process, but it does put a remarkable pressure on getting things fixed fast 
and training back up and running. I plan to slow down and take more time with future 
RL projects.

There is also the problem of agent models changing or being "quantized" as they 
are expensive services, slowly being pushed towards profits. Claude turned into 
a useless tool very fast during this semester. My $20 subscription became useless 
as the prompts would fill my 5 hour usage quota very fast near the end. Codex was 
able to keep things on track, but the idea of needing to switch between services 
mid-project, or opening a chat that does not have the context of the whole project
can lead to frustration.

All LLM services used:
Grok, ChatGPT - for theorycrafting ideas/asking advice on implementations
Claude Code, Codex - for implementing code, helping interpret diagnostic outputs