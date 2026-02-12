# Dynamic Reinforcement Learning in Gridworld Settings

The goal of this project is to develop and study RL algorithms 
in a changing gridworld environment. The end goal of this project
is to create a framework that trains NPCs to adjust their schedules
and actions to changes in the environment caused by the player's 
interactions with the world. These actions can range from a player
destroying a road's bridge or taking on a bandit role and causing 
NPCs to avoid the area they do their robbing in.

Agents should have features that conform to a world model of a 
medieval fantasy game. This includes socializing to share information 
on state space and updating their routes and routines. As NPCs aren't 
often uniform, multiple values will have to be assigned per grid square 
that represent different properties. Simply, danger level will matter 
less to a knight than to a trader, and a destroyed bridge will matter 
the same to everyone.

The overall goal of this project will be to use it as an adversarial 
AI to a procedural generation AI in order to produce functional, but 
interesting gridworld environments.

## Impelementation

This project will use gymnasium and pettingzoo for their ability 
to run parallel agents in an environment where every agent does 
their turn at the same time.

A routine will be established as a continuing MDP for each agent. 
Obstructions to their paths will be introduced, and the agents 
will have to adapt their routines. 

### Markov Decision Process

This is a continueing MDP, so state is defined by `s = (x, y, k)` 
where k is the route index. and x and y are the coords. 

Transitions are "if waypoint reached, `k' = (k + 1) mod 4` 
otherwise `k' = k`.

Rewards will be shaped by progress towards the current route goal 
with the largest reward received when each goal is reached.

### Dynamic Programming



### Agent Framework

Agents have a route of four coordinates to follow. They are rewarded 
when they reach the current goal coord, then the goal coord 
is changed to the next coord in the route. Their actions are simply 
move up, down, right or left. For now, there is no benefit to staying 
in one place, so it is not included.

## Install and Run

Windows

Activate the venv environment

`.\venv\Scripts\activate`

Install python libraries

`pip install -r requirements.txt`

Linux

Activate the venv environment

`source venv/vin/activate`

Install python libraries

`pip install -r requirements.txt`

Run on either

`python run.py`

Run with human output

`python run_human.py`
