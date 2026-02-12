import random
from copy import copy
from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np


class RoutedParallelGridworld(ParallelEnv):
    metadata = {
        "name": "routed_parallel_gridworld_v1",
    }

    def __init__(self, grid_size=7, num_agents=2, max_cycles=100, render_mode=None):
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.agents = ["agent1", "agent2"]
        self.action_spaces = {
            agent: spaces.Discrete(4)
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32
            ) for agent in self.agents
        }
        self.agent_positions = {}
        self.agent_goals = {}
        self.cycle_count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.cycle_count = 0
        self.pos = {}
        self.routes = {}
        self.route_id = {}

        for a in self.agents:
            self.pos[a] = np.array([0,0], dtype=np.float32)
            self.routes[a] = [
                np.array([1,1]),
                np.array([8,1]),
                np.array([8,8]),
                np.array([1,8]),
            ]
        observations = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

        # self.cycle_count = 0

        # positions = []
        # while len(positions) < self._num_agents:
        #     # assuming NxM grid with no gaps for now
        #     pos = (np.random.randint(0, self.grid_size),
        #            np.random.randint(0, self.grid_size))
        #     # for now, no overlap, may use overlap to 
        #     # share information between agents later (TODO)
        #     if pos not in positions:
        #         positions.append(pos)

        # goals = []
        # for i in range(self._num_agents):
        #     # set to one initial position to be immediately
        #     # truthy in while that will give random coord
        #     goal = positions[i]
        #     while goal == positions[i] or goal in goals:
        #         goal = (np.random.randint(0, self.grid_size),
        #                np.random.randint(0, self.grid_size))
        #     goals.append(goal)

        # observations = {
        #     agent: np.array(self.agent_positions[agent], dtype=np.float32)
        #     for agent in self.agents
        # }

        # infos = {agent: {} for agent in self.agents}

        # return observations, infos

    def step(self, actions):
        self.cycle_count += 1

        action_deltas = {
            0: (-1,  0), # up
            1: ( 1,  0), # down
            2: ( 0, -1), # left
            3: ( 0,  1)  # right
        }

        for agent in self.agents:
            action = actions[agent]
            dx, dy = action_deltas(action)
            x, y = self.agent_positions[agent]
            # TODO: don't assume square shape of grid
            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)
            self.agent_positions[agent] = (new_x, new_y)
        
        observations = {}
        rewards = {}
        terminations = {} # working with no terminal condition, ignore
        truncations = {}
        infos = {}

        for agent, action in actions.items():
            prev_pos = self.pos[agent].copy()
            self._move(agent, action)

            observations[agent] = np.array(
                self.agent_positions[agent],
                dtype=np.float32
            )

            if self.agent_positions[agent] == self.agent_goals[agent]:
                rewards[agent] = 10.0 # TODO: env vars
            else:
                rewards[agent] = -0.1

            truncations[agent] = self.cycle_count >= self.max_cycles

            goal_x, goal_y = self.agent_goals[agent]
            pos_x, pos_y = self.agent_positions[agent]
            distance = abs(goal_x - pos_x) + abs(goal_y - pos_y)
            infos[agent] = {
                "distance_to_goal": distance,
                "at_goal": terminations[agent]
            }
            terminations[agent] = False # always

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), " ")
        for agent in self.agents:
            grid(self.pos[agent][0], self.pos[agent][1]) = "A"
        print(f"{grid} \n")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]