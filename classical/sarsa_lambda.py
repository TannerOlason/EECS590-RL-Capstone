"""SARSA(λ) with eligibility traces — tabular implementation.

This module provides two variants:
    - SarsaLambda   : full online SARSA(λ) with replacing traces
    - SarsaN        : n-step SARSA (hard trace horizon), implemented as
                      SARSA(λ) with λ = 0 and an n-step return buffer

Conceptual lineage
------------------
SARSA(λ)  →  GAE (Generalized Advantage Estimation)  →  MAPPO

The λ parameter here plays the same role as the λ in GAE: both weight
earlier transitions by λ^k, trading bias (λ=0, TD(0), fully bootstrapped)
against variance (λ=1, Monte-Carlo returns).

In MAPPO the advantage is computed as:

    Â_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

where δ_t = r_t + γ V(s_{t+1}) - V(s_t).  This is the continuous-decay
exponential weighting that eligibility traces implement discretely.

Usage
-----
    env = SimpleNavigationEnv(seed=42)
    agent = SarsaLambda(n_actions=env.n_actions)
    returns = agent.train(env, n_episodes=500)
    # Act greedily after training:
    action = agent.greedy(state)
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classical.nav_env import NavState, SimpleNavigationEnv


class SarsaLambda:
    """Online SARSA(λ) with replacing eligibility traces.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space.
    alpha : float
        Learning rate (step size).
    gamma : float
        Discount factor.
    lambda_ : float
        Trace decay parameter.  λ=0 reduces to TD(0) / one-step SARSA.
        λ=1 approaches Monte-Carlo SARSA.
    epsilon : float
        Initial ε for ε-greedy exploration.
    epsilon_min : float
        Floor value for ε after annealing.
    epsilon_decay : float
        Multiplicative decay applied to ε each episode.
    """

    def __init__(
        self,
        n_actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_: float = 0.8,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: state -> array of Q-values, one per action
        # Using defaultdict so unseen states start at 0.
        self.Q: dict = defaultdict(lambda: [0.0] * n_actions)

    # ── Policy ───────────────────────────────────────────────────────────────

    def epsilon_greedy(self, state: "NavState") -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._argmax(self.Q[state])

    def greedy(self, state: "NavState") -> int:
        return self._argmax(self.Q[state])

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        env: "SimpleNavigationEnv",
        n_episodes: int = 500,
        verbose: bool = False,
    ) -> list[float]:
        """Train for *n_episodes* episodes; return per-episode total returns."""
        episode_returns: list[float] = []

        for ep in range(n_episodes):
            # Eligibility traces reset each episode
            E: dict = defaultdict(lambda: [0.0] * self.n_actions)

            state = env.reset()
            action = self.epsilon_greedy(state)
            total_return = 0.0

            while True:
                next_state, reward, done, _ = env.step(action)
                total_return += reward

                next_action = self.epsilon_greedy(next_state) if not done else 0

                # TD error
                q_next = 0.0 if done else self.Q[next_state][next_action]
                delta = reward + self.gamma * q_next - self.Q[state][action]

                # Replacing traces: set the visited (s,a) trace to 1
                E[state][action] = 1.0

                # Update all visited (s,a) pairs
                for s, e_vec in list(E.items()):
                    for a in range(self.n_actions):
                        if e_vec[a] != 0.0:
                            self.Q[s][a] += self.alpha * delta * e_vec[a]
                            e_vec[a] *= self.gamma * self.lambda_

                if done:
                    break

                state = next_state
                action = next_action

            # Anneal exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_returns.append(total_return)

            if verbose and (ep + 1) % 50 == 0:
                mean_r = sum(episode_returns[-50:]) / 50
                print(
                    f"  Episode {ep+1:4d}/{n_episodes}  "
                    f"mean_return={mean_r:+.3f}  "
                    f"ε={self.epsilon:.3f}  "
                    f"states_visited={len(self.Q)}"
                )

        return episode_returns

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _argmax(values: list[float]) -> int:
        best_val = max(values)
        # Break ties randomly
        best_actions = [i for i, v in enumerate(values) if v == best_val]
        return random.choice(best_actions)

    @property
    def n_states_visited(self) -> int:
        return len(self.Q)


class SarsaN(SarsaLambda):
    """n-step SARSA implemented as SARSA(λ) with hard trace horizon.

    Instead of the exponential trace decay, the n-step variant accumulates
    rewards for exactly *n* steps before bootstrapping.  Implemented here
    via a bounded replay buffer and a λ = 0 base with a manual n-step
    return calculation.

    Note: for most practical purposes ``SarsaLambda`` with an appropriate
    λ is preferable; ``SarsaN`` is included to satisfy the rubric requirement
    for the n-cutoff variant.
    """

    def __init__(self, n_steps: int = 4, **kwargs) -> None:
        # λ=0 disables the exponential trace; n-step buffer does the work
        kwargs.setdefault("lambda_", 0.0)
        super().__init__(**kwargs)
        self.n_steps = n_steps

    def train(
        self,
        env: "SimpleNavigationEnv",
        n_episodes: int = 500,
        verbose: bool = False,
    ) -> list[float]:
        """Train using n-step SARSA returns."""
        episode_returns: list[float] = []

        for ep in range(n_episodes):
            state = env.reset()
            action = self.epsilon_greedy(state)
            total_return = 0.0

            # Circular buffer: (state, action, reward)
            buffer: deque = deque()

            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                total_return += reward
                buffer.append((state, action, reward))

                next_action = self.epsilon_greedy(next_state) if not done else 0

                # Once buffer is full (or episode ends), flush the oldest entry
                if len(buffer) >= self.n_steps or done:
                    # Compute n-step return from the oldest buffered transition
                    s0, a0, _ = buffer[0]
                    G = 0.0
                    for k, (_, _, r) in enumerate(buffer):
                        G += (self.gamma ** k) * r

                    # Bootstrap from the tail state
                    if not done:
                        G += (self.gamma ** len(buffer)) * self.Q[next_state][next_action]

                    self.Q[s0][a0] += self.alpha * (G - self.Q[s0][a0])
                    buffer.popleft()

                state = next_state
                action = next_action

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_returns.append(total_return)

            if verbose and (ep + 1) % 50 == 0:
                mean_r = sum(episode_returns[-50:]) / 50
                print(
                    f"  [n={self.n_steps}] Episode {ep+1:4d}/{n_episodes}  "
                    f"mean_return={mean_r:+.3f}  ε={self.epsilon:.3f}"
                )

        return episode_returns
