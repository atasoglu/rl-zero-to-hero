from collections import defaultdict

import numpy as np


class MonteCarloAgent:
    """First-visit MC control with ε-greedy policy."""

    def __init__(
        self,
        n_actions: int,
        gamma: float = 1.0,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q and visit counts stored as dicts — Blackjack state is a tuple
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))
        self.returns_count: dict = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, episode: list[tuple]):
        """Update Q-table from a completed episode using first-visit MC."""
        visited = set()
        G = 0.0
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns_count[state][action] += 1
                n = self.returns_count[state][action]
                # Incremental mean update
                self.q_table[state][action] += (G - self.q_table[state][action]) / n

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
