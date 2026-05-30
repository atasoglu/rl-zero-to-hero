import random
from collections import defaultdict

import numpy as np


class DynaQAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        n_planning: int = 10,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_planning = n_planning

        self.q_table = np.zeros((n_states, n_actions))
        # model[(s, a)] = list of (s', r) observations
        self.model: dict[tuple, list] = defaultdict(list)
        self._observed: list[tuple] = []  # unique (s, a) pairs seen

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        # Q-learning update
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

        # Model update: store observed transition
        key = (state, action)
        if key not in self.model or not self.model[key]:
            self._observed.append(key)
        self.model[key].append((next_state, reward))

    def plan(self) -> None:
        if not self._observed:
            return
        for _ in range(self.n_planning):
            s, a = random.choice(self._observed)
            s_next, r = random.choice(self.model[(s, a)])
            target = r + self.gamma * np.max(self.q_table[s_next])
            self.q_table[s, a] += self.alpha * (target - self.q_table[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
