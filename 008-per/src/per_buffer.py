import numpy as np


class SumTree:
    """Binary sum tree for O(log n) priority updates and O(log n) stratified sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._pos = 0
        self._size = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float) -> None:
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float) -> int:
        leaf_idx = self._pos + self.capacity - 1
        self.update(leaf_idx, priority)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return leaf_idx

    def get(self, value: float) -> tuple[int, float]:
        """Return (leaf_idx, priority) for cumulative priority value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = 2 * idx + 2
        return idx, self.tree[idx]

    @property
    def total(self) -> float:
        return self.tree[0]

    def __len__(self) -> int:
        return self._size


class PrioritizedReplayBuffer:
    """Proportional PER with importance sampling weights."""

    def __init__(self, capacity: int, obs_dim: int | tuple, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)

        obs_shape = (obs_dim,) if isinstance(obs_dim, int) else obs_dim
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._max_priority = 1.0

    def push(self, obs, action, reward, next_obs, done) -> None:
        pos = self.tree._pos
        self.obs[pos] = obs
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.next_obs[pos] = next_obs
        self.dones[pos] = float(done)
        self.tree.add(self._max_priority ** self.alpha)

    def sample(self, batch_size: int, beta: float = 0.4):
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority = self.tree.get(val)
            data_idx = idx - (self.tree.capacity - 1)
            indices.append(data_idx)
            priorities.append(priority)

        probs = np.array(priorities) / self.tree.total
        weights = (len(self.tree) * probs) ** (-beta)
        weights /= weights.max()

        idx_arr = np.array(indices)
        return (
            self.obs[idx_arr],
            self.actions[idx_arr],
            self.rewards[idx_arr],
            self.next_obs[idx_arr],
            self.dones[idx_arr],
            np.array(indices),
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, priority in zip(indices, priorities):
            leaf_idx = idx + self.tree.capacity - 1
            self.tree.update(leaf_idx, float(priority))
        self._max_priority = max(self._max_priority, priorities.max())

    def __len__(self) -> int:
        return len(self.tree)
