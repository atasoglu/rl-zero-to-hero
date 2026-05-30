import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int | tuple, action_dim: int = 1):
        self.capacity = capacity
        obs_shape = (obs_dim,) if isinstance(obs_dim, int) else obs_dim
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def push(self, obs: np.ndarray, action, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_obs[self._pos] = next_obs
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size
