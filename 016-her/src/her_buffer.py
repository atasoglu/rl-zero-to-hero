import numpy as np


class HERReplayBuffer:
    """Replay buffer with Hindsight Experience Replay (future strategy).

    Stores transitions normally; at episode end, adds k relabeled copies
    where the desired_goal is replaced by a future achieved_goal.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, k: int = 4):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.k = k

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0

        # Temporary episode buffer: list of (obs, action, reward, next_obs, done, next_achieved_goal)
        self._episode: list[tuple] = []

    def _push_one(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_obs[self._pos] = next_obs
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_step(self, obs_flat, action, reward, next_obs_flat, done, next_achieved_goal) -> None:
        """Buffer one step of an episode (does not write to main buffer yet)."""
        self._episode.append((obs_flat, action, reward, next_obs_flat, done, next_achieved_goal))

    def finalize_episode(self, obs_size: int, compute_reward_fn) -> None:
        """Flush episode to main buffer with original + hindsight transitions."""
        T = len(self._episode)
        if T == 0:
            return

        achieved_goals = np.array([step[5] for step in self._episode])  # (T, goal_dim)

        for t, (obs_flat, action, reward, next_obs_flat, done, next_ag) in enumerate(self._episode):
            # Original transition
            self._push_one(obs_flat, action, reward, next_obs_flat, done)

            # k hindsight transitions: pick k future indices
            future_indices = np.random.randint(t, T, size=self.k)
            for fi in future_indices:
                new_goal = achieved_goals[fi]
                # Replace goal in obs: last goal_dim elements of obs_flat
                goal_dim = len(new_goal)
                new_obs = obs_flat.copy()
                new_obs[obs_size:obs_size + goal_dim] = new_goal
                new_next_obs = next_obs_flat.copy()
                new_next_obs[obs_size:obs_size + goal_dim] = new_goal
                new_reward = float(compute_reward_fn(next_ag, new_goal, {}))
                new_done = new_reward == 0.0
                self._push_one(new_obs, action, new_reward, new_next_obs, new_done)

        self._episode = []

    def sample(self, batch_size: int):
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
