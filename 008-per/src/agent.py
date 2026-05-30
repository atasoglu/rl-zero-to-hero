import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from per_buffer import PrioritizedReplayBuffer


class PERAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,
        epsilon_min: float = 0.01,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dim: int = 128,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self._update_count = 0
        self.per_beta = per_beta_start
        self.per_beta_end = per_beta_end

        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_size, obs_dim, alpha=per_alpha)

    def select_action(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.q_net(obs_t).argmax(dim=1).item()

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones, indices, weights = self.buffer.sample(
            self.batch_size, beta=self.per_beta
        )

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions.astype(np.int64).squeeze(-1)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.q_net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs_t).gather(1, best_actions).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        td_errors = (targets - q_values).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # weighted MSE: high-priority samples get lower weight to correct bias
        loss = (weights_t * F.mse_loss(q_values, targets, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def anneal_beta(self, fraction: float) -> None:
        """Linearly anneal beta from beta_start to beta_end over training."""
        self.per_beta = self.per_beta + fraction * (self.per_beta_end - self.per_beta)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
