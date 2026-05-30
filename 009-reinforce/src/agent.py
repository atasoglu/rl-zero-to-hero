import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class REINFORCEAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    def select_action(self, obs: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.policy(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    def update(self) -> float:
        """Compute discounted returns and update policy. Call once per episode."""
        T = len(self._rewards)
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)
        # normalize returns for stable gradients
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(self._log_probs)
        loss = -(log_probs_t * returns_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._log_probs.clear()
        self._rewards.clear()
        return loss.item()
