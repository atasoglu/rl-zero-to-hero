import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value


class A2CAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.device = torch.device(device)

        self.net = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._entropies: list[torch.Tensor] = []

    def select_action(self, obs: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        self._values.append(value)
        self._entropies.append(dist.entropy())
        return action.item()

    def store_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    def update(self) -> float:
        T = len(self._rewards)
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)
        values_t = torch.stack(self._values).squeeze(-1)
        log_probs_t = torch.stack(self._log_probs)
        entropies_t = torch.stack(self._entropies)

        advantages = returns_t - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = (returns_t - values_t).pow(2).mean()
        entropy_loss = -entropies_t.mean()

        loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._entropies.clear()
        return loss.item()
