import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal


class DiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
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
        return Categorical(logits=self.actor(features)), self.critic(features).squeeze(-1)


class ContinuousActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std), self.critic(features).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        continuous: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.continuous = continuous
        self.device = torch.device(device)

        if continuous:
            self.net = ContinuousActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.net = DiscreteActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.net(obs_t)
            action = dist.sample()
        if self.continuous:
            return action.cpu().numpy().squeeze(0), dist.log_prob(action).sum(-1).item(), value.item()
        return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, trajectories: list[dict]) -> None:
        obs = torch.FloatTensor(np.concatenate([t["obs"] for t in trajectories])).to(self.device)
        old_log_probs = torch.FloatTensor(np.concatenate([t["log_probs"] for t in trajectories])).to(self.device)
        rewards = torch.FloatTensor(np.concatenate([t["rewards"] for t in trajectories])).to(self.device)
        dones = torch.FloatTensor(np.concatenate([t["dones"] for t in trajectories])).to(self.device)
        values = torch.FloatTensor(np.concatenate([t["values"] for t in trajectories])).to(self.device)

        if self.continuous:
            actions = torch.FloatTensor(np.concatenate([t["actions"] for t in trajectories])).to(self.device)
        else:
            actions = torch.LongTensor(np.concatenate([t["actions"] for t in trajectories])).to(self.device)

        # GAE
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(obs)
        for _ in range(self.n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                mb = idx[start:start + self.batch_size]
                dist, vals = self.net(obs[mb])
                if self.continuous:
                    log_probs = dist.log_prob(actions[mb]).sum(-1)
                    entropy = dist.entropy().sum(-1)
                else:
                    log_probs = dist.log_prob(actions[mb])
                    entropy = dist.entropy()

                ratio = torch.exp(log_probs - old_log_probs[mb])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * advantages[mb], clipped * advantages[mb]).mean()
                critic_loss = F.mse_loss(vals, returns[mb].detach())
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()
