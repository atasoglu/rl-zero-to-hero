import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from rl_common import ReplayBuffer

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_scale: float, hidden_dim: int = 256):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale
        # Enforcing action bounds: log_prob correction for tanh squashing
        log_prob = dist.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_tune_alpha = auto_tune_alpha
        self.device = torch.device(device)

        self.actor = Actor(obs_dim, action_dim, action_scale, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        # Automatic temperature tuning
        if auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                action, _ = self.actor.sample(obs_t)
            else:
                mean, _ = self.actor(obs_t)
                action = torch.tanh(mean) * self.actor.action_scale
        return action.cpu().numpy()[0]

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs_t)
            q1_target, q2_target = self.critic_target(next_obs_t, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards_t + self.gamma * min_q_target * (1 - dones_t)

        q1, q2 = self.critic(obs_t, actions_t)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(obs_t)
        q1, q2 = self.critic(obs_t, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (temperature) update
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
