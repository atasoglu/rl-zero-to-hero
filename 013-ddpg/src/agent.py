import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_common import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_scale: float, hidden_dim: int = 256):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * self.action_scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        noise_sigma: float = 0.2,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_scale = action_scale
        self.device = torch.device(device)

        self.actor = Actor(obs_dim, action_dim, action_scale, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)
        self.noise = OUNoise(action_dim, sigma=noise_sigma)

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if explore:
            action += self.noise.sample()
        return np.clip(action, -self.action_scale, self.action_scale)

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
            next_actions = self.actor_target(next_obs_t)
            target_q = rewards_t + self.gamma * self.critic_target(next_obs_t, next_actions) * (1 - dones_t)
        critic_loss = F.mse_loss(self.critic(obs_t, actions_t), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(obs_t, self.actor(obs_t)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_noise(self) -> None:
        self.noise.reset()
