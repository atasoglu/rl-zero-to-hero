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
        # Twin Q-networks in a single module
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

    def q1_forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1)


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        explore_noise: float = 0.1,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.action_scale = action_scale
        self.explore_noise = explore_noise
        self.device = torch.device(device)
        self._update_count = 0

        self.actor = Actor(obs_dim, action_dim, action_scale, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if explore:
            noise = np.random.normal(0, self.explore_noise, size=action.shape)
            action = np.clip(action + noise, -self.action_scale, self.action_scale)
        return action

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

        # Target policy smoothing: add clipped noise to target actions
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(actions_t) * self.policy_noise,
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(
                self.actor_target(next_obs_t) + noise,
                -self.action_scale, self.action_scale
            )
            q1_target, q2_target = self.critic_target(next_obs_t, next_actions)
            target_q = rewards_t + self.gamma * torch.min(q1_target, q2_target) * (1 - dones_t)

        # Critic update
        q1, q2 = self.critic(obs_t, actions_t)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._update_count += 1
        # Delayed actor and target updates
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(obs_t, self.actor(obs_t)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
