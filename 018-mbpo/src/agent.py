import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from model import EnsembleDynamicsModel
from rl_common import ReplayBuffer

LOG_STD_MIN, LOG_STD_MAX = -5, 2


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
        f = self.net(x)
        mean = self.mean_layer(f)
        log_std = torch.clamp(self.log_std_layer(f), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        x = Normal(mean, std).rsample()
        y = torch.tanh(x)
        action = y * self.action_scale
        log_prob = Normal(mean, std).log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1)


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


class MBPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float,
        lr: float = 3e-4,
        model_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        real_buffer_size: int = 100_000,
        model_buffer_size: int = 400_000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        n_model_members: int = 5,
        rollout_length: int = 1,
        rollout_batch_size: int = 400,
        model_train_freq: int = 250,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.model_train_freq = model_train_freq
        self.device = torch.device(device)
        self._step = 0

        self.actor = Actor(obs_dim, action_dim, action_scale, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.dynamics = EnsembleDynamicsModel(obs_dim, action_dim, n_members=n_model_members).to(self.device)
        self.model_optimizer = optim.Adam(self.dynamics.parameters(), lr=model_lr)

        self.real_buffer = ReplayBuffer(real_buffer_size, obs_dim, action_dim)
        self.model_buffer = ReplayBuffer(model_buffer_size, obs_dim, action_dim)

        if auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
            self.log_alpha = None

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
        self.real_buffer.push(obs, action, reward, next_obs, done)
        self._step += 1

    def _train_model(self) -> None:
        if len(self.real_buffer) < 1000:
            return
        for _ in range(50):
            obs, actions, rewards, next_obs, _ = self.real_buffer.sample(min(len(self.real_buffer), 256))
            obs_t = torch.FloatTensor(obs).to(self.device)
            act_t = torch.FloatTensor(actions).to(self.device)
            rew_t = torch.FloatTensor(rewards).to(self.device)
            next_t = torch.FloatTensor(next_obs).to(self.device)
            loss = self.dynamics.loss(obs_t, act_t, next_t, rew_t)
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

    def _rollout_model(self) -> None:
        if len(self.real_buffer) < self.rollout_batch_size:
            return
        obs, _, _, _, _ = self.real_buffer.sample(self.rollout_batch_size)
        obs_t = torch.FloatTensor(obs).to(self.device)
        for _ in range(self.rollout_length):
            with torch.no_grad():
                action, _ = self.actor.sample(obs_t)
            next_obs_t, reward_t = self.dynamics.sample(obs_t, action)
            for i in range(len(obs_t)):
                self.model_buffer.push(
                    obs_t[i].cpu().numpy(), action[i].cpu().numpy(),
                    reward_t[i].item(), next_obs_t[i].cpu().numpy(), False
                )
            obs_t = next_obs_t

    def _sac_update(self) -> None:
        real_n = self.batch_size // 5
        model_n = self.batch_size - real_n
        if len(self.real_buffer) < real_n or len(self.model_buffer) < model_n:
            return

        obs_r, act_r, rew_r, nobs_r, don_r = self.real_buffer.sample(real_n)
        obs_m, act_m, rew_m, nobs_m, don_m = self.model_buffer.sample(model_n)

        obs = torch.FloatTensor(np.concatenate([obs_r, obs_m])).to(self.device)
        actions = torch.FloatTensor(np.concatenate([act_r, act_m])).to(self.device)
        rewards = torch.FloatTensor(np.concatenate([rew_r, rew_m])).to(self.device)
        next_obs = torch.FloatTensor(np.concatenate([nobs_r, nobs_m])).to(self.device)
        dones = torch.FloatTensor(np.concatenate([don_r, don_m])).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_actions)
            target_q = rewards + self.gamma * (torch.min(q1_t, q2_t) - self.alpha * next_log_probs) * (1 - dones)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(obs)
        q1, q2 = self.critic(obs, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self) -> None:
        if self._step % self.model_train_freq == 0:
            self._train_model()
            self._rollout_model()
        self._sac_update()
