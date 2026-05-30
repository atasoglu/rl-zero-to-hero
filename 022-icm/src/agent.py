import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import AtariActorCritic, ICMNetwork


class ICMAgent:
    def __init__(
        self,
        n_actions: int,
        feature_dim: int = 512,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.1,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
        icm_beta: float = 0.2,       # weight for forward vs inverse loss
        icm_eta: float = 0.01,       # intrinsic reward scale
        icm_lambda: float = 0.1,     # weight of ICM loss vs policy loss
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.icm_beta = icm_beta
        self.icm_eta = icm_eta
        self.icm_lambda = icm_lambda
        self.device = torch.device(device)

        self.ac = AtariActorCritic(n_actions, feature_dim).to(self.device)
        self.icm = ICMNetwork(n_actions, feature_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.ac.parameters()) + list(self.icm.parameters()), lr=lr
        )

    def select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.ac(obs_t)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def compute_intrinsic_rewards(self, obs: np.ndarray, actions: np.ndarray,
                                  next_obs: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        with torch.no_grad():
            phi_s = self.icm.encode(obs_t)
            phi_s_next = self.icm.encode(next_obs_t)
            _, r_int = self.icm.forward_loss_and_reward(phi_s, actions_t, phi_s_next)
        return r_int.cpu().numpy() * self.icm_eta

    def update(self, rollout: dict) -> None:
        obs = torch.FloatTensor(rollout["obs"]).to(self.device)
        next_obs = torch.FloatTensor(rollout["next_obs"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        rewards = torch.FloatTensor(rollout["total_rewards"]).to(self.device)
        dones = torch.FloatTensor(rollout["dones"]).to(self.device)
        values = torch.FloatTensor(rollout["values"]).to(self.device)

        # GAE advantages
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

                # PPO loss
                dist, vals = self.ac(obs[mb])
                log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy()
                ratio = torch.exp(log_probs - old_log_probs[mb])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * advantages[mb], clipped * advantages[mb]).mean()
                critic_loss = F.mse_loss(vals, returns[mb].detach())
                ppo_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy.mean()

                # ICM loss
                phi_s = self.icm.encode(obs[mb])
                phi_s_next = self.icm.encode(next_obs[mb])
                inv_loss = self.icm.inverse_loss(phi_s, phi_s_next, actions[mb])
                fwd_loss, _ = self.icm.forward_loss_and_reward(phi_s, actions[mb], phi_s_next)
                icm_loss = (1 - self.icm_beta) * inv_loss + self.icm_beta * fwd_loss

                loss = (1 - self.icm_lambda) * ppo_loss + self.icm_lambda * icm_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.ac.parameters()) + list(self.icm.parameters()), max_norm=0.5
                )
                self.optimizer.step()
