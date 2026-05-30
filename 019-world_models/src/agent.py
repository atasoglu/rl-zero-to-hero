import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Controller, MDNRNN, VAE


class WorldModelAgent:
    def __init__(
        self,
        action_dim: int,
        z_dim: int = 32,
        h_dim: int = 256,
        n_gaussians: int = 5,
        vae_lr: float = 1e-3,
        rnn_lr: float = 1e-3,
        ctrl_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.vae = VAE(z_dim).to(self.device)
        self.rnn = MDNRNN(z_dim, action_dim, h_dim, n_gaussians).to(self.device)
        self.controller = Controller(z_dim, h_dim, action_dim).to(self.device)

        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=vae_lr)
        self.rnn_optimizer = optim.Adam(self.rnn.parameters(), lr=rnn_lr)
        self.ctrl_optimizer = optim.Adam(self.controller.parameters(), lr=ctrl_lr)

    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """frame: (H, W, 3) uint8 → (z, z_mu)"""
        x = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        if x.shape[-2:] != (64, 64):
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        mu, log_var = self.vae.encode(x)
        z = self.vae.reparameterize(mu, log_var)
        return z.squeeze(0), mu.squeeze(0)

    def select_action(self, z: torch.Tensor, h: torch.Tensor) -> tuple[int, float, float]:
        with torch.no_grad():
            dist, value = self.controller(z.unsqueeze(0), h.unsqueeze(0))
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def init_hidden(self) -> torch.Tensor:
        return self.rnn.init_hidden(1, self.device).squeeze(0)

    @torch.no_grad()
    def step_rnn(self, z: torch.Tensor, action: int, h: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(torch.tensor([action], device=self.device), self.action_dim).float()
        _, _, _, _, h_next = self.rnn(z.unsqueeze(0), a_onehot, h.unsqueeze(0))
        return h_next.squeeze(0)

    def train_vae(self, frames: np.ndarray, n_epochs: int = 5) -> float:
        """frames: (N, H, W, 3) uint8"""
        x = torch.FloatTensor(frames).permute(0, 3, 1, 2).to(self.device) / 255.0
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        total_loss = 0.0
        for _ in range(n_epochs):
            idx = torch.randperm(len(x))
            for start in range(0, len(x), self.batch_size):
                mb = idx[start:start + self.batch_size]
                recon, mu, log_var = self.vae(x[mb])
                loss = self.vae.loss(x[mb], recon, mu, log_var)
                self.vae_optimizer.zero_grad()
                loss.backward()
                self.vae_optimizer.step()
                total_loss += loss.item()
        return total_loss

    def train_rnn(self, episodes_z: list, episodes_a: list, n_epochs: int = 5) -> float:
        """episodes_z: list of (T, z_dim) tensors; episodes_a: list of (T,) int arrays"""
        total_loss = 0.0
        for _ in range(n_epochs):
            for zs, acts in zip(episodes_z, episodes_a):
                if len(zs) < 2:
                    continue
                z_seq = torch.FloatTensor(zs).to(self.device)       # (T, z_dim)
                a_seq = torch.LongTensor(acts).to(self.device)       # (T,)
                h = self.rnn.init_hidden(1, self.device)

                loss = torch.tensor(0.0, device=self.device)
                for t in range(len(z_seq) - 1):
                    a_oh = F.one_hot(a_seq[t:t+1], self.action_dim).float()
                    pi_logits, mu, sigma, _, h = self.rnn(z_seq[t:t+1], a_oh, h)
                    loss += self.rnn.loss(pi_logits, mu, sigma, z_seq[t+1:t+2])

                self.rnn_optimizer.zero_grad()
                (loss / max(len(z_seq) - 1, 1)).backward()
                nn.utils.clip_grad_norm_(self.rnn.parameters(), 1.0)
                self.rnn_optimizer.step()
                total_loss += loss.item()
        return total_loss

    def update_controller(self, rollout: dict) -> None:
        """PPO update for controller using latent space rollout."""
        z = torch.FloatTensor(rollout["z"]).to(self.device)
        h = torch.FloatTensor(rollout["h"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        rewards = torch.FloatTensor(rollout["rewards"]).to(self.device)
        dones = torch.FloatTensor(rollout["dones"]).to(self.device)
        values = torch.FloatTensor(rollout["values"]).to(self.device)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1].item() if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(z)
        for _ in range(self.n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                mb = idx[start:start + self.batch_size]
                dist, vals = self.controller(z[mb], h[mb])
                log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy()
                ratio = torch.exp(log_probs - old_log_probs[mb])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * advantages[mb], clipped * advantages[mb]).mean()
                critic_loss = F.mse_loss(vals, returns[mb].detach())
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
                self.ctrl_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
                self.ctrl_optimizer.step()
