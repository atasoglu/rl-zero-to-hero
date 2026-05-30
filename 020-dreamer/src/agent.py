import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from model import ActorCritic, WorldModel
from rl_common import ReplayBuffer


class DreamerAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float,
        h_dim: int = 200,
        z_dim: int = 30,
        lr_model: float = 6e-4,
        lr_actor: float = 8e-5,
        lr_critic: float = 8e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        imagine_horizon: int = 15,
        buffer_size: int = 100_000,
        batch_size: int = 50,
        seq_len: int = 50,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.imagine_horizon = imagine_horizon
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device)

        self.world_model = WorldModel(obs_dim, action_dim, h_dim, z_dim).to(self.device)
        self.ac = ActorCritic(h_dim, z_dim, action_dim).to(self.device)

        self.model_optimizer = optim.Adam(self.world_model.parameters(), lr=lr_model)
        self.actor_optimizer = optim.Adam(self.ac.actor_net.parameters(), lr=lr_actor)
        # Also include actor_mean and actor_log_std
        self.actor_optimizer = optim.Adam(
            list(self.ac.actor_net.parameters()) +
            list(self.ac.actor_mean.parameters()) +
            [self.ac.actor_log_std], lr=lr_actor
        )
        self.critic_optimizer = optim.Adam(self.ac.critic_net.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)
        self._h = torch.zeros(1, h_dim, device=self.device)
        self._z = torch.zeros(1, z_dim, device=self.device)

    def reset_state(self) -> None:
        self._h = torch.zeros(1, self.h_dim, device=self.device)
        self._z = torch.zeros(1, self.z_dim, device=self.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action and advance recurrent state using posterior (real obs)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        # 1. Update state with current observation (posterior)
        self._h, self._z, _, _ = self.world_model.rssm.step(
            self._h, self._z,
            torch.zeros(1, self.action_dim, device=self.device),
            obs_t,
        )
        # 2. Select action from updated state
        if explore:
            action, _, _ = self.ac.sample_action(self._h, self._z)
        else:
            dist, _ = self.ac(self._h, self._z)
            action = torch.tanh(dist.mean)
        return (action * self.action_scale).cpu().numpy()[0]

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action / self.action_scale, float(reward), next_obs, float(done))

    def world_model_update(self) -> None:
        """Train world model on sequences from buffer."""
        min_needed = self.batch_size * self.seq_len
        if len(self.buffer) < min_needed:
            return

        # Sample random starting indices for sequences
        obs_all = self.buffer.obs[:len(self.buffer)]
        act_all = self.buffer.actions[:len(self.buffer)]
        rew_all = self.buffer.rewards[:len(self.buffer)]
        don_all = self.buffer.dones[:len(self.buffer)]

        N = len(self.buffer) - self.seq_len
        if N <= 0:
            return
        starts = np.random.randint(0, N, size=self.batch_size)

        obs_seq = torch.FloatTensor(
            np.stack([obs_all[s:s + self.seq_len] for s in starts])
        ).to(self.device)
        act_seq = torch.FloatTensor(
            np.stack([act_all[s:s + self.seq_len] for s in starts])
        ).to(self.device)
        rew_seq = torch.FloatTensor(
            np.stack([rew_all[s:s + self.seq_len] for s in starts])
        ).to(self.device)
        don_seq = torch.FloatTensor(
            np.stack([don_all[s:s + self.seq_len] for s in starts])
        ).to(self.device)

        # Roll out RSSM
        h, z = self.world_model.rssm.initial_state(self.batch_size, self.device)
        kl_loss = torch.tensor(0.0, device=self.device)
        recon_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)

        for t in range(self.seq_len):
            h, z, prior, post = self.world_model.rssm.step(
                h, z, act_seq[:, t], obs_seq[:, t]
            )
            kl = torch.distributions.kl_divergence(post, prior).sum(-1).mean()
            kl_loss += torch.clamp(kl, min=1.0)
            recon_loss += F.mse_loss(self.world_model.decode(h, z), obs_seq[:, t])
            reward_loss += F.mse_loss(self.world_model.reward(h, z), rew_seq[:, t])

        T = self.seq_len
        total_loss = recon_loss / T + reward_loss / T + 0.1 * kl_loss / T
        self.model_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.model_optimizer.step()

    def actor_critic_update(self) -> None:
        """Train actor-critic via latent imagination."""
        min_needed = self.batch_size * self.seq_len
        if len(self.buffer) < min_needed:
            return

        # Start imagination from random latent states
        obs_all = self.buffer.obs[:len(self.buffer)]
        act_all = self.buffer.actions[:len(self.buffer)]
        N = len(self.buffer) - self.seq_len
        if N <= 0:
            return
        starts = np.random.randint(0, N, size=self.batch_size)

        # Warm up hidden state for batch_size sequences
        h, z = self.world_model.rssm.initial_state(self.batch_size, self.device)
        with torch.no_grad():
            warmup = min(10, self.seq_len)
            obs_seq = torch.FloatTensor(
                np.stack([obs_all[s:s + warmup] for s in starts])
            ).to(self.device)
            act_seq = torch.FloatTensor(
                np.stack([act_all[s:s + warmup] for s in starts])
            ).to(self.device)
            for t in range(warmup):
                h, z, _, _ = self.world_model.rssm.step(h, z, act_seq[:, t], obs_seq[:, t])
        h, z = h.detach(), z.detach()

        # Imagine trajectories
        imagined_h, imagined_z, imagined_rewards, imagined_cont = [], [], [], []
        log_probs = []
        for _ in range(self.imagine_horizon):
            action, log_prob, _ = self.ac.sample_action(h, z)
            with torch.no_grad():
                h, z, _, _ = self.world_model.rssm.step(h, z, action)
                reward = self.world_model.reward(h, z)
                cont = self.world_model.continue_prob(h, z)
            imagined_h.append(h)
            imagined_z.append(z)
            imagined_rewards.append(reward)
            imagined_cont.append(cont)
            log_probs.append(log_prob)

        # Compute λ-returns
        _, values = self.ac(
            torch.stack(imagined_h), torch.stack(imagined_z)
        )
        rewards = torch.stack(imagined_rewards)  # (H, B)
        conts = torch.stack(imagined_cont)

        returns = torch.zeros_like(rewards)
        last_val = values[-1].detach()
        for t in reversed(range(self.imagine_horizon)):
            last_val = rewards[t] + self.gamma * conts[t] * (
                (1 - self.gae_lambda) * values[t].detach() + self.gae_lambda * last_val
            )
            returns[t] = last_val

        # Actor loss: maximize returns
        log_probs_stack = torch.stack(log_probs)
        actor_loss = -(log_probs_stack * (returns - values.detach())).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            list(self.ac.actor_net.parameters()) + list(self.ac.actor_mean.parameters()) +
            [self.ac.actor_log_std], 100.0
        )
        self.actor_optimizer.step()

        # Critic loss
        _, values2 = self.ac(
            torch.stack([h.detach() for h in imagined_h]),
            torch.stack([z.detach() for z in imagined_z])
        )
        critic_loss = F.mse_loss(values2, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.critic_net.parameters(), 100.0)
        self.critic_optimizer.step()

    def update(self) -> None:
        self.world_model_update()
        self.actor_critic_update()
