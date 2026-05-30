import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import AtariActorCritic, RNDNetwork


class RunningMeanStd:
    """Tracks running mean and variance for normalization."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RNDAgent:
    def __init__(
        self,
        n_actions: int,
        feature_dim: int = 512,
        lr: float = 1e-4,
        gamma_ext: float = 0.999,
        gamma_int: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.1,
        entropy_coef: float = 0.001,
        value_loss_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
        int_coef: float = 1.0,
        ext_coef: float = 2.0,
        device: str = "cpu",
    ):
        self.gamma_ext = gamma_ext
        self.gamma_int = gamma_int
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.device = torch.device(device)

        self.ac = AtariActorCritic(n_actions, feature_dim).to(self.device)
        self.rnd = RNDNetwork(out_dim=feature_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.ac.parameters()) + list(self.rnd.predictor.parameters()), lr=lr
        )

        self.int_reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, 84, 84))

    def select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v_ext, v_int = self.ac(obs_t)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), v_ext.item(), v_int.item()

    def compute_intrinsic_rewards(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            r_int = self.rnd.intrinsic_reward(obs_t).cpu().numpy()
        self.int_reward_rms.update(r_int)
        return r_int / (np.sqrt(self.int_reward_rms.var) + 1e-8)

    def _gae(self, rewards, values, dones, gamma):
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def update(self, rollout: dict) -> None:
        obs = torch.FloatTensor(rollout["obs"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        ext_rewards = rollout["ext_rewards"]
        int_rewards = rollout["int_rewards"]
        dones = rollout["dones"]
        v_ext_arr = rollout["v_ext"]
        v_int_arr = rollout["v_int"]

        adv_ext = self._gae(ext_rewards, v_ext_arr, dones, self.gamma_ext)
        adv_int = self._gae(int_rewards, v_int_arr, np.zeros_like(dones), self.gamma_int)  # non-episodic
        advantages = torch.FloatTensor(
            self.ext_coef * adv_ext + self.int_coef * adv_int
        ).to(self.device)
        returns_ext = torch.FloatTensor(adv_ext + v_ext_arr).to(self.device)
        returns_int = torch.FloatTensor(adv_int + v_int_arr).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(obs)
        for _ in range(self.n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                mb = idx[start:start + self.batch_size]

                dist, v_ext, v_int = self.ac(obs[mb])
                log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy()

                ratio = torch.exp(log_probs - old_log_probs[mb])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * advantages[mb], clipped * advantages[mb]).mean()
                critic_loss = (F.mse_loss(v_ext, returns_ext[mb].detach()) +
                               F.mse_loss(v_int, returns_int[mb].detach()))
                rnd_loss = self.rnd.predictor_loss(obs[mb])

                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy.mean() + rnd_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.ac.parameters()) + list(self.rnd.predictor.parameters()), max_norm=0.5
                )
                self.optimizer.step()
