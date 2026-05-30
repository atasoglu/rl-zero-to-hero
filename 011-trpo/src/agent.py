import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return Categorical(logits=self.net(x))


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _flat_grad(loss, params, create_graph=False, retain_graph=False):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=retain_graph)
    return torch.cat([g.contiguous().view(-1) for g in grads])


def _conjugate_gradient(Avp_fn, b, n_steps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(n_steps):
        Ap = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x


class TRPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        max_kl: float = 0.01,
        damping: float = 0.1,
        value_lr: float = 1e-3,
        value_iters: int = 5,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.damping = damping
        self.value_iters = value_iters
        self.device = torch.device(device)

        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)

    def select_action(self, obs: np.ndarray) -> tuple[int, float]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.policy(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def _gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:T]
        return advantages, returns

    def update(self, trajectories: list[dict]) -> None:
        obs_all = torch.FloatTensor(np.concatenate([t["obs"] for t in trajectories])).to(self.device)
        actions_all = torch.LongTensor(np.concatenate([t["actions"] for t in trajectories])).to(self.device)
        old_log_probs = torch.FloatTensor(np.concatenate([t["log_probs"] for t in trajectories])).to(self.device)
        rewards_all = torch.FloatTensor(np.concatenate([t["rewards"] for t in trajectories])).to(self.device)
        dones_all = torch.FloatTensor(np.concatenate([t["dones"] for t in trajectories])).to(self.device)

        with torch.no_grad():
            values_all = self.value(obs_all)

        advantages, returns = self._gae(rewards_all, values_all, dones_all)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value function
        for _ in range(self.value_iters):
            value_loss = F.mse_loss(self.value(obs_all), returns.detach())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # Policy gradient
        dist = self.policy(obs_all)
        # detach a fixed snapshot for KL reference in Fvp
        with torch.no_grad():
            dist_fixed = self.policy(obs_all)
        log_probs = dist.log_prob(actions_all)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages.detach()).mean()

        params = list(self.policy.parameters())
        pg = _flat_grad(policy_loss, params, retain_graph=True)

        # Fisher-vector product for conjugate gradient (recompute dist each call)
        def Fvp(v):
            d = self.policy(obs_all)
            kl = torch.distributions.kl_divergence(dist_fixed, d).mean()
            grads = torch.autograd.grad(kl, params, create_graph=True)
            flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
            kl_v = (flat_grads * v).sum()
            grads2 = torch.autograd.grad(kl_v, params)
            flat_grads2 = torch.cat([g.contiguous().view(-1) for g in grads2])
            return flat_grads2 + self.damping * v

        step_dir = _conjugate_gradient(Fvp, -pg)
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum()
        lagrange_mult = torch.sqrt(shs / self.max_kl)
        step = step_dir / (lagrange_mult + 1e-8)

        # Line search
        old_params = torch.cat([p.data.view(-1) for p in params])
        step_size = 1.0
        for _ in range(10):
            new_params = old_params + step_size * step
            offset = 0
            for p in params:
                numel = p.numel()
                p.data.copy_(new_params[offset:offset + numel].view_as(p))
                offset += numel

            with torch.no_grad():
                new_dist = self.policy(obs_all)
                new_log_probs = new_dist.log_prob(actions_all)
                new_ratio = torch.exp(new_log_probs - old_log_probs)
                new_loss = -(new_ratio * advantages.detach()).mean()
                kl_div = torch.distributions.kl_divergence(dist, new_dist).mean()

            if new_loss < policy_loss and kl_div <= self.max_kl:
                break
            step_size *= 0.5
        else:
            # restore old params if line search fails
            offset = 0
            for p in params:
                numel = p.numel()
                p.data.copy_(old_params[offset:offset + numel].view_as(p))
                offset += numel
