import torch
import torch.nn as nn


class EnsembleDynamicsModel(nn.Module):
    """Ensemble of probabilistic MLP dynamics models.

    Each member predicts Δobs and reward as Gaussian (mean, log_var).
    Ensemble disagreement provides epistemic uncertainty for rollout truncation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 200,
                 n_members: int = 5):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_members = n_members

        # Each member: shared trunk + mean head + log_var head
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            for _ in range(n_members)
        ])
        self.mean_heads = nn.ModuleList([
            nn.Linear(hidden_dim, obs_dim + 1) for _ in range(n_members)
        ])
        self.log_var_heads = nn.ModuleList([
            nn.Linear(hidden_dim, obs_dim + 1) for _ in range(n_members)
        ])
        # Learnable bounds for log_var
        self.max_log_var = nn.Parameter(torch.ones(obs_dim + 1) * 0.5)
        self.min_log_var = nn.Parameter(-torch.ones(obs_dim + 1) * 10.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        means, log_vars = [], []
        for trunk, mean_h, logvar_h in zip(self.members, self.mean_heads, self.log_var_heads):
            h = trunk(x)
            mean = mean_h(h)
            log_var = logvar_h(h)
            log_var = self.max_log_var - nn.functional.softplus(self.max_log_var - log_var)
            log_var = self.min_log_var + nn.functional.softplus(log_var - self.min_log_var)
            means.append(mean)
            log_vars.append(log_var)
        return torch.stack(means), torch.stack(log_vars)  # (n_members, batch, obs_dim+1)

    def loss(self, obs: torch.Tensor, action: torch.Tensor,
             next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        targets = torch.cat([next_obs - obs, reward.unsqueeze(-1)], dim=-1)
        means, log_vars = self.forward(obs, action)
        inv_var = torch.exp(-log_vars)
        nll = (inv_var * (means - targets.unsqueeze(0)).pow(2) + log_vars).mean()
        # Regularize log_var bounds
        reg = 0.01 * (self.max_log_var.sum() - self.min_log_var.sum())
        return nll + reg

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample next_obs and reward from a random ensemble member."""
        means, log_vars = self.forward(obs, action)
        idx = torch.randint(self.n_members, (obs.shape[0],), device=obs.device)
        mean = means[idx, torch.arange(obs.shape[0])]
        std = torch.exp(0.5 * log_vars[idx, torch.arange(obs.shape[0])])
        sample = mean + std * torch.randn_like(mean)
        delta_obs = sample[..., :self.obs_dim]
        reward = sample[..., -1]
        return obs + delta_obs, reward
