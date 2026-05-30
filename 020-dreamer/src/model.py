import torch
import torch.nn as nn
from torch.distributions import Normal


class RSSM(nn.Module):
    """Recurrent State Space Model: deterministic h + stochastic z.

    Prior: p(z_t | h_t)
    Posterior: q(z_t | h_t, obs_t)
    Transition: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 h_dim: int = 200, z_dim: int = 30):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
        )

        # GRU transition
        self.gru = nn.GRUCell(z_dim + action_dim, h_dim)

        # Prior p(z | h)
        self.prior = nn.Sequential(nn.Linear(h_dim, 200), nn.ELU())
        self.prior_mu = nn.Linear(200, z_dim)
        self.prior_log_std = nn.Linear(200, z_dim)

        # Posterior q(z | h, obs_embed)
        self.posterior = nn.Sequential(nn.Linear(h_dim + 400, 200), nn.ELU())
        self.post_mu = nn.Linear(200, z_dim)
        self.post_log_std = nn.Linear(200, z_dim)

    def initial_state(self, batch_size: int, device):
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def step(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor,
             obs: torch.Tensor | None = None):
        """One step forward. Returns (h_next, z_next, prior_dist, post_dist or None)."""
        h_next = self.gru(torch.cat([z, action], dim=-1), h)

        # Prior
        p_feat = self.prior(h_next)
        p_mu = self.prior_mu(p_feat)
        p_std = nn.functional.softplus(self.prior_log_std(p_feat)) + 0.1
        prior_dist = Normal(p_mu, p_std)

        if obs is not None:
            obs_embed = self.obs_encoder(obs)
            q_feat = self.posterior(torch.cat([h_next, obs_embed], dim=-1))
            q_mu = self.post_mu(q_feat)
            q_std = nn.functional.softplus(self.post_log_std(q_feat)) + 0.1
            post_dist = Normal(q_mu, q_std)
            z_next = post_dist.rsample()
        else:
            post_dist = None
            z_next = prior_dist.rsample()

        return h_next, z_next, prior_dist, post_dist


class WorldModel(nn.Module):
    """World model: RSSM + obs decoder + reward head + continue head."""

    def __init__(self, obs_dim: int, action_dim: int,
                 h_dim: int = 200, z_dim: int = 30):
        super().__init__()
        self.rssm = RSSM(obs_dim, action_dim, h_dim, z_dim)
        feat_dim = h_dim + z_dim

        self.obs_decoder = nn.Sequential(
            nn.Linear(feat_dim, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, 400), nn.ELU(),
            nn.Linear(400, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(feat_dim, 400), nn.ELU(),
            nn.Linear(400, 1),
            nn.Sigmoid(),
        )

    def feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)

    def decode(self, h, z):
        return self.obs_decoder(self.feat(h, z))

    def reward(self, h, z):
        return self.reward_head(self.feat(h, z)).squeeze(-1)

    def continue_prob(self, h, z):
        return self.continue_head(self.feat(h, z)).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor and critic operating in latent space (h, z)."""

    def __init__(self, h_dim: int, z_dim: int, action_dim: int, hidden_dim: int = 400):
        super().__init__()
        feat_dim = h_dim + z_dim
        self.actor_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        self.critic_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor):
        feat = torch.cat([h, z], dim=-1)
        ac_feat = self.actor_net(feat)
        mean = self.actor_mean(ac_feat)
        std = torch.exp(torch.clamp(self.actor_log_std, -4, 2)).expand_as(mean)
        return Normal(mean, std), self.critic_net(feat).squeeze(-1)

    def sample_action(self, h: torch.Tensor, z: torch.Tensor) -> tuple:
        dist, value = self.forward(h, z)
        action = torch.tanh(dist.rsample())
        log_prob = (dist.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob, value
