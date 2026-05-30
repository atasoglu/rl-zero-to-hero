import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent


class VAE(nn.Module):
    """Convolutional VAE: (3,64,64) frame → z_dim latent vector."""

    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),   # 31x31
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # 14x14
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(), # 6x6
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),# 2x2
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, z_dim)
        self.fc_log_var = nn.Linear(256 * 2 * 2, z_dim)

        self.decoder_fc = nn.Linear(z_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),  # 9x9
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),    # 21x21
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),     # 46x46
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid(),   # 96x96 → crop to 64x64
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 1024, 1, 1)
        out = self.decoder(h)
        return out[:, :, :64, :64]  # crop to target size

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def loss(self, x: torch.Tensor, recon: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()
        return recon_loss + kl


class MDNRNN(nn.Module):
    """MDN-RNN: GRU + Mixture Density Network over next latent z."""

    def __init__(self, z_dim: int = 32, action_dim: int = 3,
                 hidden_dim: int = 256, n_gaussians: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.z_dim = z_dim

        self.gru = nn.GRUCell(z_dim + action_dim, hidden_dim)
        # MDN outputs: pi (mix weights), mu, sigma per gaussian component
        self.mdn = nn.Linear(hidden_dim, n_gaussians * (1 + z_dim + z_dim))
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor,
                h: torch.Tensor) -> tuple:
        x = torch.cat([z, action], dim=-1)
        h_next = self.gru(x, h)
        out = self.mdn(h_next)
        pi_logits, mu, log_sigma = out.split(
            [self.n_gaussians, self.n_gaussians * self.z_dim, self.n_gaussians * self.z_dim],
            dim=-1
        )
        mu = mu.view(-1, self.n_gaussians, self.z_dim)
        sigma = torch.exp(log_sigma).view(-1, self.n_gaussians, self.z_dim)
        reward = self.reward_head(h_next).squeeze(-1)
        return pi_logits, mu, sigma, reward, h_next

    def loss(self, pi_logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
             z_next: torch.Tensor) -> torch.Tensor:
        mix = Categorical(logits=pi_logits)
        comp = Independent(Normal(mu, sigma), 1)  # event_shape=(z_dim,)
        gmm = MixtureSameFamily(mix, comp)
        return -gmm.log_prob(z_next).mean()

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class Controller(nn.Module):
    """Linear controller: (z, h) → action logits / values for PPO."""

    def __init__(self, z_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        in_dim = z_dim + hidden_dim
        self.actor = nn.Linear(in_dim, n_actions)
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor, h: torch.Tensor):
        from torch.distributions import Categorical
        feat = torch.cat([z, h], dim=-1)
        return Categorical(logits=self.actor(feat)), self.critic(feat).squeeze(-1)
