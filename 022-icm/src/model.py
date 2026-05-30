import torch
import torch.nn as nn


class AtariEncoder(nn.Module):
    """Shared CNN feature extractor for (4, 84, 84) grayscale frame stacks."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, feature_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.float() / 255.0))


class AtariActorCritic(nn.Module):
    """PPO actor-critic with shared CNN encoder for Atari."""

    def __init__(self, n_actions: int, feature_dim: int = 512):
        super().__init__()
        self.encoder = AtariEncoder(feature_dim)
        self.actor = nn.Linear(feature_dim, n_actions)
        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor):
        from torch.distributions import Categorical
        features = self.encoder(x)
        return Categorical(logits=self.actor(features)), self.critic(features).squeeze(-1)


class ICMNetwork(nn.Module):
    """Intrinsic Curiosity Module: inverse + forward dynamics models."""

    def __init__(self, n_actions: int, feature_dim: int = 512):
        super().__init__()
        self.encoder = AtariEncoder(feature_dim)

        # Inverse model: (φ(s), φ(s')) → predicted action logits
        self.inverse = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        # Forward model: (φ(s), one_hot(a)) → predicted φ(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        self.n_actions = n_actions

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def inverse_loss(self, phi_s: torch.Tensor, phi_s_next: torch.Tensor,
                     actions: torch.Tensor) -> torch.Tensor:
        logits = self.inverse(torch.cat([phi_s, phi_s_next], dim=-1))
        return nn.functional.cross_entropy(logits, actions)

    def forward_loss_and_reward(self, phi_s: torch.Tensor, actions: torch.Tensor,
                                phi_s_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_onehot = nn.functional.one_hot(actions, self.n_actions).float()
        phi_s_next_hat = self.forward_model(torch.cat([phi_s, a_onehot], dim=-1))
        error = (phi_s_next_hat - phi_s_next.detach()).pow(2).mean(dim=-1)
        return error.mean(), error.detach()
