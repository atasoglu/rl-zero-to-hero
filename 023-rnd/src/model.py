import torch
import torch.nn as nn


class AtariActorCritic(nn.Module):
    """PPO actor-critic with CNN encoder. Separate value heads for ext/int rewards."""

    def __init__(self, n_actions: int, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, feature_dim), nn.ReLU())
        self.actor = nn.Linear(feature_dim, n_actions)
        self.critic_ext = nn.Linear(feature_dim, 1)   # extrinsic value
        self.critic_int = nn.Linear(feature_dim, 1)   # intrinsic value

    def forward(self, x: torch.Tensor):
        from torch.distributions import Categorical
        features = self.fc(self.conv(x.float() / 255.0))
        dist = Categorical(logits=self.actor(features))
        v_ext = self.critic_ext(features).squeeze(-1)
        v_int = self.critic_int(features).squeeze(-1)
        return dist, v_ext, v_int


class RNDNetwork(nn.Module):
    """Random Network Distillation: fixed target + trainable predictor."""

    def __init__(self, out_dim: int = 512):
        super().__init__()
        # Target: random init, never updated
        self.target = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor: same architecture, trained to match target
        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Use only the last frame (index -1) for novelty detection."""
        frame = obs[:, -1:, :, :].float() / 255.0
        with torch.no_grad():
            target_feat = self.target(frame)
        pred_feat = self.predictor(frame)
        return (pred_feat - target_feat).pow(2).mean(dim=-1)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        frame = obs[:, -1:, :, :].float() / 255.0
        with torch.no_grad():
            target_feat = self.target(frame)
        pred_feat = self.predictor(frame)
        return (pred_feat - target_feat).pow(2).mean()
