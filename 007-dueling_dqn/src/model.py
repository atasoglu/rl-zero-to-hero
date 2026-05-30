import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Separate value and advantage streams, combined as Q = V + (A - mean(A))."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        features = self.shared(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # subtract mean advantage so V and A are identifiable
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class AtariQNetwork(nn.Module):
    """CNN encoder for raw Atari frames (84x84 grayscale), then dueling heads."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_dim = 64 * 7 * 7
        self.value_stream = nn.Sequential(nn.Linear(conv_out_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(conv_out_dim, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        x = x.float() / 255.0
        features = self.conv(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
