import torch
import torch.nn as nn


class RepresentationNetwork(nn.Module):
    """obs → hidden state s^0"""

    def __init__(self, obs_dim: int, hidden_dim: int = 64, state_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DynamicsNetwork(nn.Module):
    """(s, a_onehot) → (next_s, reward)"""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, state_dim), nn.Tanh())
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor):
        x = self.net(torch.cat([state, action_onehot], dim=-1))
        return self.state_head(x), self.reward_head(x).squeeze(-1)


class PredictionNetwork(nn.Module):
    """s → (policy logits, value)"""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.net(state)
        return self.policy_head(features), self.value_head(features).squeeze(-1)
