import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mcts import Node, expand, run_mcts
from model import DynamicsNetwork, PredictionNetwork, RepresentationNetwork


class MuZeroAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        state_dim: int = 64,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.997,
        n_simulations: int = 50,
        buffer_size: int = 10_000,
        batch_size: int = 128,
        td_steps: int = 5,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_simulations = n_simulations
        self.batch_size = batch_size
        self.td_steps = td_steps
        self.device = torch.device(device)

        self.repr_net = RepresentationNetwork(obs_dim, hidden_dim, state_dim).to(self.device)
        self.dyn_net = DynamicsNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.pred_net = PredictionNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.repr_net.parameters()) +
            list(self.dyn_net.parameters()) +
            list(self.pred_net.parameters()),
            lr=lr,
        )

        self.buffer: deque = deque(maxlen=buffer_size)

    def select_action(self, obs: np.ndarray, temperature: float = 1.0) -> tuple[int, np.ndarray]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = self.repr_net(obs_t)
            policy_logits, _ = self.pred_net(hidden)

        root = Node(prior=1.0)
        expand(root, policy_logits[0], hidden[0].cpu().numpy(), 0.0, self.n_actions)

        action_probs = run_mcts(
            root, (self.dyn_net, self.pred_net),
            self.n_simulations, self.n_actions, self.gamma
        )

        if temperature == 0.0:
            action = int(np.argmax(action_probs))
        else:
            probs = action_probs ** (1.0 / temperature)
            probs /= probs.sum()
            action = int(np.random.choice(self.n_actions, p=probs))

        return action, action_probs

    def store(self, trajectory: list[dict]) -> None:
        """Store a completed episode as individual (obs, action_probs, n-step target) records."""
        T = len(trajectory)
        for t in range(T):
            # N-step value target
            G = 0.0
            for k in range(self.td_steps):
                if t + k < T:
                    G += (self.gamma ** k) * trajectory[t + k]["reward"]
            self.buffer.append({
                "obs": trajectory[t]["obs"],
                "action": trajectory[t]["action"],
                "action_probs": trajectory[t]["action_probs"],
                "value_target": G,
                "reward_target": trajectory[t]["reward"],
            })

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        obs = torch.FloatTensor(np.array([b["obs"] for b in batch])).to(self.device)
        actions = torch.LongTensor([b["action"] for b in batch]).to(self.device)
        target_probs = torch.FloatTensor(np.array([b["action_probs"] for b in batch])).to(self.device)
        target_values = torch.FloatTensor([b["value_target"] for b in batch]).to(self.device)
        target_rewards = torch.FloatTensor([b["reward_target"] for b in batch]).to(self.device)

        # Initial inference
        hidden = self.repr_net(obs)
        policy_logits, values = self.pred_net(hidden)

        # Dynamics rollout for reward prediction
        a_onehot = F.one_hot(actions, self.n_actions).float()
        _, pred_rewards = self.dyn_net(hidden, a_onehot)

        # Losses
        value_loss = F.mse_loss(values, target_values)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        policy_loss = -(target_probs * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

        loss = value_loss + reward_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.repr_net.parameters()) +
            list(self.dyn_net.parameters()) +
            list(self.pred_net.parameters()),
            max_norm=5.0,
        )
        self.optimizer.step()
