import os

import torch
import torch.nn.functional as F

from model import RewardModel


class RewardModelAgent:
    """
    Trains a reward model on human preference pairs using Bradley-Terry loss:

        L = -log σ(r_chosen - r_rejected)

    The model learns to assign higher scores to preferred responses.
    """

    def __init__(self, model_name: str, lr: float, device: str):
        self.device = torch.device(device)
        self.model = RewardModel(model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[float, float]:
        self.model.train()
        r_chosen = self.model(chosen_ids.to(self.device), chosen_mask.to(self.device))
        r_rejected = self.model(rejected_ids.to(self.device), rejected_mask.to(self.device))

        # Bradley-Terry: probability that chosen is preferred = σ(r_c - r_r)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        margin = (r_chosen - r_rejected).mean().item()
        return loss.item(), margin

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        self.model.eval()
        return self.model(
            input_ids.to(self.device), attention_mask.to(self.device)
        ).item()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
