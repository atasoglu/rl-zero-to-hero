import os

import torch
import torch.nn.functional as F

from model import ProcessRewardModel


class PRMAgent:
    """
    Trains a Process Reward Model on step-level correctness labels using BCE loss.

    Each training batch contains full tokenized solutions together with the token
    positions of each step boundary and binary labels (1.0 = step on a path to
    the correct answer, 0.0 = step leads away from the correct answer).

    A single forward pass produces logits for every token; we then gather the
    logits at step-boundary positions and compute BCE against the labels.
    """

    def __init__(self, model_name: str, lr: float, device: str):
        self.device = torch.device(device)
        self.model = ProcessRewardModel(model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )

    def train_step(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        step_positions: torch.Tensor,   # (B, max_steps) token indices; -1 = padding
        step_labels: torch.Tensor,      # (B, max_steps) 0.0/1.0;     -1 = padding
    ) -> tuple[float, float]:
        self.model.train()

        logits = self.model(
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )  # (B, L)

        positions = step_positions.to(self.device)
        labels = step_labels.to(self.device)

        valid = positions >= 0                              # (B, max_steps) bool mask
        safe_pos = positions.clamp(min=0)                  # avoid -1 index
        step_logits = logits.gather(1, safe_pos)           # (B, max_steps)

        flat_logits = step_logits[valid]
        flat_labels = labels[valid]

        loss = F.binary_cross_entropy_with_logits(flat_logits, flat_labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        acc = ((flat_logits > 0) == (flat_labels > 0.5)).float().mean().item()
        return loss.item(), acc

    @torch.no_grad()
    def score_steps(
        self,
        input_ids: torch.Tensor,       # (1, L)
        attention_mask: torch.Tensor,  # (1, L)
        step_positions: list[int],
    ) -> list[float]:
        """Return per-step correctness probabilities for a single solution."""
        self.model.eval()
        logits = self.model(
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )  # (1, L)
        return [torch.sigmoid(logits[0, pos]).item() for pos in step_positions]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
