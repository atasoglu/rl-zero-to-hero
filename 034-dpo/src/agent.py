import torch
import torch.nn as nn
import torch.nn.functional as F

from model import make_policy, make_reference


def _sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Sum of per-token log probs for the label tokens.

    input_ids : (B, T)  — full sequence (prompt + response)
    labels    : (B, T)  — same shape; positions to ignore are set to -100

    Returns (B,) — one scalar per example.
    """
    logits = model(input_ids).logits                 # (B, T, V)
    # shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]                 # (B, T-1, V)
    shift_labels = labels[:, 1:]                     # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Gather log prob of each label token; ignore padding (-100)
    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    per_token = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (per_token * mask).sum(dim=-1)            # (B,)


class DPOAgent:
    """
    Direct Preference Optimization.

    Reparameterises the RLHF objective to eliminate the explicit reward model:

        L_DPO = -log σ(β · [(log π(y_w|x) - log π_ref(y_w|x))
                             - (log π(y_l|x) - log π_ref(y_l|x))])

    where y_w = chosen, y_l = rejected, π_ref is the frozen reference policy.
    β controls the KL budget: small β → more deviation from reference allowed.
    """

    def __init__(self, model_name: str, beta: float, lr: float, device: str):
        self.device = torch.device(device)
        self.beta = beta

        self.policy = make_policy(model_name).to(self.device)
        self.ref = make_reference(model_name).to(self.device)
        self.ref.eval()

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=0.01)

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_step(self, batch: dict) -> tuple[float, float]:
        """
        batch keys: chosen_ids, chosen_labels, rejected_ids, rejected_labels
        Returns (loss, implicit_reward_margin).
        """
        b = self._move(batch)

        # Policy log probs
        self.policy.train()
        pi_lp_w = _sequence_log_probs(self.policy, b["chosen_ids"],   b["chosen_labels"])
        pi_lp_l = _sequence_log_probs(self.policy, b["rejected_ids"],  b["rejected_labels"])

        # Reference log probs (no grad)
        self.ref.eval()
        with torch.no_grad():
            ref_lp_w = _sequence_log_probs(self.ref, b["chosen_ids"],  b["chosen_labels"])
            ref_lp_l = _sequence_log_probs(self.ref, b["rejected_ids"], b["rejected_labels"])

        # DPO loss
        log_ratio_w = pi_lp_w - ref_lp_w   # (B,)  implicit reward for chosen
        log_ratio_l = pi_lp_l - ref_lp_l   # (B,)  implicit reward for rejected
        loss = -F.logsigmoid(self.beta * (log_ratio_w - log_ratio_l)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        margin = (log_ratio_w - log_ratio_l).mean().item()
        return loss.item(), margin

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, pad_token_id: int) -> torch.Tensor:
        self.policy.eval()
        out = self.policy.generate(
            prompt_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=pad_token_id,
        )
        return out[:, prompt_ids.shape[1]:]
