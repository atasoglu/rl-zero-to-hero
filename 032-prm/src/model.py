import torch
import torch.nn as nn
from transformers import AutoModel


class ProcessRewardModel(nn.Module):
    """
    LM backbone with a per-token step head.

    Unlike an outcome reward model (which pools at the last token of the whole
    response), a PRM produces a logit at every token position so that logits at
    step-boundary positions can be extracted and trained independently.

    forward() returns (B, L) logits — one per token. The caller selects the
    positions that correspond to step boundaries (the last token of each "ки"
    marker in Math-Shepherd) and computes BCE loss only at those positions.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        H = self.backbone.config.hidden_size
        self.step_head = nn.Linear(H, 1, bias=False)
        nn.init.zeros_(self.step_head.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                       # (B, L, H)
        h = h.to(self.step_head.weight.dtype)           # match possible bf16 backbone
        return self.step_head(h).squeeze(-1)            # (B, L)
