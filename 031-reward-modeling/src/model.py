import torch
import torch.nn as nn
from transformers import AutoModel


class RewardModel(nn.Module):
    """
    Wraps a pretrained LM backbone and adds a scalar value head.

    The reward for a (prompt, response) pair is the value head's output
    at the last token position — following the convention used in RLHF
    literature (last token aggregates the full sequence).
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.value_head.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # last non-padding token's hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]           # (B, H)
        last_hidden = last_hidden.to(self.value_head.weight.dtype)  # backbone may be bf16
        return self.value_head(last_hidden).squeeze(-1)             # (B,)
