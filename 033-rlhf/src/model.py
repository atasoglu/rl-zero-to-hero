import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class PolicyWithValueHead(nn.Module):
    """
    Language model policy augmented with a scalar value head for PPO.

    - lm_head produces token logits for generation and log-prob computation
    - value_head produces V(s_t) at each token position for advantage estimation
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.lm.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.value_head.weight)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.lm(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits                          # (B, T, V)
        last_hidden = outputs.hidden_states[-1]          # (B, T, H)
        values = self.value_head(last_hidden).squeeze(-1)  # (B, T)
        return logits, values

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.lm.generate(input_ids, **kwargs)
