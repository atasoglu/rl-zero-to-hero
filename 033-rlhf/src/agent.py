import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import PolicyWithValueHead

# Import reward model from sibling module at runtime via path
import importlib.util, sys, os


def _load_reward_model(rm_path: str | None, model_name: str, device: torch.device):
    """Load reward model from 031 checkpoint if provided, else return None."""
    spec = importlib.util.spec_from_file_location(
        "reward_model",
        os.path.join(os.path.dirname(__file__), "..", "..", "031-reward-modeling", "src", "model.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rm = mod.RewardModel(model_name).to(device)
    if rm_path and os.path.exists(rm_path):
        rm.load_state_dict(torch.load(rm_path, map_location=device))
        print(f"Loaded reward model from {rm_path}")
    else:
        print("No reward model checkpoint found — using random-init RM (scores will be noisy)")
    for p in rm.parameters():
        p.requires_grad_(False)
    rm.eval()
    return rm


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation over a response sequence.

    rewards, values: (B, T)
    returns: advantages (B, T), returns (B, T)
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_val = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    returns = advantages + values
    return advantages, returns


def _response_log_probs(
    logits: torch.Tensor,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Per-token log probs for the response portion of full_ids.

    logits[t] predicts token[t+1], so response token at position prompt_len+k
    is predicted by logits at prompt_len+k-1.
    """
    resp_len = full_ids.shape[1] - prompt_len
    # logits slice: positions [prompt_len-1 .. prompt_len+resp_len-2]
    resp_logits = logits[:, prompt_len - 1 : prompt_len + resp_len - 1, :]  # (B, resp_len, V)
    resp_labels = full_ids[:, prompt_len:]                                   # (B, resp_len)
    log_probs = F.log_softmax(resp_logits, dim=-1)
    return log_probs.gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)       # (B, resp_len)


class RLHFAgent:
    """
    RLHF agent: fine-tunes an LLM with PPO using a reward model signal.

    Total per-token reward:
        r_t = -kl_coef * KL_t   for t < T-1
        r_T = rm_score - kl_coef * KL_T

    KL_t = log π_θ(a_t|s_t) - log π_ref(a_t|s_t)  (per token)
    """

    def __init__(
        self,
        model_name: str,
        rm_path: str | None,
        kl_coef: float,
        clip_eps: float,
        lr: float,
        ppo_epochs: int,
        device: str,
    ):
        self.device = torch.device(device)
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs

        self.policy = PolicyWithValueHead(model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        # Reference model: frozen copy of the initial policy
        self.ref = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()

        self.rm = _load_reward_model(rm_path, model_name, self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Generate response tokens for a batch of prompts."""
        output = self.policy.generate(
            prompt_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=pad_token_id,
        )
        return output[:, prompt_ids.shape[1]:]  # response only

    def ppo_step(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> dict[str, float]:
        prompt_ids = prompt_ids.to(self.device)
        response_ids = response_ids.to(self.device)
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (B, P+R)
        prompt_len = prompt_ids.shape[1]
        B, R = response_ids.shape
        attn_mask = (full_ids != 0).long()

        # ── reward model score (scalar per response) ──────────────────────
        with torch.no_grad():
            rm_scores = self.rm(full_ids, attn_mask)  # (B,)

        # ── reference log probs ──────────────────────────────────────────
        with torch.no_grad():
            ref_logits = self.ref(full_ids).logits
            ref_log_probs = _response_log_probs(ref_logits, full_ids, prompt_len)  # (B, R)

        # ── old policy: log probs + values (before any PPO update) ───────
        with torch.no_grad():
            old_logits, old_values_full = self.policy(full_ids, attn_mask)
            old_log_probs = _response_log_probs(old_logits, full_ids, prompt_len)
            old_values = old_values_full[:, prompt_len - 1 : prompt_len + R - 1]  # (B, R)

        # ── per-token augmented rewards ──────────────────────────────────
        kl = old_log_probs - ref_log_probs                  # (B, R)
        token_rewards = -self.kl_coef * kl                  # (B, R)
        token_rewards[:, -1] = token_rewards[:, -1] + rm_scores  # RM at last token

        # ── GAE advantages & returns ─────────────────────────────────────
        advantages, returns = _gae(token_rewards, old_values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update loop ───────────────────────────────────────────────
        for _ in range(self.ppo_epochs):
            logits, values_full = self.policy(full_ids, attn_mask)
            new_log_probs = _response_log_probs(logits, full_ids, prompt_len)
            new_values = values_full[:, prompt_len - 1 : prompt_len + R - 1]

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            pg_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            ).mean()

            value_loss = F.mse_loss(new_values, returns.detach())

            loss = pg_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        return {
            "pg_loss":   pg_loss.item(),
            "value_loss": value_loss.item(),
            "rm_score":  rm_scores.mean().item(),
            "kl":        kl.mean().item(),
        }
