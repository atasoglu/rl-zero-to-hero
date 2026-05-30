import argparse
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from agent import RLHFAgent
from rl_common import plot_rewards

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "trl-lib/ultrafeedback_binarized"
DEFAULT_RM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "031-reward-modeling", "checkpoints", "rm.pt"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="RLHF: fine-tune an LLM with PPO + reward model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="pretrained LM to fine-tune (policy and reference are both initialised from this)")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HuggingFace dataset; only the prompt field is used")
    p.add_argument("--dataset-size", type=int, default=1000,
                   help="number of prompts to load for training")
    p.add_argument("--steps", type=int, default=200,
                   help="number of PPO rollout+update steps")
    p.add_argument("--batch-size", type=int, default=2,
                   help="prompts per rollout batch")
    p.add_argument("--max-prompt-len", type=int, default=128,
                   help="max tokenised length for the prompt")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="max response tokens to generate per prompt")
    p.add_argument("--ppo-epochs", type=int, default=2,
                   help="number of PPO gradient epochs per rollout batch")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="AdamW learning rate for the policy")
    p.add_argument("--kl-coef", type=float, default=0.1,
                   help="β — weight of the per-token KL penalty against the reference policy")
    p.add_argument("--clip-eps", type=float, default=0.2,
                   help="PPO clipping epsilon for the surrogate objective")
    p.add_argument("--reward-model-path", default=DEFAULT_RM_PATH,
                   help="path to trained reward model checkpoint from module 031")
    p.add_argument("--generate", type=int, default=3, metavar="N",
                   help="generate N example responses after training (0 to skip)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="compute device; auto selects cuda > mps > cpu")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_user_message(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return messages[0]["content"]


class PromptDataset(Dataset):
    def __init__(self, raw, tokenizer, max_prompt_len: int):
        self.data = raw
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        prompt = f"Human: {_get_user_message(self.data[idx]['chosen'])}\nAssistant:"
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def pad_responses(response_list: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Pad variable-length response tensors to the same length."""
    max_len = max(r.shape[0] for r in response_list)
    padded = torch.full((len(response_list), max_len), pad_id, dtype=torch.long)
    for i, r in enumerate(response_list):
        padded[i, : r.shape[0]] = r
    return padded


def train(agent: RLHFAgent, loader: DataLoader, steps: int, tokenizer) -> list[float]:
    rm_scores = []
    it = iter(loader)
    pbar = tqdm(range(steps), desc="RLHF training")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        prompt_ids = batch["input_ids"].to(agent.device)

        # Step 1: generate responses with the current policy
        response_ids = agent.generate(
            prompt_ids,
            max_new_tokens=agent.policy.lm.config.max_position_embeddings
            if hasattr(agent.policy.lm.config, "max_position_embeddings")
            else 64,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Use args-specified max_new_tokens passed via closure
        # (re-generate with correct length — see _max_new_tokens below)

        # Step 2: PPO update
        stats = agent.ppo_step(prompt_ids, response_ids)
        rm_scores.append(stats["rm_score"])

        pbar.set_postfix(
            rm=f"{stats['rm_score']:.3f}",
            kl=f"{stats['kl']:.4f}",
            pg=f"{stats['pg_loss']:.4f}",
        )
    return rm_scores


def _train_with_max_tokens(
    agent: RLHFAgent, loader: DataLoader, steps: int, tokenizer, max_new_tokens: int
) -> list[float]:
    rm_scores = []
    it = iter(loader)
    pbar = tqdm(range(steps), desc="RLHF training")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        prompt_ids = batch["input_ids"].to(agent.device)
        attention_mask = batch["attention_mask"]

        response_ids = agent.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
        )

        stats = agent.ppo_step(prompt_ids, response_ids)
        rm_scores.append(stats["rm_score"])

        pbar.set_postfix(
            rm=f"{stats['rm_score']:.3f}",
            kl=f"{stats['kl']:.4f}",
            pg=f"{stats['pg_loss']:.4f}",
        )
    return rm_scores


def show_examples(agent: RLHFAgent, tokenizer, raw_data, n: int, max_prompt_len: int, max_new_tokens: int):
    print(f"\n{'─'*60}")
    print(f"{'Generated examples':^60}")
    print(f"{'─'*60}")
    for i in range(min(n, len(raw_data))):
        prompt_text = f"Human: {raw_data[i]['prompt']}\nAssistant:"
        enc = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        response_ids = agent.generate(
            enc["input_ids"].to(agent.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print(f"\nPrompt  : {_get_user_message(raw_data[i]['chosen'])[:100]}...")
        print(f"Response: {response_text[:200]}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device : {device}")
    print(f"Model  : {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading dataset...")
    raw = load_dataset(args.dataset, split="train")
    raw = raw.select(range(min(args.dataset_size, len(raw))))

    dataset = PromptDataset(raw, tokenizer, args.max_prompt_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    agent = RLHFAgent(
        model_name=args.model,
        rm_path=args.reward_model_path,
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    rm_scores = _train_with_max_tokens(agent, loader, args.steps, tokenizer, args.max_new_tokens)

    plot_rewards(rm_scores, "RLHF — Reward Model Score per Step")

    if args.generate > 0:
        show_examples(agent, tokenizer, raw, args.generate, args.max_prompt_len, args.max_new_tokens)


if __name__ == "__main__":
    main()
