import argparse
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from agent import DPOAgent
from rl_common import plot_rewards

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "trl-lib/ultrafeedback_binarized"


def parse_args():
    p = argparse.ArgumentParser(
        description="DPO: Direct Preference Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="pretrained LM to fine-tune (policy and reference are both initialised from this)")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HuggingFace preference dataset with chosen/rejected fields")
    p.add_argument("--dataset-size", type=int, default=2000,
                   help="number of training examples to load")
    p.add_argument("--steps", type=int, default=500,
                   help="number of gradient update steps")
    p.add_argument("--batch-size", type=int, default=4,
                   help="preference pairs per batch")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="AdamW learning rate")
    p.add_argument("--beta", type=float, default=0.1,
                   help="KL budget β: higher = policy stays closer to the reference")
    p.add_argument("--max-length", type=int, default=512,
                   help="max tokenised length for prompt+response")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="max tokens to generate when sampling examples with --generate")
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


def _get_assistant_response(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg["content"]
    return messages[-1]["content"]


def _make_labels(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Mask prompt tokens with -100 so loss is only computed on the response."""
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return labels


class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int):
        self.data = data
        self.tok = tokenizer
        self.max_length = max_length

    def _encode_pair(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_enc = self.tok(
            f"Human: {prompt}\nAssistant:",
            add_special_tokens=False,
        )
        prompt_len = len(prompt_enc["input_ids"])

        full_enc = self.tok(
            f"Human: {prompt}\nAssistant: {response}",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)
        labels = _make_labels(input_ids.unsqueeze(0), prompt_len).squeeze(0)
        return input_ids, labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex = self.data[idx]
        prompt = _get_user_message(ex["chosen"])
        chosen_ids, chosen_labels = self._encode_pair(
            prompt, _get_assistant_response(ex["chosen"])
        )
        rejected_ids, rejected_labels = self._encode_pair(
            prompt, _get_assistant_response(ex["rejected"])
        )
        return {
            "chosen_ids":      chosen_ids,
            "chosen_labels":   chosen_labels,
            "rejected_ids":    rejected_ids,
            "rejected_labels": rejected_labels,
        }


def train(agent: DPOAgent, loader: DataLoader, steps: int) -> tuple[list, list]:
    losses, margins = [], []
    it = iter(loader)
    pbar = tqdm(range(steps), desc="DPO training")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        loss, margin = agent.train_step(batch)
        losses.append(loss)
        margins.append(margin)
        pbar.set_postfix(loss=f"{loss:.4f}", margin=f"{margin:.3f}")
    return losses, margins


def show_examples(
    agent: DPOAgent,
    tokenizer,
    raw_data,
    n: int,
    max_prompt_len: int,
    max_new_tokens: int,
):
    print(f"\n{'─'*60}")
    print(f"{'Generated examples (DPO policy)':^60}")
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
            enc["input_ids"], max_new_tokens, tokenizer.pad_token_id
        )
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print(f"\nPrompt  : {_get_user_message(raw_data[i]['chosen'])[:100]}...")
        print(f"Response: {response_text[:200]}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"β      : {args.beta}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    raw = load_dataset(args.dataset, split="train")
    raw = raw.select(range(min(args.dataset_size, len(raw))))

    dataset = DPODataset(raw, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    agent = DPOAgent(
        model_name=args.model,
        beta=args.beta,
        lr=args.lr,
        device=device,
    )

    losses, margins = train(agent, loader, args.steps)

    plot_rewards(margins, f"DPO — Implicit Reward Margin (β={args.beta})")

    if args.generate > 0:
        show_examples(agent, tokenizer, raw, args.generate, args.max_length, args.max_new_tokens)


if __name__ == "__main__":
    main()
