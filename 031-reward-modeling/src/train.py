import argparse
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from agent import RewardModelAgent
from rl_common import plot_rewards

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "trl-lib/ultrafeedback_binarized"


def parse_args():
    p = argparse.ArgumentParser(
        description="Reward Modeling: Bradley-Terry preference learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="pretrained backbone from HuggingFace Hub")
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
    p.add_argument("--max-length", type=int, default=512,
                   help="max tokenised length for prompt+response")
    p.add_argument("--save-path", default="checkpoints/rm.pt",
                   help="where to save the trained reward model checkpoint")
    p.add_argument("--generate", type=int, default=3, metavar="N",
                   help="score N examples after training (0 to skip)")
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


def _get_assistant_response(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg["content"]
    return messages[-1]["content"]


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode(self, prompt: str, response: str) -> dict:
        text = f"Human: {prompt}\nAssistant: {response}"
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex = self.data[idx]
        prompt = ex["prompt"]
        enc_c = self._encode(prompt, _get_assistant_response(ex["chosen"]))
        enc_r = self._encode(prompt, _get_assistant_response(ex["rejected"]))
        return {
            "chosen_ids":   enc_c["input_ids"].squeeze(0),
            "chosen_mask":  enc_c["attention_mask"].squeeze(0),
            "rejected_ids": enc_r["input_ids"].squeeze(0),
            "rejected_mask": enc_r["attention_mask"].squeeze(0),
        }


def train(agent: RewardModelAgent, loader: DataLoader, steps: int):
    losses, margins = [], []
    it = iter(loader)
    pbar = tqdm(range(steps), desc="Training")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        loss, margin = agent.train_step(
            batch["chosen_ids"], batch["chosen_mask"],
            batch["rejected_ids"], batch["rejected_mask"],
        )
        losses.append(loss)
        margins.append(margin)
        pbar.set_postfix(loss=f"{loss:.4f}", margin=f"{margin:.3f}")
    return losses, margins


@torch.no_grad()
def evaluate(agent: RewardModelAgent, loader: DataLoader, n_batches: int = 25) -> float:
    agent.model.eval()
    correct = total = 0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        r_c = agent.model(
            batch["chosen_ids"].to(agent.device),
            batch["chosen_mask"].to(agent.device),
        )
        r_r = agent.model(
            batch["rejected_ids"].to(agent.device),
            batch["rejected_mask"].to(agent.device),
        )
        correct += (r_c > r_r).sum().item()
        total += r_c.shape[0]
    return correct / total if total else 0.0


def show_examples(agent: RewardModelAgent, tokenizer, raw_data, n: int, max_length: int):
    print(f"\n{'─'*60}")
    print(f"{'Scored examples':^60}")
    print(f"{'─'*60}")
    agent.model.eval()
    for i in range(min(n, len(raw_data))):
        ex = raw_data[i]
        prompt = ex["prompt"]
        chosen_resp = _get_assistant_response(ex["chosen"])
        rejected_resp = _get_assistant_response(ex["rejected"])

        def _score(text: str) -> float:
            enc = tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            return agent.score(enc["input_ids"], enc["attention_mask"])

        s_c = _score(f"Human: {prompt}\nAssistant: {chosen_resp}")
        s_r = _score(f"Human: {prompt}\nAssistant: {rejected_resp}")

        verdict = "✓" if s_c > s_r else "✗"
        print(f"\nPrompt   : {prompt[:100]}...")
        print(f"Chosen   ({s_c:+.3f}): {chosen_resp[:80]}...")
        print(f"Rejected ({s_r:+.3f}): {rejected_resp[:80]}...")
        print(f"Correct  : {verdict}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Dataset: {args.dataset} (n={args.dataset_size})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    raw = load_dataset(args.dataset, split="train")
    raw = raw.select(range(min(args.dataset_size, len(raw))))

    dataset = PreferenceDataset(raw, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    agent = RewardModelAgent(args.model, args.lr, device)

    losses, margins = train(agent, loader, args.steps)

    acc = evaluate(agent, loader)
    print(f"\nAccuracy (chosen > rejected): {acc:.1%}")

    agent.save(args.save_path)

    plot_rewards(margins, "Reward Margin (chosen − rejected)")

    if args.generate > 0:
        show_examples(agent, tokenizer, raw, args.generate, args.max_length)


if __name__ == "__main__":
    main()
