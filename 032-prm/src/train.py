import argparse
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from agent import PRMAgent
from rl_common import plot_rewards

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "peiyi9979/Math-Shepherd"

# Math-Shepherd step separator token (Cyrillic "ki")
STEP_SEP = " ки"
# Upper bound on steps per solution in the dataset (~10 observed; 16 for headroom)
MAX_STEPS = 16


def parse_args():
    p = argparse.ArgumentParser(
        description="PRM: train a Process Reward Model on step-level math reasoning labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="pretrained backbone from HuggingFace Hub")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HuggingFace dataset with input/label fields (Math-Shepherd format)")
    p.add_argument("--dataset-size", type=int, default=2000,
                   help="number of training solutions to load")
    p.add_argument("--steps", type=int, default=500,
                   help="number of gradient update steps")
    p.add_argument("--batch-size", type=int, default=4,
                   help="solutions per batch")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="AdamW learning rate")
    p.add_argument("--max-length", type=int, default=512,
                   help="max tokenised length per solution")
    p.add_argument("--save-path", default="checkpoints/prm.pt",
                   help="where to save the trained PRM checkpoint")
    p.add_argument("--generate", type=int, default=3, metavar="N",
                   help="score N example solutions after training (0 to skip)")
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


def _parse_steps(input_text: str, label_text: str) -> list[tuple[str, float]]:
    """
    Parse a Math-Shepherd example into a list of (prefix_text, label) pairs.

    input_text lines end with ' ки'; the corresponding label_text lines end
    with ' +' (correct) or ' -' (incorrect).  A prefix includes all text up
    to and including the current step separator.
    """
    input_lines = input_text.split("\n")
    label_lines = label_text.split("\n")

    steps = []
    cumulative = ""
    for inp_line, lbl_line in zip(input_lines, label_lines):
        if not inp_line.strip():
            continue
        cumulative = cumulative + ("\n" if cumulative else "") + inp_line
        if inp_line.endswith(STEP_SEP):
            last_char = lbl_line.rstrip()[-1] if lbl_line.rstrip() else "+"
            label = 1.0 if last_char == "+" else 0.0
            steps.append((cumulative, label))
    return steps


class StepDataset(Dataset):
    """
    Each item is one full math solution.  Preprocessing tokenizes each step
    prefix to find the step-boundary token positions, then tokenizes the full
    solution once for the forward pass.

    step_positions and step_labels are fixed-length tensors padded with -1.
    """

    def __init__(self, raw, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict] = []

        for ex in tqdm(raw, desc="Preprocessing", leave=False):
            item = self._build(ex["input"], ex["label"])
            if item is not None:
                self.examples.append(item)

    def _build(self, input_text: str, label_text: str) -> dict | None:
        steps = _parse_steps(input_text, label_text)
        if not steps:
            return None

        step_positions = [-1] * MAX_STEPS
        step_labels = [-1.0] * MAX_STEPS

        for i, (prefix, lbl) in enumerate(steps[:MAX_STEPS]):
            enc = self.tokenizer(
                prefix,
                truncation=True,
                max_length=self.max_length,
            )
            # Last token of the prefix = last token of the step separator
            pos = min(len(enc["input_ids"]) - 1, self.max_length - 1)
            step_positions[i] = pos
            step_labels[i] = lbl

        full_enc = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids":       full_enc["input_ids"].squeeze(0),
            "attention_mask":  full_enc["attention_mask"].squeeze(0),
            "step_positions":  torch.tensor(step_positions, dtype=torch.long),
            "step_labels":     torch.tensor(step_labels, dtype=torch.float),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def train(agent: PRMAgent, loader: DataLoader, steps: int) -> tuple[list, list]:
    losses, accs = [], []
    it = iter(loader)
    pbar = tqdm(range(steps), desc="Training")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        loss, acc = agent.train_step(
            batch["input_ids"],
            batch["attention_mask"],
            batch["step_positions"],
            batch["step_labels"],
        )
        losses.append(loss)
        accs.append(acc)
        pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")
    return losses, accs


@torch.no_grad()
def evaluate(agent: PRMAgent, loader: DataLoader, n_batches: int = 25) -> float:
    agent.model.eval()
    correct = total = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        logits = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
        )
        positions = batch["step_positions"].to(agent.device)
        labels = batch["step_labels"].to(agent.device)

        valid = positions >= 0
        step_logits = logits.gather(1, positions.clamp(min=0))
        correct += ((step_logits[valid] > 0) == (labels[valid] > 0.5)).sum().item()
        total += valid.sum().item()
    return correct / total if total else 0.0


def show_examples(agent: PRMAgent, tokenizer, raw, n: int, max_length: int):
    print(f"\n{'─' * 60}")
    print(f"{'Step-level scoring':^60}")
    print(f"{'─' * 60}")

    for ex in list(raw)[:n]:
        steps = _parse_steps(ex["input"], ex["label"])
        if not steps:
            continue

        step_positions = []
        for prefix, _ in steps:
            enc = tokenizer(prefix, truncation=True, max_length=max_length)
            step_positions.append(min(len(enc["input_ids"]) - 1, max_length - 1))

        full_enc = tokenizer(
            ex["input"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        scores = agent.score_steps(
            full_enc["input_ids"],
            full_enc["attention_mask"],
            step_positions,
        )

        # Print the problem (first sentence of first line)
        problem_line = ex["input"].split("\n")[0]
        problem_preview = problem_line[:80] + ("…" if len(problem_line) > 80 else "")
        print(f"\nProblem: {problem_preview}")

        # Print each step with its predicted score and ground truth label
        input_lines = [l for l in ex["input"].split("\n") if l.endswith(STEP_SEP)]
        label_lines = [l for l in ex["label"].split("\n") if l.rstrip() and l.rstrip()[-1] in "+-"]

        for j, (score, inp_line, lbl_line) in enumerate(
            zip(scores, input_lines, label_lines)
        ):
            gt = lbl_line.rstrip()[-1]
            verdict = "✓" if (score > 0.5) == (gt == "+") else "✗"
            step_text = inp_line.replace(STEP_SEP, "").strip()
            print(
                f"  Step {j + 1} [{gt}] p={score:.3f} {verdict} | "
                f"{step_text[:55]}{'…' if len(step_text) > 55 else ''}"
            )

        # Solution-level score: minimum step probability (product drops too fast)
        solution_score = min(scores)
        print(f"  → solution score (min-step): {solution_score:.3f}")


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

    dataset = StepDataset(raw, tokenizer, args.max_length)
    print(f"Dataset built: {len(dataset)} solutions")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    agent = PRMAgent(args.model, args.lr, device)

    losses, accs = train(agent, loader, args.steps)

    acc = evaluate(agent, loader)
    print(f"\nStep accuracy (p>0.5 matches label): {acc:.1%}")

    agent.save(args.save_path)

    plot_rewards(accs, "PRM — Step-Level Accuracy")

    if args.generate > 0:
        show_examples(agent, tokenizer, raw, args.generate, args.max_length)


if __name__ == "__main__":
    main()
