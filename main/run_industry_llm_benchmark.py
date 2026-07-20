"""Industry-style small LLM optimizer benchmark.

This uses a recognized language-modeling benchmark dataset (WikiText-2),
GPT-2 BPE tokenization, and a tiny randomly initialized GPT-2 model. It is
still CPU-scale, but the data path and optimizer set are much closer to real
LLM training than the character-level smoke benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.optimization import Adafactor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main.run_llm_benchmark import HamiltonianGeometricTorch, Lion
from main.device_utils import add_torch_device_argument, resolve_torch_device


@dataclass(frozen=True)
class RunResult:
    optimizer: str
    runtime_s: float
    history: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]


class MuonLite(torch.optim.Optimizer):
    """Muon-style optimizer for CPU-scale tests.

    Matrix parameters use momentum followed by Newton-Schulz orthogonalization.
    Vector parameters use AdamW-style moments, matching common Muon training
    recipes where embeddings, biases, and normalization weights are handled by
    AdamW while dense matrices use Muon.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        adam_lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            adam_lr=adam_lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            adam_lr = group["adam_lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if parameter.ndim >= 2:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    buffer = state["momentum_buffer"]
                    if weight_decay:
                        parameter.mul_(1.0 - lr * weight_decay)
                    buffer.mul_(momentum).add_(grad)
                    update = _orthogonalize(buffer)
                    parameter.add_(update, alpha=-lr)
                else:
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    if weight_decay:
                        parameter.mul_(1.0 - adam_lr * weight_decay)
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** state["step"]
                    bias2 = 1.0 - beta2 ** state["step"]
                    parameter.addcdiv_(
                        exp_avg / bias1,
                        (exp_avg_sq / bias2).sqrt().add_(eps),
                        value=-adam_lr,
                    )
        return loss


def _orthogonalize(update: torch.Tensor, steps: int = 5) -> torch.Tensor:
    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / (matrix.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.T
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix
    if transposed:
        matrix = matrix.T
    return matrix.reshape(original_shape)


OPTIMIZER_NAMES = ("adamw", "adafactor", "lion", "muon_lite", "hamiltonian_geometric")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLM optimizers on WikiText-2 with GPT-2 tokenization.")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run all optimizers for multiple seeds and write an aggregate report.",
    )
    parser.add_argument("--hg-lr", type=float, default=3e-4)
    parser.add_argument("--hg-beta", type=float, default=0.9)
    parser.add_argument("--hg-metric-decay", type=float, default=0.99)
    parser.add_argument("--hg-metric-epsilon", type=float, default=1e-8)
    parser.add_argument("--hg-memory-decay", type=float, default=0.9)
    parser.add_argument("--hg-memory-coupling", type=float, default=0.01)
    parser.add_argument("--hg-weight-decay", type=float, default=0.01)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "downloads" / "huggingface")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "industry_llm_benchmark",
    )
    add_torch_device_argument(parser)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        device = resolve_torch_device(args.device, torch)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    train_tokens, val_tokens, vocab_size = load_tokenized_wikitext(args)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=None,
        eos_token_id=None,
    )
    param_count = sum(p.numel() for p in GPT2LMHeadModel(config).parameters())

    print("Industry-style LLM optimizer benchmark")
    print(f"dataset = {args.dataset}/{args.dataset_config}")
    print(f"tokenizer = {args.tokenizer}, vocab_size = {vocab_size}")
    print(f"device = {device}, parameters = {param_count:,}")

    seeds = args.seeds or [args.seed]
    seeded_results: list[tuple[int, list[RunResult]]] = []
    for seed in seeds:
        output_dir = args.output_dir if len(seeds) == 1 else args.output_dir / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        results = run_seed(seed, config, train_tokens, val_tokens, args, device)
        seeded_results.append((seed, results))
        write_outputs(results, args, device, param_count, vocab_size, output_dir, seed)
        print_results(results)
    if len(seeded_results) > 1:
        write_aggregate_outputs(seeded_results, args.output_dir)


def run_seed(
    seed: int,
    config: GPT2Config,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[RunResult]:
    set_seed(seed)
    initial_state = GPT2LMHeadModel(config).state_dict()
    results = []
    for name in OPTIMIZER_NAMES:
        # Reset model/dropout randomness and use a separate deterministic batch
        # generator so every optimizer sees the same samples for this seed.
        set_seed(seed)
        model = GPT2LMHeadModel(config).to(device)
        model.load_state_dict(initial_state)
        start = time.perf_counter()
        history = train_one(name, model, train_tokens, val_tokens, args, device, seed)
        results.append(RunResult(name, time.perf_counter() - start, history))
    return results


def print_results(results: list[RunResult]) -> None:
    print("optimizer,final_train_loss,final_val_loss,final_val_perplexity,runtime_s")
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        print(
            f"{result.optimizer},{final['train_loss']:.6e},{final['val_loss']:.6e},"
            f"{final['val_perplexity']:.6e},{result.runtime_s:.6e}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def load_tokenized_wikitext(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    dataset = load_dataset(args.dataset, args.dataset_config, cache_dir=str(args.cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=str(args.cache_dir))

    def encode(split: str) -> torch.Tensor:
        text = "\n\n".join(row["text"] for row in dataset[split] if row["text"].strip())
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    return encode("train"), encode("validation"), int(tokenizer.vocab_size)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def make_optimizer(name: str, model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    if name == "adafactor":
        return Adafactor(model.parameters(), lr=1e-3, relative_step=False, scale_parameter=False, warmup_init=False)
    if name == "lion":
        return Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)
    if name == "muon_lite":
        return MuonLite(model.parameters(), lr=2e-4, adam_lr=3e-4, weight_decay=0.01)
    if name == "hamiltonian_geometric":
        return HamiltonianGeometricTorch(
            model.parameters(),
            lr=args.hg_lr,
            beta=args.hg_beta,
            metric_decay=args.hg_metric_decay,
            metric_epsilon=args.hg_metric_epsilon,
            memory_decay=args.hg_memory_decay,
            memory_coupling=args.hg_memory_coupling,
            weight_decay=args.hg_weight_decay,
        )
    raise ValueError(f"unknown optimizer {name!r}")


def train_one(
    optimizer_name: str,
    model: nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    optimizer = make_optimizer(optimizer_name, model, args)
    train_generator = torch.Generator().manual_seed(seed)
    history = []
    running_loss = 0.0
    running_count = 0
    for step in range(1, args.max_steps + 1):
        model.train()
        inputs, targets = get_batch(train_tokens, args.block_size, args.batch_size, device, train_generator)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=inputs, labels=targets).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.item())
        running_count += 1
        if step % args.eval_every == 0 or step == args.max_steps:
            train_loss = running_loss / running_count
            running_loss = 0.0
            running_count = 0
            val_loss = evaluate(model, val_tokens, args, device, seed)
            history.append(
                {
                    "step": float(step),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": math.exp(min(val_loss, 20.0)),
                }
            )
            print(f"{optimizer_name} step={step} val_loss={val_loss:.4f}", flush=True)
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    losses = []
    eval_generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(args.eval_batches):
        inputs, targets = get_batch(val_tokens, args.block_size, args.batch_size, device, eval_generator)
        losses.append(float(model(input_ids=inputs, labels=targets).loss.item()))
    return sum(losses) / len(losses)


def write_outputs(
    results: list[RunResult],
    args: argparse.Namespace,
    device: torch.device,
    param_count: int,
    vocab_size: int,
    output_dir: Path,
    seed: int,
) -> None:
    history_path = output_dir / "industry_llm_training_history.csv"
    summary_path = output_dir / "industry_llm_summary.csv"
    loss_plot_path = output_dir / "industry_llm_validation_loss.png"
    perplexity_plot_path = output_dir / "industry_llm_validation_perplexity.png"
    report_path = output_dir / "industry_llm_results.md"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "step", "train_loss", "val_loss", "val_perplexity"])
        for result in results:
            for row in result.history:
                writer.writerow([result.optimizer, int(row["step"]), row["train_loss"], row["val_loss"], row["val_perplexity"]])

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "final_train_loss", "final_val_loss", "final_val_perplexity", "runtime_s"])
        for result in sorted(results, key=lambda item: item.final["val_loss"]):
            row = result.final
            writer.writerow([result.optimizer, row["train_loss"], row["val_loss"], row["val_perplexity"], result.runtime_s])

    export_plot(results, loss_plot_path, "val_loss", "validation loss")
    export_plot(results, perplexity_plot_path, "val_perplexity", "validation perplexity")

    lines = [
        "# Industry-Style LLM Optimizer Benchmark",
        "",
        "Tiny randomly initialized GPT-2 model trained on WikiText-2 with GPT-2 BPE tokenization.",
        "This is CPU-scale, not a substitute for billion-token pretraining, but it uses a standard LM dataset/tokenizer path.",
        "",
        f"Dataset: `{args.dataset}/{args.dataset_config}`",
        f"Tokenizer: `{args.tokenizer}`",
        f"Device: `{device}`",
        f"Vocabulary size: `{vocab_size}`",
        f"Model parameters: `{param_count:,}`",
        f"Layers: `{args.n_layer}`, heads: `{args.n_head}`, embedding dim: `{args.n_embd}`, context length: `{args.block_size}`",
        f"Training steps: `{args.max_steps}`, batch size: `{args.batch_size}`",
        f"Seed: `{seed}` (identical training and evaluation samples for every optimizer)",
        f"Hamiltonian-geometric settings: lr=`{args.hg_lr}`, beta=`{args.hg_beta}`, metric decay=`{args.hg_metric_decay}`, memory decay=`{args.hg_memory_decay}`, memory coupling=`{args.hg_memory_coupling}`, weight decay=`{args.hg_weight_decay}`",
        "",
        "| optimizer | final val loss | final val perplexity | runtime s |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        lines.append(f"| {result.optimizer} | {final['val_loss']:.4f} | {final['val_perplexity']:.3f} | {result.runtime_s:.2f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in (history_path, summary_path, loss_plot_path, perplexity_plot_path, report_path):
        print(f"exported = {path}")


def write_aggregate_outputs(seeded_results: list[tuple[int, list[RunResult]]], output_dir: Path) -> None:
    csv_path = output_dir / "industry_llm_seed_aggregate.csv"
    report_path = output_dir / "industry_llm_seed_aggregate.md"
    rows = []
    for optimizer in OPTIMIZER_NAMES:
        matches = [next(result for result in results if result.optimizer == optimizer) for _, results in seeded_results]
        losses = [result.final["val_loss"] for result in matches]
        perplexities = [result.final["val_perplexity"] for result in matches]
        runtimes = [result.runtime_s for result in matches]
        rows.append(
            {
                "optimizer": optimizer,
                "seeds": len(matches),
                "mean_val_loss": statistics.fmean(losses),
                "std_val_loss": statistics.stdev(losses),
                "mean_val_perplexity": statistics.fmean(perplexities),
                "std_val_perplexity": statistics.stdev(perplexities),
                "mean_runtime_s": statistics.fmean(runtimes),
            }
        )
    rows.sort(key=lambda row: row["mean_val_loss"])
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Industry LLM Multi-Seed Aggregate",
        "",
        f"Seeds: `{', '.join(str(seed) for seed, _ in seeded_results)}`",
        "",
        "| optimizer | mean val loss | std | mean perplexity | std | mean runtime s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['optimizer']} | {row['mean_val_loss']:.4f} | {row['std_val_loss']:.4f} | "
            f"{row['mean_val_perplexity']:.2f} | {row['std_val_perplexity']:.2f} | {row['mean_runtime_s']:.2f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exported = {csv_path}")
    print(f"exported = {report_path}")


def export_plot(results: list[RunResult], path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    for result in results:
        steps = [row["step"] for row in result.history]
        values = [row[metric] for row in result.history]
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.9
        ax.plot(steps, values, marker="o", markersize=3.5, linewidth=linewidth, label=result.optimizer)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"WikiText-2 GPT-2-tokenized {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
