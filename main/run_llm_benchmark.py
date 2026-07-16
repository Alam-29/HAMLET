"""Tiny character-level GPT-2 optimizer benchmark.

Compares hamiltonian_geometric against the optimizers actually used to train
large language models -- AdamW (GPT-2/3, LLaMA, ...), Adafactor (T5, PaLM),
and Lion (a newer sign-based optimizer used in some LLM training recipes) --
on a real transformers.GPT2LMHeadModel, trained from scratch (no pretrained
weights are downloaded) on character-level next-token prediction over the
"tiny Shakespeare" corpus.

This is intentionally toy-scale: a full-size LLM comparison is not feasible
on a CPU-only machine. What's real here is the architecture (an actual
Hugging Face GPT-2 model), the loss (actual causal language-modeling
cross-entropy), and the optimizers (the literal classes/algorithms used for
real LLM pretraining) -- just sized so the whole thing runs in a few minutes
on CPU, the same spirit as the project's MNIST/CIFAR-10 benchmarks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.optimization import Adafactor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
FALLBACK_CORPUS = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they would yield us but
the superfluity, while it were wholesome, we might guess they relieved us
humanely; but they think we are too dear.
""".strip()


@dataclass(frozen=True)
class LLMRunResult:
    optimizer: str
    runtime_s: float
    history: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]


class HamiltonianGeometricTorch(torch.optim.Optimizer):
    """Same diagonal-metric PyTorch adaptation used in the MNIST/CIFAR-10 benchmarks."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        beta: float = 0.9,
        metric_decay: float = 0.99,
        metric_epsilon: float = 1e-8,
        memory_decay: float = 0.9,
        memory_coupling: float = 0.01,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta=beta,
            metric_decay=metric_decay,
            metric_epsilon=metric_epsilon,
            memory_decay=memory_decay,
            memory_coupling=memory_coupling,
            weight_decay=weight_decay,
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
            beta = group["beta"]
            metric_decay = group["metric_decay"]
            metric_epsilon = group["metric_epsilon"]
            memory_decay = group["memory_decay"]
            memory_coupling = group["memory_coupling"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("HamiltonianGeometricTorch does not support sparse gradients")
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                    state["metric"] = torch.zeros_like(parameter)
                    state["memory"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                momentum = state["momentum"]
                metric = state["metric"]
                memory = state["memory"]
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                metric.mul_(metric_decay).addcmul_(grad, grad, value=1.0 - metric_decay)
                memory.mul_(memory_decay).add_(grad, alpha=1.0 - memory_decay)
                force = grad.add(memory - grad, alpha=memory_coupling)
                momentum.mul_(beta).add_(force, alpha=1.0 - beta)
                m_hat = momentum / (1.0 - beta**step)
                v_hat = metric / (1.0 - metric_decay**step)
                parameter.addcdiv_(m_hat, v_hat.sqrt().add_(metric_epsilon), value=-lr)
        return loss


class Lion(torch.optim.Optimizer):
    """Sign-based optimizer from "Symbolic Discovery of Optimization
    Algorithms" (Chen et al., 2023), used in some LLM training recipes for
    its lower memory footprint (one momentum buffer, no second moment)."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                update = exp_avg * beta1 + grad * (1.0 - beta1)
                parameter.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark optimizers used for LLM training on a tiny character-level GPT-2."
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "downloads" / "tiny_shakespeare" / "input.txt",
        help="Cached tiny Shakespeare corpus path. Downloaded once when network is available.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "llm_benchmark"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text, corpus_name = load_corpus(args.corpus_path)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    print("LLM optimizer benchmark (tiny character-level GPT-2)")
    print(f"corpus = {corpus_name}, {len(text)} chars, vocab_size = {vocab_size}")
    print(f"device = {device}")

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
    initial_state = GPT2LMHeadModel(config).state_dict()
    param_count = sum(p.numel() for p in GPT2LMHeadModel(config).parameters())
    print(f"model = GPT2LMHeadModel, {param_count:,} parameters, {args.n_layer} layers")

    results = []
    for optimizer_name in ("adamw", "adafactor", "lion", "hamiltonian_geometric"):
        model = GPT2LMHeadModel(config).to(device)
        model.load_state_dict(initial_state)
        start = time.perf_counter()
        history = train_one_optimizer(
            optimizer_name, model, train_data, val_data, args, device
        )
        results.append(LLMRunResult(optimizer_name, time.perf_counter() - start, history))

    write_outputs(results, args, device, vocab_size, param_count, corpus_name)
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


def load_corpus(corpus_path: Path) -> tuple[str, str]:
    if corpus_path.exists():
        return corpus_path.read_text(encoding="utf-8"), f"cached tiny Shakespeare ({corpus_path})"
    try:
        text = fetch_tiny_shakespeare()
    except RuntimeError as error:
        repeated = "\n\n".join(FALLBACK_CORPUS for _ in range(900))
        print(f"using built-in fallback corpus because tiny Shakespeare could not be fetched: {error}")
        return repeated, "built-in Shakespeare excerpt fallback"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(text, encoding="utf-8")
    return text, f"downloaded tiny Shakespeare ({corpus_path})"


def fetch_tiny_shakespeare(retries: int = 3, timeout: float = 45.0) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(TINY_SHAKESPEARE_URL, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as error:
            last_error = error
            print(f"fetch_tiny_shakespeare: attempt {attempt + 1}/{retries} failed ({error}), retrying...")
    raise RuntimeError(f"failed to fetch tiny Shakespeare corpus after {retries} attempts") from last_error


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def make_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    if name == "adafactor":
        return Adafactor(
            model.parameters(),
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    if name == "lion":
        return Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)
    if name == "hamiltonian_geometric":
        return HamiltonianGeometricTorch(model.parameters(), lr=3e-4, metric_decay=0.99, memory_coupling=0.01)
    raise ValueError(f"unknown optimizer {name!r}")


def train_one_optimizer(
    optimizer_name: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = make_optimizer(optimizer_name, model)
    history = []
    running_loss = 0.0
    running_count = 0
    for step in range(1, args.max_steps + 1):
        model.train()
        inputs, targets = get_batch(train_data, args.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        running_count += 1

        if step % args.eval_every == 0 or step == args.max_steps:
            train_loss = running_loss / running_count
            running_loss = 0.0
            running_count = 0
            val_loss = evaluate(model, val_data, args, device)
            history.append(
                {
                    "step": float(step),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": math.exp(min(val_loss, 20.0)),
                }
            )
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module, val_data: torch.Tensor, args: argparse.Namespace, device: torch.device, batches: int = 10
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(val_data, args.block_size, args.batch_size, device)
        outputs = model(input_ids=inputs, labels=targets)
        losses.append(outputs.loss.item())
    return sum(losses) / len(losses)


def write_outputs(
    results: list[LLMRunResult],
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
    param_count: int,
    corpus_name: str,
) -> None:
    history_path = args.output_dir / "llm_training_history.csv"
    summary_path = args.output_dir / "llm_summary.csv"
    loss_plot_path = args.output_dir / "llm_validation_loss.png"
    perplexity_plot_path = args.output_dir / "llm_validation_perplexity.png"
    report_path = args.output_dir / "llm_results.md"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "step", "train_loss", "val_loss", "val_perplexity"])
        for result in results:
            for row in result.history:
                writer.writerow(
                    [
                        result.optimizer,
                        f"{row['step']:.0f}",
                        f"{row['train_loss']:.12g}",
                        f"{row['val_loss']:.12g}",
                        f"{row['val_perplexity']:.12g}",
                    ]
                )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "final_train_loss", "final_val_loss", "final_val_perplexity", "runtime_s"])
        for result in sorted(results, key=lambda item: item.final["val_loss"]):
            row = result.final
            writer.writerow(
                [
                    result.optimizer,
                    f"{row['train_loss']:.12g}",
                    f"{row['val_loss']:.12g}",
                    f"{row['val_perplexity']:.12g}",
                    f"{result.runtime_s:.12g}",
                ]
            )

    export_plot(results, loss_plot_path, "val_loss", "validation loss (cross-entropy)")
    export_plot(results, perplexity_plot_path, "val_perplexity", "validation perplexity")

    lines = [
        "# LLM Optimizer Benchmark (Tiny Character-Level GPT-2)",
        "",
        "Real transformers.GPT2LMHeadModel (random init, no pretrained weights downloaded), "
        "real causal language-modeling loss, real optimizers used for LLM pretraining "
        "(AdamW, Adafactor, Lion) compared against hamiltonian_geometric. Sized to run on CPU.",
        "",
        f"Device: `{device}`",
        f"Corpus: {corpus_name} (character-level)",
        f"Vocabulary size: `{vocab_size}`",
        f"Model parameters: `{param_count:,}`",
        f"Layers: `{args.n_layer}`, heads: `{args.n_head}`, embedding dim: `{args.n_embd}`, "
        f"context length: `{args.block_size}`",
        f"Training steps: `{args.max_steps}`, batch size: `{args.batch_size}`",
        "",
        "| optimizer | final val loss | final val perplexity | runtime s |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        lines.append(
            f"| {result.optimizer} | {final['val_loss']:.4f} | {final['val_perplexity']:.3f} | "
            f"{result.runtime_s:.2f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in (history_path, summary_path, loss_plot_path, perplexity_plot_path, report_path):
        print(f"exported = {path}")


def export_plot(results: list[LLMRunResult], path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for result in results:
        steps = [row["step"] for row in result.history]
        values = [row[metric] for row in result.history]
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.9
        ax.plot(steps, values, marker="o", markersize=3.5, label=result.optimizer, linewidth=linewidth)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Tiny GPT-2 {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
