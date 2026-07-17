# /// script
# dependencies = [
#   "datasets>=2.18.0",
#   "huggingface_hub>=0.24.0",
#   "matplotlib>=3.8.0",
#   "torch>=2.3.0",
#   "transformers>=4.41.0",
# ]
# ///
"""Standalone Hugging Face Jobs LLM optimizer benchmark.

Run this with:

    hf jobs uv run hf_jobs/hf_industry_llm_benchmark.py --flavor a10g-small -- \
      --max-steps 1000 --upload-repo YOUR_USERNAME/hg-optimizer-llm-results

The script is intentionally self-contained so Hugging Face Jobs can execute it
without cloning this whole repository.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import time
from typing import Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from datasets import Dataset, DatasetDict, DownloadConfig, config as datasets_config, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.optimization import Adafactor


@dataclass(frozen=True)
class RunResult:
    optimizer: str
    learning_rate: float | None
    runtime_s: float
    history: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]


class BatchSource(Protocol):
    def reset(self) -> None:
        ...

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class HamiltonianGeometricTorch(torch.optim.Optimizer):
    """Diagonal Hamiltonian-geometric optimizer adaptation for PyTorch tensors."""

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
    """Lion optimizer, used in some modern LLM recipes for reduced state memory."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

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


class MuonLite(torch.optim.Optimizer):
    """Small Muon-style baseline with orthogonalized matrix updates."""

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
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, adam_lr=adam_lr, betas=betas, weight_decay=weight_decay, eps=eps),
        )

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
                    parameter.add_(_orthogonalize(buffer), alpha=-lr)
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
                    parameter.addcdiv_(exp_avg / bias1, (exp_avg_sq / bias2).sqrt().add_(eps), value=-adam_lr)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an HF-hosted LLM optimizer benchmark.")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--offline", action="store_true", help="Use only locally cached dataset/tokenizer files.")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset rows from Hugging Face instead of downloading the dataset.")
    parser.add_argument("--dataset-retries", type=int, default=5, help="Dataset/tokenizer load attempts for transient Hub errors.")
    parser.add_argument("--optimizers", default="adamw,adafactor,lion,muon_lite,hamiltonian_geometric")
    parser.add_argument("--learning-rate", type=float, default=None, help="Use one shared learning rate for every optimizer.")
    parser.add_argument("--tune-learning-rates", action="store_true", help="Run every optimizer over --lr-sweep and compare each optimizer's best result.")
    parser.add_argument("--lr-sweep", default="1e-4,3e-4,1e-3", help="Comma-separated learning rates used with --tune-learning-rates.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Use one shared weight decay for every optimizer that supports it.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:N. Default auto uses CUDA when available.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("hf_llm_benchmark_results"))
    parser.add_argument("--upload-repo", default="", help="Optional HF dataset repo id, e.g. username/hg-llm-results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.learning_rate is not None and args.tune_learning_rates:
        raise ValueError("--learning-rate and --tune-learning-rates are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_tokens, val_tokens, vocab_size = load_tokenized_dataset(args)
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

    print("HF Jobs LLM optimizer benchmark")
    print(f"dataset={args.dataset}/{args.dataset_config}")
    print(f"tokenizer={args.tokenizer}, vocab_size={vocab_size}")
    print(f"device={device}, parameters={param_count:,}")
    if args.learning_rate is not None:
        print(f"shared_learning_rate={args.learning_rate:.6e}")
    if args.tune_learning_rates:
        print(f"lr_sweep={','.join(f'{item:.6e}' for item in parse_lr_sweep(args.lr_sweep))}")
    if args.weight_decay is not None:
        print(f"shared_weight_decay={args.weight_decay:.6e}")

    results = []
    for name in [item.strip() for item in args.optimizers.split(",") if item.strip()]:
        for learning_rate in learning_rates_for_optimizer(args):
            run_args = argparse.Namespace(**vars(args))
            run_args.learning_rate = learning_rate
            model = GPT2LMHeadModel(config).to(device)
            model.load_state_dict(initial_state)
            start = time.perf_counter()
            history = train_one(name, model, train_tokens, val_tokens, run_args, device)
            results.append(RunResult(name, learning_rate, time.perf_counter() - start, history))

    write_outputs(results, args, device, param_count, vocab_size)
    print("optimizer,learning_rate,final_train_loss,final_val_loss,final_val_perplexity,runtime_s")
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        lr_text = "" if result.learning_rate is None else f"{result.learning_rate:.6e}"
        print(f"{result.optimizer},{lr_text},{final['train_loss']:.6e},{final['val_loss']:.6e},{final['val_perplexity']:.6e},{result.runtime_s:.6e}")
    if args.upload_repo:
        upload_outputs(args.upload_repo, args.output_dir)


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"requested --device {requested!r}, but PyTorch cannot access CUDA")
        if ":" in requested:
            _prefix, index_text = requested.split(":", 1)
            try:
                index = int(index_text)
            except ValueError as error:
                raise ValueError(f"invalid CUDA device {requested!r}; use cuda or cuda:N") from error
            device_count = torch.cuda.device_count()
            if index < 0 or index >= device_count:
                raise ValueError(
                    f"requested --device {requested!r}, but only {device_count} CUDA device(s) are visible"
                )
        return torch.device(requested)
    raise ValueError(f"unsupported --device {requested!r}; use auto, cpu, cuda, or cuda:N")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def parse_lr_sweep(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--lr-sweep must contain at least one learning rate")
    if any(value <= 0 for value in values):
        raise ValueError("--lr-sweep values must be positive")
    return values


def learning_rates_for_optimizer(args: argparse.Namespace) -> list[float | None]:
    if args.tune_learning_rates:
        return parse_lr_sweep(args.lr_sweep)
    return [args.learning_rate]


def load_tokenized_dataset(args: argparse.Namespace) -> tuple[torch.Tensor | BatchSource, torch.Tensor | BatchSource, int]:
    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        datasets_config.HF_DATASETS_OFFLINE = True
    if args.offline and args.streaming:
        raise ValueError("--offline and --streaming are mutually exclusive")
    dataset = _load_cached_arrow_dataset(args) if args.offline and args.cache_dir is not None else None
    download_config = DownloadConfig(local_files_only=args.offline)
    last_error: Exception | None = None
    attempts = 1 if args.offline else max(1, args.dataset_retries)
    for attempt in range(1, attempts + 1):
        try:
            if dataset is None:
                dataset = _load_streaming_parquet_dataset(args, cache_dir, download_config) or load_dataset(
                    args.dataset,
                    args.dataset_config,
                    cache_dir=cache_dir,
                    download_config=download_config,
                    streaming=args.streaming,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                cache_dir=cache_dir,
                local_files_only=args.offline,
            )
            break
        except Exception as error:
            last_error = error
            if attempt == attempts:
                mode = "cached files" if args.offline else "Hugging Face Hub/cache"
                raise RuntimeError(f"Could not load {args.dataset}/{args.dataset_config} from {mode}") from error
            delay = min(60, 2 ** (attempt - 1))
            print(f"Dataset/tokenizer load failed on attempt {attempt}/{attempts}: {error}")
            print(f"Retrying in {delay}s...")
            time.sleep(delay)
    else:
        raise RuntimeError("Could not load dataset/tokenizer") from last_error

    if args.streaming:
        return (
            StreamingTokenBatchSource(dataset["train"], tokenizer, args.block_size, args.batch_size),
            StreamingTokenBatchSource(dataset["validation"], tokenizer, args.block_size, args.batch_size),
            int(tokenizer.vocab_size),
        )

    def encode(split: str) -> torch.Tensor:
        text = "\n\n".join(row["text"] for row in dataset[split] if row["text"].strip())
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    return encode("train"), encode("validation"), int(tokenizer.vocab_size)


def _load_streaming_parquet_dataset(
    args: argparse.Namespace,
    cache_dir: str | None,
    download_config: DownloadConfig,
):
    if not args.streaming or args.dataset != "Salesforce/wikitext":
        return None
    base_url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/{args.dataset_config}"
    data_files = {
        "train": f"{base_url}/train-00000-of-00001.parquet",
        "validation": f"{base_url}/validation-00000-of-00001.parquet",
        "test": f"{base_url}/test-00000-of-00001.parquet",
    }
    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=cache_dir,
        download_config=download_config,
        streaming=True,
    )


def _load_cached_arrow_dataset(args: argparse.Namespace) -> DatasetDict | None:
    if args.dataset != "Salesforce/wikitext":
        return None
    cache_root = args.cache_dir / "Salesforce___wikitext" / args.dataset_config / "0.0.0"
    candidates = sorted(
        [path for path in cache_root.glob("*") if path.is_dir() and (path / "wikitext-train.arrow").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    dataset_dir = candidates[0]
    return DatasetDict(
        {
            "train": Dataset.from_file(str(dataset_dir / "wikitext-train.arrow")),
            "validation": Dataset.from_file(str(dataset_dir / "wikitext-validation.arrow")),
            "test": Dataset.from_file(str(dataset_dir / "wikitext-test.arrow")),
        }
    )


class StreamingTokenBatchSource:
    def __init__(self, split, tokenizer, block_size: int, batch_size: int) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.separator_id = tokenizer.eos_token_id
        self.reset()

    def reset(self) -> None:
        self.iterator = iter(self.split)
        self.buffer: list[int] = []

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self._next_sequence() for _ in range(self.batch_size)]
        batch = torch.tensor(rows, dtype=torch.long)
        return batch[:, :-1].to(device), batch[:, 1:].to(device)

    def _next_sequence(self) -> list[int]:
        while len(self.buffer) < self.block_size + 1:
            try:
                row = next(self.iterator)
            except StopIteration:
                self.reset()
                row = next(self.iterator)
            text = row.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            self.buffer.extend(ids)
            if self.separator_id is not None:
                self.buffer.append(int(self.separator_id))
        sequence = self.buffer[: self.block_size + 1]
        del self.buffer[: self.block_size]
        return sequence


def get_batch(
    data: torch.Tensor | BatchSource,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(data, "next_batch"):
        return data.next_batch(device)
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def make_optimizer(name: str, model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    def lr(default: float) -> float:
        return args.learning_rate if args.learning_rate is not None else default

    def weight_decay(default: float) -> float:
        return args.weight_decay if args.weight_decay is not None else default

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr(3e-4), betas=(0.9, 0.95), weight_decay=weight_decay(0.01))
    if name == "adafactor":
        return Adafactor(model.parameters(), lr=lr(1e-3), relative_step=False, scale_parameter=False, warmup_init=False)
    if name == "lion":
        return Lion(model.parameters(), lr=lr(1e-4), betas=(0.9, 0.99), weight_decay=weight_decay(0.01))
    if name == "muon_lite":
        shared_lr = lr(2e-4)
        adam_lr = args.learning_rate if args.learning_rate is not None else 3e-4
        return MuonLite(model.parameters(), lr=shared_lr, adam_lr=adam_lr, weight_decay=weight_decay(0.01))
    if name == "hamiltonian_geometric":
        return HamiltonianGeometricTorch(
            model.parameters(),
            lr=lr(3e-4),
            metric_decay=0.99,
            memory_coupling=0.01,
            weight_decay=weight_decay(0.01),
        )
    raise ValueError(f"unknown optimizer {name!r}")


def train_one(
    optimizer_name: str,
    model: nn.Module,
    train_tokens: torch.Tensor | BatchSource,
    val_tokens: torch.Tensor | BatchSource,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    set_seed(args.seed)
    if hasattr(train_tokens, "reset"):
        train_tokens.reset()
    optimizer = make_optimizer(optimizer_name, model, args)
    history = []
    running_loss = 0.0
    running_count = 0
    for step in range(1, args.max_steps + 1):
        model.train()
        inputs, targets = get_batch(train_tokens, args.block_size, args.batch_size, device)
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
            val_loss = evaluate(model, val_tokens, args, device)
            row = {
                "step": float(step),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": math.exp(min(val_loss, 20.0)),
            }
            history.append(row)
            lr_text = "" if args.learning_rate is None else f" lr={args.learning_rate:.1e}"
            print(f"{optimizer_name}{lr_text} step={step} val_loss={val_loss:.4f} ppl={row['val_perplexity']:.2f}", flush=True)
    return history


@torch.no_grad()
def evaluate(model: nn.Module, val_tokens: torch.Tensor | BatchSource, args: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    if hasattr(val_tokens, "reset"):
        val_tokens.reset()
    losses = []
    for _ in range(args.eval_batches):
        inputs, targets = get_batch(val_tokens, args.block_size, args.batch_size, device)
        losses.append(float(model(input_ids=inputs, labels=targets).loss.item()))
    return sum(losses) / len(losses)


def write_outputs(results: list[RunResult], args: argparse.Namespace, device: torch.device, param_count: int, vocab_size: int) -> None:
    history_path = args.output_dir / "hf_llm_training_history.csv"
    summary_path = args.output_dir / "hf_llm_summary.csv"
    best_summary_path = args.output_dir / "hf_llm_best_by_optimizer.csv"
    loss_plot_path = args.output_dir / "hf_llm_validation_loss.png"
    perplexity_plot_path = args.output_dir / "hf_llm_validation_perplexity.png"
    report_path = args.output_dir / "hf_llm_results.md"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "learning_rate", "run_label", "step", "train_loss", "val_loss", "val_perplexity"])
        for result in results:
            for row in result.history:
                writer.writerow(
                    [
                        result.optimizer,
                        result.learning_rate,
                        run_label(result),
                        int(row["step"]),
                        row["train_loss"],
                        row["val_loss"],
                        row["val_perplexity"],
                    ]
                )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "learning_rate", "run_label", "final_train_loss", "final_val_loss", "final_val_perplexity", "runtime_s"])
        for result in sorted(results, key=lambda item: item.final["val_loss"]):
            row = result.final
            writer.writerow([result.optimizer, result.learning_rate, run_label(result), row["train_loss"], row["val_loss"], row["val_perplexity"], result.runtime_s])

    best_results = best_results_by_optimizer(results)
    with best_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "best_learning_rate", "final_train_loss", "final_val_loss", "final_val_perplexity", "runtime_s"])
        for result in sorted(best_results, key=lambda item: item.final["val_loss"]):
            row = result.final
            writer.writerow([result.optimizer, result.learning_rate, row["train_loss"], row["val_loss"], row["val_perplexity"], result.runtime_s])

    export_plot(results, loss_plot_path, "val_loss", "validation loss")
    export_plot(results, perplexity_plot_path, "val_perplexity", "validation perplexity")

    lines = [
        "# Hugging Face Jobs LLM Optimizer Benchmark",
        "",
        "Randomly initialized GPT-style language model trained on a Hugging Face dataset.",
        "",
        f"Dataset: `{args.dataset}/{args.dataset_config}`",
        f"Tokenizer: `{args.tokenizer}`",
        f"Device: `{device}`",
        f"Vocabulary size: `{vocab_size}`",
        f"Model parameters: `{param_count:,}`",
        f"Layers: `{args.n_layer}`, heads: `{args.n_head}`, embedding dim: `{args.n_embd}`, context length: `{args.block_size}`",
        f"Training steps: `{args.max_steps}`, batch size: `{args.batch_size}`",
        "",
        "## All Runs",
        "",
        "| optimizer | learning rate | final val loss | final val perplexity | runtime s |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        lr_text = "default" if result.learning_rate is None else f"{result.learning_rate:.1e}"
        lines.append(f"| {result.optimizer} | {lr_text} | {final['val_loss']:.4f} | {final['val_perplexity']:.3f} | {result.runtime_s:.2f} |")
    if args.tune_learning_rates:
        lines.extend(
            [
                "",
                "## Best Learning Rate Per Optimizer",
                "",
                "| optimizer | best learning rate | final val loss | final val perplexity | runtime s |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for result in sorted(best_results, key=lambda item: item.final["val_loss"]):
            final = result.final
            lines.append(f"| {result.optimizer} | {result.learning_rate:.1e} | {final['val_loss']:.4f} | {final['val_perplexity']:.3f} | {result.runtime_s:.2f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in (history_path, summary_path, best_summary_path, loss_plot_path, perplexity_plot_path, report_path):
        print(f"exported={path}")


def export_plot(results: list[RunResult], path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for result in results:
        steps = [row["step"] for row in result.history]
        values = [row[metric] for row in result.history]
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.9
        ax.plot(steps, values, marker="o", markersize=3.3, linewidth=linewidth, label=run_label(result))
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by optimizer")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_label(result: RunResult) -> str:
    if result.learning_rate is None:
        return result.optimizer
    return f"{result.optimizer}@{result.learning_rate:.1e}"


def best_results_by_optimizer(results: list[RunResult]) -> list[RunResult]:
    best: dict[str, RunResult] = {}
    for result in results:
        current = best.get(result.optimizer)
        if current is None or result.final["val_loss"] < current.final["val_loss"]:
            best[result.optimizer] = result
    return list(best.values())


def upload_outputs(repo_id: str, output_dir: Path) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=output_dir.name,
        commit_message=f"Add LLM optimizer benchmark results from {output_dir.name}",
    )
    print(f"uploaded={repo_id}/{output_dir.name}")


if __name__ == "__main__":
    main()
