"""Local MNIST optimizer benchmark.

This is a practical MNIST comparison for the project optimizers, sibling to
main/run_cifar10_benchmark.py. It is not an official MLCommons AlgoPerf
result. By default it streams only the samples it needs directly into
memory (see src/streaming_datasets.py) and never writes anything under
--data-dir, unlike the official AlgoPerf MNIST workload
(.external/algorithmic-efficiency), which requires a pre-downloaded,
prepared local copy by design.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MnistRunResult:
    optimizer: str
    runtime_s: float
    history: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]


class SmallMnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class HamiltonianGeometricTorch(torch.optim.Optimizer):
    """Same diagonal-metric PyTorch adaptation used in run_cifar10_benchmark.py."""

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        beta: float = 0.9,
        metric_decay: float = 0.99,
        metric_epsilon: float = 1e-8,
        memory_decay: float = 0.9,
        memory_coupling: float = 0.01,
        weight_decay: float = 1e-4,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark optimizers on MNIST with a small CNN.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "mnist_benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--data-source",
        choices=["auto", "streaming", "tfds"],
        default="auto",
        help=(
            "MNIST source. 'streaming' (the default under 'auto') fetches only the needed "
            "samples into memory via src.streaming_datasets and never writes anything under "
            "--data-dir. 'tfds' downloads and caches the full dataset under --data-dir, like "
            "the official AlgoPerf MNIST workload does."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    train_loader, val_loader = make_loaders(args)
    initial_state = SmallMnistCNN().state_dict()

    results = []
    for optimizer_name in ("adamw", "nesterov", "heavy_ball", "hamiltonian_geometric"):
        model = SmallMnistCNN().to(device)
        model.load_state_dict(initial_state)
        start = time.perf_counter()
        history = train_one_optimizer(optimizer_name, model, train_loader, val_loader, args, device)
        results.append(MnistRunResult(optimizer_name, time.perf_counter() - start, history))

    write_outputs(results, args, device)
    print("MNIST optimizer benchmark")
    print(f"data_source = {args.data_source}")
    print(f"device = {device}")
    print(f"epochs = {args.epochs}")
    print(f"train_samples = {args.train_samples}")
    print(f"val_samples = {args.val_samples}")
    print("optimizer,final_train_loss,final_train_accuracy,final_val_loss,final_val_accuracy,runtime_s")
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        print(
            f"{result.optimizer},{final['train_loss']:.6e},{final['train_accuracy']:.6e},"
            f"{final['val_loss']:.6e},{final['val_accuracy']:.6e},{result.runtime_s:.6e}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.data_source == "tfds":
        return make_tfds_loaders(args)
    if args.data_source == "streaming":
        return make_streaming_loaders(args)
    try:
        return make_streaming_loaders(args)
    except Exception:
        return make_tfds_loaders(args)


def make_streaming_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Fetch only the needed MNIST samples into memory and never touch
    --data-dir. See src/streaming_datasets.py for how the bytes are pulled
    and decoded."""

    from src.streaming_datasets import fetch_mnist_arrays

    total = args.train_samples + args.val_samples
    images, labels = fetch_mnist_arrays(total, train=True)
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    x = (x - 0.1307) / 0.3081
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        dataset,
        [args.train_samples, args.val_samples],
        generator=generator,
    )
    return (
        DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0),
        DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0),
    )


def make_tfds_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    import tensorflow_datasets as tfds

    total = args.train_samples + args.val_samples
    images, labels = tfds.as_numpy(
        tfds.load(
            "mnist",
            split=f"train[:{total}]",
            batch_size=-1,
            as_supervised=True,
            data_dir=str(args.data_dir),
            download=True,
        )
    )
    x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x = (x - 0.1307) / 0.3081
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        dataset,
        [args.train_samples, args.val_samples],
        generator=generator,
    )
    return (
        DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0),
        DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0),
    )


def make_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    if name == "nesterov":
        return torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    if name == "heavy_ball":
        return torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, nesterov=False, weight_decay=1e-4)
    if name == "hamiltonian_geometric":
        return HamiltonianGeometricTorch(model.parameters(), lr=3e-3, metric_decay=0.99, memory_coupling=0.01)
    raise ValueError(f"unknown optimizer {name!r}")


def train_one_optimizer(
    optimizer_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(optimizer_name, model)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item() * targets.numel()
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_count += targets.numel()
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / train_count,
                "train_accuracy": train_correct / train_count,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss_sum += loss.item() * targets.numel()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += targets.numel()
    return loss_sum / count, correct / count


def write_outputs(results: list[MnistRunResult], args: argparse.Namespace, device: torch.device) -> None:
    history_path = args.output_dir / "mnist_training_history.csv"
    summary_path = args.output_dir / "mnist_summary.csv"
    loss_plot_path = args.output_dir / "mnist_validation_loss.png"
    accuracy_plot_path = args.output_dir / "mnist_validation_accuracy.png"
    report_path = args.output_dir / "mnist_results.md"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for result in results:
            for row in result.history:
                writer.writerow(
                    [
                        result.optimizer,
                        f"{row['epoch']:.0f}",
                        f"{row['train_loss']:.12g}",
                        f"{row['train_accuracy']:.12g}",
                        f"{row['val_loss']:.12g}",
                        f"{row['val_accuracy']:.12g}",
                    ]
                )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "final_train_loss", "final_train_accuracy", "final_val_loss", "final_val_accuracy", "runtime_s"])
        for result in sorted(results, key=lambda item: item.final["val_loss"]):
            row = result.final
            writer.writerow(
                [
                    result.optimizer,
                    f"{row['train_loss']:.12g}",
                    f"{row['train_accuracy']:.12g}",
                    f"{row['val_loss']:.12g}",
                    f"{row['val_accuracy']:.12g}",
                    f"{result.runtime_s:.12g}",
                ]
            )

    export_plot(results, loss_plot_path, "val_loss", "validation loss")
    export_plot(results, accuracy_plot_path, "val_accuracy", "validation accuracy")

    lines = [
        "# MNIST Optimizer Benchmark",
        "",
        "This is a local PyTorch MNIST run, not an official MLCommons AlgoPerf score.",
        "",
        f"Data source: `{args.data_source}`",
        f"Device: `{device}`",
        f"Epochs: `{args.epochs}`",
        f"Train samples: `{args.train_samples}`",
        f"Validation samples: `{args.val_samples}`",
        "",
        "| optimizer | final val loss | final val accuracy | runtime s |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        lines.append(f"| {result.optimizer} | {final['val_loss']:.6f} | {final['val_accuracy']:.4f} | {result.runtime_s:.2f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in (history_path, summary_path, loss_plot_path, accuracy_plot_path, report_path):
        print(f"exported = {path}")


def export_plot(results: list[MnistRunResult], path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for result in results:
        epochs = [row["epoch"] for row in result.history]
        values = [row[metric] for row in result.history]
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.9
        ax.plot(epochs, values, marker="o", label=result.optimizer, linewidth=linewidth)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"MNIST {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
