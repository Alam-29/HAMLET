"""Real deep-learning benchmark: a small CNN trained on Fashion-MNIST with
PyTorch autograd, comparing Hamiltonian-geometric against SGD+momentum,
AdamW, Lion, and Muon.

This is the one genuinely non-toy neural-network benchmark in this project:
real convolutional layers, real backprop (not finite differences), and a
real image dataset. CIFAR-10's official host was unreachable from this
environment (verified at <200 bytes/s); Fashion-MNIST is used instead, from
the same torchvision download path already used for MNIST elsewhere in this
project, downloaded from the reachable ossci-datasets S3 mirror.

Given this session's CPU-only, memory-constrained environment (no GPU, ~2GB
usable RAM), the dataset is subsampled and the network kept small -- this is
a small-scale but real CNN benchmark, not a claim of matching published
Fashion-MNIST leaderboards.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.torch_optimizers import HamiltonianGeometricTorch, LionTorch, MuonTorch

OPTIMIZER_COLORS = {
    "sgd_momentum": "#9aa3ab",
    "adamw": "#e8a33d",
    "lion": "#d1495b",
    "muon": "#2ca089",
    "hamiltonian_geometric": "#6a3d9a",
}


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data(data_dir: Path, train_samples: int, val_samples: int, seed: int):
    transform = torchvision.transforms.ToTensor()
    train_full = torchvision.datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_full = torchvision.datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    train_indices = rng.choice(len(train_full), size=train_samples, replace=False)
    val_indices = rng.choice(len(test_full), size=val_samples, replace=False)

    train_subset = torch.utils.data.Subset(train_full, train_indices.tolist())
    val_subset = torch.utils.data.Subset(test_full, val_indices.tolist())
    return train_subset, val_subset


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def build_optimizer(name: str, model: nn.Module):
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    if name == "lion":
        return LionTorch(model.parameters(), lr=0.0003, beta1=0.9, beta2=0.99, weight_decay=0.01)
    if name == "muon":
        return MuonTorch(model.parameters(), lr=0.01, momentum=0.9, fallback_lr=0.01)
    if name == "hamiltonian_geometric":
        return HamiltonianGeometricTorch(model.parameters(), lr=0.15, beta=0.9, metric_decay=0.96, metric_epsilon=0.08)
    raise ValueError(f"unknown optimizer {name!r}")


def train_one(name: str, train_subset, val_subset, args, device):
    torch.manual_seed(args.seed + {"sgd_momentum": 1, "adamw": 2, "lion": 3, "muon": 4, "hamiltonian_geometric": 5}[name])
    model = SmallCNN().to(device)
    optimizer = build_optimizer(name, model)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=256, shuffle=False)

    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        train_loss, train_accuracy = evaluate(model, train_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
        print(f"  [{name}] epoch {epoch}/{args.epochs}: val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}", flush=True)
    runtime_s = time.perf_counter() - start
    return {"name": name, "history": history, "runtime_s": runtime_s}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small CNN benchmark on Fashion-MNIST (PyTorch).")
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "fashion_mnist")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "cnn_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)

    train_subset, val_subset = load_data(args.data_dir, args.train_samples, args.val_samples, args.seed)

    results = []
    for name in ["sgd_momentum", "adamw", "lion", "muon", "hamiltonian_geometric"]:
        print(f"Training {name}...")
        results.append(train_one(name, train_subset, val_subset, args, device))

    history_path = args.output_dir / "cnn_benchmark_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
        for result in results:
            for row in result["history"]:
                writer.writerow(
                    [result["name"], row["epoch"], f"{row['train_loss']:.6f}", f"{row['val_loss']:.6f}", f"{row['train_accuracy']:.6f}", f"{row['val_accuracy']:.6f}"]
                )

    summary_path = args.output_dir / "cnn_benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "best_epoch", "best_val_loss", "best_val_accuracy", "final_val_loss", "final_val_accuracy", "runtime_s"])
        for result in sorted(results, key=lambda r: min(row["val_loss"] for row in r["history"])):
            best = min(result["history"], key=lambda row: row["val_loss"])
            final = result["history"][-1]
            writer.writerow(
                [result["name"], best["epoch"], f"{best['val_loss']:.6f}", f"{best['val_accuracy']:.6f}", f"{final['val_loss']:.6f}", f"{final['val_accuracy']:.6f}", f"{result['runtime_s']:.3f}"]
            )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for result in results:
        epochs = [row["epoch"] for row in result["history"]]
        val_loss = [row["val_loss"] for row in result["history"]]
        style = "-" if result["name"] == "hamiltonian_geometric" else "--"
        linewidth = 2.6 if result["name"] == "hamiltonian_geometric" else 1.6
        ax.plot(epochs, val_loss, style, linewidth=linewidth, label=result["name"], color=OPTIMIZER_COLORS.get(result["name"]))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (log scale)")
    ax.set_title(f"Small CNN on Fashion-MNIST ({args.train_samples} train images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "cnn_benchmark_loss.png", dpi=170)
    plt.close(fig)

    print("\nFinal ranking by best-epoch validation loss:")
    print("optimizer,best_epoch,best_val_loss,best_val_accuracy,runtime_s")
    for result in sorted(results, key=lambda r: min(row["val_loss"] for row in r["history"])):
        best = min(result["history"], key=lambda row: row["val_loss"])
        print(f"{result['name']},{best['epoch']},{best['val_loss']:.6f},{best['val_accuracy']:.6f},{result['runtime_s']:.3f}")
    print(f"exported = {history_path}")
    print(f"exported = {summary_path}")
    print(f"exported = {args.output_dir / 'cnn_benchmark_loss.png'}")


if __name__ == "__main__":
    main()
