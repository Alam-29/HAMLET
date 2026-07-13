"""Learning-rate sensitivity curves for Hamiltonian-geometric, on the two
benchmarks where tuning drove the largest swings (quantum kicked-top, modern-
optimizer spiral MLP). The question this answers: is the winning learning
rate a wide, forgiving basin, or a knife-edge that only looks good because it
was swept and picked? A wide basin around the chosen value is evidence the
tuning found a genuinely good region, not a lucky point estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main.run_deepobs_style_benchmark import MLPShape, init_parameters, make_spiral_dataset
from main.run_modern_optimizer_benchmark import train_hamiltonian_geometric as train_hg_modern
from src.quantum_chaos import QuantumChaosConfig, build_problem, train_quantum_optimizer


def quantum_lr_sweep(learning_rates, iterations: int = 80) -> list[dict]:
    config = QuantumChaosConfig(spin_j=4, steps=14, kick_strength=7.0, target_control_amplitude=0.42, seed=41)
    problem = build_problem(config)
    base = {
        "memory_coupling": 0.0, "memory_decay": 0.85, "metric_regularization": 0.02,
        "use_geometric_correction": False, "beta": 0.7,
        "max_energy_backtracks": 30, "energy_backtrack_factor": 0.85,
    }
    rows = []
    for lr in learning_rates:
        result = train_quantum_optimizer(problem, "hamiltonian_geometric", iterations, {**base, "learning_rate": lr})
        rows.append({"learning_rate": lr, "final_loss": result.final_loss, "final_fidelity": result.final_fidelity})
        print(f"quantum lr={lr}: fidelity={result.final_fidelity:.6f}", flush=True)
    return rows


def modern_benchmark_lr_sweep(learning_rates, args) -> list[dict]:
    shape = MLPShape(hidden_dim=args.hidden_dim)
    x_train, y_train = make_spiral_dataset(args.train_samples, seed=args.seed)
    x_val, y_val = make_spiral_dataset(args.val_samples, seed=args.seed + 1)
    theta0 = init_parameters(shape, seed=args.seed + 2)

    import inspect

    source = inspect.getsource(train_hg_modern)
    rows = []
    for lr in learning_rates:
        namespace: dict = {}
        patched = source.replace(
            "learning_rate, beta, memory_coupling = 0.19, 0.9, 0.0",
            f"learning_rate, beta, memory_coupling = {lr}, 0.9, 0.0",
        )
        exec(patched, train_hg_modern.__globals__, namespace)
        result = namespace["train_hamiltonian_geometric"](theta0, shape, x_train, y_train, x_val, y_val, args)
        best = result.best
        rows.append({"learning_rate": lr, "best_val_loss": best["val_loss"], "best_val_accuracy": best["val_accuracy"]})
        print(f"modern-benchmark lr={lr}: best_val_loss={best['val_loss']:.6f}", flush=True)
    return rows


def export_quantum_plot(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    lrs = [row["learning_rate"] for row in rows]
    fidelities = [row["final_fidelity"] for row in rows]
    ax.plot(lrs, fidelities, "-o", color="#6a3d9a", linewidth=2.0)
    ax.axvline(0.35, color="#999999", linestyle="--", linewidth=1.2, label="chosen lr = 0.35")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final fidelity")
    ax.set_title("Quantum kicked-top: HG fidelity vs. learning rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def export_modern_plot(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    lrs = [row["learning_rate"] for row in rows]
    losses = [row["best_val_loss"] for row in rows]
    ax.plot(lrs, losses, "-o", color="#6a3d9a", linewidth=2.0)
    ax.axvline(0.19, color="#999999", linestyle="--", linewidth=1.2, label="chosen lr = 0.19")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best-epoch validation loss (log scale)")
    ax.set_title("Modern-optimizer benchmark: HG loss vs. learning rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning-rate sensitivity curves for Hamiltonian-geometric.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=900)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "hyperparameter_sensitivity"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quantum_lrs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.38, 0.4, 0.45, 0.5]
    quantum_rows = quantum_lr_sweep(quantum_lrs)
    with (args.output_dir / "quantum_lr_sensitivity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["learning_rate", "final_loss", "final_fidelity"])
        writer.writeheader()
        writer.writerows(quantum_rows)
    export_quantum_plot(quantum_rows, args.output_dir / "quantum_lr_sensitivity.png")

    modern_lrs = [0.08, 0.1, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.28, 0.32]
    modern_rows = modern_benchmark_lr_sweep(modern_lrs, args)
    with (args.output_dir / "modern_benchmark_lr_sensitivity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["learning_rate", "best_val_loss", "best_val_accuracy"])
        writer.writeheader()
        writer.writerows(modern_rows)
    export_modern_plot(modern_rows, args.output_dir / "modern_benchmark_lr_sensitivity.png")

    print("\nDone.")
    for name in (
        "quantum_lr_sensitivity.csv",
        "quantum_lr_sensitivity.png",
        "modern_benchmark_lr_sensitivity.csv",
        "modern_benchmark_lr_sensitivity.png",
    ):
        print(f"exported = {args.output_dir / name}")


if __name__ == "__main__":
    main()
