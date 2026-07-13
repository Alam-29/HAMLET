"""Robustness follow-up to run_modern_optimizer_benchmark.py.

The single-seed benchmark found Hamiltonian-geometric in a near three-way tie
with Adam and Muon. This script checks whether that holds up (1) across
seeds, at the benchmark's default architecture, and (2) across problem size,
at the benchmark's default seed, using the exact same tuned hyperparameters
found there -- no re-tuning per seed or per size, so this is a direct test of
robustness, not a search for a new best case.
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
from main.run_modern_optimizer_benchmark import (
    train_adam_baseline,
    train_hamiltonian_geometric,
    train_lion_or_adamw,
    train_shampoo_or_muon,
)

OPTIMIZER_COLORS = {
    "adam": "#4f83b5",
    "hamiltonian_geometric": "#6a3d9a",
    "muon": "#2ca089",
    "adamw": "#e8a33d",
    "lion": "#d1495b",
    "shampoo": "#9aa3ab",
}


def train_one(name: str, theta0, shape, x_train, y_train, x_val, y_val, args):
    if name == "adam":
        return train_adam_baseline(theta0, shape, x_train, y_train, x_val, y_val, args)
    if name in ("lion", "adamw"):
        return train_lion_or_adamw(name, theta0, shape, x_train, y_train, x_val, y_val, args)
    if name in ("shampoo", "muon"):
        return train_shampoo_or_muon(name, theta0, shape, x_train, y_train, x_val, y_val, args)
    if name == "hamiltonian_geometric":
        return train_hamiltonian_geometric(theta0, shape, x_train, y_train, x_val, y_val, args)
    raise ValueError(f"unknown optimizer {name!r}")


OPTIMIZERS = ["adam", "hamiltonian_geometric", "muon", "adamw", "lion", "shampoo"]


def run_seed_sweep(base_args, seeds) -> list[dict]:
    rows = []
    for seed in seeds:
        args = argparse.Namespace(**{**vars(base_args), "seed": seed})
        shape = MLPShape(hidden_dim=args.hidden_dim)
        x_train, y_train = make_spiral_dataset(args.train_samples, seed=args.seed)
        x_val, y_val = make_spiral_dataset(args.val_samples, seed=args.seed + 1)
        theta0 = init_parameters(shape, seed=args.seed + 2)
        for name in OPTIMIZERS:
            result = train_one(name, theta0, shape, x_train, y_train, x_val, y_val, args)
            best = result.best
            rows.append(
                {
                    "seed": seed,
                    "optimizer": name,
                    "best_val_loss": best["val_loss"],
                    "best_val_accuracy": best["val_accuracy"],
                    "runtime_s": result.runtime_s,
                }
            )
            print(f"seed={seed} {name}: best_val_loss={best['val_loss']:.5f}", flush=True)
    return rows


def run_scaling_sweep(base_args, hidden_dims) -> list[dict]:
    rows = []
    for hidden_dim in hidden_dims:
        args = argparse.Namespace(**{**vars(base_args), "hidden_dim": hidden_dim})
        shape = MLPShape(hidden_dim=hidden_dim)
        x_train, y_train = make_spiral_dataset(args.train_samples, seed=args.seed)
        x_val, y_val = make_spiral_dataset(args.val_samples, seed=args.seed + 1)
        theta0 = init_parameters(shape, seed=args.seed + 2)
        for name in OPTIMIZERS:
            result = train_one(name, theta0, shape, x_train, y_train, x_val, y_val, args)
            best = result.best
            rows.append(
                {
                    "hidden_dim": hidden_dim,
                    "optimizer": name,
                    "best_val_loss": best["val_loss"],
                    "best_val_accuracy": best["val_accuracy"],
                    "runtime_s": result.runtime_s,
                }
            )
            print(f"hidden_dim={hidden_dim} {name}: best_val_loss={best['val_loss']:.5f}", flush=True)
    return rows


def export_seed_sweep_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["seed", "optimizer", "best_val_loss", "best_val_accuracy", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)


def export_scaling_sweep_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["hidden_dim", "optimizer", "best_val_loss", "best_val_accuracy", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)


def export_seed_sweep_boxplot(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for name in OPTIMIZERS:
        losses = [row["best_val_loss"] for row in rows if row["optimizer"] == name]
        data.append(losses)
        labels.append(name)
    box = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for patch, name in zip(box["boxes"], labels):
        patch.set_facecolor(OPTIMIZER_COLORS.get(name, "#cccccc"))
        patch.set_alpha(0.65)
    ax.set_yscale("log")
    ax.set_ylabel("best-epoch validation loss (log scale)")
    ax.set_title("Multi-seed spread of best-epoch validation loss")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def export_scaling_plot(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    hidden_dims = sorted({row["hidden_dim"] for row in rows})
    for name in OPTIMIZERS:
        ys = [
            next(row["best_val_loss"] for row in rows if row["optimizer"] == name and row["hidden_dim"] == hd)
            for hd in hidden_dims
        ]
        style = "-o" if name == "hamiltonian_geometric" else "--o"
        linewidth = 2.6 if name == "hamiltonian_geometric" else 1.6
        ax.plot(hidden_dims, ys, style, linewidth=linewidth, label=name, color=OPTIMIZER_COLORS.get(name))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("hidden layer width")
    ax.set_ylabel("best-epoch validation loss (log scale)")
    ax.set_title("Best-epoch validation loss vs. model width (fixed hyperparameters)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def export_pareto_plot(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in OPTIMIZERS:
        matching = [row for row in rows if row["optimizer"] == name]
        mean_runtime = float(np.mean([row["runtime_s"] for row in matching]))
        mean_loss = float(np.mean([row["best_val_loss"] for row in matching]))
        ax.scatter(mean_runtime, mean_loss, s=110, color=OPTIMIZER_COLORS.get(name), label=name, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(name, (mean_runtime, mean_loss), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean runtime (s, log scale)")
    ax.set_ylabel("mean best-epoch validation loss (log scale)")
    ax.set_title("Compute cost vs. accuracy (mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed and scaling robustness study for the modern-optimizer benchmark.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=900)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "modern_optimizer_benchmark"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + 100 * i for i in range(args.num_seeds)]
    print(f"Running seed sweep over {len(seeds)} seeds: {seeds}")
    seed_rows = run_seed_sweep(args, seeds)
    export_seed_sweep_csv(seed_rows, args.output_dir / "modern_optimizer_seed_sweep.csv")
    export_seed_sweep_boxplot(seed_rows, args.output_dir / "modern_optimizer_seed_sweep.png")
    export_pareto_plot(seed_rows, args.output_dir / "modern_optimizer_pareto.png")

    print("\nSeed-sweep summary (mean +/- std of best-epoch val loss):")
    for name in OPTIMIZERS:
        losses = [row["best_val_loss"] for row in seed_rows if row["optimizer"] == name]
        print(f"{name}: {np.mean(losses):.5f} +/- {np.std(losses):.5f} (n={len(losses)})")

    hidden_dims = [8, 16, 24, 48, 96]
    print(f"\nRunning scaling sweep over hidden_dims: {hidden_dims}")
    scaling_rows = run_scaling_sweep(args, hidden_dims)
    export_scaling_sweep_csv(scaling_rows, args.output_dir / "modern_optimizer_scaling_sweep.csv")
    export_scaling_plot(scaling_rows, args.output_dir / "modern_optimizer_scaling_sweep.png")

    print("\nDone.")
    for name in (
        "modern_optimizer_seed_sweep.csv",
        "modern_optimizer_seed_sweep.png",
        "modern_optimizer_pareto.png",
        "modern_optimizer_scaling_sweep.csv",
        "modern_optimizer_scaling_sweep.png",
    ):
        print(f"exported = {args.output_dir / name}")


if __name__ == "__main__":
    main()
