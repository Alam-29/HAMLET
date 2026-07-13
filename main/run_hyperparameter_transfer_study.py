"""Hyperparameter-transfer study: tune the learning rate on task A, freeze it,
and measure how much performance is lost when it is applied unmodified to a
different task B -- versus retuning directly on B (the oracle upper bound).

Task A: rotated_quadratic(dim=10, condition_number=1e4)
Task B: rotated_quadratic(dim=16, condition_number=1e6)

Both tasks share the same generator (src.geometric_evidence.rotated_quadratic)
and the same five optimizers (src.geometric_evidence.run_optimizer), so the
only things that differ between A and B are dimensionality and conditioning
-- exactly the kind of task shift a hyperparameter chosen on one problem has
to survive to be useful on another. Only the learning rate is tuned; every
other per-optimizer constant (beta, decay rates, metric regularization) is
fixed inside run_optimizer, so this is a single-hyperparameter transfer test
applied identically to all five optimizers.
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

from src.geometric_evidence import rotated_quadratic, run_optimizer

OPTIMIZERS = ("sgd", "heavy_ball", "adam", "entropy_descent", "hamiltonian_geometric")
LEARNING_RATE_GRID = np.geomspace(1e-4, 1.0, 13)

OPTIMIZER_COLORS = {
    "sgd": "#9aa3ab",
    "heavy_ball": "#4c72b0",
    "adam": "#e8a33d",
    "entropy_descent": "#2ca089",
    "hamiltonian_geometric": "#6a3d9a",
}


def median_final_loss(name: str, lr: float, dim: int, condition: float, seeds, steps: int) -> float:
    losses = []
    for seed in seeds:
        loss_fn, grad_fn, metric_fn, _a = rotated_quadratic(dim, condition, seed=seed)
        rng = np.random.default_rng(2000 + seed)
        theta0 = rng.normal(size=dim)
        trace = run_optimizer(name, theta0, loss_fn, grad_fn, metric_fn, steps, lr)
        final_loss = float(trace.loss[-1])
        losses.append(final_loss if np.isfinite(final_loss) else 1e30)
    return float(np.median(losses))


def tune_learning_rate(name: str, dim: int, condition: float, seeds, steps: int):
    best_lr, best_loss = None, float("inf")
    for lr in LEARNING_RATE_GRID:
        loss = median_final_loss(name, lr, dim, condition, seeds, steps)
        if loss < best_loss:
            best_lr, best_loss = float(lr), loss
    return best_lr, best_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter-transfer study across a task shift.")
    parser.add_argument("--dim-a", type=int, default=10)
    parser.add_argument("--condition-a", type=float, default=1e4)
    parser.add_argument("--dim-b", type=int, default=16)
    parser.add_argument("--condition-b", type=float, default=1e6)
    parser.add_argument("--tune-seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(5, 15)))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "hyperparameter_transfer_study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in OPTIMIZERS:
        lr_a, loss_a_tuned = tune_learning_rate(name, args.dim_a, args.condition_a, args.tune_seeds, args.steps)
        lr_b, loss_b_oracle_tuned = tune_learning_rate(name, args.dim_b, args.condition_b, args.tune_seeds, args.steps)

        loss_transferred = median_final_loss(name, lr_a, args.dim_b, args.condition_b, args.eval_seeds, args.steps)
        loss_oracle = median_final_loss(name, lr_b, args.dim_b, args.condition_b, args.eval_seeds, args.steps)

        transfer_gap = loss_transferred / max(loss_oracle, 1e-300)
        rows.append(
            {
                "optimizer": name,
                "lr_tuned_on_a": lr_a,
                "lr_tuned_on_b": lr_b,
                "loss_transferred_a_to_b": loss_transferred,
                "loss_oracle_tuned_on_b": loss_oracle,
                "transfer_gap_ratio": transfer_gap,
            }
        )

    csv_path = args.output_dir / "hyperparameter_transfer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(list(rows[0].keys()))
        for row in rows:
            writer.writerow([row["optimizer"], f"{row['lr_tuned_on_a']:.6g}", f"{row['lr_tuned_on_b']:.6g}",
                              f"{row['loss_transferred_a_to_b']:.6g}", f"{row['loss_oracle_tuned_on_b']:.6g}",
                              f"{row['transfer_gap_ratio']:.6g}"])

    summary_lines = [
        "# Hyperparameter-Transfer Study",
        "",
        f"Task A: rotated quadratic, dim={args.dim_a}, condition_number={args.condition_a:.0e} "
        f"(learning rate tuned here, {len(args.tune_seeds)} seeds).",
        f"Task B: rotated quadratic, dim={args.dim_b}, condition_number={args.condition_b:.0e} "
        f"(different dimensionality and conditioning; evaluated on {len(args.eval_seeds)} held-out seeds).",
        "",
        "transfer_gap_ratio = (median loss using the task-A-tuned learning rate on task B) / "
        "(median loss using the learning rate retuned directly on task B). 1.0x = no cost to "
        "transferring; larger values mean the tuned hyperparameter generalized poorly.",
        "",
        "| optimizer | lr tuned on A | lr tuned on B | loss (A-tuned lr, on B) | loss (B-tuned lr, on B) | transfer gap |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["transfer_gap_ratio"]):
        summary_lines.append(
            f"| {row['optimizer']} | {row['lr_tuned_on_a']:.4g} | {row['lr_tuned_on_b']:.4g} | "
            f"{row['loss_transferred_a_to_b']:.4g} | {row['loss_oracle_tuned_on_b']:.4g} | "
            f"{row['transfer_gap_ratio']:.3g}x |"
        )
    summary_lines.append("")
    summary_lines.append(
        "Interpretation: this reports what was measured, not an assumed ranking -- an optimizer "
        "whose best learning rate is stable across this task shift will show a transfer gap near "
        "1x; one whose optimal learning rate is task-specific will show a larger gap."
    )
    (args.output_dir / "hyperparameter_transfer_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    sorted_rows = sorted(rows, key=lambda r: r["transfer_gap_ratio"])
    names = [r["optimizer"] for r in sorted_rows]
    gaps = [r["transfer_gap_ratio"] for r in sorted_rows]
    colors = [OPTIMIZER_COLORS[n] for n in names]
    ax.bar(names, gaps, color=colors)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="no transfer cost")
    ax.set_yscale("log")
    ax.set_ylabel("transfer gap ratio (log scale)")
    ax.set_title("Hyperparameter-transfer gap: A-tuned lr applied to B, vs. B-tuned oracle")
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(args.output_dir / "hyperparameter_transfer.png", dpi=170)
    plt.close(fig)

    print("\n".join(summary_lines))
    print(f"exported = {csv_path}")
    print(f"exported = {args.output_dir / 'hyperparameter_transfer_summary.md'}")
    print(f"exported = {args.output_dir / 'hyperparameter_transfer.png'}")


if __name__ == "__main__":
    main()
