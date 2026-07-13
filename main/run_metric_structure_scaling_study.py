"""Metric-structure scaling study: scalar -> diagonal -> block-diagonal -> full.

Tests the framework's own central claim directly: on the SAME rotated,
ill-conditioned quadratic task, does the Hamiltonian-geometric optimizer's
advantage grow as the metric g is allowed to capture more of the true
curvature's off-diagonal coupling? All four metric variants below are exposed
as a metric_fn over the identical underlying Hessian `a` from
src.geometric_evidence.rotated_quadratic -- only how much of `a` the metric
is allowed to see changes, everything else (optimizer, learning rate, steps,
seeds, theta0) is held fixed so the comparison is fair.

  scalar:    c * I,  c = trace(a) / dim            (one number, no coupling)
  diagonal:  diag(a)                                (per-coordinate curvature)
  block:     a restricted to blocks of BLOCK_SIZE   (partial coupling)
  full:      a                                       (exact curvature + coupling)

Expectation being tested (not assumed): richer metrics should degrade more
gracefully as condition_number grows, because they capture more of the
coupling that makes the naive coordinate directions a poor eigenbasis.
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

STRUCTURE_COLORS = {
    "scalar": "#9aa3ab",
    "diagonal": "#e8a33d",
    "block": "#2ca089",
    "full": "#6a3d9a",
}


def make_metric_variant(a: np.ndarray, structure: str, block_size: int):
    dim = a.shape[0]
    if structure == "scalar":
        scale = float(np.trace(a) / dim)
        fixed = scale * np.eye(dim)
    elif structure == "diagonal":
        fixed = np.diag(np.diag(a))
    elif structure == "block":
        fixed = np.zeros_like(a)
        for start in range(0, dim, block_size):
            end = min(start + block_size, dim)
            fixed[start:end, start:end] = a[start:end, start:end]
    elif structure == "full":
        fixed = a.copy()
    else:
        raise ValueError(f"unknown structure {structure!r}")

    def metric_fn(_theta: np.ndarray) -> np.ndarray:
        return fixed

    return metric_fn


def run_study(
    condition_numbers: list[float],
    seeds: list[int],
    dim: int,
    block_size: int,
    steps: int,
    learning_rate: float,
) -> list[dict]:
    rows: list[dict] = []
    for condition in condition_numbers:
        for seed in seeds:
            loss_fn, grad_fn, _true_metric_fn, a = rotated_quadratic(dim, condition, seed=seed)
            rng = np.random.default_rng(1000 + seed)
            theta0 = rng.normal(size=dim)

            for structure in ("scalar", "diagonal", "block", "full"):
                metric_fn = make_metric_variant(a, structure, block_size)
                trace = run_optimizer(
                    "hamiltonian_geometric", theta0, loss_fn, grad_fn, metric_fn, steps, learning_rate
                )
                final_loss = float(trace.loss[-1])
                diverged = not np.isfinite(final_loss) or final_loss > 1e6 * max(trace.loss[0], 1e-300)
                rows.append(
                    {
                        "condition_number": float(condition),
                        "seed": seed,
                        "structure": structure,
                        "initial_loss": float(trace.loss[0]),
                        "final_loss": final_loss if np.isfinite(final_loss) else float("nan"),
                        "diverged": float(diverged),
                        "loss_reduction": float(trace.loss[0] / max(final_loss, 1e-300)) if np.isfinite(final_loss) else 0.0,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric-structure scaling ablation.")
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--condition-numbers", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "metric_structure_scaling_study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_study(args.condition_numbers, args.seeds, args.dim, args.block_size, args.steps, args.learning_rate)

    csv_path = args.output_dir / "metric_structure_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["condition_number", "seed", "structure", "initial_loss", "final_loss", "diverged", "loss_reduction"])
        for row in rows:
            writer.writerow(
                [
                    row["condition_number"], row["seed"], row["structure"],
                    f"{row['initial_loss']:.6g}", f"{row['final_loss']:.6g}" if np.isfinite(row["final_loss"]) else "nan",
                    int(row["diverged"]), f"{row['loss_reduction']:.6g}",
                ]
            )

    structures = ("scalar", "diagonal", "block", "full")
    summary_lines = ["# Metric-Structure Scaling Study", ""]
    summary_lines.append(
        f"Setting: dim={args.dim}, block_size={args.block_size}, steps={args.steps}, "
        f"learning_rate={args.learning_rate}, seeds={args.seeds}."
    )
    summary_lines.append("")
    summary_lines.append("Median final loss (lower is better) and divergence rate, by condition number and metric structure:")
    summary_lines.append("")
    summary_lines.append("| condition_number | " + " | ".join(structures) + " |")
    summary_lines.append("|---|" + "---|" * len(structures))

    plot_data = {structure: {"x": [], "y": [], "yerr": []} for structure in structures}

    for condition in args.condition_numbers:
        cells = []
        for structure in structures:
            subset = [r for r in rows if r["condition_number"] == condition and r["structure"] == structure]
            finite_losses = [r["final_loss"] for r in subset if np.isfinite(r["final_loss"])]
            divergence_rate = sum(r["diverged"] for r in subset) / len(subset)
            median_loss = float(np.median(finite_losses)) if finite_losses else float("nan")
            cells.append(f"{median_loss:.4g} (div={divergence_rate:.0%})")
            if finite_losses:
                plot_data[structure]["x"].append(condition)
                plot_data[structure]["y"].append(median_loss)
        summary_lines.append(f"| {condition:.0e} | " + " | ".join(cells) + " |")

    summary_lines.append("")
    summary_lines.append(
        "Interpretation: as condition_number grows, metric variants that see less of the true "
        "curvature's off-diagonal coupling (scalar, diagonal) are expected to show higher median "
        "final loss and/or higher divergence rate than block/full, since a coordinate-aligned "
        "metric is a progressively worse approximation to the rotated Hessian's eigenbasis. This "
        "table reports what was actually measured, including any cases where that expectation did "
        "not hold."
    )

    (args.output_dir / "metric_structure_scaling_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for structure in structures:
        data = plot_data[structure]
        if not data["x"]:
            continue
        ax.plot(data["x"], data["y"], marker="o", label=structure, color=STRUCTURE_COLORS[structure])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("condition number")
    ax.set_ylabel("median final loss (log scale)")
    ax.set_title(f"Metric-structure scaling (dim={args.dim}, {len(args.seeds)} seeds/point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "metric_structure_scaling.png", dpi=170)
    plt.close(fig)

    print("\n".join(summary_lines))
    print(f"exported = {csv_path}")
    print(f"exported = {args.output_dir / 'metric_structure_scaling_summary.md'}")
    print(f"exported = {args.output_dir / 'metric_structure_scaling.png'}")


if __name__ == "__main__":
    main()
