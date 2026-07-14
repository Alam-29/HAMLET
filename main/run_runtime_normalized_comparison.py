"""Runtime-normalized comparison: does the advantage survive when optimizers
are compared by wall-clock time and function-evaluation count, not just by
step count?

Every other table in this project reports final loss after a fixed number of
*steps*. That is fair only if every optimizer's step costs the same amount of
compute. It does not: Hamiltonian-geometric with the energy-backtracking
safeguard enabled can spend up to (max_energy_backtracks + 1) loss
evaluations on a single step, versus exactly one gradient evaluation and zero
loss evaluations for plain SGD. This script instruments every optimizer with
call counters and a wall-clock timer on the identical rotated-quadratic task,
then reports final loss against three different budgets: optimizer steps,
wall-clock time, and total (loss + gradient) function evaluations -- so a
reader can see directly whether the ranking changes once compute cost is
accounted for.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_evidence import rotated_quadratic
from src.hamiltonian_geometric import HamiltonianGeometricConfig, hamiltonian_geometric_step, initial_state

OPTIMIZER_COLORS = {
    "sgd": "#9aa3ab",
    "heavy_ball": "#4c72b0",
    "adam": "#e8a33d",
    "entropy_descent": "#2ca089",
    "hamiltonian_geometric_no_backtrack": "#c96f9a",
    "hamiltonian_geometric_with_backtrack": "#6a3d9a",
}


class CallCounter:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def run_instrumented(name: str, theta0: np.ndarray, loss_fn, grad_fn, metric_fn, steps: int, learning_rate: float):
    counted_loss = CallCounter(loss_fn)
    counted_grad = CallCounter(grad_fn)

    start = time.perf_counter()
    theta = theta0.astype(float, copy=True)
    velocity = np.zeros_like(theta)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    metric_accumulator = np.zeros_like(theta)

    if name in ("hamiltonian_geometric_no_backtrack", "hamiltonian_geometric_with_backtrack"):
        with_backtrack = name.endswith("with_backtrack")
        state = initial_state(theta.size)
        state = state.__class__(parameters=theta.copy(), momentum=state.momentum, memory=state.memory, memory_metric=state.memory_metric)
        config = HamiltonianGeometricConfig(
            learning_rate=learning_rate, beta=0.7, metric_regularization=1e-3, memory_coupling=0.0,
            use_geometric_correction=False,
            max_energy_backtracks=8 if with_backtrack else 0,
            energy_backtrack_factor=0.5,
        )
        for _ in range(steps):
            state = hamiltonian_geometric_step(
                state, counted_grad, metric_fn, config,
                loss_fn=counted_loss if with_backtrack else None,
            )
        theta = state.parameters
        final_loss = float(counted_loss(theta))
    else:
        for step in range(1, steps + 1):
            grad = counted_grad(theta)
            if name == "sgd":
                theta = theta - learning_rate * grad
            elif name == "heavy_ball":
                velocity = 0.85 * velocity + learning_rate * grad
                theta = theta - velocity
            elif name == "adam":
                adam_m = 0.9 * adam_m + 0.1 * grad
                adam_v = 0.999 * adam_v + 0.001 * grad**2
                m_hat = adam_m / (1.0 - 0.9**step)
                v_hat = adam_v / (1.0 - 0.999**step)
                theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif name == "entropy_descent":
                metric_accumulator = 0.95 * metric_accumulator + 0.05 * grad**2
                theta = theta - learning_rate * grad / (np.sqrt(metric_accumulator) + 0.05)
            else:
                raise ValueError(f"unknown optimizer {name!r}")
        final_loss = float(counted_loss(theta))

    wall_clock_s = time.perf_counter() - start
    return {
        "name": name,
        "final_loss": final_loss,
        "wall_clock_s": wall_clock_s,
        "loss_calls": counted_loss.calls,
        "grad_calls": counted_grad.calls,
        "total_fn_evals": counted_loss.calls + counted_grad.calls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime-normalized optimizer comparison.")
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--condition-number", type=float, default=1e5)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "runtime_normalized_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    optimizer_lrs = {
        "sgd": 0.5 / args.condition_number,
        "heavy_ball": 0.3 / args.condition_number,
        "adam": 0.05,
        "entropy_descent": 0.15,
        "hamiltonian_geometric_no_backtrack": 0.18,
        "hamiltonian_geometric_with_backtrack": 0.18,
    }

    rows = []
    for seed in args.seeds:
        loss_fn, grad_fn, metric_fn, _a = rotated_quadratic(args.dim, args.condition_number, seed=seed)
        rng = np.random.default_rng(3000 + seed)
        theta0 = rng.normal(size=args.dim)
        for name, lr in optimizer_lrs.items():
            result = run_instrumented(name, theta0, loss_fn, grad_fn, metric_fn, args.steps, lr)
            result["seed"] = seed
            rows.append(result)

    csv_path = args.output_dir / "runtime_normalized_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "seed", "final_loss", "wall_clock_s", "loss_calls", "grad_calls", "total_fn_evals"])
        for row in rows:
            writer.writerow([row["name"], row["seed"], f"{row['final_loss']:.6g}", f"{row['wall_clock_s']:.6f}",
                              row["loss_calls"], row["grad_calls"], row["total_fn_evals"]])

    names = list(optimizer_lrs.keys())
    summary_lines = ["# Runtime-Normalized Comparison", ""]
    summary_lines.append(
        f"Task: rotated quadratic, dim={args.dim}, condition_number={args.condition_number:.0e}, "
        f"{args.steps} steps, {len(args.seeds)} seeds. `hamiltonian_geometric_with_backtrack` enables "
        "the energy-backtracking safeguard (up to 8 extra loss evaluations per step); "
        "`hamiltonian_geometric_no_backtrack` is the same optimizer with it disabled."
    )
    summary_lines.append("")
    summary_lines.append("| optimizer | median final loss | median wall-clock (s) | median total fn evals | loss per fn-eval budget |")
    summary_lines.append("|---|---|---|---|---|")
    plot_rows = []
    for name in names:
        subset = [r for r in rows if r["name"] == name]
        median_loss = float(np.median([r["final_loss"] for r in subset]))
        median_wall = float(np.median([r["wall_clock_s"] for r in subset]))
        median_evals = float(np.median([r["total_fn_evals"] for r in subset]))
        summary_lines.append(f"| {name} | {median_loss:.4g} | {median_wall:.5f} | {median_evals:.0f} | {median_loss:.4g} |")
        plot_rows.append({"name": name, "median_loss": median_loss, "median_wall": median_wall, "median_evals": median_evals})
    summary_lines.append("")
    summary_lines.append(
        "Interpretation: comparing the two Hamiltonian-geometric rows isolates the backtracking "
        "safeguard's real compute cost -- if `with_backtrack` needs meaningfully more wall-clock "
        "time or function evaluations than `no_backtrack` for the same or similar final loss, that "
        "cost is disclosed here directly rather than hidden inside a step-count-only comparison. "
        "Rankings by step count, wall-clock time, and function-eval count are reported side by side "
        "so a step-count-only win is not silently presented as a compute-normalized win."
    )
    (args.output_dir / "runtime_normalized_comparison_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for name in names:
        subset = [r for r in rows if r["name"] == name]
        wall = [r["wall_clock_s"] for r in subset]
        evals = [r["total_fn_evals"] for r in subset]
        loss = [r["final_loss"] for r in subset]
        color = OPTIMIZER_COLORS.get(name, "#333333")
        axes[0].scatter(wall, loss, label=name, color=color, alpha=0.8)
        axes[1].scatter(evals, loss, label=name, color=color, alpha=0.8)
    for ax, xlabel in zip(axes, ["wall-clock time (s)", "total function evaluations"]):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("final loss (log scale)")
    axes[0].set_title("Final loss vs. wall-clock time")
    axes[1].set_title("Final loss vs. function evaluations")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(args.output_dir / "runtime_normalized_comparison.png", dpi=170)
    plt.close(fig)

    print("\n".join(summary_lines))
    print(f"exported = {csv_path}")
    print(f"exported = {args.output_dir / 'runtime_normalized_comparison_summary.md'}")
    print(f"exported = {args.output_dir / 'runtime_normalized_comparison.png'}")


if __name__ == "__main__":
    main()
