"""Free/open-source replacement for the Wolfram optimizer benchmark pipeline.

This runner uses only the project's Python stack (NumPy, Matplotlib, Pillow via
Matplotlib) to rerun the Hamiltonian-geometric optimizer benchmark and export
CSV, PNG, GIF, and a compact Markdown result record.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_dashboard import export_optimizer_benchmark_html
from src.pinn import (
    PINNConfig,
    export_optimizer_summary,
    export_potential_grid,
    export_training_history,
    run_optimizer_comparison,
)
from src.visualization import export_optimizer_convergence_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hamiltonian-geometric optimizer benchmark with free Python tools."
    )
    parser.add_argument("--plate-width", type=float, default=0.02)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--domain-width", type=float, default=0.08)
    parser.add_argument("--domain-height", type=float, default=0.08)
    parser.add_argument("--voltage", type=float, default=1.0)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--collocation-points", type=int, default=250)
    parser.add_argument("--plate-points", type=int, default=40)
    parser.add_argument("--outer-boundary-points", type=int, default=60)
    parser.add_argument("--boundary-weight", type=float, default=80.0)
    parser.add_argument("--outer-boundary-weight", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--disable-geometric-correction", action="store_true")
    parser.add_argument("--disable-memory-correction", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "free_optimizer_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = PINNConfig(
        plate_width=args.plate_width,
        gap=args.gap,
        domain_width=args.domain_width,
        domain_height=args.domain_height,
        voltage=args.voltage,
        hidden_features=args.features,
        collocation_points=args.collocation_points,
        plate_points=args.plate_points,
        outer_boundary_points=args.outer_boundary_points,
        boundary_weight=args.boundary_weight,
        outer_boundary_weight=args.outer_boundary_weight,
        seed=args.seed,
    )
    hamiltonian_kwargs = {
        "spectral_weight": args.spectral_weight,
        "use_geometric_correction": not args.disable_geometric_correction,
        "use_memory_correction": not args.disable_memory_correction,
    }
    model, _dataset, results = run_optimizer_comparison(
        config, steps=args.steps, hamiltonian_kwargs=hamiltonian_kwargs
    )

    history_path = args.output_dir / "free_training_history.csv"
    summary_path = args.output_dir / "free_optimizer_summary.csv"
    convergence_plot_path = args.output_dir / "free_optimizer_convergence.png"
    dashboard_path = args.output_dir / "free_optimizer_convergence_dashboard.html"
    gif_path = args.output_dir / "free_optimizer_convergence.gif"
    symbolic_path = args.output_dir / "free_symbolic_derivation_check.txt"
    report_path = args.output_dir / "free_optimizer_results.md"

    export_training_history(results, str(history_path))
    export_optimizer_summary(results, str(summary_path))
    export_optimizer_convergence_png(results, str(convergence_plot_path))
    export_optimizer_benchmark_html(results, str(dashboard_path))
    export_convergence_gif(history_path, gif_path)

    best_result = min(results, key=lambda result: result.final_loss)
    potential_path = args.output_dir / f"{best_result.optimizer}_potential_grid.csv"
    export_potential_grid(model, best_result.parameters, config, str(potential_path))

    write_symbolic_check(symbolic_path)
    write_markdown_report(report_path, args, results, best_result)

    print("Free/open-source Hamiltonian-geometric optimizer benchmark")
    print(f"features = {config.hidden_features}")
    print(f"steps = {args.steps}")
    print("optimizer,final_loss,pde_loss,plate_loss,outer_loss,gradient_norm,spectral_entropy")
    for result in results:
        print(
            f"{result.optimizer},"
            f"{result.final_loss:.6e},"
            f"{result.pde_loss:.6e},"
            f"{result.plate_loss:.6e},"
            f"{result.outer_loss:.6e},"
            f"{result.gradient_norm:.6e},"
            f"{result.spectral_entropy:.6e}"
        )
    print(f"best_optimizer = {best_result.optimizer}")
    for path in (
        history_path,
        summary_path,
        convergence_plot_path,
        dashboard_path,
        gif_path,
        symbolic_path,
        report_path,
        potential_path,
    ):
        print(f"exported = {path}")


def export_convergence_gif(history_path: Path, gif_path: Path) -> None:
    rows: list[dict[str, float | int | str]] = []
    with history_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "step": int(row["step"]),
                    "optimizer": row["optimizer"],
                    "loss": float(row["loss"]),
                }
            )

    optimizers: list[str] = []
    for row in rows:
        optimizer = str(row["optimizer"])
        if optimizer not in optimizers:
            optimizers.append(optimizer)

    series = {
        optimizer: [
            (int(row["step"]), float(row["loss"]))
            for row in rows
            if row["optimizer"] == optimizer
        ]
        for optimizer in optimizers
    }
    max_step = max(int(row["step"]) for row in rows)
    min_loss = min(float(row["loss"]) for row in rows)
    max_loss = max(float(row["loss"]) for row in rows)
    frame_count = min(36, max_step)
    frame_steps = sorted(
        {round(1 + index * (max_step - 1) / max(1, frame_count - 1)) for index in range(frame_count)}
    )
    colors = {
        "sgd": "#9aa3ab",
        "falling_ball": "#4f83b5",
        "adam": "#e8a33d",
        "entropy_descent": "#c0392b",
        "hamiltonian_geometric": "#6a3d9a",
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    def draw(frame_step: int):
        ax.clear()
        for optimizer in optimizers:
            points = [(x, y) for x, y in series[optimizer] if x <= frame_step]
            if not points:
                continue
            xs, ys = zip(*points)
            linewidth = 3.0 if optimizer in {"entropy_descent", "hamiltonian_geometric"} else 1.8
            ax.plot(xs, ys, label=optimizer, color=colors.get(optimizer, "#333333"), linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlim(1, max_step)
        ax.set_ylim(min_loss * 0.98, max_loss * 1.02)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss (log scale)")
        ax.set_title(f"Free Python optimizer benchmark, step {frame_step}/{max_step}")
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="upper right", fontsize=8)
        return ax.lines

    animation = FuncAnimation(fig, draw, frames=frame_steps, interval=90, blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def write_symbolic_check(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Hamiltonian-geometric symbolic derivation check",
                "",
                "H(theta, p) = 1/2 p^T g^{-1}(theta) p + L(theta)",
                "",
                "d theta / dt = partial H / partial p = g^{-1}(theta) p",
                "",
                "d p_i / dt = -partial H / partial theta_i",
                "           = -partial_i L(theta) - 1/2 p^T partial_i(g^{-1}(theta)) p",
                "",
                "With Rayleigh damping, memory, and spectral regularization:",
                "p_{t+1} = beta p_t - eta [grad L + F_geo + F_mem + alpha grad S(g)]",
                "theta_{t+1} = theta_t + eta g^{-1} p_{t+1}",
                "",
                "This is the update implemented in src/hamiltonian_geometric.py.",
            ]
        ),
        encoding="utf-8",
    )


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    results,
    best_result,
) -> None:
    ordered = sorted(results, key=lambda result: result.final_loss)
    lines = [
        "# Free/Open-Source Optimizer Benchmark Results",
        "",
        "Toolchain: Python, NumPy, Matplotlib, Pillow. No Mathematica/Wolfram dependency.",
        "",
        f"Steps: {args.steps}",
        f"Features: {args.features}",
        f"Collocation points: {args.collocation_points}",
        f"Best optimizer: `{best_result.optimizer}`",
        "",
        "| optimizer | final loss | gradient norm | spectral entropy |",
        "|---|---:|---:|---:|",
    ]
    for result in ordered:
        lines.append(
            f"| {result.optimizer} | {result.final_loss:.12g} | "
            f"{result.gradient_norm:.12g} | {result.spectral_entropy:.12g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
