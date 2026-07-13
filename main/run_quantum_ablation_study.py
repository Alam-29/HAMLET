import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum_chaos import QuantumChaosConfig, build_problem, train_quantum_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a critical ablation study for Hamiltonian-geometric quantum control."
    )
    parser.add_argument("--spin-j", type=int, default=4)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--kick-strength", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "quantum_ablation_study",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = QuantumChaosConfig(
        spin_j=args.spin_j,
        steps=args.steps,
        kick_strength=args.kick_strength,
        seed=args.seed,
    )
    problem = build_problem(config)
    variants = [
        (
            "hg_current_default",
            "hamiltonian_geometric",
            {
                "learning_rate": 0.16,
                "beta": 0.7,
                "memory_coupling": 0.0,
                "metric_regularization": 5e-2,
                "use_geometric_correction": False,
            },
        ),
        (
            "hg_lower_lr",
            "hamiltonian_geometric",
            {
                "learning_rate": 0.08,
                "beta": 0.7,
                "memory_coupling": 0.0,
                "metric_regularization": 5e-2,
                "use_geometric_correction": False,
            },
        ),
        (
            "hg_low_momentum",
            "hamiltonian_geometric",
            {
                "learning_rate": 0.16,
                "beta": 0.3,
                "memory_coupling": 0.0,
                "metric_regularization": 5e-2,
                "use_geometric_correction": False,
            },
        ),
        (
            "hg_with_memory",
            "hamiltonian_geometric",
            {
                "learning_rate": 0.16,
                "beta": 0.7,
                "memory_coupling": 0.02,
                "memory_decay": 0.85,
                "metric_regularization": 5e-2,
                "use_geometric_correction": False,
            },
        ),
        (
            "hg_with_geometric_force",
            "hamiltonian_geometric",
            {
                "learning_rate": 0.16,
                "beta": 0.7,
                "memory_coupling": 0.0,
                "metric_regularization": 5e-2,
                "use_geometric_correction": True,
            },
        ),
        (
            "hg_with_spectral_force",
            "hamiltonian_geometric",
            {
                "learning_rate": 0.16,
                "beta": 0.7,
                "memory_coupling": 0.0,
                "metric_regularization": 5e-2,
                "spectral_weight": 0.01,
                "use_geometric_correction": False,
            },
        ),
        ("entropy_descent", "entropy_descent", {"learning_rate": 0.08, "metric_regularization": 5e-2}),
        ("heavy_ball", "heavy_ball", {"learning_rate": 0.05, "momentum": 0.82}),
        ("adamw", "adamw", {"learning_rate": 0.035, "beta1": 0.9, "beta2": 0.995, "weight_decay": 1e-4}),
    ]
    results = []
    for label, optimizer, hyperparameters in variants:
        result = train_quantum_optimizer(problem, optimizer, args.iterations, hyperparameters)
        results.append((label, optimizer, hyperparameters, result))

    summary_path = args.output_dir / "quantum_ablation_summary.csv"
    history_path = args.output_dir / "quantum_ablation_history.csv"
    plot_path = args.output_dir / "quantum_ablation_fidelity.png"
    loss_plot_path = args.output_dir / "quantum_ablation_loss.png"
    report_path = args.output_dir / "quantum_ablation_report.md"
    write_summary(results, summary_path)
    write_history(results, history_path)
    export_plot(results, plot_path, "fidelity")
    export_plot(results, loss_plot_path, "loss")
    write_report(config, args.iterations, results, report_path)

    print("Quantum Hamiltonian-geometric ablation study")
    print(f"spin_j = {config.spin_j}")
    print(f"steps = {config.steps}")
    print(f"iterations = {args.iterations}")
    print("variant,optimizer,final_loss,final_fidelity,runtime_s,hyperparameters")
    for label, optimizer, hyperparameters, result in sorted(results, key=lambda row: row[3].final_loss):
        print(
            f"{label},{optimizer},{result.final_loss:.6e},"
            f"{result.final_fidelity:.6e},{result.runtime_s:.6e},{hyperparameters}"
        )
    for path in (summary_path, history_path, plot_path, loss_plot_path, report_path):
        print(f"exported = {path}")


def write_summary(results, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("variant,optimizer,final_loss,final_fidelity,runtime_s,spectral_entropy,hyperparameters\n")
        for label, optimizer, hyperparameters, result in sorted(results, key=lambda row: row[3].final_loss):
            file.write(
                f"{label},{optimizer},{result.final_loss:.12g},{result.final_fidelity:.12g},"
                f"{result.runtime_s:.12g},{result.spectral_entropy:.12g},\"{hyperparameters}\"\n"
            )


def write_history(results, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("iteration,variant,optimizer,loss,fidelity,gradient_norm\n")
        for label, optimizer, _hyperparameters, result in results:
            for index, loss in enumerate(result.loss_history, 1):
                file.write(
                    f"{index},{label},{optimizer},{loss:.12g},"
                    f"{result.fidelity_history[index - 1]:.12g},"
                    f"{result.gradient_norm_history[index - 1]:.12g}\n"
                )


def export_plot(results, path: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for label, optimizer, _hyperparameters, result in results:
        if metric == "fidelity":
            values = result.fidelity_history
            ylabel = "fidelity"
        else:
            values = result.loss_history
            ylabel = "infidelity loss"
            ax.set_yscale("log")
        linewidth = 2.8 if optimizer == "hamiltonian_geometric" else 1.8
        ax.plot(values, label=label, linewidth=linewidth)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Quantum ablation study: {ylabel}")
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(config: QuantumChaosConfig, iterations: int, results, path: Path) -> None:
    ordered = sorted(results, key=lambda row: row[3].final_loss)
    best_hg = min((row for row in results if row[1] == "hamiltonian_geometric"), key=lambda row: row[3].final_loss)
    best_baseline = min((row for row in results if row[1] != "hamiltonian_geometric"), key=lambda row: row[3].final_loss)
    lines = [
        "# Critical Quantum Ablation Study",
        "",
        "This ablation isolates which Hamiltonian-geometric terms help or hurt on the chaotic kicked-top control task.",
        "The result should be read as workload-specific evidence, not a universal optimizer ranking.",
        "",
        f"Spin j: {config.spin_j}",
        f"Hilbert dimension: {config.dimension}",
        f"Control steps: {config.steps}",
        f"Kick strength: {config.kick_strength}",
        f"Iterations: {iterations}",
        "",
        "| variant | optimizer | final loss | final fidelity | runtime s |",
        "|---|---|---:|---:|---:|",
    ]
    for label, optimizer, _hyperparameters, result in ordered:
        lines.append(
            f"| {label} | {optimizer} | {result.final_loss:.12g} | "
            f"{result.final_fidelity:.6f} | {result.runtime_s:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Critical Findings",
            "",
            f"Best HG variant: `{best_hg[0]}` with fidelity {best_hg[3].final_fidelity:.6f}.",
            f"Best non-HG baseline: `{best_baseline[0]}` with fidelity {best_baseline[3].final_fidelity:.6f}.",
            "",
            "The explicit geometric-force variant is expected to be slow because it differentiates a metric that",
            "already depends on finite-difference gradients. If it also underperforms, that is evidence that",
            "the current quantum metric is numerically noisy rather than a useful symmetry-exploiting object.",
            "",
            "Memory and spectral-force variants test whether extra Hamiltonian terms stabilize the chaotic",
            "landscape. If they reduce fidelity, the current issue is not missing forces; it is metric design",
            "and step scaling.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
