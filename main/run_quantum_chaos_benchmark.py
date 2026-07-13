import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum_chaos import (
    QuantumChaosConfig,
    export_quantum_controls,
    export_quantum_history,
    export_quantum_summary,
    run_quantum_optimizer_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark optimizers on a chaotic quantum kicked-top control task."
    )
    parser.add_argument("--spin-j", type=int, default=4)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--kick-strength", type=float, default=7.0)
    parser.add_argument("--target-control-amplitude", type=float, default=0.42)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "quantum_chaos_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = QuantumChaosConfig(
        spin_j=args.spin_j,
        steps=args.steps,
        kick_strength=args.kick_strength,
        target_control_amplitude=args.target_control_amplitude,
        seed=args.seed,
    )
    problem, results = run_quantum_optimizer_comparison(config, iterations=args.iterations)

    history_path = args.output_dir / "quantum_chaos_history.csv"
    summary_path = args.output_dir / "quantum_chaos_summary.csv"
    controls_path = args.output_dir / "quantum_chaos_controls.csv"
    loss_plot_path = args.output_dir / "quantum_chaos_loss.png"
    fidelity_plot_path = args.output_dir / "quantum_chaos_fidelity.png"
    controls_plot_path = args.output_dir / "quantum_chaos_controls.png"
    report_path = args.output_dir / "quantum_chaos_report.md"

    export_quantum_history(results, history_path)
    export_quantum_summary(results, summary_path)
    export_quantum_controls(problem, results, controls_path)
    export_loss_plot(results, loss_plot_path)
    export_fidelity_plot(results, fidelity_plot_path)
    export_controls_plot(problem, results, controls_plot_path)
    write_report(config, args.iterations, results, report_path)

    print("Chaotic quantum kicked-top optimizer benchmark")
    print(f"spin_j = {config.spin_j}")
    print(f"hilbert_dimension = {config.dimension}")
    print(f"control_steps = {config.steps}")
    print(f"kick_strength = {config.kick_strength:g}")
    print("optimizer,final_loss,final_fidelity,runtime_s,spectral_entropy")
    for result in sorted(results, key=lambda item: item.final_loss):
        print(
            f"{result.optimizer},"
            f"{result.final_loss:.6e},"
            f"{result.final_fidelity:.6e},"
            f"{result.runtime_s:.6e},"
            f"{result.spectral_entropy:.6e}"
        )
    for path in (
        history_path,
        summary_path,
        controls_path,
        loss_plot_path,
        fidelity_plot_path,
        controls_plot_path,
        report_path,
    ):
        print(f"exported = {path}")


def export_loss_plot(results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for result in results:
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.8
        ax.plot(result.loss_history, label=result.optimizer, linewidth=linewidth)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("infidelity loss")
    ax.set_title("Chaotic quantum control optimizer comparison")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_fidelity_plot(results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for result in results:
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.8
        ax.plot(result.fidelity_history, label=result.optimizer, linewidth=linewidth)
    ax.set_xlabel("iteration")
    ax.set_ylabel("state-transfer fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Final-state fidelity during optimization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_controls_plot(problem, results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    xs = range(1, problem.config.steps + 1)
    ax.plot(xs, problem.target_controls, label="target", color="#111827", linewidth=3.2)
    for result in sorted(results, key=lambda item: item.final_loss)[:3]:
        ax.plot(xs, result.parameters, label=result.optimizer, linewidth=2.0, alpha=0.88)
    ax.set_xlabel("kick step")
    ax.set_ylabel("control angle rad")
    ax.set_title("Target controls vs best learned controls")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(config: QuantumChaosConfig, iterations: int, results, path: Path) -> None:
    ordered = sorted(results, key=lambda item: item.final_loss)
    lines = [
        "# Chaotic Quantum Kicked-Top Optimizer Benchmark",
        "",
        "The workload optimizes a sequence of rotation-control angles for a quantum kicked top.",
        "The loss is final-state infidelity against a hidden target trajectory generated with the same chaotic dynamics.",
        "",
        f"Spin j: {config.spin_j}",
        f"Hilbert dimension: {config.dimension}",
        f"Control steps: {config.steps}",
        f"Kick strength: {config.kick_strength}",
        f"Iterations: {iterations}",
        "",
        "| optimizer | final loss | final fidelity | runtime s | spectral entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in ordered:
        lines.append(
            f"| {result.optimizer} | {result.final_loss:.12g} | "
            f"{result.final_fidelity:.8f} | {result.runtime_s:.4f} | "
            f"{result.spectral_entropy:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This is a local numerical benchmark, not a claim of universal optimizer dominance.",
            "The Hamiltonian-geometric run uses a local fidelity-sensitivity metric, while AdamW,",
            "Nesterov, heavy-ball, SGD, and entropy descent provide standard comparison points.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
