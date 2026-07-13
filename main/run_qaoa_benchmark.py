import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qaoa import QAOAConfig, export_qaoa_history, export_qaoa_summary, run_qaoa_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-statevector QAOA MaxCut optimizer benchmark.")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "qaoa_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = QAOAConfig(qubits=args.qubits, depth=args.depth, seed=args.seed)
    problem, results = run_qaoa_comparison(config, iterations=args.iterations)

    history_path = args.output_dir / "qaoa_history.csv"
    summary_path = args.output_dir / "qaoa_summary.csv"
    ratio_plot_path = args.output_dir / "qaoa_approximation_ratio.png"
    loss_plot_path = args.output_dir / "qaoa_loss.png"
    report_path = args.output_dir / "qaoa_report.md"

    export_qaoa_history(results, history_path)
    export_qaoa_summary(problem, results, summary_path)
    export_plot(results, ratio_plot_path, "ratio")
    export_plot(results, loss_plot_path, "loss")
    write_report(problem, results, report_path)

    print("QAOA MaxCut optimizer benchmark")
    print(f"qubits = {config.qubits}")
    print(f"depth = {config.depth}")
    print(f"edges = {len(problem.edges)}")
    print("optimizer,final_loss,approximation_ratio,runtime_s,spectral_entropy")
    for result in sorted(results, key=lambda item: item.final_loss):
        print(
            f"{result.optimizer},"
            f"{result.final_loss:.6e},"
            f"{result.final_ratio:.6e},"
            f"{result.runtime_s:.6e},"
            f"{result.spectral_entropy:.6e}"
        )
    for path in (history_path, summary_path, ratio_plot_path, loss_plot_path, report_path):
        print(f"exported = {path}")


def export_plot(results, path: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for result in results:
        values = result.ratio_history if metric == "ratio" else result.loss_history
        linewidth = 2.8 if result.optimizer == "hamiltonian_geometric" else 1.8
        ax.plot(values, label=result.optimizer, linewidth=linewidth)
    ax.set_xlabel("iteration")
    if metric == "ratio":
        ax.set_ylabel("approximation ratio")
        ax.set_ylim(0.0, 1.05)
    else:
        ax.set_ylabel("loss")
    ax.set_title("QAOA MaxCut optimizer benchmark")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(problem, results, path: Path) -> None:
    lines = [
        "# QAOA MaxCut Optimizer Benchmark",
        "",
        "Exact statevector QAOA on a small MaxCut graph. The loss is negative approximation ratio.",
        "",
        f"Qubits: {problem.config.qubits}",
        f"Depth: {problem.config.depth}",
        f"Edges: {len(problem.edges)}",
        f"Max cut value: {problem.max_cut:.0f}",
        "",
        "| optimizer | approximation ratio | loss | runtime s |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.final_loss):
        lines.append(
            f"| {result.optimizer} | {result.final_ratio:.6f} | "
            f"{result.final_loss:.6g} | {result.runtime_s:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
