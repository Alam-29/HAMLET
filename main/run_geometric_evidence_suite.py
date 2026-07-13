import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_evidence import (
    condition_number_experiment,
    phase_space_experiment,
    rotation_invariance_experiment,
    saddle_escape_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic geometric evidence experiments.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "geometric_evidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = condition_number_experiment()
    rotation_rows = rotation_invariance_experiment()
    traces = phase_space_experiment()
    saddle_rows = saddle_escape_experiment()

    write_csv(args.output_dir / "condition_number_scaling.csv", condition_rows)
    write_csv(args.output_dir / "rotation_invariance.csv", rotation_rows)
    write_csv(args.output_dir / "saddle_escape.csv", saddle_rows)
    export_phase_trace_csv(args.output_dir / "phase_space_traces.csv", traces)
    export_condition_plot(condition_rows, args.output_dir / "condition_number_scaling.png")
    export_rotation_plot(rotation_rows, args.output_dir / "rotation_invariance.png")
    export_phase_plot(traces, args.output_dir / "phase_space_loss_energy.png")
    export_saddle_plot(saddle_rows, args.output_dir / "saddle_escape.png")
    write_report(condition_rows, rotation_rows, traces, saddle_rows, args.output_dir / "geometric_evidence_report.md")

    print("Geometric evidence suite")
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(f"exported = {path}")


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_phase_trace_csv(path: Path, traces) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "optimizer", "loss", "energy", "gradient_norm", "final_theta_0", "final_theta_1"])
        for trace in traces:
            for step, (loss, energy, grad_norm) in enumerate(zip(trace.loss, trace.energy, trace.grad_norm), 1):
                writer.writerow([step, trace.name, loss, energy, grad_norm, trace.theta[0], trace.theta[1]])


def export_condition_plot(rows, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    by_opt = defaultdict(list)
    for row in rows:
        by_opt[row["optimizer"]].append(row)
    for optimizer, opt_rows in by_opt.items():
        opt_rows = sorted(opt_rows, key=lambda row: row["condition_number"])
        ax.plot(
            [row["condition_number"] for row in opt_rows],
            [row["final_loss"] for row in opt_rows],
            marker="o",
            label=optimizer,
            linewidth=2.8 if optimizer == "hamiltonian_geometric" else 1.8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("condition number")
    ax.set_ylabel("final loss")
    ax.set_title("Ill-conditioned rotated quadratic")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_rotation_plot(rows, path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["optimizer"]].append(row["final_loss"])
    names = sorted(grouped)
    values = [np.log10(np.array(grouped[name]) + 1e-300) for name in names]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.boxplot(values, tick_labels=names, showfliers=True)
    ax.set_ylabel("log10 final loss")
    ax.set_title("Rotation sensitivity at fixed spectrum")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_phase_plot(traces, path: Path) -> None:
    fig, (ax_loss, ax_energy) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for trace in traces:
        width = 2.8 if trace.name == "hamiltonian_geometric" else 1.8
        ax_loss.plot(trace.loss, label=trace.name, linewidth=width)
        ax_energy.plot(trace.energy, label=trace.name, linewidth=width)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Phase-space workload: loss and energy")
    ax_loss.grid(True, which="both", alpha=0.25)
    ax_energy.set_yscale("symlog", linthresh=1e-6)
    ax_energy.set_xlabel("step")
    ax_energy.set_ylabel("energy proxy")
    ax_energy.grid(True, which="both", alpha=0.25)
    ax_loss.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_saddle_plot(rows, path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["optimizer"]].append(row)
    names = sorted(grouped)
    escape = [np.mean([row["escaped"] for row in grouped[name]]) for name in names]
    loss = [np.median([row["final_loss"] for row in grouped[name]]) for name in names]
    fig, ax1 = plt.subplots(figsize=(9, 5.6))
    x = np.arange(len(names))
    ax1.bar(x - 0.18, escape, width=0.36, label="escape rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("escape rate")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, loss, width=0.36, color="#64748b", label="median final loss")
    ax2.set_ylabel("median final loss")
    ax1.set_xticks(x, names, rotation=20)
    ax1.set_title("Saddle/basin escape experiment")
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(condition_rows, rotation_rows, traces, saddle_rows, path: Path) -> None:
    best_condition = min(
        (row for row in condition_rows if row["condition_number"] == max(r["condition_number"] for r in condition_rows)),
        key=lambda row: row["final_loss"],
    )
    rotation_summary = {
        name: float(np.std(np.log10([row["final_loss"] + 1e-300 for row in rotation_rows if row["optimizer"] == name])))
        for name in {row["optimizer"] for row in rotation_rows}
    }
    saddle_summary = {
        name: float(np.mean([row["escaped"] for row in saddle_rows if row["optimizer"] == name]))
        for name in {row["optimizer"] for row in saddle_rows}
    }
    phase_best = min(traces, key=lambda trace: trace.loss[-1])
    lines = [
        "# Geometric Evidence Suite",
        "",
        "This suite targets the cases where a phase-space metric optimizer should be strongest: stiff curvature, rotations, energy behavior, and saddle/basin navigation.",
        "",
        f"Best optimizer at largest condition number: {best_condition['optimizer']} with final loss {best_condition['final_loss']:.6g}.",
        f"Best phase-space final loss: {phase_best.name} with final loss {phase_best.loss[-1]:.6g}.",
        "",
        "## Rotation Sensitivity",
        "",
        "| optimizer | std(log10 final loss) across rotations |",
        "|---|---:|",
    ]
    for name, value in sorted(rotation_summary.items(), key=lambda item: item[1]):
        lines.append(f"| {name} | {value:.6g} |")
    lines.extend(["", "## Saddle Escape", "", "| optimizer | escape rate |", "|---|---:|"])
    for name, value in sorted(saddle_summary.items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {value:.3f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
