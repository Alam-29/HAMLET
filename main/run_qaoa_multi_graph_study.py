"""Multi-seed, multi-graph-family, multi-depth QAOA MaxCut study.

The single-graph QAOA benchmark (main/run_qaoa_benchmark.py) uses one fixed
graph and one seed. This script tests whether Hamiltonian-geometric's result
there generalizes across qubit counts, circuit depths, and structurally
different graph families (Erdos-Renyi, random-regular, and two-community),
each averaged over multiple seeds -- a direct test of whether the framework's
advantage is specific to one hand-picked graph or holds more broadly.
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

from src.qaoa import QAOAConfig, build_problem, train_qaoa_optimizer

OPTIMIZER_SETTINGS = [
    ("sgd", {"learning_rate": 0.12}),
    ("heavy_ball", {"learning_rate": 0.08, "momentum": 0.8}),
    ("adamw", {"learning_rate": 0.06, "beta1": 0.9, "beta2": 0.99}),
    ("entropy_descent", {"learning_rate": 0.10, "metric_regularization": 5e-2}),
    (
        "hamiltonian_geometric",
        {"learning_rate": 0.18, "beta": 0.45, "metric_regularization": 5e-2},
    ),
]

OPTIMIZER_COLORS = {
    "sgd": "#9aa3ab",
    "heavy_ball": "#4f83b5",
    "adamw": "#e8a33d",
    "entropy_descent": "#c0392b",
    "hamiltonian_geometric": "#6a3d9a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed, multi-graph-family QAOA MaxCut generalization study."
    )
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--qubits", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--graph-families",
        type=str,
        nargs="+",
        default=["erdos_renyi", "regular", "community"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 29, 39, 49, 59])
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "qaoa_multi_graph_study"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(args.qubits) * len(args.depths) * len(args.graph_families) * len(args.seeds)
    done = 0
    for qubits in args.qubits:
        for depth in args.depths:
            for family in args.graph_families:
                for seed in args.seeds:
                    config = QAOAConfig(qubits=qubits, depth=depth, seed=seed, graph_family=family)
                    problem = build_problem(config)
                    for name, params in OPTIMIZER_SETTINGS:
                        result = train_qaoa_optimizer(problem, name, args.iterations, params)
                        rows.append(
                            {
                                "qubits": qubits,
                                "depth": depth,
                                "graph_family": family,
                                "seed": seed,
                                "optimizer": name,
                                "final_loss": result.final_loss,
                                "approximation_ratio": result.final_ratio,
                                "runtime_s": result.runtime_s,
                            }
                        )
                    done += 1
                    print(f"[{done}/{total}] qubits={qubits} depth={depth} family={family} seed={seed} done", flush=True)

    csv_path = args.output_dir / "qaoa_multi_graph_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["qubits", "depth", "graph_family", "seed", "optimizer", "final_loss", "approximation_ratio", "runtime_s"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nexported = {csv_path}")

    optimizers = [name for name, _ in OPTIMIZER_SETTINGS]

    print("\nOverall mean approximation ratio (pooled across all qubits/depths/families/seeds):")
    for name in optimizers:
        ratios = [row["approximation_ratio"] for row in rows if row["optimizer"] == name]
        print(f"{name}: {np.mean(ratios):.5f} +/- {np.std(ratios):.5f} (n={len(ratios)})")

    print("\nMean approximation ratio by graph family:")
    for family in args.graph_families:
        print(f"  {family}:")
        for name in optimizers:
            ratios = [
                row["approximation_ratio"]
                for row in rows
                if row["optimizer"] == name and row["graph_family"] == family
            ]
            print(f"    {name}: {np.mean(ratios):.5f} +/- {np.std(ratios):.5f}")

    # Rank frequency: for each (qubits, depth, family, seed) instance, which optimizer wins?
    win_counts = {name: 0 for name in optimizers}
    instances = {(row["qubits"], row["depth"], row["graph_family"], row["seed"]) for row in rows}
    for instance in instances:
        instance_rows = [
            row
            for row in rows
            if (row["qubits"], row["depth"], row["graph_family"], row["seed"]) == instance
        ]
        best = max(instance_rows, key=lambda row: row["approximation_ratio"])
        win_counts[best["optimizer"]] += 1
    print(f"\nWin counts (best approximation ratio) out of {len(instances)} instances:")
    for name in optimizers:
        print(f"  {name}: {win_counts[name]}")

    with (args.output_dir / "qaoa_multi_graph_summary.md").open("w", encoding="utf-8") as file:
        file.write("# QAOA Multi-Graph, Multi-Seed Generalization Study\n\n")
        file.write(
            f"Qubits: {args.qubits}. Depths: {args.depths}. Graph families: {args.graph_families}. "
            f"Seeds per configuration: {args.seeds}. Total instances: {len(instances)}.\n\n"
        )
        file.write("## Overall mean approximation ratio\n\n")
        file.write("| optimizer | mean | std | n |\n|---|---|---|---|\n")
        for name in optimizers:
            ratios = [row["approximation_ratio"] for row in rows if row["optimizer"] == name]
            file.write(f"| {name} | {np.mean(ratios):.5f} | {np.std(ratios):.5f} | {len(ratios)} |\n")
        file.write("\n## Mean approximation ratio by graph family\n\n")
        file.write("| graph_family | " + " | ".join(optimizers) + " |\n")
        file.write("|---|" + "---|" * len(optimizers) + "\n")
        for family in args.graph_families:
            cells = []
            for name in optimizers:
                ratios = [
                    row["approximation_ratio"]
                    for row in rows
                    if row["optimizer"] == name and row["graph_family"] == family
                ]
                cells.append(f"{np.mean(ratios):.4f}")
            file.write(f"| {family} | " + " | ".join(cells) + " |\n")
        file.write("\n## Win counts (best approximation ratio per instance)\n\n")
        file.write("| optimizer | wins | total instances |\n|---|---|---|\n")
        for name in optimizers:
            file.write(f"| {name} | {win_counts[name]} | {len(instances)} |\n")

    # Bar plot: mean approximation ratio by graph family, one bar cluster per family.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(args.graph_families))
    width = 0.15
    for i, name in enumerate(optimizers):
        means = []
        stds = []
        for family in args.graph_families:
            ratios = [
                row["approximation_ratio"]
                for row in rows
                if row["optimizer"] == name and row["graph_family"] == family
            ]
            means.append(np.mean(ratios))
            stds.append(np.std(ratios))
        ax.bar(x + (i - len(optimizers) / 2) * width, means, width, yerr=stds, label=name, color=OPTIMIZER_COLORS.get(name), capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(args.graph_families)
    ax.set_ylabel("mean approximation ratio")
    ax.set_title("QAOA approximation ratio by graph family (pooled over qubits/depths/seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "qaoa_multi_graph_by_family.png", dpi=170)
    plt.close(fig)

    print(f"exported = {args.output_dir / 'qaoa_multi_graph_summary.md'}")
    print(f"exported = {args.output_dir / 'qaoa_multi_graph_by_family.png'}")


if __name__ == "__main__":
    main()
