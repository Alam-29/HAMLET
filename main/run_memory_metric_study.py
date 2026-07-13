"""Empirical test of the memory-metric term M_ij (src/hamiltonian_geometric.py's
use_memory_metric/memory_metric_coupling) on the quantum kicked-top benchmark --
the one real (non-synthetic, non-exact-Hessian) workload in this suite where it
can be tested. M_ij is mathematically complete and unit-tested
(tests/test_hamiltonian_geometric.py) but was off in every other benchmark in
this project; this script is the reproducible record of why it stays off.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum_chaos import QuantumChaosConfig, build_problem, train_quantum_optimizer

# The tuned winning configuration from run_quantum_optimizer_comparison (see
# src/quantum_chaos.py), used unchanged as the base for every coupling tested.
BASE_HYPERPARAMETERS = {
    "memory_coupling": 0.0,
    "memory_decay": 0.85,
    "metric_regularization": 0.02,
    "use_geometric_correction": False,
    "beta": 0.7,
    "learning_rate": 0.35,
    "max_energy_backtracks": 30,
    "energy_backtrack_factor": 0.85,
}

# A second block re-tunes beta/learning_rate around each coupling, to check
# whether M_ij's effect is a fixable interaction with the existing tuning
# rather than a fundamental mismatch.
RETUNED_TRIALS = [
    {"memory_metric_coupling": 0.001, "beta": 0.7, "learning_rate": 0.35},
    {"memory_metric_coupling": 0.01, "beta": 0.6, "learning_rate": 0.25},
    {"memory_metric_coupling": 0.005, "beta": 0.6, "learning_rate": 0.25},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the memory-metric term M_ij helps on the quantum kicked-top benchmark."
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--couplings",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.01, 0.05, 0.1, 0.5],
        help="memory_metric_coupling values to test at the base (untuned) hyperparameters.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "memory_metric_study"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = QuantumChaosConfig(
        spin_j=4, steps=14, kick_strength=7.0, target_control_amplitude=0.42, seed=41
    )
    problem = build_problem(config)

    rows = []
    for coupling in args.couplings:
        hyperparameters = {
            **BASE_HYPERPARAMETERS,
            "use_memory_metric": coupling > 0.0,
            "memory_metric_coupling": coupling,
        }
        result = train_quantum_optimizer(problem, "hamiltonian_geometric", args.iterations, hyperparameters)
        rows.append(
            {
                "trial": "base",
                "memory_metric_coupling": coupling,
                "beta": BASE_HYPERPARAMETERS["beta"],
                "learning_rate": BASE_HYPERPARAMETERS["learning_rate"],
                "final_loss": result.final_loss,
                "final_fidelity": result.final_fidelity,
            }
        )
        print(
            f"[base] coupling={coupling}: loss={result.final_loss:.6f} fidelity={result.final_fidelity:.6f}",
            flush=True,
        )

    for trial in RETUNED_TRIALS:
        hyperparameters = {
            **BASE_HYPERPARAMETERS,
            "use_memory_metric": True,
            **trial,
        }
        result = train_quantum_optimizer(problem, "hamiltonian_geometric", args.iterations, hyperparameters)
        rows.append(
            {
                "trial": "retuned",
                "memory_metric_coupling": trial["memory_metric_coupling"],
                "beta": trial["beta"],
                "learning_rate": trial["learning_rate"],
                "final_loss": result.final_loss,
                "final_fidelity": result.final_fidelity,
            }
        )
        print(
            f"[retuned] {trial}: loss={result.final_loss:.6f} fidelity={result.final_fidelity:.6f}",
            flush=True,
        )

    csv_path = args.output_dir / "memory_metric_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["trial", "memory_metric_coupling", "beta", "learning_rate", "final_loss", "final_fidelity"],
        )
        writer.writeheader()
        writer.writerows(rows)

    baseline_loss = rows[0]["final_loss"]
    best_with_mij = min((row for row in rows if row["memory_metric_coupling"] > 0), key=lambda row: row["final_loss"])
    print("\nSummary:")
    print(f"baseline (no M_ij): loss={baseline_loss:.6f}")
    print(f"best with M_ij enabled (any coupling/tuning tried): {best_with_mij}")
    print(f"M_ij helped: {best_with_mij['final_loss'] < baseline_loss}")
    print(f"exported = {csv_path}")


if __name__ == "__main__":
    main()
