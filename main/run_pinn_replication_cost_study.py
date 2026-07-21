"""Quantify the cost of a multi-seed capacitor-PINN replication.

The published PINN configuration has a constant 65x65 metric.  The legacy
"full" path nevertheless evaluates 2n finite-difference metric probes per
step to confirm that the geometric force is zero.  This study times that
literal path and the mathematically equivalent analytic-zero path, retains
raw measurements, and extrapolates the prespecified 10-seed comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pinn import PINNConfig, FixedFeaturePotentialModel, build_pinn_dataset, train_hamiltonian_geometric


def timed(dataset, parameter_count: int, steps: int, geometric: bool) -> float:
    start = time.perf_counter()
    train_hamiltonian_geometric(
        dataset,
        parameter_count,
        steps=steps,
        use_geometric_correction=geometric,
        use_memory_correction=True,
    )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--target-steps", type=int, default=600)
    parser.add_argument("--target-seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "pinn_replication_cost")
    args = parser.parse_args()
    config = PINNConfig(seed=7)
    model = FixedFeaturePotentialModel(config)
    dataset = build_pinn_dataset(model, config)
    rows = []
    for geometric in (True, False):
        label = "literal_finite_difference_zero" if geometric else "analytic_zero"
        steps_to_run = args.probe_steps if geometric else [args.target_steps]
        for steps in steps_to_run:
            for repeat in range(args.repeats):
                duration = timed(dataset, model.parameter_count, steps, geometric)
                rows.append({"path": label, "steps": steps, "repeat": repeat, "runtime_s": duration})
                print(f"{label} steps={steps} repeat={repeat} runtime={duration:.3f}s", flush=True)

    literal = [row for row in rows if row["path"] == "literal_finite_difference_zero"]
    grouped = {}
    for steps in args.probe_steps:
        grouped[steps] = statistics.median(row["runtime_s"] for row in literal if row["steps"] == steps)
    slope, intercept = np.polyfit(np.asarray(list(grouped), float), np.asarray(list(grouped.values()), float), 1)
    predicted_one = max(0.0, intercept + slope * args.target_steps)
    analytic = statistics.median(
        row["runtime_s"] for row in rows if row["path"] == "analytic_zero" and row["steps"] == args.target_steps
    )
    summary = {
        "parameter_count": model.parameter_count,
        "metric_shape": [model.parameter_count, model.parameter_count],
        "central_metric_probes_per_step": 2 * model.parameter_count,
        "target_steps": args.target_steps,
        "target_seeds": args.target_seeds,
        "literal_fit_intercept_s": float(intercept),
        "literal_fit_s_per_step": float(slope),
        "predicted_literal_one_seed_s": float(predicted_one),
        "predicted_literal_ten_seed_hours": float(predicted_one * args.target_seeds / 3600.0),
        "measured_analytic_zero_one_seed_s": float(analytic),
        "estimated_speedup": float(predicted_one / analytic),
        "equivalence_basis": "metric is constant in theta, hence F_geo is identically zero",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pinn_cost_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "pinn_cost_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
