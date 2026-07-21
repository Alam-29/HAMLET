"""Write an executable audit of the approximate-metric stability theorem."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.approximate_metric_theory import verify_relative_spectrum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "approximate_metric_theorem_check.csv")
    args = parser.parse_args()
    rows = []
    for condition_number in (1.0, 1e2, 1e4, 1e8):
        # A's absolute spectrum changes by eight orders of magnitude, while
        # B is constructed to keep B^{-1/2} A B^{-1/2} in [0.8, 1.2].
        absolute = np.geomspace(1.0, condition_number, 32)
        relative = np.linspace(0.8, 1.2, absolute.size)
        metric = absolute / relative
        observed_relative = absolute / metric
        row = verify_relative_spectrum(observed_relative, delta=0.2, learning_rate=0.9, momentum=0.7)
        row["condition_number_A"] = condition_number
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not all(row["theorem_condition"] and row["interval_ok"] and row["empirically_stable"] for row in rows):
        raise SystemExit("approximate-metric theorem audit failed")
    print(f"wrote {args.output.relative_to(ROOT)}; max rho={max(float(row['max_spectral_radius']) for row in rows):.6f}")


if __name__ == "__main__":
    main()
