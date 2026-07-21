"""Confirmatory optimizer study with equal, seed-averaged tuning budgets.

Each optimizer receives exactly six predeclared configurations. Configurations
are ranked by mean validation loss over three tuning seeds, then the selected
configuration is frozen and evaluated on ten disjoint seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.run_algoperf_style_benchmark import search_space, train_optimizer
from main.run_deepobs_style_benchmark import MLPShape, init_parameters, make_spiral_dataset

OPTIMIZERS = ("adamw", "nag", "heavy_ball", "hamiltonian_geometric")


def problem(seed: int, args):
    shape = MLPShape(hidden_dim=args.hidden_dim)
    return (
        shape,
        make_spiral_dataset(args.train_samples, seed=seed),
        make_spiral_dataset(args.val_samples, seed=seed + 1),
        init_parameters(shape, seed=seed + 2),
    )


def run_once(optimizer: str, hp: dict[str, float], seed: int, args) -> float:
    shape, (x_train, y_train), (x_val, y_val), theta0 = problem(seed, args)
    args.seed = seed
    result = train_optimizer(optimizer, theta0, shape, x_train, y_train, x_val, y_val, args, hp)
    return float(result.final["val_loss"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "equal_budget_tuning")
    cli = parser.parse_args()
    args = argparse.Namespace(
        epochs=90, batch_size=64, hidden_dim=24,
        train_samples=900, val_samples=300, seed=0,
    )
    tuning_rows = []
    selected = {}
    for optimizer in OPTIMIZERS:
        candidates = search_space(optimizer)[: cli.trials]
        if len(candidates) != cli.trials:
            raise ValueError(f"{optimizer} does not provide {cli.trials} trials")
        means = []
        for trial, hp in enumerate(candidates, 1):
            losses = []
            for seed in cli.tuning_seeds:
                loss = run_once(optimizer, hp, seed, args)
                losses.append(loss)
                tuning_rows.append({"optimizer": optimizer, "trial": trial, "seed": seed,
                                    "val_loss": loss, "hyperparameters": json.dumps(hp, sort_keys=True)})
            means.append(statistics.fmean(losses))
            print(f"tune {optimizer} trial={trial} mean={means[-1]:.6f}", flush=True)
        best_index = int(np.argmin(means))
        selected[optimizer] = candidates[best_index]

    evaluation_rows = []
    by_optimizer = {}
    for optimizer in OPTIMIZERS:
        losses = []
        for seed in cli.evaluation_seeds:
            loss = run_once(optimizer, selected[optimizer], seed, args)
            losses.append(loss)
            evaluation_rows.append({"optimizer": optimizer, "seed": seed, "val_loss": loss,
                                    "hyperparameters": json.dumps(selected[optimizer], sort_keys=True)})
        by_optimizer[optimizer] = losses
        print(f"eval {optimizer} mean={statistics.fmean(losses):.6f}", flush=True)

    comparison_rows = []
    hg = np.asarray(by_optimizer["hamiltonian_geometric"])
    for optimizer in OPTIMIZERS[:-1]:
        other = np.asarray(by_optimizer[optimizer])
        difference = hg - other
        test = stats.ttest_rel(hg, other)
        comparison_rows.append({
            "comparison": f"hamiltonian_geometric-vs-{optimizer}",
            "n": len(hg), "hg_mean": hg.mean(), "other_mean": other.mean(),
            "mean_difference": difference.mean(), "cohen_dz": difference.mean() / difference.std(ddof=1),
            "paired_t": test.statistic, "paired_p": test.pvalue,
            "hg_wins": int(np.sum(difference < 0)),
        })
    order = np.argsort([row["paired_p"] for row in comparison_rows])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * comparison_rows[index]["paired_p"]))
        comparison_rows[index]["holm_p"] = running

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("tuning_raw.csv", tuning_rows), ("evaluation_raw.csv", evaluation_rows),
                           ("paired_comparisons.csv", comparison_rows)):
        with (cli.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    manifest = {"protocol": "six trials per optimizer, mean over disjoint tuning seeds, frozen evaluation",
                "tuning_seeds": cli.tuning_seeds, "evaluation_seeds": cli.evaluation_seeds,
                "trials_per_optimizer": cli.trials, "selected": selected}
    (cli.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
