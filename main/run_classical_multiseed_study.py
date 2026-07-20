"""Multi-seed replication of the four classical benchmark tables.

Tables 4-7 of the consolidated report (MNIST, AlgoPerf-style MLP, DeepOBS-style
MLP, capacitor PINN) are single-seed point estimates -- a limitation the
report states explicitly. This script re-runs each benchmark's existing CLI
across multiple seeds (invoking the checked-in scripts themselves, not a
reimplementation, so the numbers are produced by the same code the single-seed
tables come from), extracts the Hamiltonian-geometric result and its best
comparator from each seed's own summary CSV, and reports the mean, sample
standard deviation, a normal-approximation 95% CI, and a paired t-test
(seeds are shared between the two optimizers within a run) for each workload.

Usage (from repo root):
    python main/run_classical_multiseed_study.py --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(script: str, seed: int, out_dir: Path, extra: list[str]) -> None:
    command = [PYTHON, str(PROJECT_ROOT / "main" / script), "--seed", str(seed), "--output-dir", str(out_dir), *extra]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"benchmark failed (exit {completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )


def _extract(rows: list[dict], value_key: str) -> tuple[float, float, str]:
    hg = float(next(r for r in rows if r["optimizer"] == "hamiltonian_geometric")[value_key])
    competitors = [(float(r[value_key]), r["optimizer"]) for r in rows if r["optimizer"] != "hamiltonian_geometric"]
    best_other, comparator = min(competitors)
    return hg, best_other, comparator


def mnist_seed(seed: int, tmp: Path) -> tuple[float, float, str]:
    out = tmp / f"mnist_{seed}"
    if not (out / "mnist_optimizer_summary.csv").exists():
        _run("run_mnist_optimizer_benchmark.py", seed, out, [])
    rows = _read_csv(out / "mnist_optimizer_summary.csv")
    return _extract(rows, "final_test_loss")


def algoperf_seed(seed: int, tmp: Path) -> tuple[float, float, str]:
    out = tmp / f"algoperf_{seed}"
    if not (out / "algoperf_style_best_summary.csv").exists():
        _run("run_algoperf_style_benchmark.py", seed, out, [])
    rows = _read_csv(out / "algoperf_style_best_summary.csv")
    return _extract(rows, "final_val_loss")


def deepobs_seed(seed: int, tmp: Path) -> tuple[float, float, str]:
    out = tmp / f"deepobs_{seed}"
    if not (out / "deepobs_style_optimizer_summary.csv").exists():
        _run("run_deepobs_style_benchmark.py", seed, out, [])
    rows = _read_csv(out / "deepobs_style_optimizer_summary.csv")
    return _extract(rows, "final_val_loss")


def pinn_seed(seed: int, tmp: Path) -> tuple[float, float, str]:
    out = tmp / f"pinn_{seed}"
    if not (out / "pinn_optimizer_summary.csv").exists():
        _run("run_pinn_benchmark.py", seed, out, [])
    rows = _read_csv(out / "pinn_optimizer_summary.csv")
    return _extract(rows, "final_loss")


def llm_seed(seed: int, tmp: Path) -> tuple[float, float, str]:
    out = tmp / f"llm_{seed}"
    if not (out / "industry_llm_summary.csv").exists():
        _run("run_industry_llm_benchmark.py", seed, out, [])
    rows = _read_csv(out / "industry_llm_summary.csv")
    return _extract(rows, "final_val_loss")


WORKLOADS = {
    "mnist": ("MNIST softmax (test loss)", mnist_seed),
    "algoperf": ("AlgoPerf-style MLP (val loss)", algoperf_seed),
    "deepobs": ("DeepOBS-style MLP (val loss)", deepobs_seed),
    "pinn": ("Capacitor PINN (final loss)", pinn_seed),
    "llm": ("WikiText-2 LLM, 200 steps (val loss)", llm_seed),
}


def summarize(name: str, hg_values: list[float], other_values: list[float]) -> dict:
    hg = stats.describe(hg_values)
    other = stats.describe(other_values)
    diff = [h - o for h, o in zip(hg_values, other_values)]
    t_result = stats.ttest_rel(hg_values, other_values)
    try:
        wilcoxon_p = float(stats.wilcoxon(hg_values, other_values, zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    n = len(hg_values)
    hg_ci = stats.t.interval(0.95, n - 1, loc=hg.mean, scale=(hg.variance ** 0.5) / (n ** 0.5)) if n > 1 else (hg.mean, hg.mean)
    other_ci = stats.t.interval(0.95, n - 1, loc=other.mean, scale=(other.variance ** 0.5) / (n ** 0.5)) if n > 1 else (other.mean, other.mean)
    return {
        "workload": name, "n": n,
        "hg_mean": hg.mean, "hg_std": hg.variance ** 0.5, "hg_ci_low": hg_ci[0], "hg_ci_high": hg_ci[1],
        "other_mean": other.mean, "other_std": other.variance ** 0.5, "other_ci_low": other_ci[0], "other_ci_high": other_ci[1],
        "mean_diff": sum(diff) / n, "paired_t": t_result.statistic, "paired_p": t_result.pvalue,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": (sum(diff) / n) / stats.tstd(diff) if n > 1 and stats.tstd(diff) > 0 else 0.0,
        "hg_wins": sum(1 for h, o in zip(hg_values, other_values) if h < o),
    }


def add_holm_correction(summaries: list[dict], source: str, target: str) -> None:
    order = sorted(range(len(summaries)), key=lambda index: float(summaries[index][source]))
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(order) - rank) * float(summaries[index][source]))
        running = max(running, adjusted)
        summaries[index][target] = running


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed classical-benchmark replication.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    parser.add_argument("--tmp-dir", type=Path, default=PROJECT_ROOT / "results" / "_classical_multiseed_tmp")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "results" / "classical_multiseed_summary.csv")
    parser.add_argument("--raw-output-csv", type=Path, default=PROJECT_ROOT / "results" / "classical_multiseed_raw.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results" / "classical_multiseed_manifest.json")
    parser.add_argument("--cleanup-tmp", action="store_true", help="Delete per-seed benchmark artifacts after successful export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    raw_rows = []
    for key in args.workloads:
        label, fn = WORKLOADS[key]
        hg_values, other_values = [], []
        for seed in args.seeds:
            hg, other, comparator = fn(seed, args.tmp_dir)
            hg_values.append(hg)
            other_values.append(other)
            raw_rows.append({"workload_key": key, "workload": label, "seed": seed,
                             "hg_value": hg, "best_other_value": other,
                             "best_other_optimizer": comparator, "difference_hg_minus_other": hg - other})
            print(f"{key} seed={seed} hg={hg:.6g} best_other={other:.6g}", flush=True)
        summary = summarize(label, hg_values, other_values)
        summaries.append(summary)
        print(f"=== {label}: HG {summary['hg_mean']:.4g} +/- {summary['hg_std']:.4g}, "
              f"best-other {summary['other_mean']:.4g} +/- {summary['other_std']:.4g}, "
              f"paired t={summary['paired_t']:.3f} p={summary['paired_p']:.4f}, "
              f"HG wins {summary['hg_wins']}/{summary['n']} ===", flush=True)

    add_holm_correction(summaries, "paired_p", "holm_paired_p")
    add_holm_correction(summaries, "wilcoxon_p", "holm_wilcoxon_p")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with args.raw_output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    manifest = {
        "python": sys.version, "platform": platform.platform(), "seeds": args.seeds,
        "workloads": args.workloads, "comparison": "HG versus the lowest-loss non-HG optimizer within each seed",
        "selection_warning": "The comparator is an oracle best-of-baselines per seed; this favors the comparator and its identity may vary.",
        "summary_csv": str(args.output_csv.resolve()), "raw_csv": str(args.raw_output_csv.resolve()),
        "per_seed_artifacts": str(args.tmp_dir.resolve()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported = {args.output_csv}")
    print(f"exported = {args.raw_output_csv}")
    print(f"exported = {args.manifest}")

    if args.cleanup_tmp:
        shutil.rmtree(args.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
