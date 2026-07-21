"""Verify the manuscript's authoritative numerical evidence and provenance.

This intentionally does not accept the superseded classical_multiseed_*.csv
files. The paired files are the only classical replication used by the paper.
Run from any directory with Python 3.11+::

    python scripts/verify_submission_package.py
    python scripts/verify_submission_package.py --write-manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE = (
    "docs/hamiltonian_geometric_consolidated_report.tex",
    "docs/hamiltonian_geometric_consolidated_report.pdf",
    "results/classical_multiseed_paired_raw.csv",
    "results/classical_multiseed_paired_summary.csv",
    "results/classical_multiseed_paired_manifest.json",
    "results/classical_multiseed_raw_llm.csv",
    "results/classical_multiseed_summary_llm_audit.csv",
    "results/classical_multiseed_manifest_llm.json",
    "results/industry_llm_compute_audit/industry_llm_seed_aggregate.csv",
    "results/industry_llm_compute_audit/seed_0/industry_llm_summary.csv",
    "results/industry_llm_compute_audit/seed_1/industry_llm_summary.csv",
    "results/industry_llm_compute_audit/seed_2/industry_llm_summary.csv",
    "results/approximate_metric_theorem_check.csv",
    "results/official_deepobs/mnist_mlp_summary.csv",
    "results/official_deepobs/lr_sweep_notes.md",
    "results/official_deepobs/adamw/mnist_mlp/AdamWFixedDecay/num_epochs__20__batch_size__128__lr__3.e-03/random_seed__42__2026-07-21-10-26-02.json",
    "results/official_deepobs/hamiltonian_geometric/mnist_mlp/HamiltonianGeometricTorch/num_epochs__20__batch_size__128__beta__9.e-01__lr__1.e-01/random_seed__42__2026-07-21-09-52-53.json",
    "results/official_deepobs/sgd_momentum/mnist_mlp/SGD/num_epochs__20__batch_size__128__lr__1.e-01__momentum__9.e-01/random_seed__42__2026-07-21-10-09-14.json",
    "results/pinn_multiseed_raw.csv",
    "results/pinn_multiseed_summary.csv",
    "results/pinn_multiseed_manifest.json",
    "results/pinn_replication_cost/pinn_cost_raw.csv",
    "results/pinn_replication_cost/pinn_cost_summary.json",
    "results/equal_budget_tuning/tuning_raw.csv",
    "results/equal_budget_tuning/evaluation_raw.csv",
    "results/equal_budget_tuning/paired_comparisons.csv",
    "results/equal_budget_tuning/manifest.json",
    "ablation styudy/results/architecture_raw.csv",
    "ablation styudy/results/architecture_summary.csv",
    "ablation styudy/results/quantum_raw.csv",
    "ablation styudy/results/quantum_summary.csv",
    "ablation styudy/results/gpu_raw.csv",
    "ablation styudy/results/gpu_summary.csv",
    "ablation styudy/results/paired_tests.csv",
    "ablation styudy/results/factorial_effects.csv",
    "ablation styudy/results/metadata.json",
    "ablation styudy/results/cuda_environment.json",
    "ppt_assets/test_result_media/runtime_normalized_comparison/runtime_normalized_comparison.csv",
    "ppt_assets/test_result_media/modern_optimizer_benchmark_paired/modern_optimizer_seed_sweep.csv",
    "ppt_assets/test_result_media/modern_optimizer_benchmark_paired/modern_optimizer_scaling_sweep.csv",
)


def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify() -> dict[str, object]:
    missing = [name for name in AUTHORITATIVE if not (ROOT / name).is_file()]
    require(not missing, f"missing authoritative files: {missing}")

    classical = rows("results/classical_multiseed_paired_raw.csv")
    require(len(classical) == 30, "paired classical raw CSV must contain 30 rows")
    for key in ("mnist", "algoperf", "deepobs"):
        subset = [row for row in classical if row["workload_key"] == key]
        require({int(row["seed"]) for row in subset} == set(range(10)), f"{key}: expected seeds 0--9")
        require(len(subset) == 10, f"{key}: expected exactly one row per seed")

    llm = rows("results/classical_multiseed_raw_llm.csv")
    require(len(llm) == 3 and {int(row["seed"]) for row in llm} == {0, 1, 2}, "LLM seeds must be 0,1,2")
    llm_compute = rows("results/industry_llm_compute_audit/industry_llm_seed_aggregate.csv")
    require(len(llm_compute) == 5, "LLM compute audit must contain five optimizers")
    require(all(row["target_val_loss"] == "7.5" for row in llm_compute), "LLM target must be 7.5")
    theorem = rows("results/approximate_metric_theorem_check.csv")
    require(len(theorem) == 4, "approximate-metric theorem audit must contain four condition numbers")
    require(all(row["theorem_condition"] == "True" and row["empirically_stable"] == "True" for row in theorem),
            "approximate-metric theorem audit is not stable")
    official_deepobs = rows("results/official_deepobs/mnist_mlp_summary.csv")
    require(len(official_deepobs) == 3, "official DeepOBS summary must contain three optimizers")
    require({row["optimizer"] for row in official_deepobs} ==
            {"hamiltonian_geometric", "sgd_momentum", "adamw"},
            "official DeepOBS optimizer set changed")
    require(all(row["testproblem"] == "mnist_mlp" and int(row["epochs"]) == 20
                for row in official_deepobs),
            "official DeepOBS result must be the reported 20-epoch mnist_mlp run")
    deepobs_by_optimizer = {row["optimizer"]: row for row in official_deepobs}
    require(float(deepobs_by_optimizer["hamiltonian_geometric"]["final_test_acc"]) >
            max(float(deepobs_by_optimizer[name]["final_test_acc"])
                for name in ("sgd_momentum", "adamw")),
            "official DeepOBS accuracy ordering changed")
    require(float(deepobs_by_optimizer["sgd_momentum"]["final_test_loss"]) <
            min(float(deepobs_by_optimizer[name]["final_test_loss"])
                for name in ("hamiltonian_geometric", "adamw")),
            "official DeepOBS loss ordering changed")
    deepobs_raw_paths = {
        "adamw": "results/official_deepobs/adamw/mnist_mlp/AdamWFixedDecay/num_epochs__20__batch_size__128__lr__3.e-03/random_seed__42__2026-07-21-10-26-02.json",
        "hamiltonian_geometric": "results/official_deepobs/hamiltonian_geometric/mnist_mlp/HamiltonianGeometricTorch/num_epochs__20__batch_size__128__beta__9.e-01__lr__1.e-01/random_seed__42__2026-07-21-09-52-53.json",
        "sgd_momentum": "results/official_deepobs/sgd_momentum/mnist_mlp/SGD/num_epochs__20__batch_size__128__lr__1.e-01__momentum__9.e-01/random_seed__42__2026-07-21-10-09-14.json",
    }
    expected_raw_names = {
        "adamw": "AdamWFixedDecay",
        "hamiltonian_geometric": "HamiltonianGeometricTorch",
        "sgd_momentum": "SGD",
    }
    for name, relative in deepobs_raw_paths.items():
        raw = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        require(raw["testproblem"] == "mnist_mlp" and raw["num_epochs"] == 20 and
                raw["random_seed"] == 42 and raw["optimizer_name"] == expected_raw_names[name],
                f"official DeepOBS raw metadata changed for {name}")
        summary = deepobs_by_optimizer[name]
        require(abs(raw["valid_losses"][-1] - float(summary["final_val_loss"])) < 1e-12 and
                abs(raw["valid_accuracies"][-1] - float(summary["final_val_acc"])) < 1e-12 and
                abs(raw["test_losses"][-1] - float(summary["final_test_loss"])) < 1e-12 and
                abs(raw["test_accuracies"][-1] - float(summary["final_test_acc"])) < 1e-12,
                f"official DeepOBS summary does not match raw runner output for {name}")
    manuscript = (ROOT / "docs/hamiltonian_geometric_consolidated_report.tex").read_text(encoding="utf-8")
    table_start = manuscript.find(r"\label{tab:deepobs-official}")
    table_end = manuscript.find(r"\end{table*}", table_start)
    require(table_start >= 0 and table_end > table_start,
            "official DeepOBS manuscript table is missing")
    deepobs_table = manuscript[table_start:table_end]
    display_names = {
        "hamiltonian_geometric": "Hamiltonian-geometric",
        "sgd_momentum": "SGD + momentum",
        "adamw": "AdamW",
    }
    for name, row in deepobs_by_optimizer.items():
        expected_values = (
            f'{float(row["final_val_loss"]):.4f}',
            f'{float(row["final_val_acc"]):.4f}',
            f'{float(row["final_test_loss"]):.4f}',
            f'{float(row["final_test_acc"]):.4f}',
            f'{float(row["runtime_s"]):.1f}',
        )
        require(display_names[name] in deepobs_table and
                all(value in deepobs_table for value in expected_values),
                f"official DeepOBS manuscript row is stale for {name}")
    interpretation = manuscript[table_end:table_end + 1400].lower()
    require("single seed" in interpretation and "no significance claim" in interpretation,
            "official DeepOBS manuscript caveat is missing")
    pinn = rows("results/pinn_multiseed_raw.csv")
    require(len(pinn) == 10 and {int(row["seed"]) for row in pinn} == set(range(10)),
            "PINN replication must contain seeds 0--9")
    require(all(row["best_other_optimizer"] == "entropy_descent" for row in pinn),
            "PINN comparator identity changed")
    cost = json.loads((ROOT / "results/pinn_replication_cost/pinn_cost_summary.json").read_text(encoding="utf-8"))
    require(cost["central_metric_probes_per_step"] == 130 and cost["estimated_speedup"] > 100,
            "PINN cost audit is incomplete")
    tuning = rows("results/equal_budget_tuning/tuning_raw.csv")
    evaluation = rows("results/equal_budget_tuning/evaluation_raw.csv")
    require(len(tuning) == 72 and len(evaluation) == 40,
            "equal-budget tuning must contain 4x6x3 tuning and 4x10 evaluation rows")
    require({int(row["seed"]) for row in tuning} == {100, 101, 102} and
            {int(row["seed"]) for row in evaluation} == set(range(10)),
            "equal-budget tuning/evaluation seeds overlap or are incomplete")

    expected_counts = {
        "ablation styudy/results/architecture_raw.csv": 640,
        "ablation styudy/results/quantum_raw.csv": 110,
        "ablation styudy/results/gpu_raw.csv": 60,
    }
    for name, expected in expected_counts.items():
        require(len(rows(name)) == expected, f"{name}: expected {expected} rows")
    gpu = rows("ablation styudy/results/gpu_raw.csv")
    require(all(row.get("target_accuracy") == "0.9" for row in gpu), "CUDA neural target must be 0.9")
    require(all(row.get("peak_gpu_memory_mb") for row in gpu), "CUDA neural peak memory is missing")

    require(
        len(rows("ppt_assets/test_result_media/modern_optimizer_benchmark_paired/modern_optimizer_seed_sweep.csv")) == 60,
        "modern optimizer paired seed sweep must contain 10 seeds x 6 optimizers",
    )
    require(
        len(rows("ppt_assets/test_result_media/modern_optimizer_benchmark_paired/modern_optimizer_scaling_sweep.csv")) == 30,
        "modern optimizer scaling sweep must contain 5 widths x 6 optimizers",
    )

    cuda = json.loads((ROOT / "ablation styudy/results/cuda_environment.json").read_text(encoding="utf-8"))
    require(cuda.get("cuda_available") is True, "CUDA ablation provenance does not report CUDA available")
    require(bool(cuda.get("device")), "CUDA device name is missing")

    return {
        "schema_version": 1,
        "authoritative_files": [
            {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)}
            for name in AUTHORITATIVE
        ],
        "validated_counts": {"paired_classical": 30, "official_deepobs": 3, "pinn": 10, "equal_budget_tuning": 72, "equal_budget_evaluation": 40, "llm": 3, "llm_compute_optimizers": 5, "architecture": 640, "quantum": 110, "gpu": 60,
                             "modern_optimizer_seeds": 60, "modern_optimizer_scaling": 30},
        "cuda_device": cuda["device"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    manifest = verify()
    if args.write_manifest:
        output = ROOT / "results/submission_artifact_manifest.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output.relative_to(ROOT)}")
    print(json.dumps({key: value for key, value in manifest.items() if key != "authoritative_files"}, indent=2))
    print(f"verified {len(AUTHORITATIVE)} authoritative files")


if __name__ == "__main__":
    main()
