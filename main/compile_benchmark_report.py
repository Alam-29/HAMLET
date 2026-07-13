import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_dir = PROJECT_ROOT / "visualizations" / "benchmark_suite_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_suite_report.md"
    sections = [
        "# Benchmark Suite Report",
        "",
        "This report aggregates the local benchmark artifacts generated in this workspace.",
        "Official MLCommons AlgoPerf harness results are not included because `.external/algorithmic-efficiency`",
        "is not present in this checkout; the available runs are local development benchmarks.",
        "",
    ]
    sections.extend(section_from_csv("MNIST softmax", PROJECT_ROOT / "visualizations/mnist_optimizer_benchmark/mnist_optimizer_summary.csv"))
    sections.extend(section_from_csv("AlgoPerf-style local MLP", PROJECT_ROOT / "visualizations/algoperf_style_benchmark/algoperf_style_best_summary.csv"))
    sections.extend(section_from_csv("DeepOBS-style local MLP", PROJECT_ROOT / "visualizations/deepobs_style_benchmark/deepobs_style_optimizer_summary.csv"))
    sections.extend(section_from_csv("PINN capacitor benchmark", PROJECT_ROOT / "visualizations/pinn_benchmark/pinn_optimizer_summary.csv"))
    sections.extend(section_from_csv("Quantum kicked-top benchmark", PROJECT_ROOT / "visualizations/quantum_chaos_benchmark/quantum_chaos_summary.csv"))
    sections.extend(section_from_csv("QAOA MaxCut benchmark", PROJECT_ROOT / "visualizations/qaoa_benchmark/qaoa_summary.csv"))
    sections.extend(section_from_csv("Quantum HG ablation", PROJECT_ROOT / "visualizations/quantum_ablation_study/quantum_ablation_summary.csv"))
    sections.extend(architecture_ablation_section(PROJECT_ROOT / "visualizations/ablation_study/ablation_raw_data.npz"))
    sections.extend(section_from_csv("Geometric evidence: condition scaling", PROJECT_ROOT / "visualizations/geometric_evidence/condition_number_scaling.csv"))
    sections.extend(section_from_csv("Geometric evidence: saddle escape", PROJECT_ROOT / "visualizations/geometric_evidence/saddle_escape.csv"))
    sections.extend(
        [
            "## Critical Takeaways",
            "",
            "- Hamiltonian-geometric is strong on the local AlgoPerf-style MLP and small PINN benchmark.",
            "- It is competitive on MNIST softmax regression by test loss, though test accuracy is effectively tied with several baselines.",
            "- Adam is strongest on the DeepOBS-style MLP in this run.",
            "- Heavy ball and AdamW dominate the chaotic quantum kicked-top default workload.",
            "- QAOA MaxCut gives a cleaner VQA benchmark where Hamiltonian-geometric ties heavy-ball for best approximation ratio on the small exact-statevector instance.",
            "- Quantum ablation shows low momentum is crucial for HG; memory, spectral force, and explicit finite-difference geometric force hurt this metric design.",
            "- Architecture ablation shows metric choice dominates: the spectral term is highly significant under the Hessian metric but inert under an unrelated toy metric.",
            "- The full Hessian-metric HG architecture is one of only two Hessian configurations with zero divergent seeds in the controlled ablation.",
            "- Geometric evidence experiments show the clearest HG advantage on full-metric ill-conditioned rotated quadratics.",
            "- Phase-space and saddle tests are more mixed: HG escapes saddles reliably, but entropy descent can match or beat it on final loss in the tested double-well landscape.",
            "- The explicit geometric-force quantum variant is much slower, so a serious next step is a better analytic or adjoint quantum metric rather than deeper finite differences.",
            "",
            "## Visual Summary",
            "",
            "- MNIST test-loss and test-accuracy curves were generated.",
            "- Local AlgoPerf-style validation loss and accuracy curves were generated.",
            "- DeepOBS-style loss, accuracy, and convergence animation were generated.",
            "- PINN optimizer convergence and dashboard visualizations were generated.",
            "- Quantum kicked-top fidelity, loss, and learned-control curves were generated.",
            "- QAOA MaxCut approximation-ratio and loss curves were generated.",
            "- Quantum ablation fidelity and loss curves were generated.",
            "- Controlled architecture ablation panels were generated.",
            "- Geometric evidence plots for conditioning, rotations, phase-space energy, and saddle escape were generated.",
            "",
        ]
    )
    report_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"exported = {report_path}")


def section_from_csv(title: str, path: Path) -> list[str]:
    if not path.exists():
        return [f"## {title}", "", f"Missing artifact: `{path}`", ""]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows:
        return [f"## {title}", "", "Empty artifact.", ""]
    header = rows[0]
    body = rows[1:8]
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def architecture_ablation_section(path: Path) -> list[str]:
    if not path.exists():
        return ["## Architecture Ablation", "", "Architecture ablation data is not available.", ""]
    data = np.load(path, allow_pickle=True)
    rows: list[tuple[str, str, float, int]] = []
    for metric in ("toy", "hessian"):
        for config in (
            "G0M0S0",
            "G0M0S1",
            "G0M1S0",
            "G0M1S1",
            "G1M0S0",
            "G1M0S1",
            "G1M1S0",
            "G1M1S1",
        ):
            history = data[f"history_{metric}_{config}"]
            final = history[:, -1]
            rows.append((metric, config, float(np.median(final)), int(np.sum(np.abs(final) > 100.0))))
    lines = [
        "## Architecture Ablation",
        "",
        "| metric | configuration | median final loss | divergent seeds |",
        "|---|---|---:|---:|",
    ]
    for metric, config, median_loss, diverged in rows:
        lines.append(f"| {metric} | {config} | {median_loss:.12g} | {diverged} |")
    lines.extend(
        [
            "",
            "Configuration key: G = geometric correction, M = memory correction, S = spectral term.",
            "The controlled study uses 40 paired random seeds per configuration and 300 optimization steps.",
            "",
        ]
    )
    return lines


if __name__ == "__main__":
    main()
