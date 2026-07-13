from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normal_modes import (
    compute_normal_modes,
    export_normal_mode_markdown,
    run_all_checks,
    summarize_conditioning,
)
from src.pinn import FixedFeaturePotentialModel, PINNConfig, build_pinn_dataset
from src.visualization import export_normal_mode_spectrum_png


def main() -> None:
    output_path = PROJECT_ROOT / "docs" / "normal_mode_analysis.md"
    spectrum_path = PROJECT_ROOT / "visualizations" / "pinn_benchmark" / "normal_mode_spectrum.png"
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)

    config = PINNConfig()
    model = FixedFeaturePotentialModel(config)
    dataset = build_pinn_dataset(model, config)
    modes = compute_normal_modes(dataset)
    conditioning = summarize_conditioning(modes)
    export_normal_mode_spectrum_png(modes, conditioning, str(spectrum_path))

    checks = run_all_checks(config)
    markdown = export_normal_mode_markdown(modes, checks, conditioning)
    output_path.write_text(markdown, encoding="utf-8")

    print("Normal-mode / action-angle analysis")
    print("passed,kind,identifier,max_error,tolerance,section")
    failed = 0
    for check in checks:
        if not check.passed:
            failed += 1
        print(
            f"{check.passed},"
            f"{check.kind},"
            f"{check.identifier},"
            f"{check.max_error:.3e},"
            f"{check.tolerance:.3e},"
            f"{check.section}"
        )
    print()
    print(f"condition_number = {conditioning['condition_number']:.6g}")
    print(f"worst_mode_rate_plain = {conditioning['worst_mode_rate_plain']:.8f}")
    print(f"worst_mode_rate_preconditioned = {conditioning['worst_mode_rate_preconditioned']:.8f}")
    print(f"exported = {output_path}")
    if failed:
        print(f"WARNING: {failed} check(s) failed -- see the report for detail")


if __name__ == "__main__":
    main()
