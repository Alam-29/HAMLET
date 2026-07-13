from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.report import build_pdf_report


def main() -> None:
    output_path = PROJECT_ROOT / "docs" / "hamiltonian_geometric_optimizer_report.pdf"
    benchmark_dir = PROJECT_ROOT / "visualizations" / "pinn_benchmark"

    build_pdf_report(
        output_path=str(output_path),
        project_root=PROJECT_ROOT,
        benchmark_summary_path=benchmark_dir / "pinn_optimizer_summary.csv",
        potential_png=PROJECT_ROOT / "visualizations" / "potential_field.png",
        fringing_png=PROJECT_ROOT / "visualizations" / "fringing_field_lines.png",
        convergence_png=benchmark_dir / "optimizer_convergence.png",
        spectrum_png=benchmark_dir / "normal_mode_spectrum.png",
    )

    print(f"exported = {output_path}")


if __name__ == "__main__":
    main()
