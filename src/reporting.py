"""Generate compact research reports from model outputs."""

from __future__ import annotations

from datetime import datetime

from src.laplace2d import SolverResult
from src.studies import ConvergenceRow, DomainSizeRow, GapSweepRow


def write_validation_report(
    path: str,
    result: SolverResult,
    convergence_rows: list[ConvergenceRow],
    domain_rows: list[DomainSizeRow],
    gap_rows: list[GapSweepRow] | None = None,
) -> None:
    """Write a Markdown report summarizing the latest model run."""

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Capacitor Fringing Validation Report\n\n")
        file.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        file.write("## Main Field Solve\n\n")
        file.write(f"- Converged: {result.converged}\n")
        file.write(f"- Iterations: {result.iterations}\n")
        file.write(f"- Solver method: {result.config.method}\n")
        file.write(f"- Relaxation factor: {result.config.relaxation:.3f}\n")
        file.write(f"- Final max delta: {result.max_delta:.6e} V\n")
        file.write(f"- Dimensionless Laplace residual: {result.residual_norm:.6e}\n")
        file.write(f"- Requested gap: {result.config.gap:.6e} m\n")
        file.write(f"- Grid electrode gap: {result.electrode_gap:.6e} m\n")
        file.write(f"- Requested plate width: {result.config.plate_width:.6e} m\n")
        file.write(f"- Grid electrode width: {result.electrode_width:.6e} m\n")
        file.write(
            f"- Energy capacitance per depth: {result.capacitance_per_depth:.6e} F/m\n"
        )
        file.write(
            f"- Charge capacitance per depth: "
            f"{result.charge_capacitance_per_depth:.6e} F/m\n"
        )
        file.write(
            f"- Method relative difference: "
            f"{result.capacitance_estimate_relative_difference:.6f}\n"
        )
        file.write(f"- Energy fringe ratio: {result.fringe_ratio:.6f}\n")
        file.write(f"- Charge fringe ratio: {result.charge_fringe_ratio:.6f}\n\n")

        file.write("## Grid Convergence\n\n")
        file.write(
            "| nx | residual | gap error m | plate width error m | energy F/m | "
            "charge F/m | method diff | change from previous |\n"
        )
        file.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in convergence_rows:
            file.write(
                f"| {row.nx} | {row.residual_norm:.3e} | "
                f"{row.gap_error:.3e} | "
                f"{row.plate_width_error:.3e} | "
                f"{row.capacitance_per_depth:.6e} | "
                f"{row.charge_capacitance_per_depth:.6e} | "
                f"{row.capacitance_estimate_relative_difference:.6f} | "
                f"{_optional_float(row.relative_change_from_previous)} |\n"
            )

        file.write("\n## Domain Size Sensitivity\n\n")
        file.write(
            "| domain width m | nx | residual | energy F/m | charge F/m | "
            "method diff | change from previous |\n"
        )
        file.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in domain_rows:
            file.write(
                f"| {row.domain_width:.3f} | {row.nx} | "
                f"{row.residual_norm:.3e} | "
                f"{row.capacitance_per_depth:.6e} | "
                f"{row.charge_capacitance_per_depth:.6e} | "
                f"{row.capacitance_estimate_relative_difference:.6f} | "
                f"{_optional_float(row.relative_change_from_previous)} |\n"
            )

        if gap_rows is not None:
            file.write("\n## Numerical Gap Sweep\n\n")
            file.write(
                "| gap / width | requested gap m | actual gap m | residual | energy ratio | "
                "charge ratio | effective-area ratio | method diff |\n"
            )
            file.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in gap_rows:
                file.write(
                    f"| {row.gap_to_width:.3f} | "
                    f"{row.requested_gap:.4f} | {row.electrode_gap:.4f} | "
                    f"{row.residual_norm:.3e} | "
                    f"{row.energy_fringe_ratio:.6f} | "
                    f"{row.charge_fringe_ratio:.6f} | "
                    f"{row.effective_area_fringe_ratio:.6f} | "
                    f"{row.capacitance_estimate_relative_difference:.6f} |\n"
                )

        file.write("\n## Interpretation\n\n")
        file.write(
            "A small final domain-size change suggests the outer boundary is far "
            "enough for the tested geometry. A nonzero energy-vs-charge method "
            "difference should be treated as numerical uncertainty, especially "
            "near thin electrode edges. The gap sweep is a trend comparison, "
            "not a final validation, until grid convergence is stronger.\n"
        )


def _optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"
