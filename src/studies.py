"""Research-study helpers built on top of the capacitor solvers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.capacitance import RectangularCapacitor, fringe_ratio
from src.laplace2d import SolverConfig, solve_parallel_plate_2d


@dataclass(frozen=True)
class ConvergenceRow:
    """One grid-resolution result from a convergence study."""

    nx: int
    ny: int
    dx: float
    dy: float
    requested_gap: float
    electrode_gap: float
    gap_error: float
    requested_plate_width: float
    electrode_width: float
    plate_width_error: float
    converged: bool
    iterations: int
    max_delta: float
    residual_norm: float
    capacitance_per_depth: float
    charge_capacitance_per_depth: float
    capacitance_estimate_relative_difference: float
    ideal_capacitance_per_depth: float
    fringe_ratio: float
    charge_fringe_ratio: float
    relative_change_from_previous: float | None


@dataclass(frozen=True)
class DomainSizeRow:
    """One domain-size result from a boundary sensitivity study."""

    domain_width: float
    domain_height: float
    nx: int
    ny: int
    dx: float
    dy: float
    requested_gap: float
    electrode_gap: float
    gap_error: float
    requested_plate_width: float
    electrode_width: float
    plate_width_error: float
    converged: bool
    iterations: int
    max_delta: float
    residual_norm: float
    capacitance_per_depth: float
    charge_capacitance_per_depth: float
    capacitance_estimate_relative_difference: float
    ideal_capacitance_per_depth: float
    fringe_ratio: float
    charge_fringe_ratio: float
    relative_change_from_previous: float | None


@dataclass(frozen=True)
class GapSweepRow:
    """One physical gap result from a numerical gap sweep."""

    requested_gap: float
    gap_to_width: float
    electrode_gap: float
    gap_error: float
    nx: int
    ny: int
    dx: float
    dy: float
    requested_plate_width: float
    electrode_width: float
    plate_width_error: float
    converged: bool
    iterations: int
    max_delta: float
    residual_norm: float
    capacitance_per_depth: float
    charge_capacitance_per_depth: float
    capacitance_estimate_relative_difference: float
    ideal_capacitance_per_depth: float
    energy_fringe_ratio: float
    charge_fringe_ratio: float
    effective_area_fringe_ratio: float


def grid_convergence_study(
    base_config: SolverConfig,
    grid_sizes: list[int],
) -> list[ConvergenceRow]:
    """Run the same physical model across several square grid resolutions.

    The returned relative change is measured against the previous grid size:

        abs(C_current - C_previous) / abs(C_previous)

    A stable sequence gives more confidence that the result is controlled by
    the modeled geometry instead of by the grid spacing.
    """

    if len(grid_sizes) < 2:
        raise ValueError("grid_sizes must contain at least two resolutions")
    if any(size < 5 for size in grid_sizes):
        raise ValueError("each grid size must be at least 5")
    if grid_sizes != sorted(grid_sizes):
        raise ValueError("grid_sizes must be sorted from coarse to fine")
    if len(set(grid_sizes)) != len(grid_sizes):
        raise ValueError("grid_sizes must not contain duplicates")

    rows: list[ConvergenceRow] = []
    previous_capacitance: float | None = None
    for size in grid_sizes:
        config = replace(base_config, nx=size, ny=size)
        result = solve_parallel_plate_2d(config)

        relative_change = None
        if previous_capacitance is not None:
            relative_change = abs(
                result.capacitance_per_depth - previous_capacitance
            ) / abs(previous_capacitance)

        rows.append(
            ConvergenceRow(
                nx=config.nx,
                ny=config.ny,
                dx=config.dx,
                dy=config.dy,
                requested_gap=config.gap,
                electrode_gap=result.electrode_gap,
                gap_error=result.electrode_gap - config.gap,
                requested_plate_width=config.plate_width,
                electrode_width=result.electrode_width,
                plate_width_error=result.electrode_width - config.plate_width,
                converged=result.converged,
                iterations=result.iterations,
                max_delta=result.max_delta,
                residual_norm=result.residual_norm,
                capacitance_per_depth=result.capacitance_per_depth,
                charge_capacitance_per_depth=result.charge_capacitance_per_depth,
                capacitance_estimate_relative_difference=(
                    result.capacitance_estimate_relative_difference
                ),
                ideal_capacitance_per_depth=result.ideal_capacitance_per_depth,
                fringe_ratio=result.fringe_ratio,
                charge_fringe_ratio=result.charge_fringe_ratio,
                relative_change_from_previous=relative_change,
            )
        )
        previous_capacitance = result.capacitance_per_depth

    return rows


def numerical_gap_sweep(
    base_config: SolverConfig,
    gaps: list[float],
    edge_extension_fraction: float = 0.5,
) -> list[GapSweepRow]:
    """Run the 2D solver across several plate gaps.

    The domain size and grid spacing from `base_config` are kept fixed. Choose
    gaps that land close to grid rows when possible.
    """

    if len(gaps) < 2:
        raise ValueError("gaps must contain at least two values")
    if any(gap <= 0 for gap in gaps):
        raise ValueError("each gap must be positive")
    if any(gap >= base_config.domain_height for gap in gaps):
        raise ValueError("each gap must be smaller than the domain height")
    if gaps != sorted(gaps):
        raise ValueError("gaps must be sorted from small to large")
    if len(set(gaps)) != len(gaps):
        raise ValueError("gaps must not contain duplicates")

    rows: list[GapSweepRow] = []
    for gap in gaps:
        config = replace(base_config, gap=gap)
        result = solve_parallel_plate_2d(config)
        analytical_capacitor = RectangularCapacitor(
            length=config.plate_width,
            width=1.0,
            gap=gap,
            relative_permittivity=config.relative_permittivity,
        )
        rows.append(
            GapSweepRow(
                requested_gap=config.gap,
                gap_to_width=config.gap / config.plate_width,
                electrode_gap=result.electrode_gap,
                gap_error=result.electrode_gap - config.gap,
                nx=config.nx,
                ny=config.ny,
                dx=config.dx,
                dy=config.dy,
                requested_plate_width=config.plate_width,
                electrode_width=result.electrode_width,
                plate_width_error=result.electrode_width - config.plate_width,
                converged=result.converged,
                iterations=result.iterations,
                max_delta=result.max_delta,
                residual_norm=result.residual_norm,
                capacitance_per_depth=result.capacitance_per_depth,
                charge_capacitance_per_depth=result.charge_capacitance_per_depth,
                capacitance_estimate_relative_difference=(
                    result.capacitance_estimate_relative_difference
                ),
                ideal_capacitance_per_depth=result.ideal_capacitance_per_depth,
                energy_fringe_ratio=result.fringe_ratio,
                charge_fringe_ratio=result.charge_fringe_ratio,
                effective_area_fringe_ratio=fringe_ratio(
                    analytical_capacitor,
                    edge_extension_fraction=edge_extension_fraction,
                ),
            )
        )

    return rows


def adaptive_numerical_gap_sweep(
    base_config: SolverConfig,
    gaps: list[float],
    min_cells_across_smallest_gap: int = 4,
    edge_extension_fraction: float = 0.5,
) -> list[GapSweepRow]:
    """Run a gap sweep after refining the grid from the smallest gap.

    This avoids comparing a small-gap case represented by too few grid cells
    with large-gap cases represented by many cells.
    """

    if min_cells_across_smallest_gap < 2:
        raise ValueError("min_cells_across_smallest_gap must be at least 2")
    if len(gaps) < 2:
        raise ValueError("gaps must contain at least two values")
    if gaps != sorted(gaps):
        raise ValueError("gaps must be sorted from small to large")
    if any(gap <= 0 for gap in gaps):
        raise ValueError("each gap must be positive")

    target_spacing = gaps[0] / min_cells_across_smallest_gap
    nx = int(round(base_config.domain_width / target_spacing)) + 1
    ny = int(round(base_config.domain_height / target_spacing)) + 1
    refined_config = replace(base_config, nx=nx, ny=ny)
    return numerical_gap_sweep(
        refined_config,
        gaps,
        edge_extension_fraction=edge_extension_fraction,
    )


def domain_size_study(
    base_config: SolverConfig,
    domain_sizes: list[float],
) -> list[DomainSizeRow]:
    """Run the model while moving the artificial boundary farther away.

    The grid spacing from `base_config` is kept approximately fixed by deriving
    `nx` and `ny` from each requested square domain size. This makes the study
    focus on boundary placement instead of mixing in a changing resolution.
    """

    if len(domain_sizes) < 2:
        raise ValueError("domain_sizes must contain at least two widths")
    if any(size <= base_config.plate_width for size in domain_sizes):
        raise ValueError("each domain size must exceed the plate width")
    if any(size <= base_config.gap for size in domain_sizes):
        raise ValueError("each domain size must exceed the plate gap")
    if domain_sizes != sorted(domain_sizes):
        raise ValueError("domain_sizes must be sorted from small to large")
    if len(set(domain_sizes)) != len(domain_sizes):
        raise ValueError("domain_sizes must not contain duplicates")

    rows: list[DomainSizeRow] = []
    previous_capacitance: float | None = None
    target_dx = base_config.dx
    for size in domain_sizes:
        grid_points = int(round(size / target_dx)) + 1
        config = replace(
            base_config,
            domain_width=size,
            domain_height=size,
            nx=grid_points,
            ny=grid_points,
        )
        result = solve_parallel_plate_2d(config)

        relative_change = None
        if previous_capacitance is not None:
            relative_change = abs(
                result.capacitance_per_depth - previous_capacitance
            ) / abs(previous_capacitance)

        rows.append(
            DomainSizeRow(
                domain_width=config.domain_width,
                domain_height=config.domain_height,
                nx=config.nx,
                ny=config.ny,
                dx=config.dx,
                dy=config.dy,
                requested_gap=config.gap,
                electrode_gap=result.electrode_gap,
                gap_error=result.electrode_gap - config.gap,
                requested_plate_width=config.plate_width,
                electrode_width=result.electrode_width,
                plate_width_error=result.electrode_width - config.plate_width,
                converged=result.converged,
                iterations=result.iterations,
                max_delta=result.max_delta,
                residual_norm=result.residual_norm,
                capacitance_per_depth=result.capacitance_per_depth,
                charge_capacitance_per_depth=result.charge_capacitance_per_depth,
                capacitance_estimate_relative_difference=(
                    result.capacitance_estimate_relative_difference
                ),
                ideal_capacitance_per_depth=result.ideal_capacitance_per_depth,
                fringe_ratio=result.fringe_ratio,
                charge_fringe_ratio=result.charge_fringe_ratio,
                relative_change_from_previous=relative_change,
            )
        )
        previous_capacitance = result.capacitance_per_depth

    return rows


def export_gap_sweep_csv(rows: list[GapSweepRow], path: str) -> None:
    """Export numerical gap-sweep rows for plotting or spreadsheet analysis."""

    with open(path, "w", encoding="utf-8") as file:
        file.write(
            "requested_gap_m,gap_to_width,electrode_gap_m,gap_error_m,nx,ny,dx_m,dy_m,"
            "requested_plate_width_m,electrode_width_m,plate_width_error_m,"
            "converged,iterations,max_delta_V,residual_norm,"
            "capacitance_per_depth_F_per_m,"
            "charge_capacitance_per_depth_F_per_m,"
            "capacitance_estimate_relative_difference,"
            "ideal_capacitance_per_depth_F_per_m,energy_fringe_ratio,"
            "charge_fringe_ratio,effective_area_fringe_ratio\n"
        )
        for row in rows:
            file.write(
                f"{row.requested_gap:.12g},"
                f"{row.gap_to_width:.12g},"
                f"{row.electrode_gap:.12g},"
                f"{row.gap_error:.12g},"
                f"{row.nx},"
                f"{row.ny},"
                f"{row.dx:.12g},"
                f"{row.dy:.12g},"
                f"{row.requested_plate_width:.12g},"
                f"{row.electrode_width:.12g},"
                f"{row.plate_width_error:.12g},"
                f"{int(row.converged)},"
                f"{row.iterations},"
                f"{row.max_delta:.12g},"
                f"{row.residual_norm:.12g},"
                f"{row.capacitance_per_depth:.12g},"
                f"{row.charge_capacitance_per_depth:.12g},"
                f"{row.capacitance_estimate_relative_difference:.12g},"
                f"{row.ideal_capacitance_per_depth:.12g},"
                f"{row.energy_fringe_ratio:.12g},"
                f"{row.charge_fringe_ratio:.12g},"
                f"{row.effective_area_fringe_ratio:.12g}\n"
            )


def export_convergence_csv(rows: list[ConvergenceRow], path: str) -> None:
    """Export convergence-study rows for plotting or spreadsheet analysis."""

    with open(path, "w", encoding="utf-8") as file:
        file.write(
            "nx,ny,dx_m,dy_m,requested_gap_m,electrode_gap_m,gap_error_m,"
            "requested_plate_width_m,electrode_width_m,plate_width_error_m,"
            "converged,iterations,max_delta_V,residual_norm,"
            "capacitance_per_depth_F_per_m,"
            "charge_capacitance_per_depth_F_per_m,"
            "capacitance_estimate_relative_difference,"
            "ideal_capacitance_per_depth_F_per_m,fringe_ratio,charge_fringe_ratio,"
            "relative_change_from_previous\n"
        )
        for row in rows:
            relative_change = (
                ""
                if row.relative_change_from_previous is None
                else f"{row.relative_change_from_previous:.12g}"
            )
            file.write(
                f"{row.nx},"
                f"{row.ny},"
                f"{row.dx:.12g},"
                f"{row.dy:.12g},"
                f"{row.requested_gap:.12g},"
                f"{row.electrode_gap:.12g},"
                f"{row.gap_error:.12g},"
                f"{row.requested_plate_width:.12g},"
                f"{row.electrode_width:.12g},"
                f"{row.plate_width_error:.12g},"
                f"{int(row.converged)},"
                f"{row.iterations},"
                f"{row.max_delta:.12g},"
                f"{row.residual_norm:.12g},"
                f"{row.capacitance_per_depth:.12g},"
                f"{row.charge_capacitance_per_depth:.12g},"
                f"{row.capacitance_estimate_relative_difference:.12g},"
                f"{row.ideal_capacitance_per_depth:.12g},"
                f"{row.fringe_ratio:.12g},"
                f"{row.charge_fringe_ratio:.12g},"
                f"{relative_change}\n"
            )


def export_domain_size_csv(rows: list[DomainSizeRow], path: str) -> None:
    """Export domain-size sensitivity rows for plotting or spreadsheet analysis."""

    with open(path, "w", encoding="utf-8") as file:
        file.write(
            "domain_width_m,domain_height_m,nx,ny,dx_m,dy_m,requested_gap_m,"
            "electrode_gap_m,gap_error_m,requested_plate_width_m,"
            "electrode_width_m,plate_width_error_m,converged,iterations,max_delta_V,"
            "residual_norm,capacitance_per_depth_F_per_m,"
            "charge_capacitance_per_depth_F_per_m,"
            "capacitance_estimate_relative_difference,"
            "ideal_capacitance_per_depth_F_per_m,fringe_ratio,charge_fringe_ratio,"
            "relative_change_from_previous\n"
        )
        for row in rows:
            relative_change = (
                ""
                if row.relative_change_from_previous is None
                else f"{row.relative_change_from_previous:.12g}"
            )
            file.write(
                f"{row.domain_width:.12g},"
                f"{row.domain_height:.12g},"
                f"{row.nx},"
                f"{row.ny},"
                f"{row.dx:.12g},"
                f"{row.dy:.12g},"
                f"{row.requested_gap:.12g},"
                f"{row.electrode_gap:.12g},"
                f"{row.gap_error:.12g},"
                f"{row.requested_plate_width:.12g},"
                f"{row.electrode_width:.12g},"
                f"{row.plate_width_error:.12g},"
                f"{int(row.converged)},"
                f"{row.iterations},"
                f"{row.max_delta:.12g},"
                f"{row.residual_norm:.12g},"
                f"{row.capacitance_per_depth:.12g},"
                f"{row.charge_capacitance_per_depth:.12g},"
                f"{row.capacitance_estimate_relative_difference:.12g},"
                f"{row.ideal_capacitance_per_depth:.12g},"
                f"{row.fringe_ratio:.12g},"
                f"{row.charge_fringe_ratio:.12g},"
                f"{relative_change}\n"
            )
