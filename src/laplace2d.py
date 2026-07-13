"""Finite-difference Laplace solver for a capacitor cross-section.

The solver models two finite, thin plate electrodes in a 2D rectangular domain.
It estimates capacitance per unit depth from stored electrostatic energy:

    U' = 0.5 * epsilon * integral(|E|^2 dA)
    C' = 2 U' / V^2

It also computes an independent charge-based estimate from the flux leaving
the positive electrode.

This is a numerical research model, not a closed-form law. Results depend on
grid resolution and on how far the artificial simulation boundary is from the
plates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.capacitance import EPSILON_0


@dataclass(frozen=True)
class SolverConfig:
    """Grid and physical settings for the 2D cross-section solver."""

    plate_width: float
    gap: float
    domain_width: float
    domain_height: float
    nx: int = 121
    ny: int = 121
    voltage: float = 1.0
    relative_permittivity: float = 1.0
    tolerance: float = 1e-5
    max_iterations: int = 20_000
    relaxation: float = 1.7
    method: str = "sor"

    def __post_init__(self) -> None:
        positive_values = {
            "plate_width": self.plate_width,
            "gap": self.gap,
            "domain_width": self.domain_width,
            "domain_height": self.domain_height,
            "voltage": self.voltage,
            "relative_permittivity": self.relative_permittivity,
            "tolerance": self.tolerance,
            "relaxation": self.relaxation,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")

        if self.nx < 5 or self.ny < 5:
            raise ValueError("nx and ny must each be at least 5")
        if self.plate_width >= self.domain_width:
            raise ValueError("plate_width must be smaller than domain_width")
        if self.gap >= self.domain_height:
            raise ValueError("gap must be smaller than domain_height")
        if self.method not in {"jacobi", "sor"}:
            raise ValueError("method must be either 'jacobi' or 'sor'")
        if self.method == "jacobi" and not 0.0 < self.relaxation <= 1.0:
            raise ValueError("jacobi relaxation must be in the range (0.0, 1.0]")
        if self.method == "sor" and not 0.0 < self.relaxation < 2.0:
            raise ValueError("sor relaxation must be in the range (0.0, 2.0)")

    @property
    def dx(self) -> float:
        return self.domain_width / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.domain_height / (self.ny - 1)

    @property
    def permittivity(self) -> float:
        return EPSILON_0 * self.relative_permittivity

    @property
    def ideal_capacitance_per_depth(self) -> float:
        return self.permittivity * self.plate_width / self.gap


@dataclass(frozen=True)
class SolverResult:
    """Outputs from a solved 2D capacitor cross-section."""

    potential: np.ndarray
    electrode_mask: np.ndarray
    x: np.ndarray
    y: np.ndarray
    iterations: int
    max_delta: float
    residual_norm: float
    capacitance_per_depth: float
    charge_capacitance_per_depth: float
    energy_per_depth: float
    positive_charge_per_depth: float
    electrode_gap: float
    electrode_width: float
    config: SolverConfig

    @property
    def ideal_capacitance_per_depth(self) -> float:
        return self.config.permittivity * self.config.plate_width / self.electrode_gap

    @property
    def fringe_ratio(self) -> float:
        return self.capacitance_per_depth / self.ideal_capacitance_per_depth

    @property
    def charge_fringe_ratio(self) -> float:
        return self.charge_capacitance_per_depth / self.ideal_capacitance_per_depth

    @property
    def capacitance_estimate_relative_difference(self) -> float:
        return abs(
            self.capacitance_per_depth - self.charge_capacitance_per_depth
        ) / self.capacitance_per_depth

    @property
    def converged(self) -> bool:
        return self.max_delta <= self.config.tolerance


def solve_parallel_plate_2d(config: SolverConfig) -> SolverResult:
    """Solve the 2D potential around finite parallel plates."""

    x = np.linspace(-config.domain_width / 2.0, config.domain_width / 2.0, config.nx)
    y = np.linspace(-config.domain_height / 2.0, config.domain_height / 2.0, config.ny)
    potential = np.zeros((config.ny, config.nx), dtype=float)
    electrode_mask = np.zeros_like(potential, dtype=bool)

    center_row = (config.ny - 1) // 2
    half_gap_steps = max(1, int(round((config.gap / 2.0) / config.dy)))
    lower_row = center_row - half_gap_steps
    upper_row = center_row + half_gap_steps
    if lower_row <= 0 or upper_row >= config.ny - 1:
        raise ValueError("gap places electrodes too close to the domain boundary")

    geometry_tolerance = max(config.dx, config.dy) * 1e-9
    plate_columns = np.abs(x) <= config.plate_width / 2.0 + geometry_tolerance
    electrode_gap = float(y[upper_row] - y[lower_row])
    electrode_width = float(x[plate_columns][-1] - x[plate_columns][0])

    electrode_mask[lower_row, plate_columns] = True
    electrode_mask[upper_row, plate_columns] = True
    positive_electrode_mask = np.zeros_like(potential, dtype=bool)
    positive_electrode_mask[upper_row, plate_columns] = True
    potential[lower_row, plate_columns] = -config.voltage / 2.0
    potential[upper_row, plate_columns] = config.voltage / 2.0

    max_delta = float("inf")
    omega = config.relaxation
    inverse_dx2 = 1.0 / (config.dx**2)
    inverse_dy2 = 1.0 / (config.dy**2)
    denominator = 2.0 * (inverse_dx2 + inverse_dy2)
    interior_rows, interior_columns = np.indices((config.ny - 2, config.nx - 2))
    free_node_mask = ~electrode_mask[1:-1, 1:-1]
    red_node_mask = free_node_mask & ((interior_rows + interior_columns) % 2 == 0)
    black_node_mask = free_node_mask & ~red_node_mask

    for iteration in range(1, config.max_iterations + 1):
        old = potential.copy()

        _apply_open_boundary(potential)
        if config.method == "jacobi":
            average = (
                (potential[1:-1, 2:] + potential[1:-1, :-2]) * inverse_dx2
                + (potential[2:, 1:-1] + potential[:-2, 1:-1]) * inverse_dy2
            ) / denominator
            potential[1:-1, 1:-1] = (1.0 - omega) * old[1:-1, 1:-1] + omega * average
        else:
            interior = potential[1:-1, 1:-1]
            for node_mask in (red_node_mask, black_node_mask):
                average = (
                    (potential[1:-1, 2:] + potential[1:-1, :-2]) * inverse_dx2
                    + (potential[2:, 1:-1] + potential[:-2, 1:-1]) * inverse_dy2
                ) / denominator
                interior[node_mask] += omega * (average[node_mask] - interior[node_mask])

        _apply_electrode_voltages(potential, lower_row, upper_row, plate_columns, config.voltage)

        max_delta = float(np.max(np.abs(potential - old)))
        if max_delta <= config.tolerance:
            break

    residual_norm = laplace_residual_norm(potential, electrode_mask, config)
    energy = electric_energy_per_depth(potential, config)
    capacitance = 2.0 * energy / (config.voltage**2)
    charge = electrode_charge_per_depth(
        potential,
        positive_electrode_mask,
        config,
        electrode_voltage=config.voltage / 2.0,
    )
    charge_capacitance = charge / config.voltage
    return SolverResult(
        potential=potential,
        electrode_mask=electrode_mask,
        x=x,
        y=y,
        iterations=iteration,
        max_delta=max_delta,
        residual_norm=residual_norm,
        capacitance_per_depth=capacitance,
        charge_capacitance_per_depth=charge_capacitance,
        energy_per_depth=energy,
        positive_charge_per_depth=charge,
        electrode_gap=electrode_gap,
        electrode_width=electrode_width,
        config=config,
    )


def _apply_open_boundary(potential: np.ndarray) -> None:
    # Zero-normal-gradient outer boundaries keep the box from acting like a
    # grounded conductor. The domain should still be made generously large.
    potential[0, :] = potential[1, :]
    potential[-1, :] = potential[-2, :]
    potential[:, 0] = potential[:, 1]
    potential[:, -1] = potential[:, -2]


def _apply_electrode_voltages(
    potential: np.ndarray,
    lower_row: int,
    upper_row: int,
    plate_columns: np.ndarray,
    voltage: float,
) -> None:
    potential[lower_row, plate_columns] = -voltage / 2.0
    potential[upper_row, plate_columns] = voltage / 2.0


def laplace_residual_norm(
    potential: np.ndarray,
    electrode_mask: np.ndarray,
    config: SolverConfig,
) -> float:
    """Return a dimensionless max residual for Laplace's equation."""

    residual = (
        (potential[1:-1, 2:] - 2.0 * potential[1:-1, 1:-1] + potential[1:-1, :-2])
        / (config.dx**2)
        + (potential[2:, 1:-1] - 2.0 * potential[1:-1, 1:-1] + potential[:-2, 1:-1])
        / (config.dy**2)
    )
    free_nodes = ~electrode_mask[1:-1, 1:-1]
    if not np.any(free_nodes):
        return 0.0
    scale = config.voltage / min(config.dx, config.dy) ** 2
    return float(np.max(np.abs(residual[free_nodes])) / scale)


def electric_field(potential: np.ndarray, config: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return Ex and Ey arrays from the solved potential grid."""

    dphi_dy, dphi_dx = np.gradient(potential, config.dy, config.dx)
    return -dphi_dx, -dphi_dy


def electric_energy_per_depth(potential: np.ndarray, config: SolverConfig) -> float:
    """Estimate stored electrostatic energy per unit depth."""

    ex, ey = electric_field(potential, config)
    energy_density = 0.5 * config.permittivity * (ex**2 + ey**2)
    return float(np.sum(energy_density) * config.dx * config.dy)


def electrode_charge_per_depth(
    potential: np.ndarray,
    electrode_mask: np.ndarray,
    config: SolverConfig,
    electrode_voltage: float,
) -> float:
    """Estimate electrode charge per unit depth from adjacent-grid flux."""

    charge = 0.0
    rows, columns = np.nonzero(electrode_mask)
    neighbor_specs = (
        (-1, 0, config.dy, config.dx),
        (1, 0, config.dy, config.dx),
        (0, -1, config.dx, config.dy),
        (0, 1, config.dx, config.dy),
    )
    for row, column in zip(rows, columns):
        for row_offset, column_offset, distance, surface_length in neighbor_specs:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if (
                neighbor_row < 0
                or neighbor_row >= potential.shape[0]
                or neighbor_column < 0
                or neighbor_column >= potential.shape[1]
                or electrode_mask[neighbor_row, neighbor_column]
            ):
                continue
            charge += (
                config.permittivity
                * (electrode_voltage - potential[neighbor_row, neighbor_column])
                / distance
                * surface_length
            )
    return float(charge)


def export_potential_csv(result: SolverResult, path: str) -> None:
    """Export x, y, potential, and electrode-mask columns for external plotting."""

    with open(path, "w", encoding="utf-8") as file:
        file.write("x_m,y_m,potential_V,is_electrode\n")
        for row, y_value in enumerate(result.y):
            for column, x_value in enumerate(result.x):
                file.write(
                    f"{x_value:.12g},"
                    f"{y_value:.12g},"
                    f"{result.potential[row, column]:.12g},"
                    f"{int(result.electrode_mask[row, column])}\n"
                )
