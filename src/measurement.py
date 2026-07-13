"""Detector/photo-plate observations of capacitor fringing fields.

This module turns the solved field into a prototype measurement system. It can
sample a detector plate near the capacitor, add external DC/EMI/stochastic field
interference, and report maxima/minima that an optimizer can try to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.laplace2d import SolverResult, electric_field
from src.physical_system import Environment


@dataclass(frozen=True)
class DetectorPlate:
    """Line detector placed near the capacitor in the 2D cross-section."""

    center_x: float = 0.0
    center_y: float = 0.012
    length: float = 0.05
    samples: int = 121
    normal_x: float = 0.0
    normal_y: float = 1.0

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("length must be positive")
        if self.samples < 3:
            raise ValueError("samples must be at least 3")
        if self.normal_x == 0.0 and self.normal_y == 0.0:
            raise ValueError("detector normal vector must be nonzero")

    @property
    def unit_normal(self) -> tuple[float, float]:
        norm = math.hypot(self.normal_x, self.normal_y)
        return self.normal_x / norm, self.normal_y / norm


@dataclass(frozen=True)
class ExternalInterference:
    """External field perturbations seen by the detector setup."""

    dc_field_x_v_per_m: float = 0.0
    dc_field_y_v_per_m: float = 0.0
    emi_amplitude_v_per_m: float = 0.0
    emi_spatial_frequency_per_m: float = 80.0
    emi_phase_rad: float = 0.0
    stochastic_std_v_per_m: float = 0.0
    humidity_noise_gain: float = 1.0
    temperature_noise_gain_per_c: float = 0.01
    seed: int = 11

    def __post_init__(self) -> None:
        if self.emi_amplitude_v_per_m < 0.0:
            raise ValueError("emi_amplitude_v_per_m must be non-negative")
        if self.emi_spatial_frequency_per_m < 0.0:
            raise ValueError("emi_spatial_frequency_per_m must be non-negative")
        if self.stochastic_std_v_per_m < 0.0:
            raise ValueError("stochastic_std_v_per_m must be non-negative")


@dataclass(frozen=True)
class DetectorObservation:
    """One sampled detector point and its observed field."""

    x: float
    y: float
    ex_v_per_m: float
    ey_v_per_m: float
    magnitude_v_per_m: float
    normal_component_v_per_m: float
    tangential_component_v_per_m: float
    direction_deg: float
    is_local_maximum: bool
    is_local_minimum: bool


@dataclass(frozen=True)
class DetectorSummary:
    """Aggregate detector-plate observables for optimization targets."""

    max_magnitude_v_per_m: float
    min_magnitude_v_per_m: float
    max_normal_v_per_m: float
    min_normal_v_per_m: float
    peak_to_peak_magnitude_v_per_m: float
    edge_to_center_ratio: float
    mean_direction_deg: float
    direction_std_deg: float
    fringing_tangential_ratio: float
    local_maxima_count: int
    local_minima_count: int


def observe_detector_plate(
    result: SolverResult,
    detector: DetectorPlate,
    interference: ExternalInterference | None = None,
    environment: Environment | None = None,
) -> tuple[list[DetectorObservation], DetectorSummary]:
    """Sample the solved field along a detector plate with optional interference."""

    interference = interference or ExternalInterference()
    environment = environment or Environment()
    ex_grid, ey_grid = electric_field(result.potential, result.config)

    points = _detector_points(detector)
    base_ex = _bilinear_sample(result.x, result.y, ex_grid, points[:, 0], points[:, 1])
    base_ey = _bilinear_sample(result.x, result.y, ey_grid, points[:, 0], points[:, 1])
    perturbed_ex, perturbed_ey = _apply_interference(
        points,
        base_ex,
        base_ey,
        interference,
        environment,
    )

    normal_x, normal_y = detector.unit_normal
    tangent_x, tangent_y = -normal_y, normal_x
    normal_component = perturbed_ex * normal_x + perturbed_ey * normal_y
    tangential_component = perturbed_ex * tangent_x + perturbed_ey * tangent_y
    magnitude = np.hypot(perturbed_ex, perturbed_ey)
    direction = np.degrees(np.arctan2(perturbed_ey, perturbed_ex))
    maxima = _local_maxima(magnitude)
    minima = _local_minima(magnitude)

    observations = [
        DetectorObservation(
            x=float(points[index, 0]),
            y=float(points[index, 1]),
            ex_v_per_m=float(perturbed_ex[index]),
            ey_v_per_m=float(perturbed_ey[index]),
            magnitude_v_per_m=float(magnitude[index]),
            normal_component_v_per_m=float(normal_component[index]),
            tangential_component_v_per_m=float(tangential_component[index]),
            direction_deg=float(direction[index]),
            is_local_maximum=bool(maxima[index]),
            is_local_minimum=bool(minima[index]),
        )
        for index in range(detector.samples)
    ]
    return observations, _detector_summary(
        magnitude,
        normal_component,
        tangential_component,
        direction,
        maxima,
        minima,
    )


def export_detector_observations_csv(
    observations: list[DetectorObservation],
    summary: DetectorSummary,
    observation_path: str,
    summary_path: str,
) -> None:
    """Export detector samples and aggregate fringing extrema diagnostics."""

    with open(observation_path, "w", encoding="utf-8") as file:
        file.write(
            "x_m,y_m,ex_v_per_m,ey_v_per_m,magnitude_v_per_m,"
            "normal_component_v_per_m,tangential_component_v_per_m,"
            "direction_deg,is_local_maximum,is_local_minimum\n"
        )
        for row in observations:
            file.write(
                f"{row.x:.12g},"
                f"{row.y:.12g},"
                f"{row.ex_v_per_m:.12g},"
                f"{row.ey_v_per_m:.12g},"
                f"{row.magnitude_v_per_m:.12g},"
                f"{row.normal_component_v_per_m:.12g},"
                f"{row.tangential_component_v_per_m:.12g},"
                f"{row.direction_deg:.12g},"
                f"{int(row.is_local_maximum)},"
                f"{int(row.is_local_minimum)}\n"
            )

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("observable,value\n")
        for name, value in summary.__dict__.items():
            file.write(f"{name},{value:.12g}\n")


def _detector_points(detector: DetectorPlate) -> np.ndarray:
    offsets = np.linspace(-detector.length / 2.0, detector.length / 2.0, detector.samples)
    normal_x, normal_y = detector.unit_normal
    tangent_x, tangent_y = -normal_y, normal_x
    return np.column_stack(
        [
            detector.center_x + tangent_x * offsets,
            detector.center_y + tangent_y * offsets,
        ]
    )


def _apply_interference(
    points: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    interference: ExternalInterference,
    environment: Environment,
) -> tuple[np.ndarray, np.ndarray]:
    emi = interference.emi_amplitude_v_per_m * np.sin(
        2.0
        * math.pi
        * interference.emi_spatial_frequency_per_m
        * points[:, 0]
        + interference.emi_phase_rad
    )
    humidity_factor = 1.0 + interference.humidity_noise_gain * environment.relative_humidity
    temperature_factor = 1.0 + interference.temperature_noise_gain_per_c * abs(
        environment.temperature_c - 25.0
    )
    noise_std = interference.stochastic_std_v_per_m * humidity_factor * temperature_factor
    rng = np.random.default_rng(interference.seed)
    noise_x = rng.normal(0.0, noise_std, size=points.shape[0])
    noise_y = rng.normal(0.0, noise_std, size=points.shape[0])
    return (
        ex + interference.dc_field_x_v_per_m + emi + noise_x,
        ey + interference.dc_field_y_v_per_m + 0.35 * emi + noise_y,
    )


def _detector_summary(
    magnitude: np.ndarray,
    normal_component: np.ndarray,
    tangential_component: np.ndarray,
    direction: np.ndarray,
    maxima: np.ndarray,
    minima: np.ndarray,
) -> DetectorSummary:
    center_index = magnitude.size // 2
    edge_count = max(1, magnitude.size // 10)
    edge_mean = float(np.mean(np.concatenate([magnitude[:edge_count], magnitude[-edge_count:]])))
    center_value = max(float(magnitude[center_index]), 1e-30)
    return DetectorSummary(
        max_magnitude_v_per_m=float(np.max(magnitude)),
        min_magnitude_v_per_m=float(np.min(magnitude)),
        max_normal_v_per_m=float(np.max(normal_component)),
        min_normal_v_per_m=float(np.min(normal_component)),
        peak_to_peak_magnitude_v_per_m=float(np.max(magnitude) - np.min(magnitude)),
        edge_to_center_ratio=edge_mean / center_value,
        mean_direction_deg=float(np.mean(direction)),
        direction_std_deg=float(np.std(direction)),
        fringing_tangential_ratio=float(
            np.mean(np.abs(tangential_component)) / max(np.mean(np.abs(normal_component)), 1e-30)
        ),
        local_maxima_count=int(np.count_nonzero(maxima)),
        local_minima_count=int(np.count_nonzero(minima)),
    )


def _local_maxima(values: np.ndarray) -> np.ndarray:
    flags = np.zeros(values.shape, dtype=bool)
    flags[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    return flags


def _local_minima(values: np.ndarray) -> np.ndarray:
    flags = np.zeros(values.shape, dtype=bool)
    flags[1:-1] = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    return flags


def _bilinear_sample(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
) -> np.ndarray:
    x_points = np.clip(x_points, x_grid[0], x_grid[-1])
    y_points = np.clip(y_points, y_grid[0], y_grid[-1])
    x_index = np.searchsorted(x_grid, x_points, side="right") - 1
    y_index = np.searchsorted(y_grid, y_points, side="right") - 1
    x_index = np.clip(x_index, 0, len(x_grid) - 2)
    y_index = np.clip(y_index, 0, len(y_grid) - 2)

    x0 = x_grid[x_index]
    x1 = x_grid[x_index + 1]
    y0 = y_grid[y_index]
    y1 = y_grid[y_index + 1]
    tx = (x_points - x0) / np.maximum(x1 - x0, 1e-30)
    ty = (y_points - y0) / np.maximum(y1 - y0, 1e-30)

    v00 = values[y_index, x_index]
    v10 = values[y_index, x_index + 1]
    v01 = values[y_index + 1, x_index]
    v11 = values[y_index + 1, x_index + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
