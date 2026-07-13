"""Analytical capacitor models.

These functions are intentionally small and explicit so the assumptions can be
checked while the research model evolves.
"""

from __future__ import annotations

from dataclasses import dataclass


EPSILON_0 = 8.854_187_8128e-12


@dataclass(frozen=True)
class RectangularCapacitor:
    """Geometry and material parameters for a rectangular parallel-plate capacitor.

    All dimensions are in meters.
    """

    length: float
    width: float
    gap: float
    relative_permittivity: float = 1.0

    def __post_init__(self) -> None:
        values = {
            "length": self.length,
            "width": self.width,
            "gap": self.gap,
            "relative_permittivity": self.relative_permittivity,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")

    @property
    def area(self) -> float:
        return self.length * self.width

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.length + self.width)

    @property
    def permittivity(self) -> float:
        return EPSILON_0 * self.relative_permittivity


def ideal_parallel_plate(capacitor: RectangularCapacitor) -> float:
    """Return ideal capacitance in farads.

    This assumes perfectly uniform field lines and ignores all edge effects.
    """

    return capacitor.permittivity * capacitor.area / capacitor.gap


def effective_area_fringe(
    capacitor: RectangularCapacitor,
    edge_extension_fraction: float = 0.5,
) -> float:
    """Approximate capacitance by expanding each plate's effective dimensions.

    The model uses

        L_eff = L + alpha * d
        W_eff = W + alpha * d
        C = epsilon * L_eff * W_eff / d

    where `alpha` is `edge_extension_fraction`.

    This is not a universal fringing law. It is a tunable baseline that should
    later be compared against a numerical field solution or measurements.
    """

    if edge_extension_fraction < 0:
        raise ValueError("edge_extension_fraction must be non-negative")

    effective_length = capacitor.length + edge_extension_fraction * capacitor.gap
    effective_width = capacitor.width + edge_extension_fraction * capacitor.gap
    return capacitor.permittivity * effective_length * effective_width / capacitor.gap


def fringe_ratio(
    capacitor: RectangularCapacitor,
    edge_extension_fraction: float = 0.5,
) -> float:
    """Return approximate fringing capacitance divided by ideal capacitance."""

    return effective_area_fringe(capacitor, edge_extension_fraction) / ideal_parallel_plate(
        capacitor
    )


def sweep_gap(
    length: float,
    width: float,
    gaps: list[float],
    relative_permittivity: float = 1.0,
    edge_extension_fraction: float = 0.5,
) -> list[dict[str, float]]:
    """Compare ideal and approximate fringing capacitance for multiple gaps."""

    rows: list[dict[str, float]] = []
    for gap in gaps:
        capacitor = RectangularCapacitor(
            length=length,
            width=width,
            gap=gap,
            relative_permittivity=relative_permittivity,
        )
        ideal = ideal_parallel_plate(capacitor)
        fringed = effective_area_fringe(capacitor, edge_extension_fraction)
        rows.append(
            {
                "gap_m": gap,
                "ideal_F": ideal,
                "fringed_F": fringed,
                "ratio": fringed / ideal,
            }
        )
    return rows
