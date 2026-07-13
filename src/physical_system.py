"""System-level physical parameters for a real capacitor setup.

This module keeps non-ideal observables explicit: temperature, humidity,
dielectric drift, leakage, dielectric loss, generator loss, heat dissipation,
and breakdown margin. The formulas are engineering models, not universal
material laws; coefficients must be chosen for the actual dielectric and build.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.capacitance import RectangularCapacitor, ideal_parallel_plate


@dataclass(frozen=True)
class Environment:
    """Ambient operating conditions around the capacitor."""

    temperature_c: float = 25.0
    relative_humidity: float = 0.45
    pressure_pa: float = 101_325.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative_humidity must be between 0.0 and 1.0")
        if self.pressure_pa <= 0.0:
            raise ValueError("pressure_pa must be positive")
        if self.temperature_c <= -273.15:
            raise ValueError("temperature_c must be above absolute zero")

    @property
    def temperature_k(self) -> float:
        return self.temperature_c + 273.15


@dataclass(frozen=True)
class DielectricMaterial:
    """Linearized dielectric material response around a reference state."""

    reference_relative_permittivity: float = 1.0006
    reference_temperature_c: float = 25.0
    reference_relative_humidity: float = 0.45
    temperature_coefficient_per_c: float = -4.0e-4
    humidity_coefficient: float = 0.03
    volume_resistivity_ohm_m: float = 1.0e14
    humidity_leakage_coefficient: float = 8.0
    loss_tangent: float = 2.0e-4
    dielectric_strength_v_per_m: float = 3.0e6
    strength_temperature_coefficient_per_c: float = -2.0e-3
    strength_humidity_coefficient: float = -0.25

    def __post_init__(self) -> None:
        positive_values = {
            "reference_relative_permittivity": self.reference_relative_permittivity,
            "volume_resistivity_ohm_m": self.volume_resistivity_ohm_m,
            "dielectric_strength_v_per_m": self.dielectric_strength_v_per_m,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        if self.loss_tangent < 0.0:
            raise ValueError("loss_tangent must be non-negative")
        if not 0.0 <= self.reference_relative_humidity <= 1.0:
            raise ValueError("reference_relative_humidity must be between 0.0 and 1.0")

    def relative_permittivity(self, environment: Environment) -> float:
        """Return effective relative permittivity under current environment."""

        temperature_factor = 1.0 + self.temperature_coefficient_per_c * (
            environment.temperature_c - self.reference_temperature_c
        )
        humidity_factor = 1.0 + self.humidity_coefficient * (
            environment.relative_humidity - self.reference_relative_humidity
        )
        return max(1e-9, self.reference_relative_permittivity * temperature_factor * humidity_factor)

    def resistivity(self, environment: Environment) -> float:
        """Return humidity-adjusted volume resistivity."""

        humidity_delta = environment.relative_humidity - self.reference_relative_humidity
        return self.volume_resistivity_ohm_m * math.exp(
            -self.humidity_leakage_coefficient * humidity_delta
        )

    def dielectric_strength(self, environment: Environment) -> float:
        """Return reduced dielectric strength under current environment."""

        temperature_factor = 1.0 + self.strength_temperature_coefficient_per_c * (
            environment.temperature_c - self.reference_temperature_c
        )
        humidity_factor = 1.0 + self.strength_humidity_coefficient * (
            environment.relative_humidity - self.reference_relative_humidity
        )
        return max(
            1.0,
            self.dielectric_strength_v_per_m * temperature_factor * humidity_factor,
        )


@dataclass(frozen=True)
class GeneratorDrive:
    """Electrical/mechanical drive parameters feeding the capacitor."""

    voltage: float = 1.0
    frequency_hz: float = 0.0
    source_resistance_ohm: float = 1.0
    mechanical_power_w: float = 0.0
    mechanical_efficiency: float = 1.0
    friction_coefficient: float = 0.0
    shaft_speed_rad_per_s: float = 0.0
    normal_force_n: float = 0.0
    friction_radius_m: float = 0.0

    def __post_init__(self) -> None:
        if self.voltage < 0.0:
            raise ValueError("voltage must be non-negative")
        if self.frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative")
        if self.source_resistance_ohm < 0.0:
            raise ValueError("source_resistance_ohm must be non-negative")
        if self.mechanical_power_w < 0.0:
            raise ValueError("mechanical_power_w must be non-negative")
        if not 0.0 < self.mechanical_efficiency <= 1.0:
            raise ValueError("mechanical_efficiency must be in the range (0.0, 1.0]")
        for name, value in {
            "friction_coefficient": self.friction_coefficient,
            "shaft_speed_rad_per_s": self.shaft_speed_rad_per_s,
            "normal_force_n": self.normal_force_n,
            "friction_radius_m": self.friction_radius_m,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def friction_power_w(self) -> float:
        torque = self.friction_coefficient * self.normal_force_n * self.friction_radius_m
        return torque * self.shaft_speed_rad_per_s

    @property
    def generator_heat_w(self) -> float:
        conversion_loss = self.mechanical_power_w * (1.0 - self.mechanical_efficiency)
        return conversion_loss + self.friction_power_w


@dataclass(frozen=True)
class ThermalModel:
    """Lumped heat capacity and heat rejection for the built device."""

    heat_capacity_j_per_k: float = 12.0
    thermal_conductance_w_per_k: float = 0.18
    surface_area_m2: float = 0.01
    emissivity: float = 0.85
    ambient_temperature_c: float = 25.0

    def __post_init__(self) -> None:
        if self.heat_capacity_j_per_k <= 0.0:
            raise ValueError("heat_capacity_j_per_k must be positive")
        if self.thermal_conductance_w_per_k < 0.0:
            raise ValueError("thermal_conductance_w_per_k must be non-negative")
        if self.surface_area_m2 < 0.0:
            raise ValueError("surface_area_m2 must be non-negative")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must be between 0.0 and 1.0")
        if self.ambient_temperature_c <= -273.15:
            raise ValueError("ambient_temperature_c must be above absolute zero")


@dataclass(frozen=True)
class CapacitorPhysicalState:
    """Computed internal observables for the non-ideal capacitor system."""

    relative_permittivity: float
    capacitance_f: float
    effective_capacitance_f: float
    electric_field_v_per_m: float
    stored_energy_j: float
    charge_c: float
    leakage_resistance_ohm: float
    leakage_current_a: float
    leakage_power_w: float
    dielectric_loss_power_w: float
    source_resistor_power_w: float
    generator_heat_w: float
    total_heat_generation_w: float
    heat_dissipation_w: float
    net_heat_w: float
    temperature_rate_k_per_s: float
    dielectric_strength_v_per_m: float
    breakdown_margin: float
    rc_time_constant_s: float
    quality_factor: float


def evaluate_physical_state(
    geometry: RectangularCapacitor,
    environment: Environment | None = None,
    dielectric: DielectricMaterial | None = None,
    drive: GeneratorDrive | None = None,
    thermal: ThermalModel | None = None,
    fringe_multiplier: float = 1.0,
    device_temperature_c: float | None = None,
) -> CapacitorPhysicalState:
    """Compute non-ideal observables for a capacitor under operating conditions."""

    if fringe_multiplier <= 0.0:
        raise ValueError("fringe_multiplier must be positive")

    environment = environment or Environment()
    dielectric = dielectric or DielectricMaterial()
    drive = drive or GeneratorDrive()
    thermal = thermal or ThermalModel(ambient_temperature_c=environment.temperature_c)
    device_temperature_c = (
        environment.temperature_c if device_temperature_c is None else device_temperature_c
    )

    effective_environment = Environment(
        temperature_c=device_temperature_c,
        relative_humidity=environment.relative_humidity,
        pressure_pa=environment.pressure_pa,
    )
    relative_permittivity = dielectric.relative_permittivity(effective_environment)
    adjusted_geometry = RectangularCapacitor(
        length=geometry.length,
        width=geometry.width,
        gap=geometry.gap,
        relative_permittivity=relative_permittivity,
    )
    capacitance = ideal_parallel_plate(adjusted_geometry)
    effective_capacitance = capacitance * fringe_multiplier

    electric_field = 0.0 if geometry.gap == 0.0 else drive.voltage / geometry.gap
    stored_energy = 0.5 * effective_capacitance * drive.voltage**2
    charge = effective_capacitance * drive.voltage

    resistivity = dielectric.resistivity(effective_environment)
    leakage_resistance = resistivity * geometry.gap / geometry.area
    leakage_current = 0.0 if leakage_resistance == 0.0 else drive.voltage / leakage_resistance
    leakage_power = drive.voltage * leakage_current

    angular_frequency = 2.0 * math.pi * drive.frequency_hz
    dielectric_loss_power = (
        drive.voltage**2 * angular_frequency * effective_capacitance * dielectric.loss_tangent
    )
    source_resistor_power = leakage_current**2 * drive.source_resistance_ohm
    generator_heat = drive.generator_heat_w
    total_heat = leakage_power + dielectric_loss_power + source_resistor_power + generator_heat

    heat_dissipation = _heat_dissipation_w(device_temperature_c, thermal)
    net_heat = total_heat - heat_dissipation
    temperature_rate = net_heat / thermal.heat_capacity_j_per_k

    dielectric_strength = dielectric.dielectric_strength(effective_environment)
    breakdown_margin = (
        math.inf if electric_field == 0.0 else dielectric_strength / electric_field
    )
    rc_time_constant = leakage_resistance * effective_capacitance
    quality_factor = (
        math.inf
        if drive.frequency_hz == 0.0 or dielectric.loss_tangent == 0.0
        else 1.0 / dielectric.loss_tangent
    )

    return CapacitorPhysicalState(
        relative_permittivity=relative_permittivity,
        capacitance_f=capacitance,
        effective_capacitance_f=effective_capacitance,
        electric_field_v_per_m=electric_field,
        stored_energy_j=stored_energy,
        charge_c=charge,
        leakage_resistance_ohm=leakage_resistance,
        leakage_current_a=leakage_current,
        leakage_power_w=leakage_power,
        dielectric_loss_power_w=dielectric_loss_power,
        source_resistor_power_w=source_resistor_power,
        generator_heat_w=generator_heat,
        total_heat_generation_w=total_heat,
        heat_dissipation_w=heat_dissipation,
        net_heat_w=net_heat,
        temperature_rate_k_per_s=temperature_rate,
        dielectric_strength_v_per_m=dielectric_strength,
        breakdown_margin=breakdown_margin,
        rc_time_constant_s=rc_time_constant,
        quality_factor=quality_factor,
    )


def _heat_dissipation_w(device_temperature_c: float, thermal: ThermalModel) -> float:
    conductive = thermal.thermal_conductance_w_per_k * (
        device_temperature_c - thermal.ambient_temperature_c
    )
    sigma = 5.670_374_419e-8
    device_k = device_temperature_c + 273.15
    ambient_k = thermal.ambient_temperature_c + 273.15
    radiative = (
        thermal.emissivity
        * sigma
        * thermal.surface_area_m2
        * (device_k**4 - ambient_k**4)
    )
    return conductive + radiative
