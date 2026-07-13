import math
import unittest

from src.capacitance import RectangularCapacitor
from src.physical_system import (
    DielectricMaterial,
    Environment,
    GeneratorDrive,
    ThermalModel,
    evaluate_physical_state,
)


class PhysicalSystemTests(unittest.TestCase):
    def test_temperature_and_humidity_change_effective_permittivity(self) -> None:
        geometry = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)
        dielectric = DielectricMaterial(
            reference_relative_permittivity=2.2,
            temperature_coefficient_per_c=-1e-3,
            humidity_coefficient=0.2,
        )

        dry_cool = evaluate_physical_state(
            geometry,
            environment=Environment(temperature_c=20.0, relative_humidity=0.2),
            dielectric=dielectric,
        )
        humid_hot = evaluate_physical_state(
            geometry,
            environment=Environment(temperature_c=45.0, relative_humidity=0.9),
            dielectric=dielectric,
        )

        self.assertNotEqual(dry_cool.relative_permittivity, humid_hot.relative_permittivity)
        self.assertNotEqual(dry_cool.effective_capacitance_f, humid_hot.effective_capacitance_f)

    def test_humidity_increases_leakage_for_default_material(self) -> None:
        geometry = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)
        drive = GeneratorDrive(voltage=50.0)

        dry = evaluate_physical_state(
            geometry,
            environment=Environment(relative_humidity=0.2),
            drive=drive,
        )
        humid = evaluate_physical_state(
            geometry,
            environment=Environment(relative_humidity=0.9),
            drive=drive,
        )

        self.assertGreater(humid.leakage_current_a, dry.leakage_current_a)
        self.assertLess(humid.leakage_resistance_ohm, dry.leakage_resistance_ohm)

    def test_generator_friction_and_losses_feed_heat_balance(self) -> None:
        geometry = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)
        drive = GeneratorDrive(
            voltage=100.0,
            frequency_hz=1000.0,
            mechanical_power_w=2.0,
            mechanical_efficiency=0.75,
            friction_coefficient=0.2,
            normal_force_n=4.0,
            friction_radius_m=0.01,
            shaft_speed_rad_per_s=200.0,
        )
        thermal = ThermalModel(
            heat_capacity_j_per_k=10.0,
            thermal_conductance_w_per_k=0.1,
            ambient_temperature_c=25.0,
        )

        state = evaluate_physical_state(
            geometry,
            environment=Environment(temperature_c=25.0),
            drive=drive,
            thermal=thermal,
        )

        self.assertGreater(state.generator_heat_w, 0.0)
        self.assertGreater(state.total_heat_generation_w, state.leakage_power_w)
        self.assertGreater(state.temperature_rate_k_per_s, 0.0)

    def test_hot_device_dissipates_heat_to_environment(self) -> None:
        geometry = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)
        thermal = ThermalModel(
            heat_capacity_j_per_k=10.0,
            thermal_conductance_w_per_k=0.5,
            ambient_temperature_c=25.0,
        )

        state = evaluate_physical_state(
            geometry,
            environment=Environment(temperature_c=25.0),
            drive=GeneratorDrive(voltage=1.0),
            thermal=thermal,
            device_temperature_c=60.0,
        )

        self.assertGreater(state.heat_dissipation_w, 0.0)
        self.assertLess(state.temperature_rate_k_per_s, 0.0)

    def test_breakdown_margin_and_energy_are_reported(self) -> None:
        geometry = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)
        state = evaluate_physical_state(
            geometry,
            drive=GeneratorDrive(voltage=100.0, frequency_hz=60.0),
            fringe_multiplier=1.2,
        )

        self.assertGreater(state.stored_energy_j, 0.0)
        self.assertGreater(state.charge_c, 0.0)
        self.assertGreater(state.breakdown_margin, 1.0)
        self.assertTrue(math.isfinite(state.quality_factor))

    def test_rejects_invalid_physical_parameters(self) -> None:
        with self.assertRaises(ValueError):
            Environment(relative_humidity=1.2)
        with self.assertRaises(ValueError):
            GeneratorDrive(mechanical_efficiency=0.0)
        with self.assertRaises(ValueError):
            ThermalModel(heat_capacity_j_per_k=0.0)


if __name__ == "__main__":
    unittest.main()
