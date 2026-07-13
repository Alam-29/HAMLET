import tempfile
import unittest
from pathlib import Path

from src.laplace2d import SolverConfig, solve_parallel_plate_2d
from src.measurement import (
    DetectorPlate,
    ExternalInterference,
    export_detector_observations_csv,
    observe_detector_plate,
)
from src.physical_system import Environment


class MeasurementTests(unittest.TestCase):
    def test_detector_samples_field_and_reports_summary(self) -> None:
        result = _small_solution()
        detector = DetectorPlate(center_y=0.012, length=0.04, samples=41)

        observations, summary = observe_detector_plate(result, detector)

        self.assertEqual(len(observations), detector.samples)
        self.assertGreater(summary.max_magnitude_v_per_m, summary.min_magnitude_v_per_m)
        self.assertGreaterEqual(summary.local_maxima_count, 0)
        self.assertGreaterEqual(summary.local_minima_count, 0)
        self.assertGreaterEqual(summary.fringing_tangential_ratio, 0.0)

    def test_external_interference_changes_detector_observations(self) -> None:
        result = _small_solution()
        detector = DetectorPlate(center_y=0.012, length=0.04, samples=41)

        clean, clean_summary = observe_detector_plate(result, detector)
        noisy, noisy_summary = observe_detector_plate(
            result,
            detector,
            interference=ExternalInterference(
                dc_field_x_v_per_m=100.0,
                emi_amplitude_v_per_m=75.0,
                stochastic_std_v_per_m=25.0,
                seed=4,
            ),
            environment=Environment(temperature_c=40.0, relative_humidity=0.8),
        )

        self.assertNotEqual(clean[0].ex_v_per_m, noisy[0].ex_v_per_m)
        self.assertNotEqual(clean_summary.direction_std_deg, noisy_summary.direction_std_deg)

    def test_detector_export_writes_csv_files(self) -> None:
        result = _small_solution()
        detector = DetectorPlate(center_y=0.012, length=0.04, samples=21)
        observations, summary = observe_detector_plate(result, detector)

        with tempfile.TemporaryDirectory() as directory:
            observation_path = Path(directory) / "observations.csv"
            summary_path = Path(directory) / "summary.csv"
            export_detector_observations_csv(
                observations,
                summary,
                str(observation_path),
                str(summary_path),
            )

            self.assertIn("direction_deg", observation_path.read_text(encoding="utf-8"))
            self.assertIn("edge_to_center_ratio", summary_path.read_text(encoding="utf-8"))

    def test_detector_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            DetectorPlate(samples=2)
        with self.assertRaises(ValueError):
            DetectorPlate(normal_x=0.0, normal_y=0.0)


def _small_solution():
    config = SolverConfig(
        plate_width=0.02,
        gap=0.004,
        domain_width=0.06,
        domain_height=0.06,
        nx=41,
        ny=41,
        tolerance=1e-4,
        max_iterations=10_000,
    )
    return solve_parallel_plate_2d(config)


if __name__ == "__main__":
    unittest.main()
