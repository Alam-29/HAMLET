import csv
import tempfile
import unittest
from pathlib import Path

from src.laplace2d import SolverConfig
from src.studies import (
    adaptive_numerical_gap_sweep,
    domain_size_study,
    export_convergence_csv,
    export_domain_size_csv,
    export_gap_sweep_csv,
    grid_convergence_study,
    numerical_gap_sweep,
)


class ConvergenceStudyTests(unittest.TestCase):
    def test_grid_convergence_study_reports_relative_changes(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
            tolerance=1e-4,
            max_iterations=10_000,
        )

        rows = grid_convergence_study(base_config, [41, 81])

        self.assertEqual([row.nx for row in rows], [41, 81])
        self.assertIsNone(rows[0].relative_change_from_previous)
        self.assertIsNotNone(rows[1].relative_change_from_previous)
        self.assertAlmostEqual(rows[0].gap_error, 0.0)
        self.assertAlmostEqual(rows[1].gap_error, 0.0)
        self.assertAlmostEqual(rows[0].plate_width_error, 0.0)
        self.assertAlmostEqual(rows[1].plate_width_error, 0.0)
        self.assertGreater(rows[1].capacitance_per_depth, 0.0)
        self.assertGreater(rows[1].charge_capacitance_per_depth, 0.0)
        self.assertGreaterEqual(rows[1].capacitance_estimate_relative_difference, 0.0)
        self.assertTrue(rows[1].converged)

    def test_grid_convergence_rejects_unsorted_or_too_short_input(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
        )

        with self.assertRaises(ValueError):
            grid_convergence_study(base_config, [61])
        with self.assertRaises(ValueError):
            grid_convergence_study(base_config, [61, 41])

    def test_export_convergence_csv_writes_expected_columns(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
            tolerance=1e-4,
            max_iterations=10_000,
        )
        rows = grid_convergence_study(base_config, [41, 61])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "convergence.csv"
            export_convergence_csv(rows, str(path))

            with open(path, newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))

        self.assertEqual(len(csv_rows), 2)
        self.assertIn("gap_error_m", csv_rows[0])
        self.assertIn("plate_width_error_m", csv_rows[0])
        self.assertIn("charge_capacitance_per_depth_F_per_m", csv_rows[0])
        self.assertIn("capacitance_estimate_relative_difference", csv_rows[0])
        self.assertIn("fringe_ratio", csv_rows[0])
        self.assertIn("charge_fringe_ratio", csv_rows[0])
        self.assertEqual(csv_rows[0]["relative_change_from_previous"], "")

    def test_domain_size_study_keeps_spacing_and_reports_changes(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.06,
            domain_height=0.06,
            nx=61,
            ny=61,
            tolerance=1e-4,
            max_iterations=10_000,
        )

        rows = domain_size_study(base_config, [0.06, 0.08])

        self.assertEqual([row.domain_width for row in rows], [0.06, 0.08])
        self.assertEqual([row.nx for row in rows], [61, 81])
        self.assertAlmostEqual(rows[0].dx, rows[1].dx)
        self.assertAlmostEqual(rows[0].plate_width_error, 0.0)
        self.assertAlmostEqual(rows[1].plate_width_error, 0.0)
        self.assertIsNone(rows[0].relative_change_from_previous)
        self.assertIsNotNone(rows[1].relative_change_from_previous)
        self.assertGreater(rows[1].charge_capacitance_per_depth, 0.0)
        self.assertGreaterEqual(rows[1].capacitance_estimate_relative_difference, 0.0)
        self.assertTrue(rows[1].converged)

    def test_domain_size_study_rejects_invalid_sequences(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.06,
            domain_height=0.06,
            nx=61,
            ny=61,
        )

        with self.assertRaises(ValueError):
            domain_size_study(base_config, [0.06])
        with self.assertRaises(ValueError):
            domain_size_study(base_config, [0.08, 0.06])
        with self.assertRaises(ValueError):
            domain_size_study(base_config, [0.02, 0.06])

    def test_export_domain_size_csv_writes_expected_columns(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.06,
            domain_height=0.06,
            nx=61,
            ny=61,
            tolerance=1e-4,
            max_iterations=10_000,
        )
        rows = domain_size_study(base_config, [0.06, 0.08])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "domain_size.csv"
            export_domain_size_csv(rows, str(path))

            with open(path, newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))

        self.assertEqual(len(csv_rows), 2)
        self.assertIn("domain_width_m", csv_rows[0])
        self.assertIn("plate_width_error_m", csv_rows[0])
        self.assertIn("charge_capacitance_per_depth_F_per_m", csv_rows[0])
        self.assertIn("capacitance_estimate_relative_difference", csv_rows[0])
        self.assertIn("fringe_ratio", csv_rows[0])
        self.assertIn("charge_fringe_ratio", csv_rows[0])
        self.assertEqual(csv_rows[0]["relative_change_from_previous"], "")

    def test_numerical_gap_sweep_compares_against_effective_area_model(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
            nx=81,
            ny=81,
            tolerance=1e-4,
            max_iterations=10_000,
        )

        rows = numerical_gap_sweep(base_config, [0.002, 0.004])

        self.assertEqual([row.requested_gap for row in rows], [0.002, 0.004])
        self.assertAlmostEqual(rows[0].gap_to_width, 0.1)
        self.assertGreater(rows[0].energy_fringe_ratio, 0.0)
        self.assertGreater(rows[0].charge_fringe_ratio, 0.0)
        self.assertGreater(rows[0].effective_area_fringe_ratio, 1.0)
        self.assertTrue(rows[0].converged)

    def test_numerical_gap_sweep_rejects_invalid_sequences(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
        )

        with self.assertRaises(ValueError):
            numerical_gap_sweep(base_config, [0.004])
        with self.assertRaises(ValueError):
            numerical_gap_sweep(base_config, [0.004, 0.002])

    def test_adaptive_gap_sweep_refines_grid_from_smallest_gap(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.04,
            domain_height=0.04,
            nx=21,
            ny=21,
            tolerance=1e-4,
            max_iterations=10_000,
        )

        rows = adaptive_numerical_gap_sweep(
            base_config,
            [0.004, 0.008],
            min_cells_across_smallest_gap=4,
        )

        self.assertEqual([row.requested_gap for row in rows], [0.004, 0.008])
        self.assertGreater(rows[0].nx, base_config.nx)
        self.assertAlmostEqual(rows[0].dx, 0.001)
        self.assertTrue(rows[0].converged)

    def test_export_gap_sweep_csv_writes_expected_columns(self) -> None:
        base_config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
            nx=81,
            ny=81,
            tolerance=1e-4,
            max_iterations=10_000,
        )
        rows = numerical_gap_sweep(base_config, [0.002, 0.004])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gap_sweep.csv"
            export_gap_sweep_csv(rows, str(path))

            with open(path, newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))

        self.assertEqual(len(csv_rows), 2)
        self.assertIn("energy_fringe_ratio", csv_rows[0])
        self.assertIn("gap_to_width", csv_rows[0])
        self.assertIn("charge_fringe_ratio", csv_rows[0])
        self.assertIn("effective_area_fringe_ratio", csv_rows[0])


if __name__ == "__main__":
    unittest.main()
