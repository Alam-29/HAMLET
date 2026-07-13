import unittest

import numpy as np

from src.laplace2d import SolverConfig, solve_parallel_plate_2d


class Laplace2DTests(unittest.TestCase):
    def test_solver_converges_and_keeps_electrodes_fixed(self) -> None:
        config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
            nx=61,
            ny=61,
            tolerance=5e-5,
            max_iterations=10_000,
        )

        result = solve_parallel_plate_2d(config)

        self.assertTrue(result.converged)
        self.assertGreater(result.capacitance_per_depth, 0.0)
        self.assertGreater(result.charge_capacitance_per_depth, 0.0)
        self.assertGreater(result.positive_charge_per_depth, 0.0)
        self.assertGreater(result.charge_fringe_ratio, 0.0)
        self.assertGreaterEqual(result.capacitance_estimate_relative_difference, 0.0)
        self.assertLessEqual(result.max_delta, config.tolerance)
        self.assertTrue(np.allclose(np.max(result.potential), config.voltage / 2.0))
        self.assertTrue(np.allclose(np.min(result.potential), -config.voltage / 2.0))
        self.assertGreater(result.electrode_gap, 0.0)
        self.assertGreater(result.electrode_width, 0.0)

    def test_fringe_ratio_exceeds_ideal_cross_section_estimate(self) -> None:
        config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.10,
            domain_height=0.10,
            nx=101,
            ny=101,
            tolerance=8e-5,
            max_iterations=12_000,
        )

        result = solve_parallel_plate_2d(config)

        self.assertGreater(result.fringe_ratio, 1.0)
        self.assertGreater(result.charge_fringe_ratio, 1.0)

    def test_solution_is_antisymmetric_about_midline(self) -> None:
        config = SolverConfig(
            plate_width=0.02,
            gap=0.004,
            domain_width=0.08,
            domain_height=0.08,
            nx=61,
            ny=61,
            tolerance=8e-5,
            max_iterations=10_000,
        )

        result = solve_parallel_plate_2d(config)
        flipped = np.flipud(result.potential)

        self.assertLess(float(np.max(np.abs(result.potential + flipped))), 0.02)

    def test_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            SolverConfig(
                plate_width=0.02,
                gap=0.004,
                domain_width=0.02,
                domain_height=0.08,
            )


if __name__ == "__main__":
    unittest.main()
