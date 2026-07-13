import tempfile
import unittest
from pathlib import Path

from src.laplace2d import SolverConfig, solve_parallel_plate_2d
from src.reporting import write_validation_report
from src.studies import domain_size_study, grid_convergence_study, numerical_gap_sweep


class ReportingTests(unittest.TestCase):
    def test_write_validation_report_includes_core_sections(self) -> None:
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
        result = solve_parallel_plate_2d(config)
        convergence_rows = grid_convergence_study(config, [41, 61])
        domain_rows = domain_size_study(config, [0.06, 0.08])
        gap_rows = numerical_gap_sweep(config, [0.002, 0.004])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_validation_report(
                str(path),
                result,
                convergence_rows,
                domain_rows,
                gap_rows,
            )
            content = path.read_text(encoding="utf-8")

        self.assertIn("# Capacitor Fringing Validation Report", content)
        self.assertIn("## Main Field Solve", content)
        self.assertIn("## Grid Convergence", content)
        self.assertIn("## Domain Size Sensitivity", content)
        self.assertIn("## Numerical Gap Sweep", content)
        self.assertIn("Method relative difference", content)


if __name__ == "__main__":
    unittest.main()
