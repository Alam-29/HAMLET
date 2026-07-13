import tempfile
import unittest
from pathlib import Path

from src.laplace2d import SolverConfig, solve_parallel_plate_2d
from src.visualization import export_fringing_field_png, export_potential_png


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class VisualizationTests(unittest.TestCase):
    def test_export_potential_png_writes_a_png_file(self) -> None:
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

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.png"
            export_potential_png(result, str(path))
            content = path.read_bytes()

        self.assertTrue(content.startswith(PNG_MAGIC))
        self.assertGreater(len(content), 0)

    def test_export_fringing_field_png_writes_a_png_file(self) -> None:
        config = SolverConfig(
            plate_width=0.02,
            gap=0.006,
            domain_width=0.06,
            domain_height=0.06,
            nx=41,
            ny=41,
            tolerance=1e-4,
            max_iterations=10_000,
        )
        result = solve_parallel_plate_2d(config)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fringing.png"
            export_fringing_field_png(result, str(path))
            content = path.read_bytes()

        self.assertTrue(content.startswith(PNG_MAGIC))
        self.assertGreater(len(content), 0)


if __name__ == "__main__":
    unittest.main()
