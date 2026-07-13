import tempfile
import unittest
from pathlib import Path

from src.benchmark_dashboard import export_optimizer_benchmark_html
from src.pinn import PINNConfig, run_optimizer_comparison


class BenchmarkDashboardTests(unittest.TestCase):
    def test_export_optimizer_benchmark_html_writes_animation(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        _model, _dataset, results = run_optimizer_comparison(config, steps=12)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dashboard.html"
            export_optimizer_benchmark_html(results, str(path))
            content = path.read_text(encoding="utf-8")

        self.assertIn("<canvas", content)
        self.assertIn("requestAnimationFrame", content)
        self.assertIn("hamiltonian_geometric", content)
        self.assertIn("entropy_descent", content)

    def test_wolfram_script_is_present(self) -> None:
        script = Path("wolfram") / "hamiltonian_geometric_benchmark.wl"

        content = script.read_text(encoding="utf-8")

        self.assertIn("hamiltonian", content.lower())
        self.assertIn("ListLogPlot", content)


if __name__ == "__main__":
    unittest.main()
