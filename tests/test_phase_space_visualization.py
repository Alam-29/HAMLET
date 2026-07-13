import csv
import os
import tempfile
import unittest

from src.phase_space_visualization import (
    compute_phase_space_projection,
    export_phase_space_html,
    export_phase_space_png,
    export_phase_space_trajectories_csv,
)
from src.pinn import PINNConfig, run_optimizer_comparison


def _small_results():
    config = PINNConfig(
        hidden_features=8,
        collocation_points=20,
        plate_points=6,
        outer_boundary_points=12,
    )
    _model, _dataset, results = run_optimizer_comparison(config, steps=15, record_theta=True)
    return results


class PhaseSpaceVisualizationTests(unittest.TestCase):
    def test_compute_phase_space_projection_shapes_and_variance(self) -> None:
        results = _small_results()

        projections, explained_variance_ratio = compute_phase_space_projection(results)

        self.assertEqual(explained_variance_ratio.shape, (3,))
        self.assertTrue((explained_variance_ratio >= -1e-9).all())
        self.assertLessEqual(float(explained_variance_ratio.sum()), 1.0 + 1e-9)
        for result in results:
            points = projections[result.optimizer]
            self.assertEqual(points.shape, (len(result.theta_history), 3))

    def test_compute_phase_space_projection_requires_recorded_theta(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        _model, _dataset, results = run_optimizer_comparison(config, steps=15)

        with self.assertRaises(ValueError):
            compute_phase_space_projection(results)

    def test_export_phase_space_trajectories_csv_writes_one_row_per_step(self) -> None:
        results = _small_results()
        projections, _explained_variance_ratio = compute_phase_space_projection(results)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "phase_space_trajectories.csv")
            export_phase_space_trajectories_csv(results, projections, path)

            with open(path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows[0].keys(), {"step", "optimizer", "loss", "pc1", "pc2", "pc3"})
        expected_rows = sum(len(result.theta_history) for result in results)
        self.assertEqual(len(rows), expected_rows)

    def test_export_phase_space_png_writes_a_file(self) -> None:
        results = _small_results()
        projections, explained_variance_ratio = compute_phase_space_projection(results)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "phase_space.png")
            export_phase_space_png(projections, explained_variance_ratio, path)

            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_export_phase_space_html_embeds_every_optimizer(self) -> None:
        results = _small_results()
        projections, explained_variance_ratio = compute_phase_space_projection(results)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "phase_space.html")
            export_phase_space_html(results, projections, explained_variance_ratio, path)

            with open(path, encoding="utf-8") as file:
                html = file.read()

        for result in results:
            self.assertIn(result.optimizer, html)


if __name__ == "__main__":
    unittest.main()
