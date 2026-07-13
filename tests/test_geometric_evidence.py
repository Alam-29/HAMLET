import unittest

import numpy as np

from src.geometric_evidence import (
    condition_number_experiment,
    phase_space_experiment,
    rotated_quadratic,
    rotation_invariance_experiment,
    saddle_escape_experiment,
)


class GeometricEvidenceTests(unittest.TestCase):
    def test_rotated_quadratic_gradient_matches_finite_difference(self) -> None:
        loss, grad, _metric, _a = rotated_quadratic(4, 100.0, seed=2)
        theta = np.array([0.2, -0.4, 0.1, 0.3])
        analytical = grad(theta)
        numerical = np.zeros_like(theta)
        step = 1e-6
        for index in range(theta.size):
            delta = np.zeros_like(theta)
            delta[index] = step
            numerical[index] = (loss(theta + delta) - loss(theta - delta)) / (2.0 * step)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-6)

    def test_condition_number_experiment_returns_all_optimizers(self) -> None:
        rows = condition_number_experiment(condition_numbers=(1e2,), dim=4, steps=5)

        self.assertEqual({row["optimizer"] for row in rows}, {"sgd", "heavy_ball", "adam", "entropy_descent", "hamiltonian_geometric"})
        self.assertTrue(all(np.isfinite(row["final_loss"]) for row in rows))

    def test_rotation_and_saddle_experiments_run(self) -> None:
        rotation_rows = rotation_invariance_experiment(dim=4, condition=100.0, steps=5)
        saddle_rows = saddle_escape_experiment(seeds=3, steps=5)

        self.assertTrue(rotation_rows)
        self.assertTrue(saddle_rows)
        self.assertTrue(all(np.isfinite(row["final_loss"]) for row in rotation_rows))
        self.assertTrue(all(np.isfinite(row["final_loss"]) for row in saddle_rows))

    def test_phase_space_experiment_returns_traces(self) -> None:
        traces = phase_space_experiment(steps=5)

        self.assertEqual(len(traces), 5)
        for trace in traces:
            self.assertEqual(trace.loss.shape, (5,))
            self.assertTrue(np.all(np.isfinite(trace.loss)))


if __name__ == "__main__":
    unittest.main()
