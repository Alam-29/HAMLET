import unittest

import numpy as np

from src.hamiltonian_geometric import (
    HamiltonianGeometricConfig,
    HamiltonianGeometricState,
    geometric_force_finite_difference,
    hamiltonian_geometric_step,
    initial_state,
    positive_definite_metric,
    spectral_entropy,
    spectral_entropy_gradient_finite_difference,
)


class HamiltonianGeometricCoreTests(unittest.TestCase):
    def test_geometric_force_is_nonzero_for_theta_dependent_metric(self) -> None:
        theta = np.array([0.35, -0.25])
        momentum = np.array([0.7, -0.4])

        def metric_fn(values: np.ndarray) -> np.ndarray:
            return np.diag([1.0 + values[0] ** 2, 2.0 + 0.5 * values[1] ** 2])

        force = geometric_force_finite_difference(
            theta, momentum, metric_fn, regularization=1e-3
        )

        self.assertTrue(np.all(np.isfinite(force)))
        self.assertGreater(np.linalg.norm(force), 0.0)

    def test_step_updates_memory_momentum_and_parameters(self) -> None:
        config = HamiltonianGeometricConfig(
            learning_rate=0.05,
            beta=0.8,
            memory_coupling=0.1,
            memory_decay=0.5,
            use_geometric_correction=False,
        )
        state = initial_state(2)

        def gradient_fn(_theta: np.ndarray) -> np.ndarray:
            return np.array([1.0, -2.0])

        def metric_fn(_theta: np.ndarray) -> np.ndarray:
            return np.eye(2)

        updated = hamiltonian_geometric_step(state, gradient_fn, metric_fn, config)

        np.testing.assert_allclose(updated.memory, np.array([1.0, -2.0]))
        self.assertGreater(np.linalg.norm(updated.momentum), 0.0)
        self.assertGreater(np.linalg.norm(updated.parameters), 0.0)
        self.assertEqual(updated.step, 1)

    def test_spectral_entropy_gradient_tracks_metric_change(self) -> None:
        theta = np.array([0.2, -0.4])

        def metric_fn(values: np.ndarray) -> np.ndarray:
            return np.diag([1.0 + values[0] ** 2, 3.0 + values[1] ** 2])

        gradient = spectral_entropy_gradient_finite_difference(
            theta, metric_fn, regularization=1e-3
        )

        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(np.linalg.norm(gradient), 0.0)

    def test_positive_definite_metric_regularizes_indefinite_matrix(self) -> None:
        metric = positive_definite_metric(np.diag([-2.0, 1.0]), regularization=1e-3)

        self.assertTrue(np.all(np.linalg.eigvalsh(metric) > 0.0))
        self.assertGreaterEqual(spectral_entropy(metric), 0.0)

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            HamiltonianGeometricConfig(memory_decay=1.0)
        with self.assertRaises(ValueError):
            HamiltonianGeometricConfig(metric_regularization=0.0)


if __name__ == "__main__":
    unittest.main()
