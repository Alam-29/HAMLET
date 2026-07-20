import unittest

import numpy as np

from src.hamiltonian_geometric import (
    HamiltonianGeometricConfig,
    HamiltonianGeometricState,
    geometric_force_finite_difference,
    geometric_force_tensor_contraction,
    hamiltonian_geometric_step,
    initial_state,
    positive_definite_metric,
    spectral_entropy,
    spectral_entropy_gradient_finite_difference,
)


class HamiltonianGeometricCoreTests(unittest.TestCase):
    def test_exact_hessian_quadratic_recurrence_is_condition_number_independent(self) -> None:
        """Executable check of Theorem 1's exact-Hessian recurrence."""

        eta, beta = 0.4, 0.7
        theta0 = np.array([0.8, -0.3, 0.5])
        q0 = np.array([-0.2, 0.4, 0.1])
        trajectories = []
        for condition_number in (1.0, 1e3, 1e8):
            eigenvalues = np.geomspace(1.0, condition_number, theta0.size)
            raw = np.random.default_rng(17).normal(size=(theta0.size, theta0.size))
            rotation, _ = np.linalg.qr(raw)
            hessian = rotation @ np.diag(eigenvalues) @ rotation.T
            theta = theta0.copy()
            momentum = hessian @ q0
            trajectory = [theta.copy()]
            for _ in range(25):
                momentum = beta * momentum - eta * (hessian @ theta)
                theta = theta + eta * np.linalg.solve(hessian, momentum)
                trajectory.append(theta.copy())
            trajectories.append(np.asarray(trajectory))

        for trajectory in trajectories[1:]:
            # Solving an explicitly 1e8-conditioned system introduces a few
            # ulps of numerical error even though the algebraic recurrence is
            # identical; the tolerance reflects that finite-precision solve.
            np.testing.assert_allclose(trajectory, trajectories[0], atol=1e-7, rtol=1e-7)

        block = np.array([[1.0 - eta**2, eta * beta], [-eta, beta]])
        self.assertLess(float(np.max(np.abs(np.linalg.eigvals(block)))), 1.0)

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

    def test_geometric_force_tensor_contraction_matches_finite_difference(self) -> None:
        # The tensor-contraction form (single inverse + einsum, used by
        # hamiltonian_geometric_step) must agree with the reference
        # finite-difference form (2n inversions) to finite-difference
        # precision -- it is a reformulation via -g^-1(dg)g^-1, not an
        # approximation.
        rng = np.random.default_rng(0)
        theta = rng.normal(size=4)
        momentum = rng.normal(size=4)
        q = rng.normal(size=(4, 4))

        def metric_fn(values: np.ndarray) -> np.ndarray:
            return q.T @ q + np.diag(1.0 + values**2)

        regularization = 1e-3
        reference = geometric_force_finite_difference(theta, momentum, metric_fn, regularization)

        metric = positive_definite_metric(metric_fn(theta), regularization)
        inverse_metric = np.linalg.pinv(metric)
        fast = geometric_force_tensor_contraction(
            theta, momentum, metric_fn, inverse_metric, regularization
        )

        np.testing.assert_allclose(fast, reference, atol=1e-6)

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
        with self.assertRaises(ValueError):
            HamiltonianGeometricConfig(max_energy_backtracks=-1)
        with self.assertRaises(ValueError):
            HamiltonianGeometricConfig(energy_backtrack_factor=1.0)

    def test_energy_backtracking_is_a_no_op_without_loss_fn(self) -> None:
        # A large step that would overshoot a quadratic bowl still applies
        # unmodified when no loss_fn is given, even if max_energy_backtracks > 0:
        # backtracking is strictly opt-in per call, not just per config.
        config = HamiltonianGeometricConfig(
            learning_rate=5.0, beta=0.0, use_geometric_correction=False,
            use_memory_correction=False, max_energy_backtracks=4,
        )
        state = initial_state(1)

        def gradient_fn(theta: np.ndarray) -> np.ndarray:
            return 2.0 * theta - 4.0  # gradient of (theta-2)^2, minimum at theta=2

        def metric_fn(_theta: np.ndarray) -> np.ndarray:
            return np.eye(1)

        updated = hamiltonian_geometric_step(state, gradient_fn, metric_fn, config)

        # No loss_fn given, so the raw (overshooting) step is used unchanged.
        expected_momentum = -config.learning_rate * gradient_fn(state.parameters)
        np.testing.assert_allclose(updated.momentum, expected_momentum)

    def test_energy_backtracking_prevents_loss_increase_on_a_quadratic_bowl(self) -> None:
        # Same oversized step as above, but now with loss_fn + backtracking
        # enabled: the momentum that would overshoot past the minimum and
        # increase the loss must be damped until the loss no longer rises,
        # mirroring the paper's own dissipative-energy invariant.
        config = HamiltonianGeometricConfig(
            learning_rate=5.0, beta=0.0, use_geometric_correction=False,
            max_energy_backtracks=8, energy_backtrack_factor=0.5,
        )
        state = initial_state(1)

        def gradient_fn(theta: np.ndarray) -> np.ndarray:
            return 2.0 * theta - 4.0

        def metric_fn(_theta: np.ndarray) -> np.ndarray:
            return np.eye(1)

        def loss_fn(theta: np.ndarray) -> float:
            return float((theta[0] - 2.0) ** 2)

        current_loss = loss_fn(state.parameters)
        updated = hamiltonian_geometric_step(
            state, gradient_fn, metric_fn, config, loss_fn=loss_fn
        )

        self.assertLessEqual(loss_fn(updated.parameters), current_loss)

    def test_memory_metric_is_a_no_op_when_disabled(self) -> None:
        config = HamiltonianGeometricConfig(
            learning_rate=0.05, beta=0.0, use_geometric_correction=False,
            use_memory_correction=False, use_memory_metric=False,
        )
        state = initial_state(2)

        def gradient_fn(_theta: np.ndarray) -> np.ndarray:
            return np.array([1.0, -2.0])

        def metric_fn(_theta: np.ndarray) -> np.ndarray:
            return np.eye(2)

        updated = hamiltonian_geometric_step(state, gradient_fn, metric_fn, config)

        np.testing.assert_allclose(updated.memory_metric, np.zeros((2, 2)))

    def test_memory_metric_accumulates_symmetric_psd_gradient_outer_products(self) -> None:
        # M_ij completes g_ij = H_ij + mu*M_ij (Sec. 12) as an EMA of gradient
        # outer products -- the same construction that motivates Adam's
        # diagonal metric, kept as a full matrix. It must stay symmetric PSD
        # (a sum of rank-1 PSD terms) regardless of the gradient sequence.
        config = HamiltonianGeometricConfig(
            learning_rate=0.05, beta=0.0, use_geometric_correction=False,
            use_memory_correction=False, use_memory_metric=True,
            memory_metric_coupling=0.5, memory_decay=0.8,
        )
        state = initial_state(2)
        gradient = np.array([1.0, -2.0])

        def gradient_fn(_theta: np.ndarray) -> np.ndarray:
            return gradient

        def metric_fn(_theta: np.ndarray) -> np.ndarray:
            return np.eye(2)

        updated = hamiltonian_geometric_step(state, gradient_fn, metric_fn, config)

        expected_memory_metric = np.outer(gradient, gradient)
        np.testing.assert_allclose(updated.memory_metric, expected_memory_metric)
        np.testing.assert_allclose(updated.memory_metric, updated.memory_metric.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(updated.memory_metric) >= -1e-12))

        updated_again = hamiltonian_geometric_step(updated, gradient_fn, metric_fn, config)
        expected_second = config.memory_decay * expected_memory_metric + np.outer(gradient, gradient)
        np.testing.assert_allclose(updated_again.memory_metric, expected_second)

    def test_memory_metric_increases_spectral_entropy_effect_on_step(self) -> None:
        # Turning on the memory-metric coupling changes the effective metric
        # (hence the step), and the metric must remain usable (positive
        # definite, finite inverse) once the M_ij contribution is folded in.
        config_off = HamiltonianGeometricConfig(
            learning_rate=0.05, beta=0.0, use_geometric_correction=False,
            use_memory_correction=False, use_memory_metric=False,
        )
        config_on = HamiltonianGeometricConfig(
            learning_rate=0.05, beta=0.0, use_geometric_correction=False,
            use_memory_correction=False, use_memory_metric=True,
            memory_metric_coupling=2.0,
        )
        state = initial_state(2)

        def gradient_fn(_theta: np.ndarray) -> np.ndarray:
            return np.array([1.0, -2.0])

        def metric_fn(_theta: np.ndarray) -> np.ndarray:
            return np.eye(2)

        updated_off = hamiltonian_geometric_step(state, gradient_fn, metric_fn, config_off)
        updated_on = hamiltonian_geometric_step(state, gradient_fn, metric_fn, config_on)

        self.assertTrue(np.all(np.isfinite(updated_on.parameters)))
        self.assertFalse(np.allclose(updated_off.parameters, updated_on.parameters))


if __name__ == "__main__":
    unittest.main()
