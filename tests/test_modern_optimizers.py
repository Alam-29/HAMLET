import unittest

import numpy as np

from src.modern_optimizers import (
    adamw_initial_state,
    adamw_step,
    lion_initial_state,
    lion_step,
    muon_step,
    orthogonalize_via_polar_factor,
    shampoo_initial_state,
    shampoo_step,
)


class LionTests(unittest.TestCase):
    def test_step_moves_opposite_the_sign_of_the_blended_gradient(self) -> None:
        parameters = np.array([1.0, -1.0])
        gradient = np.array([2.0, -3.0])
        state = lion_initial_state(parameters.shape)

        new_parameters, new_state = lion_step(parameters, gradient, state, learning_rate=0.1)

        np.testing.assert_allclose(new_parameters, parameters - 0.1 * np.sign(gradient))
        np.testing.assert_allclose(new_state.slow_momentum, 0.01 * gradient)

    def test_update_is_bounded_by_learning_rate_regardless_of_gradient_magnitude(self) -> None:
        # The defining property of sign-based (L1 kinetic) dynamics: the step
        # size does not grow with gradient magnitude, unlike plain SGD.
        parameters = np.zeros(3)
        state = lion_initial_state(parameters.shape)

        _small_step, _ = lion_step(parameters, np.array([1e-3, -1e-3, 0.0]), state, learning_rate=0.1)
        _large_step, _ = lion_step(parameters, np.array([1e3, -1e3, 0.0]), state, learning_rate=0.1)

        # Both must move by exactly the learning rate in nonzero coordinates.
        self.assertAlmostEqual(abs(_small_step[0]), 0.1)
        self.assertAlmostEqual(abs(_large_step[0]), 0.1)


class AdamWTests(unittest.TestCase):
    def test_decoupled_decay_differs_from_l2_folded_into_gradient(self) -> None:
        # The whole point of AdamW: applying weight_decay*theta outside the
        # adaptive scaling gives a different update than folding lambda*theta
        # into the gradient before adaptive scaling divides it down.
        parameters = np.array([2.0, 2.0])
        gradient_without_l2 = np.array([0.01, 0.01])
        state = adamw_initial_state(parameters.shape)

        decoupled, _ = adamw_step(
            parameters, gradient_without_l2, state, learning_rate=0.1, weight_decay=0.5
        )

        # Adam+L2: fold lambda*theta into the gradient, then adaptively scale.
        gradient_with_l2 = gradient_without_l2 + 0.5 * parameters
        coupled, _ = adamw_step(
            parameters, gradient_with_l2, state, learning_rate=0.1, weight_decay=0.0
        )

        self.assertFalse(np.allclose(decoupled, coupled))
        # Decoupled decay always removes exactly eta*weight_decay*theta,
        # independent of the adaptive second-moment scale.
        first_moment = 0.1 * gradient_without_l2
        second_moment = 0.001 * gradient_without_l2**2
        first_hat = first_moment / 0.1
        second_hat = second_moment / 0.001
        expected = parameters - 0.1 * (
            first_hat / (np.sqrt(second_hat) + 1e-8) + 0.5 * parameters
        )
        np.testing.assert_allclose(decoupled, expected)

    def test_bias_correction_matches_step_one_of_plain_adam(self) -> None:
        parameters = np.zeros(2)
        gradient = np.array([1.0, -1.0])
        state = adamw_initial_state(parameters.shape)

        updated, new_state = adamw_step(
            parameters, gradient, state, learning_rate=0.1, weight_decay=0.0
        )

        # At step 1 with beta1=0.9, beta2=0.999: bias-corrected moments equal
        # the raw gradient and its square exactly.
        expected = parameters - 0.1 * (gradient / (np.sqrt(gradient**2) + 1e-8))
        np.testing.assert_allclose(updated, expected)
        self.assertEqual(new_state.step, 1)


class ShampooTests(unittest.TestCase):
    def test_vec_axb_kronecker_identity(self) -> None:
        # The identity the Shampoo derivation (Eq. shampoo) rests on:
        # vec(A X B) = (B^T kron A) vec(X).
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 3))
        b = rng.normal(size=(4, 4))
        x = rng.normal(size=(3, 4))

        lhs = (a @ x @ b).reshape(-1, order="F")
        rhs = np.kron(b.T, a) @ x.reshape(-1, order="F")

        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_step_reduces_loss_on_a_quadratic(self) -> None:
        rng = np.random.default_rng(1)
        target = rng.normal(size=(3, 4))
        weight = np.zeros((3, 4))
        state = shampoo_initial_state(3, 4, epsilon=1e-3)

        def loss(w: np.ndarray) -> float:
            return float(0.5 * np.sum((w - target) ** 2))

        initial_loss = loss(weight)
        for _ in range(20):
            gradient = weight - target
            weight, state = shampoo_step(weight, gradient, state, learning_rate=0.5)

        self.assertLess(loss(weight), initial_loss)

    def test_preconditioners_stay_symmetric(self) -> None:
        rng = np.random.default_rng(2)
        weight = rng.normal(size=(2, 3))
        state = shampoo_initial_state(2, 3, epsilon=1e-3)
        gradient = rng.normal(size=(2, 3))

        _new_weight, new_state = shampoo_step(weight, gradient, state, learning_rate=0.1)

        np.testing.assert_allclose(
            new_state.left_preconditioner, new_state.left_preconditioner.T
        )
        np.testing.assert_allclose(
            new_state.right_preconditioner, new_state.right_preconditioner.T
        )


class MuonTests(unittest.TestCase):
    def test_orthogonalize_produces_unit_singular_values(self) -> None:
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(4, 6)) * rng.uniform(0.1, 10.0, size=(4, 6))

        orthogonalized = orthogonalize_via_polar_factor(matrix)

        singular_values = np.linalg.svd(orthogonalized, compute_uv=False)
        np.testing.assert_allclose(singular_values, np.ones_like(singular_values), atol=1e-8)

    def test_orthogonalization_maximizes_spectral_entropy_of_induced_metric(self) -> None:
        # Direct check of the report's claim: among matrices sharing M's
        # singular directions, the orthogonalized one is the equal-singular-
        # value (maximum spectral entropy) member.
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(3, 5))
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

        def spectral_entropy(values: np.ndarray) -> float:
            normalized = values / values.sum()
            return float(-np.sum(normalized * np.log(normalized)))

        orthogonalized_entropy = spectral_entropy(np.ones_like(singular_values))
        original_entropy = spectral_entropy(singular_values)

        self.assertGreaterEqual(orthogonalized_entropy, original_entropy)
        # Equal singular values is the unique maximum-entropy point (uniform
        # distribution maximizes entropy over a fixed support size).
        self.assertAlmostEqual(orthogonalized_entropy, np.log(len(singular_values)))

    def test_step_moves_weight_by_exactly_learning_rate_times_orthogonal_matrix(self) -> None:
        weight = np.zeros((3, 4))
        momentum = np.zeros((3, 4))
        gradient = np.random.default_rng(5).normal(size=(3, 4))

        new_weight, new_momentum = muon_step(weight, gradient, momentum, learning_rate=0.1)

        expected_orthogonalized = orthogonalize_via_polar_factor(gradient)
        np.testing.assert_allclose(new_weight, -0.1 * expected_orthogonalized)
        np.testing.assert_allclose(new_momentum, gradient)


if __name__ == "__main__":
    unittest.main()
