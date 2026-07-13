import unittest

import numpy as np

from src.pinn import (
    PINNConfig,
    FixedFeaturePotentialModel,
    build_pinn_dataset,
    run_optimizer_comparison,
    train_entropy_descent,
    train_falling_ball,
    train_hamiltonian_geometric,
    _positive_definite_metric,
)


class PINNBenchmarkTests(unittest.TestCase):
    def test_dataset_has_expected_parameter_columns(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        model = FixedFeaturePotentialModel(config)

        dataset = build_pinn_dataset(model, config)

        self.assertEqual(dataset.design_matrix.shape[1], model.parameter_count)
        self.assertEqual(dataset.targets.shape[0], dataset.design_matrix.shape[0])
        self.assertEqual(model.parameter_count, config.hidden_features + 1)

    def test_optimizer_comparison_reduces_loss(self) -> None:
        config = PINNConfig(
            hidden_features=12,
            collocation_points=40,
            plate_points=10,
            outer_boundary_points=16,
        )

        _model, _dataset, results = run_optimizer_comparison(config, steps=40)

        optimizer_names = {result.optimizer for result in results}
        self.assertIn("sgd", optimizer_names)
        self.assertIn("adam", optimizer_names)
        self.assertIn("falling_ball", optimizer_names)
        self.assertIn("entropy_descent", optimizer_names)
        self.assertIn("hamiltonian_geometric", optimizer_names)
        for result in results:
            self.assertTrue(np.isfinite(result.final_loss))
            self.assertLess(result.final_loss, result.loss_history[0])
            self.assertGreaterEqual(result.pde_loss, 0.0)
            self.assertGreaterEqual(result.plate_loss, 0.0)
            self.assertGreaterEqual(result.outer_loss, 0.0)

    def test_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            PINNConfig(plate_width=0.08, domain_width=0.08)

    def test_hamiltonian_geometric_reports_spectral_entropy(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        model = FixedFeaturePotentialModel(config)
        dataset = build_pinn_dataset(model, config)

        result = train_hamiltonian_geometric(dataset, model.parameter_count, steps=10)

        self.assertTrue(np.isfinite(result.spectral_entropy))
        self.assertGreaterEqual(result.spectral_entropy, 0.0)

    def test_hamiltonian_geometric_ablation_flags_still_train(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        model = FixedFeaturePotentialModel(config)
        dataset = build_pinn_dataset(model, config)

        with_memory = train_hamiltonian_geometric(dataset, model.parameter_count, steps=30)
        without_memory = train_hamiltonian_geometric(
            dataset, model.parameter_count, steps=30, use_memory_correction=False
        )
        without_geometric = train_hamiltonian_geometric(
            dataset, model.parameter_count, steps=30, use_geometric_correction=False
        )

        for result in (with_memory, without_memory, without_geometric):
            self.assertTrue(np.isfinite(result.final_loss))
            self.assertLess(result.final_loss, result.loss_history[0])

        # F_geo is exactly zero for this fixed-feature (constant-metric)
        # benchmark, so ablating it must not change the trajectory at all.
        self.assertEqual(with_memory.loss_history, without_geometric.loss_history)

    def test_positive_definite_metric_guarantees_pd_for_convex_hessian(self) -> None:
        raw_hessian = np.diag([4.0, 0.5, 2.0])

        metric = _positive_definite_metric(raw_hessian, regularization=1e-3)

        self.assertTrue(np.all(np.linalg.eigvalsh(metric) > 0.0))

    def test_falling_ball_reduces_loss(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        model = FixedFeaturePotentialModel(config)
        dataset = build_pinn_dataset(model, config)

        result = train_falling_ball(dataset, model.parameter_count, steps=200)

        self.assertTrue(np.isfinite(result.final_loss))
        self.assertLess(result.final_loss, result.loss_history[0])

    def test_falling_ball_rejects_invalid_friction(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        model = FixedFeaturePotentialModel(config)
        dataset = build_pinn_dataset(model, config)

        with self.assertRaises(ValueError):
            train_falling_ball(dataset, model.parameter_count, steps=5, friction=1.0)

    def test_entropy_descent_reduces_loss_and_reports_spectral_entropy(self) -> None:
        config = PINNConfig(
            hidden_features=8,
            collocation_points=20,
            plate_points=6,
            outer_boundary_points=12,
        )
        model = FixedFeaturePotentialModel(config)
        dataset = build_pinn_dataset(model, config)

        result = train_entropy_descent(dataset, model.parameter_count, steps=200)

        self.assertTrue(np.isfinite(result.final_loss))
        self.assertLess(result.final_loss, result.loss_history[0])
        self.assertTrue(np.isfinite(result.spectral_entropy))
        self.assertGreaterEqual(result.spectral_entropy, 0.0)

    def test_metric_preconditioned_optimizers_beat_unpreconditioned_ones(self) -> None:
        # The whole point of the metric: at a matched, small step budget, both
        # metric-preconditioned optimizers (entropy_descent,
        # hamiltonian_geometric) should make much more progress than either
        # unpreconditioned optimizer (sgd, falling_ball) on this benchmark's
        # severely ill-conditioned quadratic loss.
        config = PINNConfig(
            hidden_features=10,
            collocation_points=30,
            plate_points=8,
            outer_boundary_points=8,
        )
        _model, _dataset, results = run_optimizer_comparison(config, steps=150)
        by_name = {result.optimizer: result for result in results}

        unpreconditioned_best = min(
            by_name["sgd"].final_loss, by_name["falling_ball"].final_loss
        )
        preconditioned_worst = max(
            by_name["entropy_descent"].final_loss, by_name["hamiltonian_geometric"].final_loss
        )
        self.assertLess(preconditioned_worst, unpreconditioned_best)

    def test_positive_definite_metric_guarantees_pd_for_indefinite_hessian(self) -> None:
        # A convex quadratic loss (as used by this benchmark) never produces
        # an indefinite Hessian, but a non-convex loss could -- this checks
        # the general guarantee that motivated the adaptive regularization,
        # not just the convex case this benchmark happens to exercise.
        raw_hessian = np.diag([-5.0, 0.1, 3.0])

        metric = _positive_definite_metric(raw_hessian, regularization=1e-3)

        eigenvalues = np.linalg.eigvalsh(metric)
        self.assertTrue(np.all(eigenvalues > 0.0))


if __name__ == "__main__":
    unittest.main()
