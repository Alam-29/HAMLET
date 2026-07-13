import unittest

import numpy as np

from src.normal_modes import (
    check_conservative_action_is_linear_in_hamiltonian,
    check_metric_shares_eigenbasis_with_hessian,
    check_plain_descent_decouples_into_normal_modes,
    check_preconditioned_descent_decouples_into_normal_modes,
    compute_normal_modes,
    export_normal_mode_markdown,
    run_all_checks,
    summarize_conditioning,
)
from src.pinn import FixedFeaturePotentialModel, PINNConfig, build_pinn_dataset


def _small_dataset():
    config = PINNConfig(
        hidden_features=6, collocation_points=30, plate_points=8, outer_boundary_points=8
    )
    model = FixedFeaturePotentialModel(config)
    dataset = build_pinn_dataset(model, config)
    return model, dataset


class NormalModeTests(unittest.TestCase):
    def test_compute_normal_modes_returns_positive_metric_eigenvalues(self) -> None:
        _model, dataset = _small_dataset()

        modes = compute_normal_modes(dataset)

        self.assertTrue(np.all(modes.metric_eigenvalues > 0.0))
        self.assertTrue(np.all(modes.omega >= 0.0))
        self.assertGreaterEqual(modes.condition_number, 1.0)

    def test_metric_shares_eigenbasis_with_hessian(self) -> None:
        _model, dataset = _small_dataset()

        result = check_metric_shares_eigenbasis_with_hessian(dataset)

        self.assertTrue(result.passed)

    def test_plain_descent_decouples_into_normal_modes(self) -> None:
        model, dataset = _small_dataset()

        result = check_plain_descent_decouples_into_normal_modes(dataset, model.parameter_count)

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_preconditioned_descent_decouples_into_normal_modes(self) -> None:
        model, dataset = _small_dataset()

        result = check_preconditioned_descent_decouples_into_normal_modes(
            dataset, model.parameter_count
        )

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_conservative_action_is_linear_in_hamiltonian(self) -> None:
        _model, dataset = _small_dataset()
        modes = compute_normal_modes(dataset)

        result = check_conservative_action_is_linear_in_hamiltonian(modes)

        self.assertTrue(result.passed)

    def test_summarize_conditioning_reports_bounded_preconditioned_rate(self) -> None:
        _model, dataset = _small_dataset()
        modes = compute_normal_modes(dataset)

        conditioning = summarize_conditioning(modes, learning_rate_preconditioned=0.5)

        # The whole point of preconditioning: its worst-case per-step
        # contraction rate is bounded by (1 - eta), independent of how ill-
        # conditioned the raw Hessian is, unlike plain descent's rate.
        self.assertLessEqual(conditioning["worst_mode_rate_preconditioned"], 1.0)
        self.assertGreaterEqual(conditioning["condition_number"], 1.0)

    def test_run_all_checks_all_pass(self) -> None:
        results = run_all_checks()

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertTrue(result.passed, f"{result.identifier} failed: {result.detail}")

    def test_export_normal_mode_markdown_includes_every_check(self) -> None:
        _model, dataset = _small_dataset()
        modes = compute_normal_modes(dataset)
        checks = run_all_checks()
        conditioning = summarize_conditioning(modes)

        markdown = export_normal_mode_markdown(modes, checks, conditioning)

        for check in checks:
            self.assertIn(check.identifier, markdown)
        self.assertIn("# Normal-Mode / Action-Angle Analysis", markdown)


if __name__ == "__main__":
    unittest.main()
