import unittest

from src.paper_math_validation import (
    check_flat_metric_reduces_to_gradient_descent,
    check_geometric_force_matches_hamiltons_equation,
    check_hessian_regularization,
    check_legendre_transform,
    check_memory_recursion_matches_closed_form,
    check_rayleigh_dissipation_generalizes_to_curved_metric,
    check_spectral_entropy_gradient,
    export_numeric_checks_markdown,
    run_all_checks,
)


class PaperMathValidationTests(unittest.TestCase):
    def test_legendre_transform_holds_for_the_toy_metric(self) -> None:
        result = check_legendre_transform()

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_geometric_force_matches_hamiltons_equation(self) -> None:
        result = check_geometric_force_matches_hamiltons_equation()

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_flat_metric_reduction_is_exact(self) -> None:
        result = check_flat_metric_reduces_to_gradient_descent()

        self.assertTrue(result.passed)
        self.assertEqual(result.max_error, 0.0)

    def test_memory_recursion_matches_closed_form_sum(self) -> None:
        result = check_memory_recursion_matches_closed_form()

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_hessian_regularization_counterexample_and_fix(self) -> None:
        result = check_hessian_regularization()

        self.assertTrue(result.passed)
        self.assertIn("fails: True", result.detail)

    def test_rayleigh_dissipation_generalizes_to_curved_metric(self) -> None:
        result = check_rayleigh_dissipation_generalizes_to_curved_metric()

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_spectral_entropy_gradient_matches_finite_difference(self) -> None:
        result = check_spectral_entropy_gradient()

        self.assertTrue(result.passed)
        self.assertLess(result.max_error, result.tolerance)

    def test_run_all_checks_all_pass(self) -> None:
        results = run_all_checks()

        self.assertEqual(len(results), 7)
        for result in results:
            self.assertTrue(result.passed, f"{result.identifier} failed: {result.detail}")

    def test_export_numeric_checks_markdown_includes_every_check(self) -> None:
        results = run_all_checks()

        markdown = export_numeric_checks_markdown(results)

        for result in results:
            self.assertIn(result.identifier, markdown)
        self.assertIn("## Numerical Checks", markdown)


if __name__ == "__main__":
    unittest.main()
