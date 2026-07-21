import unittest

import numpy as np

from src.approximate_metric_theory import (
    approximate_metric_stability_bound,
    modal_spectral_radius,
    verify_relative_spectrum,
)


class ApproximateMetricTheoryTests(unittest.TestCase):
    def test_stability_condition_matches_modal_roots(self) -> None:
        self.assertTrue(approximate_metric_stability_bound(0.2, 0.9, 0.7))
        result = verify_relative_spectrum(np.linspace(0.8, 1.2, 101), 0.2, 0.9, 0.7)
        self.assertTrue(result["empirically_stable"])
        self.assertLess(result["max_spectral_radius"], 1.0)

    def test_boundary_violation_is_unstable_at_upper_mode(self) -> None:
        eta = np.sqrt(2.0 * (1.0 + 0.7) / 1.2) * 1.001
        self.assertFalse(approximate_metric_stability_bound(0.2, eta, 0.7))
        self.assertGreater(modal_spectral_radius(1.2, eta, 0.7), 1.0)

    def test_absolute_condition_number_does_not_enter_relative_modes(self) -> None:
        reference = verify_relative_spectrum([0.8, 1.0, 1.2], 0.2, 0.9, 0.7)
        scaled = verify_relative_spectrum(np.array([0.8, 1.0, 1.2]), 0.2, 0.9, 0.7)
        self.assertEqual(reference["max_spectral_radius"], scaled["max_spectral_radius"])


if __name__ == "__main__":
    unittest.main()
