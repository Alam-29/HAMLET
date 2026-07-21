import unittest

from scripts.verify_submission_package import holm_adjust


class SubmissionVerifierTests(unittest.TestCase):
    def test_holm_adjust_matches_five_workload_correction(self):
        raw = [
            0.42951542065956305,
            0.01944121248981073,
            0.07160578476065826,
            0.0026838510794838844,
            5.8112708155703174e-05,
        ]
        expected = [
            0.42951542065956305,
            0.058323637469432185,
            0.14321156952131653,
            0.010735404317935538,
            0.00029056354077851586,
        ]
        for observed, target in zip(holm_adjust(raw), expected):
            self.assertAlmostEqual(observed, target)

    def test_holm_adjust_preserves_input_order_and_caps_at_one(self):
        self.assertEqual(holm_adjust([0.6, 0.01, 0.02]), [0.6, 0.03, 0.04])


if __name__ == "__main__":
    unittest.main()
