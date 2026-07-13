import unittest

from src.paper_validation import validate_literature_review_math


class PaperValidationTests(unittest.TestCase):
    def test_validation_covers_known_mathematical_issues(self) -> None:
        findings = validate_literature_review_math()
        identifiers = {finding.identifier for finding in findings}

        self.assertIn("hamiltonian-sign", identifiers)
        self.assertIn("metric-positive-definite", identifiers)
        self.assertIn("memory-vector-vs-matrix", identifiers)
        self.assertIn("spectral-entropy-gradient", identifiers)
        self.assertIn("fringing-ground-truth", identifiers)

    def test_high_severity_findings_have_code_resolutions(self) -> None:
        findings = validate_literature_review_math()
        high_severity = [finding for finding in findings if finding.severity == "high"]

        self.assertGreaterEqual(len(high_severity), 2)
        for finding in high_severity:
            self.assertEqual(finding.status, "corrected-in-code")
            self.assertTrue(finding.code_resolution)


if __name__ == "__main__":
    unittest.main()
