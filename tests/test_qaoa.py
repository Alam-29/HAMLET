import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.qaoa import (
    QAOAConfig,
    build_problem,
    export_qaoa_history,
    export_qaoa_summary,
    loss_and_gradient,
    maxcut_values,
    qaoa_state,
    run_qaoa_comparison,
)


class QAOATests(unittest.TestCase):
    def test_maxcut_values_match_simple_edge(self) -> None:
        values = maxcut_values(2, [(0, 1)])

        np.testing.assert_allclose(values, np.array([0.0, 1.0, 1.0, 0.0]))

    def test_qaoa_state_is_normalized(self) -> None:
        problem = build_problem(QAOAConfig(qubits=4, depth=1))
        state = qaoa_state(problem, np.array([0.2, 0.3]))

        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

    def test_gradient_is_finite(self) -> None:
        problem = build_problem(QAOAConfig(qubits=4, depth=1))
        current_loss, gradient = loss_and_gradient(problem, np.zeros(2))

        self.assertTrue(np.isfinite(current_loss))
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_comparison_and_exporters_run(self) -> None:
        problem, results = run_qaoa_comparison(QAOAConfig(qubits=4, depth=1), iterations=3)

        self.assertTrue(results)
        for result in results:
            self.assertTrue(np.isfinite(result.final_loss))
            self.assertTrue(0.0 <= result.final_ratio <= 1.0 + 1e-9)

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.csv"
            summary_path = Path(temp_dir) / "summary.csv"
            export_qaoa_history(results, history_path)
            export_qaoa_summary(problem, results, summary_path)
            self.assertIn("iteration,optimizer,loss", history_path.read_text())
            self.assertIn("optimizer,final_loss,approximation_ratio", summary_path.read_text())


if __name__ == "__main__":
    unittest.main()
