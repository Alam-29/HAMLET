import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.quantum_chaos import (
    QuantumChaosConfig,
    build_problem,
    export_quantum_history,
    export_quantum_summary,
    fidelity,
    loss_and_gradient,
    run_quantum_optimizer_comparison,
    spin_operators,
    unitary_from_hermitian,
)


class QuantumChaosBenchmarkTests(unittest.TestCase):
    def test_spin_operators_have_expected_shape_and_hermiticity(self) -> None:
        jy, jz = spin_operators(3)

        self.assertEqual(jy.shape, (7, 7))
        self.assertEqual(jz.shape, (7, 7))
        np.testing.assert_allclose(jy, jy.conj().T)
        np.testing.assert_allclose(jz, jz.conj().T)

    def test_unitary_from_hermitian_is_unitary(self) -> None:
        hamiltonian = np.array([[1.0, 0.2j], [-0.2j, 2.0]], dtype=complex)

        unitary = unitary_from_hermitian(hamiltonian)

        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-12)

    def test_target_controls_reproduce_target_state(self) -> None:
        problem = build_problem(QuantumChaosConfig(spin_j=2, steps=6))

        self.assertAlmostEqual(fidelity(problem, problem.target_controls), 1.0, places=10)

    def test_loss_gradient_is_finite(self) -> None:
        problem = build_problem(QuantumChaosConfig(spin_j=2, steps=5))
        controls = np.zeros(problem.config.steps)

        current_loss, gradient = loss_and_gradient(problem, controls)

        self.assertTrue(np.isfinite(current_loss))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertEqual(gradient.shape, controls.shape)

    def test_optimizer_comparison_runs_and_improves_some_optimizer(self) -> None:
        config = QuantumChaosConfig(spin_j=2, steps=5, kick_strength=5.5)

        _problem, results = run_quantum_optimizer_comparison(config, iterations=8)

        names = {result.optimizer for result in results}
        self.assertIn("adamw", names)
        self.assertIn("nesterov", names)
        self.assertIn("heavy_ball", names)
        self.assertIn("hamiltonian_geometric", names)
        for result in results:
            self.assertTrue(np.isfinite(result.final_loss))
            self.assertTrue(0.0 <= result.final_fidelity <= 1.0 + 1e-9)
        self.assertLess(min(result.final_loss for result in results), results[0].loss_history[0])

    def test_exporters_write_csv_headers(self) -> None:
        config = QuantumChaosConfig(spin_j=2, steps=4)
        _problem, results = run_quantum_optimizer_comparison(config, iterations=3)

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.csv"
            summary_path = Path(temp_dir) / "summary.csv"
            export_quantum_history(results, history_path)
            export_quantum_summary(results, summary_path)

            self.assertIn("iteration,optimizer,loss", history_path.read_text())
            self.assertIn("optimizer,final_loss,final_fidelity", summary_path.read_text())


if __name__ == "__main__":
    unittest.main()
