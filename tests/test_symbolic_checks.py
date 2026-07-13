from pathlib import Path
import tempfile
import unittest

from src.symbolic_checks import (
    capacitor_energy_identity,
    capacitor_force_identity,
    find_cadabra_executable,
    hamiltonian_metric_positive_identity,
    run_cadabra_script,
    run_sympy_checks,
)


class SymbolicChecksTests(unittest.TestCase):
    def test_sympy_capacitor_identities_reduce_to_zero(self) -> None:
        checks = run_sympy_checks()

        self.assertTrue(checks)
        self.assertTrue(all(check.passed for check in checks))
        self.assertEqual(capacitor_energy_identity().simplified, "0")
        self.assertEqual(capacitor_force_identity().simplified, "0")

    def test_hamiltonian_one_step_metric_limit(self) -> None:
        check = hamiltonian_metric_positive_identity()

        self.assertTrue(check.passed)
        self.assertEqual(check.simplified, "0")

    def test_cadabra_runner_reports_missing_executable_cleanly(self) -> None:
        if find_cadabra_executable() is not None:
            self.skipTest("Cadabra is installed on this machine")
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "empty.cdb"
            script.write_text("", encoding="utf-8")

            result = run_cadabra_script(script)

        self.assertFalse(result.available)
        self.assertIn("not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
