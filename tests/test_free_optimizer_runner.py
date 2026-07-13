import tempfile
import unittest
from pathlib import Path

from main.run_free_optimizer_benchmark import write_symbolic_check


class FreeOptimizerRunnerTests(unittest.TestCase):
    def test_symbolic_check_records_hamiltonian_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "symbolic_check.txt"

            write_symbolic_check(path)

            content = path.read_text(encoding="utf-8")
            self.assertIn("H(theta, p)", content)
            self.assertIn("F_geo", content)
            self.assertIn("src/hamiltonian_geometric.py", content)


if __name__ == "__main__":
    unittest.main()
