import tempfile
import unittest
from pathlib import Path
import json

from main.run_algoperf_style_benchmark import search_space, write_implementation_note


class AlgoPerfStyleBenchmarkTests(unittest.TestCase):
    def test_search_spaces_include_reference_baselines(self) -> None:
        self.assertTrue(search_space("adamw"))
        self.assertTrue(search_space("nag"))
        self.assertTrue(search_space("heavy_ball"))
        self.assertTrue(search_space("hamiltonian_geometric"))

    def test_implementation_note_documents_reference_baselines(self) -> None:
        class Args:
            max_trials_per_optimizer = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "note.txt"

            write_implementation_note(Args(), path)

            content = path.read_text(encoding="utf-8")
            self.assertIn("AdamW", content)
            self.assertIn("Nesterov Momentum", content)
            self.assertIn("Heavy Ball Momentum", content)
            self.assertIn("max_trials_per_optimizer = 2", content)

    def test_official_submission_package_layout_exists(self) -> None:
        package_dir = Path("algoperf_submissions") / "external_tuning" / "hamiltonian_geometric"

        self.assertTrue((package_dir / "submission.py").exists())
        self.assertTrue((package_dir / "requirements.txt").exists())
        self.assertTrue((package_dir / "tuning_search_space.json").exists())
        search = json.loads((package_dir / "tuning_search_space.json").read_text())
        self.assertIn("learning_rate", search)
        self.assertIn("memory_coupling", search)
        self.assertIn("end_factor", search)


if __name__ == "__main__":
    unittest.main()
