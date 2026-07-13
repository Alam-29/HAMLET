import tempfile
import unittest
from pathlib import Path

from src.visualization3d import (
    Capacitor3DConfig,
    export_3d_field_animation_html,
    generate_chaotic_transient_lines,
    generate_3d_field_lines,
    generate_incoming_field_lines,
)


class Visualization3DTests(unittest.TestCase):
    def test_generate_3d_field_lines_returns_polylines(self) -> None:
        config = Capacitor3DConfig(
            field_line_rows=3,
            field_line_columns=4,
            points_per_line=12,
        )

        lines = generate_3d_field_lines(config)

        self.assertGreater(len(lines), config.field_line_rows * config.field_line_columns)
        self.assertEqual(len(lines[0]), config.points_per_line)
        self.assertEqual(len(lines[0][0]), 3)

    def test_generate_transient_phase_lines(self) -> None:
        config = Capacitor3DConfig(
            field_line_rows=3,
            field_line_columns=4,
            points_per_line=12,
        )

        incoming = generate_incoming_field_lines(config)
        chaotic = generate_chaotic_transient_lines(config)

        self.assertGreater(len(incoming), 0)
        self.assertGreater(len(chaotic), 0)
        self.assertEqual(len(incoming[0]), config.points_per_line)
        self.assertEqual(len(chaotic[0]), config.points_per_line)

    def test_export_3d_animation_writes_standalone_html(self) -> None:
        config = Capacitor3DConfig(
            field_line_rows=3,
            field_line_columns=4,
            points_per_line=12,
            emi_wobble=0.1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field3d.html"
            export_3d_field_animation_html(config, str(path))
            content = path.read_text(encoding="utf-8")

        self.assertIn("<canvas", content)
        self.assertIn("requestAnimationFrame", content)
        self.assertIn("payload.lines", content)
        self.assertIn("incoming_lines", content)
        self.assertIn("chaotic_lines", content)
        self.assertIn("capacitor plates entering", content)
        self.assertIn("Moving dots", content)

    def test_rejects_invalid_3d_config(self) -> None:
        with self.assertRaises(ValueError):
            Capacitor3DConfig(gap=0.0)
        with self.assertRaises(ValueError):
            Capacitor3DConfig(field_line_rows=1)


if __name__ == "__main__":
    unittest.main()
