import csv
import json
import os
import tempfile
import unittest

from src.capacitor_3d import (
    export_3d_solution_animation_html,
    export_3d_field_quiver_png,
    export_3d_potential_png,
    export_capacitance_comparison_png,
    load_capacitor_3d_solution,
)


def _write_synthetic_solution(directory: str) -> None:
    """Write a tiny synthetic 3D grid in the same format
    main/mathematica/capacitor_3d_solve.wls produces, so the Python loader
    and plot exporters can be tested without requiring Mathematica."""

    nx, ny, nz = 5, 5, 7
    gap = 0.004
    length, width = 0.03, 0.02
    xs = [-0.033 + i * 0.066 / (nx - 1) for i in range(nx)]
    ys = [-0.022 + j * 0.044 / (ny - 1) for j in range(ny)]
    zs = [-0.018 + k * 0.036 / (nz - 1) for k in range(nz)]
    k_top = min(range(nz), key=lambda k: abs(zs[k] - gap / 2.0))
    k_bot = min(range(nz), key=lambda k: abs(zs[k] + gap / 2.0))

    rows = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                on_plate = abs(x) <= length / 2.0 and abs(y) <= width / 2.0 and k in (k_top, k_bot)
                if k == k_top and on_plate:
                    phi = 0.5
                elif k == k_bot and on_plate:
                    phi = -0.5
                else:
                    phi = 0.5 * z / zs[-1]
                rows.append([x, y, z, phi, 0.0, 0.0, -phi / max(abs(z), 1e-6), 1 if on_plate else 0])

    with open(os.path.join(directory, "field_grid.csv"), "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)

    summary = {
        "plate_length_m": length,
        "plate_width_m": width,
        "gap_m": gap,
        "voltage_V": 1.0,
        "electrode_length_m": length * 0.9,
        "electrode_width_m": width * 0.9,
        "electrode_gap_m": gap * 0.9,
        "box_x_m": 0.066,
        "box_y_m": 0.044,
        "box_z_m": 0.036,
        "grid_nx": nx,
        "grid_ny": ny,
        "grid_nz": nz,
        "iterations": 100,
        "max_delta": 1e-6,
        "converged": True,
        "ideal_capacitance_F": 1.33e-12,
        "capacitance_3d_F": 1.7e-12,
        "fringe_ratio_3d": 1.28,
        "energy_J": 8.5e-13,
        "solver": "synthetic-test-fixture",
    }
    with open(os.path.join(directory, "capacitance_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file)


class Capacitor3DTests(unittest.TestCase):
    def test_load_capacitor_3d_solution_reshapes_grid_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_synthetic_solution(tmp_dir)
            solution = load_capacitor_3d_solution(tmp_dir)

        self.assertEqual(solution.potential.shape, (5, 5, 7))
        self.assertEqual(solution.x.shape, (5,))
        self.assertEqual(solution.y.shape, (5,))
        self.assertEqual(solution.z.shape, (7,))
        self.assertTrue(solution.is_electrode.any())
        self.assertAlmostEqual(solution.summary["capacitance_3d_F"], 1.7e-12)

    def test_export_functions_produce_nonempty_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_synthetic_solution(tmp_dir)
            solution = load_capacitor_3d_solution(tmp_dir)

            potential_path = os.path.join(tmp_dir, "potential3d.png")
            field_path = os.path.join(tmp_dir, "field3d.png")
            capacitance_path = os.path.join(tmp_dir, "capacitance.png")
            export_3d_potential_png(solution, potential_path)
            export_3d_field_quiver_png(solution, field_path, target_x=3, target_y=3, target_z=3)
            export_capacitance_comparison_png(solution, capacitance_path)

            for path in (potential_path, field_path, capacitance_path):
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)

    def test_export_solution_animation_writes_html_from_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_synthetic_solution(tmp_dir)
            solution = load_capacitor_3d_solution(tmp_dir)
            path = os.path.join(tmp_dir, "solution_animation.html")

            export_3d_solution_animation_html(solution, path, max_lines=12, points_per_line=16)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            self.assertIn("field_grid.csv", content)
            self.assertIn("payload.lines", content)
            self.assertIn("payload.external_lines", content)
            self.assertIn("drawArrowOnLine", content)
            self.assertIn("requestAnimationFrame", content)
            self.assertNotIn("detector/photo-plate", content)


if __name__ == "__main__":
    unittest.main()
