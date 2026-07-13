import unittest

from src.capacitance import (
    EPSILON_0,
    RectangularCapacitor,
    effective_area_fringe,
    fringe_ratio,
    ideal_parallel_plate,
    sweep_gap,
)


class CapacitanceModelTests(unittest.TestCase):
    def test_ideal_parallel_plate_matches_textbook_formula(self) -> None:
        capacitor = RectangularCapacitor(length=0.02, width=0.03, gap=0.001)

        capacitance = ideal_parallel_plate(capacitor)

        self.assertAlmostEqual(capacitance, EPSILON_0 * 0.02 * 0.03 / 0.001)

    def test_effective_area_model_is_never_below_ideal_for_positive_extension(
        self,
    ) -> None:
        capacitor = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)

        self.assertGreaterEqual(effective_area_fringe(capacitor), ideal_parallel_plate(capacitor))

    def test_fringe_ratio_increases_when_gap_becomes_large_relative_to_plate(
        self,
    ) -> None:
        small_gap = RectangularCapacitor(length=0.02, width=0.02, gap=0.0005)
        large_gap = RectangularCapacitor(length=0.02, width=0.02, gap=0.004)

        self.assertGreater(fringe_ratio(large_gap), fringe_ratio(small_gap))

    def test_sweep_gap_returns_one_row_per_gap(self) -> None:
        rows = sweep_gap(length=0.02, width=0.02, gaps=[0.001, 0.002])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["gap_m"], 0.001)
        self.assertIn("ratio", rows[0])

    def test_rejects_nonphysical_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            RectangularCapacitor(length=0.02, width=0.02, gap=0.0)

    def test_rejects_negative_edge_extension(self) -> None:
        capacitor = RectangularCapacitor(length=0.02, width=0.02, gap=0.001)

        with self.assertRaises(ValueError):
            effective_area_fringe(capacitor, edge_extension_fraction=-0.1)


if __name__ == "__main__":
    unittest.main()
