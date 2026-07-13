import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.capacitor_3d import export_3d_solution_animation_html, load_capacitor_3d_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an animated 3D field visualization from generated field_grid.csv data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "3d_field",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "3d_projections" / "capacitor_3d_generated_solution_animation.html",
    )
    parser.add_argument("--max-lines", type=int, default=112)
    parser.add_argument("--points-per-line", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    solution = load_capacitor_3d_solution(str(args.input_dir))
    export_3d_solution_animation_html(
        solution,
        str(args.output),
        max_lines=args.max_lines,
        points_per_line=args.points_per_line,
    )
    print("Generated-solution 3D capacitor animation")
    print(f"input_dir = {args.input_dir}")
    print(f"grid = {solution.summary['grid_nx']} x {solution.summary['grid_ny']} x {solution.summary['grid_nz']}")
    print(f"capacitance_3d_F = {solution.summary['capacitance_3d_F']:.6e}")
    print(f"fringe_ratio_3d = {solution.summary['fringe_ratio_3d']:.6f}")
    print(f"exported = {args.output}")


if __name__ == "__main__":
    main()
