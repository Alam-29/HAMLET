import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.capacitor_3d import (
    export_3d_field_quiver_png,
    export_3d_potential_png,
    export_capacitance_comparison_png,
    load_capacitor_3d_solution,
)

WOLFRAMSCRIPT_CANDIDATES = [
    "wolframscript",
    r"C:\Program Files\Wolfram Research\Wolfram Engine\15.0\wolframscript.exe",
]


def find_wolframscript() -> str:
    for candidate in WOLFRAMSCRIPT_CANDIDATES:
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "wolframscript not found. Install the Wolfram Engine (or Mathematica) "
        "and either put wolframscript on PATH or add its install path to "
        "WOLFRAMSCRIPT_CANDIDATES in this script."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real 3D finite-difference Laplace solve for a finite "
            "rectangular-plate capacitor through the Wolfram Engine, then "
            "render 3D potential/field/capacitance plots from the result."
        )
    )
    parser.add_argument("--plate-length", type=float, default=0.03)
    parser.add_argument("--plate-width", type=float, default=0.02)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--voltage", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--ny", type=int, default=77)
    parser.add_argument("--nz", type=int, default=101)
    parser.add_argument("--max-iterations", type=int, default=4500)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "3d_field"
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Reuse the existing solver output in --output-dir instead of re-running Mathematica.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = PROJECT_ROOT / "main" / "mathematica" / "capacitor_3d_solve.wls"

    if not args.skip_solve:
        wolframscript = find_wolframscript()
        command = [
            wolframscript, "-file", str(script_path),
            str(args.plate_length), str(args.plate_width), str(args.gap), str(args.voltage),
            str(args.nx), str(args.ny), str(args.nz), str(args.max_iterations),
            str(args.output_dir),
        ]
        print("Running Mathematica 3D solve:")
        print(" ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"capacitor_3d_solve.wls failed with exit code {result.returncode}")

    solution = load_capacitor_3d_solution(str(args.output_dir))

    potential_path = args.output_dir / "potential_3d.png"
    field_path = args.output_dir / "field_3d.png"
    capacitance_path = args.output_dir / "capacitance_comparison.png"
    export_3d_potential_png(solution, str(potential_path))
    export_3d_field_quiver_png(solution, str(field_path))
    export_capacitance_comparison_png(solution, str(capacitance_path))

    summary = solution.summary
    print("3D capacitor solve + visualization")
    print(f"grid = {summary['grid_nx']} x {summary['grid_ny']} x {summary['grid_nz']}")
    print(f"converged = {summary['converged']}  (max_delta = {summary['max_delta']:.3e})")
    print(f"C_ideal = {summary['ideal_capacitance_F']:.6e} F")
    print(f"C_3d    = {summary['capacitance_3d_F']:.6e} F")
    print(f"fringe_ratio_3d = {summary['fringe_ratio_3d']:.4f}")
    print(f"exported = {potential_path}")
    print(f"exported = {field_path}")
    print(f"exported = {capacitance_path}")


if __name__ == "__main__":
    main()
