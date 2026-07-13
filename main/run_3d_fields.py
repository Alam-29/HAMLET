import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization3d import Capacitor3DConfig, export_3d_field_animation_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export animated 3D capacitor field lines.")
    parser.add_argument("--plate-length", type=float, default=0.03)
    parser.add_argument("--plate-width", type=float, default=0.02)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--voltage", type=float, default=1.0)
    parser.add_argument("--field-line-rows", type=int, default=9)
    parser.add_argument("--field-line-columns", type=int, default=11)
    parser.add_argument("--points-per-line", type=int, default=72)
    parser.add_argument("--fringe-bulge", type=float, default=1.25)
    parser.add_argument("--emi-wobble", type=float, default=0.18)
    parser.add_argument("--animation-speed", type=float, default=1.05)
    parser.add_argument("--insertion-cycle-s", type=float, default=8.0)
    parser.add_argument("--chaotic-transient-strength", type=float, default=1.85)
    parser.add_argument("--plate-thickness", type=float, default=0.00045)
    parser.add_argument("--detector-distance", type=float, default=0.009)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "3d_projections" / "capacitor_3d_field_animation.html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = Capacitor3DConfig(
        plate_length=args.plate_length,
        plate_width=args.plate_width,
        gap=args.gap,
        voltage=args.voltage,
        field_line_rows=args.field_line_rows,
        field_line_columns=args.field_line_columns,
        points_per_line=args.points_per_line,
        fringe_bulge=args.fringe_bulge,
        emi_wobble=args.emi_wobble,
        animation_speed=args.animation_speed,
        insertion_cycle_s=args.insertion_cycle_s,
        chaotic_transient_strength=args.chaotic_transient_strength,
        plate_thickness=args.plate_thickness,
        detector_distance=args.detector_distance,
    )
    export_3d_field_animation_html(config, str(args.output))
    print("3D capacitor field animation")
    print(f"plate_length = {config.plate_length:g} m")
    print(f"plate_width = {config.plate_width:g} m")
    print(f"gap = {config.gap:g} m")
    print(f"fringe_bulge = {config.fringe_bulge:g}")
    print(f"emi_wobble = {config.emi_wobble:g}")
    print(f"insertion_cycle_s = {config.insertion_cycle_s:g}")
    print(f"chaotic_transient_strength = {config.chaotic_transient_strength:g}")
    print(f"plate_thickness = {config.plate_thickness:g} m")
    print(f"detector_distance = {config.detector_distance:g} m")
    print(f"exported = {args.output}")


if __name__ == "__main__":
    main()
