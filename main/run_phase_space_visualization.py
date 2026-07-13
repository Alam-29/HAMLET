"""Render the Hamiltonian-geometric optimizer's theta trajectory in 3D.

theta lives in a `--features`-dimensional space (64 by default); this script
records theta at every training step for each optimizer, fits a shared PCA
basis across all of them, and projects each trajectory onto the top 3
components so the curved path the Hamiltonian-geometric optimizer takes
through parameter space can be compared against SGD/Adam/etc. in an ordinary
3D plot.
"""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_space_visualization import (
    compute_phase_space_projection,
    export_phase_space_html,
    export_phase_space_png,
    export_phase_space_rotation_gif,
    export_phase_space_trajectories_csv,
)
from src.pinn import PINNConfig, run_optimizer_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D PCA phase-space view of the optimizer benchmark's theta trajectory."
    )
    parser.add_argument("--plate-width", type=float, default=0.02)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--domain-width", type=float, default=0.08)
    parser.add_argument("--domain-height", type=float, default=0.08)
    parser.add_argument("--voltage", type=float, default=1.0)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--collocation-points", type=int, default=700)
    parser.add_argument("--plate-points", type=int, default=80)
    parser.add_argument("--outer-boundary-points", type=int, default=100)
    parser.add_argument("--boundary-weight", type=float, default=80.0)
    parser.add_argument("--outer-boundary-weight", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--rotation-frames",
        type=int,
        default=60,
        help="Frame count for the rotating GIF (one full turn in azimuth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "phase_space",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = PINNConfig(
        plate_width=args.plate_width,
        gap=args.gap,
        domain_width=args.domain_width,
        domain_height=args.domain_height,
        voltage=args.voltage,
        hidden_features=args.features,
        collocation_points=args.collocation_points,
        plate_points=args.plate_points,
        outer_boundary_points=args.outer_boundary_points,
        boundary_weight=args.boundary_weight,
        outer_boundary_weight=args.outer_boundary_weight,
        seed=args.seed,
    )
    _model, _dataset, results = run_optimizer_comparison(
        config, steps=args.steps, record_theta=True
    )
    projections, explained_variance_ratio = compute_phase_space_projection(results)

    csv_path = args.output_dir / "phase_space_trajectories.csv"
    png_path = args.output_dir / "phase_space.png"
    gif_path = args.output_dir / "phase_space_rotation.gif"
    html_path = args.output_dir / "phase_space.html"

    export_phase_space_trajectories_csv(results, projections, str(csv_path))
    export_phase_space_png(projections, explained_variance_ratio, str(png_path))
    export_phase_space_rotation_gif(
        projections, explained_variance_ratio, str(gif_path), frames=args.rotation_frames
    )
    export_phase_space_html(results, projections, explained_variance_ratio, str(html_path))

    print("Hamiltonian-geometric optimizer: 3D theta phase-space projection")
    print(f"theta_dimension = {config.hidden_features}")
    print(f"steps = {args.steps}")
    print(
        "explained_variance_ratio = "
        + ", ".join(f"PC{i + 1}={ratio * 100:.1f}%" for i, ratio in enumerate(explained_variance_ratio))
    )
    print(f"exported = {csv_path}")
    print(f"exported = {png_path}")
    print(f"exported = {gif_path}")
    print(f"exported = {html_path}")


if __name__ == "__main__":
    main()
