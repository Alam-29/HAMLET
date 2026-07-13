import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_dashboard import export_optimizer_benchmark_html
from src.pinn import (
    PINNConfig,
    export_optimizer_summary,
    export_potential_grid,
    export_training_history,
    run_optimizer_comparison,
)
from src.visualization import export_optimizer_convergence_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Hamiltonian-geometric optimizer benchmark on a "
            "physics-informed capacitor fringing-field model."
        )
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
        "--spectral-weight",
        type=float,
        default=0.0,
        help="Coupling alpha for the spectral-entropy regularizer (paper Eq. 28-30).",
    )
    parser.add_argument(
        "--disable-geometric-correction",
        action="store_true",
        help="Ablate F_geo, the curvature correction, per the Sec. 19.3 proposed test.",
    )
    parser.add_argument(
        "--disable-memory-correction",
        action="store_true",
        help="Ablate F_mem, the exponential memory force, per the Sec. 19.3 proposed test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "pinn_benchmark",
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
    hamiltonian_kwargs = {
        "spectral_weight": args.spectral_weight,
        "use_geometric_correction": not args.disable_geometric_correction,
        "use_memory_correction": not args.disable_memory_correction,
    }
    model, _dataset, results = run_optimizer_comparison(
        config, steps=args.steps, hamiltonian_kwargs=hamiltonian_kwargs
    )

    history_path = args.output_dir / "pinn_training_history.csv"
    summary_path = args.output_dir / "pinn_optimizer_summary.csv"
    convergence_plot_path = args.output_dir / "optimizer_convergence.png"
    convergence_dashboard_path = args.output_dir / "optimizer_convergence_dashboard.html"
    export_training_history(results, str(history_path))
    export_optimizer_summary(results, str(summary_path))
    export_optimizer_convergence_png(results, str(convergence_plot_path))
    export_optimizer_benchmark_html(results, str(convergence_dashboard_path))

    best_result = min(results, key=lambda result: result.final_loss)
    potential_path = args.output_dir / f"{best_result.optimizer}_potential_grid.csv"
    export_potential_grid(model, best_result.parameters, config, str(potential_path))

    print("Physics-informed fringing-field optimizer benchmark")
    print(f"features = {config.hidden_features}")
    print(f"steps = {args.steps}")
    print("optimizer,final_loss,pde_loss,plate_loss,outer_loss,gradient_norm,spectral_entropy")
    for result in results:
        print(
            f"{result.optimizer},"
            f"{result.final_loss:.6e},"
            f"{result.pde_loss:.6e},"
            f"{result.plate_loss:.6e},"
            f"{result.outer_loss:.6e},"
            f"{result.gradient_norm:.6e},"
            f"{result.spectral_entropy:.6e}"
        )
    print(f"best_optimizer = {best_result.optimizer}")
    print(f"exported = {history_path}")
    print(f"exported = {summary_path}")
    print(f"exported = {convergence_plot_path}")
    print(f"exported = {convergence_dashboard_path}")
    print(f"exported = {potential_path}")


if __name__ == "__main__":
    main()
