import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.capacitance import RectangularCapacitor, fringe_ratio, sweep_gap
from src.laplace2d import SolverConfig, export_potential_csv, solve_parallel_plate_2d
from src.measurement import (
    DetectorPlate,
    ExternalInterference,
    export_detector_observations_csv,
    observe_detector_plate,
)
from src.physical_system import (
    DielectricMaterial,
    Environment,
    GeneratorDrive,
    ThermalModel,
    evaluate_physical_state,
)
from src.reporting import write_validation_report
from src.studies import (
    adaptive_numerical_gap_sweep,
    domain_size_study,
    export_convergence_csv,
    export_domain_size_csv,
    export_gap_sweep_csv,
    grid_convergence_study,
    numerical_gap_sweep,
)
from src.visualization import export_fringing_field_png, export_potential_png
from src.visualization3d import Capacitor3DConfig, export_3d_field_animation_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run capacitor fringing baseline models and numerical studies."
    )
    parser.add_argument("--plate-width", type=float, default=0.02)
    parser.add_argument("--plate-length", type=float, default=0.02)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--baseline-gap", type=float, default=0.001)
    parser.add_argument("--domain-width", type=float, default=0.10)
    parser.add_argument("--domain-height", type=float, default=0.10)
    parser.add_argument("--grid", type=int, default=91)
    parser.add_argument("--study-grid", type=int, default=101)
    parser.add_argument("--voltage", type=float, default=1.0)
    parser.add_argument("--relative-permittivity", type=float, default=1.0)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--relative-humidity", type=float, default=0.45)
    parser.add_argument("--pressure-pa", type=float, default=101_325.0)
    parser.add_argument("--frequency-hz", type=float, default=0.0)
    parser.add_argument("--dielectric-loss-tangent", type=float, default=2e-4)
    parser.add_argument("--dielectric-strength", type=float, default=3.0e6)
    parser.add_argument("--volume-resistivity", type=float, default=1.0e14)
    parser.add_argument("--humidity-permittivity-coefficient", type=float, default=0.03)
    parser.add_argument("--temperature-permittivity-coefficient", type=float, default=-4e-4)
    parser.add_argument("--source-resistance", type=float, default=1.0)
    parser.add_argument("--mechanical-power", type=float, default=0.0)
    parser.add_argument("--mechanical-efficiency", type=float, default=1.0)
    parser.add_argument("--friction-coefficient", type=float, default=0.0)
    parser.add_argument("--normal-force", type=float, default=0.0)
    parser.add_argument("--friction-radius", type=float, default=0.0)
    parser.add_argument("--shaft-speed", type=float, default=0.0)
    parser.add_argument("--heat-capacity", type=float, default=12.0)
    parser.add_argument("--thermal-conductance", type=float, default=0.18)
    parser.add_argument("--surface-area", type=float, default=0.01)
    parser.add_argument("--emissivity", type=float, default=0.85)
    parser.add_argument("--device-temperature-c", type=float, default=None)
    parser.add_argument("--detector-center-y", type=float, default=0.012)
    parser.add_argument("--detector-length", type=float, default=0.05)
    parser.add_argument("--detector-samples", type=int, default=121)
    parser.add_argument("--external-field-x", type=float, default=0.0)
    parser.add_argument("--external-field-y", type=float, default=0.0)
    parser.add_argument("--emi-amplitude", type=float, default=0.0)
    parser.add_argument("--emi-spatial-frequency", type=float, default=80.0)
    parser.add_argument("--emi-phase", type=float, default=0.0)
    parser.add_argument("--field-noise-std", type=float, default=0.0)
    parser.add_argument("--field-noise-seed", type=int, default=11)
    parser.add_argument("--field3d-fringe-bulge", type=float, default=1.25)
    parser.add_argument("--field3d-emi-wobble", type=float, default=0.18)
    parser.add_argument("--field3d-animation-speed", type=float, default=1.05)
    parser.add_argument("--field3d-insertion-cycle-s", type=float, default=8.0)
    parser.add_argument("--field3d-chaotic-transient-strength", type=float, default=1.85)
    parser.add_argument("--tolerance", type=float, default=8e-5)
    parser.add_argument("--study-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--method", choices=["jacobi", "sor"], default="sor")
    parser.add_argument("--relaxation", type=float, default=1.7)
    parser.add_argument("--adaptive-cells", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capacitor = RectangularCapacitor(
        length=args.plate_length,
        width=args.plate_width,
        gap=args.baseline_gap,
        relative_permittivity=args.relative_permittivity,
    )
    ratio = fringe_ratio(capacitor)

    print("Baseline rectangular capacitor")
    print(f"length = {capacitor.length:g} m")
    print(f"width  = {capacitor.width:g} m")
    print(f"gap    = {capacitor.gap:g} m")
    print(f"approximate fringing ratio = {ratio:.6f}")
    print()

    print("Gap sweep")
    print("gap_m,ideal_F,fringed_F,ratio")
    rows = sweep_gap(
        length=capacitor.length,
        width=capacitor.width,
        gaps=[
            args.baseline_gap * 0.5,
            args.baseline_gap,
            args.baseline_gap * 2.0,
            args.baseline_gap * 4.0,
        ],
        relative_permittivity=args.relative_permittivity,
    )
    for row in rows:
        print(
            f"{row['gap_m']:.6g},"
            f"{row['ideal_F']:.6e},"
            f"{row['fringed_F']:.6e},"
            f"{row['ratio']:.6f}"
        )

    print()
    print("2D finite-difference field model")
    config = SolverConfig(
        plate_width=args.plate_width,
        gap=args.gap,
        domain_width=args.domain_width,
        domain_height=args.domain_height,
        nx=args.grid,
        ny=args.grid,
        voltage=args.voltage,
        relative_permittivity=args.relative_permittivity,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        method=args.method,
    )
    result = solve_parallel_plate_2d(config)
    output_path = args.output_dir / "potential_grid.csv"
    png_path = args.output_dir / "potential_field.png"
    fringing_png_path = args.output_dir / "fringing_field_lines.png"
    field3d_dir = args.output_dir / "3d_projections"
    field3d_dir.mkdir(parents=True, exist_ok=True)
    field3d_path = field3d_dir / "capacitor_3d_field_animation.html"
    export_potential_csv(result, str(output_path))
    export_potential_png(result, str(png_path))
    export_fringing_field_png(result, str(fringing_png_path))
    export_3d_field_animation_html(
        Capacitor3DConfig(
            plate_length=args.plate_length,
            plate_width=args.plate_width,
            gap=args.gap,
            voltage=args.voltage,
            fringe_bulge=args.field3d_fringe_bulge,
            emi_wobble=args.field3d_emi_wobble,
            animation_speed=args.field3d_animation_speed,
            insertion_cycle_s=args.field3d_insertion_cycle_s,
            chaotic_transient_strength=args.field3d_chaotic_transient_strength,
        ),
        str(field3d_path),
    )
    print(f"converged = {result.converged}")
    print(f"method = {config.method}")
    print(f"relaxation = {config.relaxation:.3f}")
    print(f"iterations = {result.iterations}")
    print(f"max_delta = {result.max_delta:.3e} V")
    print(f"residual_norm = {result.residual_norm:.3e}")
    print(f"requested_gap = {config.gap:.6e} m")
    print(f"grid_electrode_gap = {result.electrode_gap:.6e} m")
    print(f"requested_plate_width = {config.plate_width:.6e} m")
    print(f"grid_electrode_width = {result.electrode_width:.6e} m")
    print(f"energy_capacitance_per_depth = {result.capacitance_per_depth:.6e} F/m")
    print(
        f"charge_capacitance_per_depth = "
        f"{result.charge_capacitance_per_depth:.6e} F/m"
    )
    print(
        f"capacitance_method_relative_difference = "
        f"{result.capacitance_estimate_relative_difference:.6f}"
    )
    print(f"ideal_per_depth_requested_gap = {config.ideal_capacitance_per_depth:.6e} F/m")
    print(f"ideal_per_depth_grid_gap = {result.ideal_capacitance_per_depth:.6e} F/m")
    print(f"energy_fringe_ratio = {result.fringe_ratio:.6f}")
    print(f"charge_fringe_ratio = {result.charge_fringe_ratio:.6f}")
    print(f"exported = {output_path}")
    print(f"exported = {png_path}")
    print(f"exported = {fringing_png_path}")
    print(f"exported = {field3d_path}")

    print()
    print("Physical system observables")
    environment = Environment(
        temperature_c=args.temperature_c,
        relative_humidity=args.relative_humidity,
        pressure_pa=args.pressure_pa,
    )
    dielectric = DielectricMaterial(
        reference_relative_permittivity=args.relative_permittivity,
        temperature_coefficient_per_c=args.temperature_permittivity_coefficient,
        humidity_coefficient=args.humidity_permittivity_coefficient,
        volume_resistivity_ohm_m=args.volume_resistivity,
        loss_tangent=args.dielectric_loss_tangent,
        dielectric_strength_v_per_m=args.dielectric_strength,
    )
    drive = GeneratorDrive(
        voltage=args.voltage,
        frequency_hz=args.frequency_hz,
        source_resistance_ohm=args.source_resistance,
        mechanical_power_w=args.mechanical_power,
        mechanical_efficiency=args.mechanical_efficiency,
        friction_coefficient=args.friction_coefficient,
        shaft_speed_rad_per_s=args.shaft_speed,
        normal_force_n=args.normal_force,
        friction_radius_m=args.friction_radius,
    )
    thermal = ThermalModel(
        heat_capacity_j_per_k=args.heat_capacity,
        thermal_conductance_w_per_k=args.thermal_conductance,
        surface_area_m2=args.surface_area,
        emissivity=args.emissivity,
        ambient_temperature_c=args.temperature_c,
    )
    physical_geometry = RectangularCapacitor(
        length=args.plate_length,
        width=args.plate_width,
        gap=args.gap,
        relative_permittivity=args.relative_permittivity,
    )
    physical_state = evaluate_physical_state(
        physical_geometry,
        environment=environment,
        dielectric=dielectric,
        drive=drive,
        thermal=thermal,
        fringe_multiplier=result.fringe_ratio,
        device_temperature_c=args.device_temperature_c,
    )
    physical_path = args.output_dir / "physical_observables.csv"
    _export_physical_observables_csv(physical_state, str(physical_path))
    print(f"effective_relative_permittivity = {physical_state.relative_permittivity:.6g}")
    print(f"effective_capacitance = {physical_state.effective_capacitance_f:.6e} F")
    print(f"electric_field = {physical_state.electric_field_v_per_m:.6e} V/m")
    print(f"leakage_current = {physical_state.leakage_current_a:.6e} A")
    print(f"total_heat_generation = {physical_state.total_heat_generation_w:.6e} W")
    print(f"heat_dissipation = {physical_state.heat_dissipation_w:.6e} W")
    print(f"temperature_rate = {physical_state.temperature_rate_k_per_s:.6e} K/s")
    print(f"breakdown_margin = {physical_state.breakdown_margin:.6g}")
    print(f"exported = {physical_path}")

    print()
    print("Detector/photo-plate field observations")
    detector = DetectorPlate(
        center_y=args.detector_center_y,
        length=args.detector_length,
        samples=args.detector_samples,
    )
    interference = ExternalInterference(
        dc_field_x_v_per_m=args.external_field_x,
        dc_field_y_v_per_m=args.external_field_y,
        emi_amplitude_v_per_m=args.emi_amplitude,
        emi_spatial_frequency_per_m=args.emi_spatial_frequency,
        emi_phase_rad=args.emi_phase,
        stochastic_std_v_per_m=args.field_noise_std,
        seed=args.field_noise_seed,
    )
    detector_rows, detector_summary = observe_detector_plate(
        result,
        detector,
        interference=interference,
        environment=environment,
    )
    detector_path = args.output_dir / "detector_field_observations.csv"
    detector_summary_path = args.output_dir / "detector_fringing_summary.csv"
    export_detector_observations_csv(
        detector_rows,
        detector_summary,
        str(detector_path),
        str(detector_summary_path),
    )
    print(f"detector_max_field = {detector_summary.max_magnitude_v_per_m:.6e} V/m")
    print(f"detector_min_field = {detector_summary.min_magnitude_v_per_m:.6e} V/m")
    print(f"detector_peak_to_peak = {detector_summary.peak_to_peak_magnitude_v_per_m:.6e} V/m")
    print(f"detector_direction_std = {detector_summary.direction_std_deg:.6e} deg")
    print(f"fringing_tangential_ratio = {detector_summary.fringing_tangential_ratio:.6e}")
    print(f"local_maxima_count = {detector_summary.local_maxima_count}")
    print(f"local_minima_count = {detector_summary.local_minima_count}")
    print(f"exported = {detector_path}")
    print(f"exported = {detector_summary_path}")

    print()
    print("Grid convergence study")
    convergence_domain = min(args.domain_width, args.domain_height) * 0.8
    convergence_grids = _study_grid_sequence(args.study_grid)
    convergence_config = SolverConfig(
        plate_width=args.plate_width,
        gap=args.gap,
        domain_width=convergence_domain,
        domain_height=convergence_domain,
        voltage=args.voltage,
        relative_permittivity=args.relative_permittivity,
        tolerance=args.study_tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        method=args.method,
    )
    convergence_rows = grid_convergence_study(convergence_config, convergence_grids)
    convergence_path = args.output_dir / "grid_convergence.csv"
    export_convergence_csv(convergence_rows, str(convergence_path))
    print(
        "nx,residual_norm,gap_error_m,plate_width_error_m,capacitance_per_depth,fringe_ratio,"
        "method_relative_difference,relative_change_from_previous"
    )
    for row in convergence_rows:
        relative_change = (
            "n/a"
            if row.relative_change_from_previous is None
            else f"{row.relative_change_from_previous:.6f}"
        )
        print(
            f"{row.nx},"
            f"{row.residual_norm:.3e},"
            f"{row.gap_error:.3e},"
            f"{row.plate_width_error:.3e},"
            f"{row.capacitance_per_depth:.6e},"
            f"{row.fringe_ratio:.6f},"
            f"{row.capacitance_estimate_relative_difference:.6f},"
            f"{relative_change}"
        )
    print(f"exported = {convergence_path}")

    print()
    print("Domain-size sensitivity study")
    base_domain = min(args.domain_width, args.domain_height)
    domain_sizes = [base_domain * 0.6, base_domain * 0.8, base_domain]
    domain_config = SolverConfig(
        plate_width=args.plate_width,
        gap=args.gap,
        domain_width=base_domain * 0.8,
        domain_height=base_domain * 0.8,
        nx=max(5, int(round(args.study_grid * 0.8))),
        ny=max(5, int(round(args.study_grid * 0.8))),
        voltage=args.voltage,
        relative_permittivity=args.relative_permittivity,
        tolerance=args.study_tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        method=args.method,
    )
    domain_rows = domain_size_study(domain_config, domain_sizes)
    domain_path = args.output_dir / "domain_size_study.csv"
    export_domain_size_csv(domain_rows, str(domain_path))
    print(
        "domain_width_m,nx,residual_norm,gap_error_m,capacitance_per_depth,fringe_ratio,"
        "method_relative_difference,relative_change_from_previous"
    )
    for row in domain_rows:
        relative_change = (
            "n/a"
            if row.relative_change_from_previous is None
            else f"{row.relative_change_from_previous:.6f}"
        )
        print(
            f"{row.domain_width:.3f},"
            f"{row.nx},"
            f"{row.residual_norm:.3e},"
            f"{row.gap_error:.3e},"
            f"{row.capacitance_per_depth:.6e},"
            f"{row.fringe_ratio:.6f},"
            f"{row.capacitance_estimate_relative_difference:.6f},"
            f"{relative_change}"
        )
    print(f"exported = {domain_path}")

    print()
    print("Numerical gap sweep")
    gap_sweep_values = [args.gap * 0.5, args.gap, args.gap * 1.5, args.gap * 2.0]
    gap_config = SolverConfig(
        plate_width=args.plate_width,
        gap=args.gap,
        domain_width=args.domain_width,
        domain_height=args.domain_height,
        nx=args.study_grid,
        ny=args.study_grid,
        voltage=args.voltage,
        relative_permittivity=args.relative_permittivity,
        tolerance=args.study_tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        method=args.method,
    )
    gap_rows = numerical_gap_sweep(gap_config, gap_sweep_values)
    gap_path = args.output_dir / "numerical_gap_sweep.csv"
    export_gap_sweep_csv(gap_rows, str(gap_path))
    print(
        "gap_m,gap_to_width,residual_norm,energy_fringe_ratio,charge_fringe_ratio,"
        "effective_area_fringe_ratio,method_relative_difference"
    )
    for row in gap_rows:
        print(
            f"{row.requested_gap:.4f},"
            f"{row.gap_to_width:.3f},"
            f"{row.residual_norm:.3e},"
            f"{row.energy_fringe_ratio:.6f},"
            f"{row.charge_fringe_ratio:.6f},"
            f"{row.effective_area_fringe_ratio:.6f},"
            f"{row.capacitance_estimate_relative_difference:.6f}"
        )
    print(f"exported = {gap_path}")

    print()
    print("Adaptive numerical gap sweep")
    adaptive_gap_rows = adaptive_numerical_gap_sweep(
        gap_config,
        gap_sweep_values,
        min_cells_across_smallest_gap=args.adaptive_cells,
    )
    adaptive_gap_path = args.output_dir / "adaptive_gap_sweep.csv"
    export_gap_sweep_csv(adaptive_gap_rows, str(adaptive_gap_path))
    print(
        "gap_m,gap_to_width,nx,residual_norm,energy_fringe_ratio,charge_fringe_ratio,"
        "effective_area_fringe_ratio,method_relative_difference"
    )
    for row in adaptive_gap_rows:
        print(
            f"{row.requested_gap:.4f},"
            f"{row.gap_to_width:.3f},"
            f"{row.nx},"
            f"{row.residual_norm:.3e},"
            f"{row.energy_fringe_ratio:.6f},"
            f"{row.charge_fringe_ratio:.6f},"
            f"{row.effective_area_fringe_ratio:.6f},"
            f"{row.capacitance_estimate_relative_difference:.6f}"
        )
    print(f"exported = {adaptive_gap_path}")

    report_path = args.output_dir / "validation_report.md"
    write_validation_report(
        str(report_path),
        result,
        convergence_rows,
        domain_rows,
        adaptive_gap_rows,
    )
    print(f"exported = {report_path}")


def _study_grid_sequence(center_grid: int) -> list[int]:
    center_grid = max(9, center_grid)
    coarse = max(5, center_grid - 20)
    middle = center_grid + 20
    fine = center_grid + 60
    grids = sorted({coarse, middle, fine})
    if len(grids) < 2:
        grids.append(grids[-1] + 40)
    return grids


def _export_physical_observables_csv(state, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("observable,value\n")
        for name, value in state.__dict__.items():
            file.write(f"{name},{value:.12g}\n")


if __name__ == "__main__":
    main()
