"""Loader and 3D visualizations for the real 3D capacitor solve.

Unlike src/visualization3d.py (a hand-animated visual prototype with no
physics behind it) and src/laplace2d.py (a 2D cross-section that assumes an
infinitely long plate), this module consumes the output of
main/mathematica/capacitor_3d_solve.wls -- a genuine 3D finite-difference
solve of Laplace's equation for a finite rectangular parallel-plate
capacitor, run through the Wolfram Engine -- and turns it into 3D matplotlib
plots of potential, electric field, and capacitance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PLATE_POSITIVE_COLOR = "#c0392b"
PLATE_NEGATIVE_COLOR = "#2f5fa8"


@dataclass(frozen=True)
class Capacitor3DSolution:
    """A solved 3D potential/field grid plus its run summary."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    potential: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    is_electrode: np.ndarray
    summary: dict


def load_capacitor_3d_solution(data_dir: str) -> Capacitor3DSolution:
    """Load field_grid.csv, potential_slices.csv metadata, and
    capacitance_summary.json written by capacitor_3d_solve.wls."""

    directory = Path(data_dir)
    with open(directory / "capacitance_summary.json", "r", encoding="utf-8") as file:
        summary = json.load(file)

    nx = int(summary["grid_nx"])
    ny = int(summary["grid_ny"])
    nz = int(summary["grid_nz"])

    rows: list[list[float]] = []
    with open(directory / "field_grid.csv", "r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            # Wolfram's InputForm writes scientific notation as "*^-6"
            # rather than "e-6"; translate before parsing as float.
            rows.append([float(value.replace("*^", "e")) for value in row])

    data = np.array(rows, dtype=float)
    expected_rows = nx * ny * nz
    if data.shape[0] != expected_rows:
        raise ValueError(
            f"field_grid.csv has {data.shape[0]} rows, expected nx*ny*nz = {expected_rows}"
        )

    shape = (nx, ny, nz)
    x_grid, y_grid, z_grid, phi, ex, ey, ez, fixed = (
        column.reshape(shape) for column in data.T
    )

    return Capacitor3DSolution(
        x=x_grid[:, 0, 0],
        y=y_grid[0, :, 0],
        z=z_grid[0, 0, :],
        potential=phi,
        ex=ex,
        ey=ey,
        ez=ez,
        is_electrode=fixed > 0.5,
        summary=summary,
    )


def _plate_polygon(solution: Capacitor3DSolution, z_value: float) -> np.ndarray:
    length = solution.summary["electrode_length_m"]
    width = solution.summary["electrode_width_m"]
    hx, hy = length / 2.0, width / 2.0
    return np.array(
        [
            [-hx, -hy, z_value],
            [hx, -hy, z_value],
            [hx, hy, z_value],
            [-hx, hy, z_value],
        ]
    )


def _draw_plates_3d(ax: "plt.Axes", solution: Capacitor3DSolution) -> None:
    gap = solution.summary["electrode_gap_m"]
    top = Poly3DCollection([_plate_polygon(solution, gap / 2.0)], alpha=0.55)
    top.set_facecolor(PLATE_POSITIVE_COLOR)
    top.set_edgecolor("black")
    bottom = Poly3DCollection([_plate_polygon(solution, -gap / 2.0)], alpha=0.55)
    bottom.set_facecolor(PLATE_NEGATIVE_COLOR)
    bottom.set_edgecolor("black")
    ax.add_collection3d(top)
    ax.add_collection3d(bottom)


def _set_physical_box_aspect(ax: "plt.Axes", solution: Capacitor3DSolution) -> None:
    span_x = solution.x.max() - solution.x.min()
    span_y = solution.y.max() - solution.y.min()
    span_z = solution.z.max() - solution.z.min()
    ax.set_box_aspect((span_x, span_y, max(span_z, 0.18 * max(span_x, span_y))))


def export_3d_potential_png(solution: Capacitor3DSolution, path: str, dpi: int = 200) -> None:
    """Export a 3D view of the potential field as stacked filled-contour
    slices at several heights, from the electrode plane out into the
    fringing region above it, plus the capacitance computed from this same
    3D solve."""

    x, y, z, phi = solution.x, solution.y, solution.z, solution.potential
    view_x = min(x.max(), solution.summary["electrode_length_m"] * 1.9)
    view_y = min(y.max(), solution.summary["electrode_width_m"] * 1.9)
    x_mask = np.abs(x) <= view_x
    y_mask = np.abs(y) <= view_y
    x_view, y_view = x[x_mask], y[y_mask]
    xg, yg = np.meshgrid(x_view, y_view, indexing="ij")

    # Slices are kept clear of z = +-gap/2 on purpose: that is exactly where
    # the opaque plate polygons are drawn, and a filled contour at the same
    # height as an opaque plate either hides it or fights it for the same
    # depth in the 3D render. Placing slices strictly outside the gap lets
    # the plates themselves show the polarity, and the slices show how the
    # field spreads out above/below/beside them.
    gap = solution.summary["electrode_gap_m"]
    z_targets = [-gap * 2.5, 0.0, gap * 2.5, gap * 5.0]
    z_indices = sorted({int(np.argmin(np.abs(z - target))) for target in z_targets})

    vmax = float(np.percentile(np.abs(phi), 99.0))
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(9.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    for z_index in z_indices:
        slice_values = phi[np.ix_(x_mask, y_mask, [z_index])][:, :, 0]
        ax.contourf(
            xg, yg, slice_values, zdir="z", offset=float(z[z_index]),
            levels=18, cmap=cmap, norm=norm, alpha=0.85,
        )

    _draw_plates_3d(ax, solution)
    _set_physical_box_aspect(ax, solution)
    ax.set_zlim(z[z_indices[0]] - gap * 0.5, z[z_indices[-1]] + gap * 0.5)
    ax.set_xlim(-view_x, view_x)
    ax.set_ylim(-view_y, view_y)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D capacitor potential field (stacked slices)")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="Potential (V)")

    capacitance = solution.summary["capacitance_3d_F"]
    ideal = solution.summary["ideal_capacitance_F"]
    fringe_ratio = solution.summary["fringe_ratio_3d"]
    ax.text2D(
        0.02, 0.02,
        f"C (3D solve) = {capacitance * 1e12:.4f} pF\n"
        f"C (ideal, no fringing) = {ideal * 1e12:.4f} pF\n"
        f"fringe ratio = {fringe_ratio:.3f}",
        transform=ax.transAxes, fontsize=8.5,
        bbox={"boxstyle": "round, pad=0.35", "facecolor": "white", "edgecolor": "#888888", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_3d_field_quiver_png(
    solution: Capacitor3DSolution, path: str, dpi: int = 200,
    target_x: int = 11, target_y: int = 9, target_z: int = 7,
) -> None:
    """Export a 3D quiver plot of the electric field around the plates,
    colored by field magnitude, subsampled to roughly target_x/y/z arrows
    per axis regardless of the source grid's resolution. The view is
    cropped close to the plate footprint (rather than the whole solve box)
    so the near-edge fringing bow is legible instead of being buried under
    a dense curtain of far-field return arrows."""

    x, y, z = solution.x, solution.y, solution.z
    view_x = min(x.max(), solution.summary["electrode_length_m"] * 0.85)
    view_y = min(y.max(), solution.summary["electrode_width_m"] * 0.95)
    gap = solution.summary["electrode_gap_m"]
    view_z = gap * 2.2

    x_mask = np.abs(x) <= view_x
    y_mask = np.abs(y) <= view_y
    z_mask = np.abs(z) <= view_z
    x_idx_all = np.nonzero(x_mask)[0]
    y_idx_all = np.nonzero(y_mask)[0]
    z_idx_all = np.nonzero(z_mask)[0]
    xi = x_idx_all[:: max(1, len(x_idx_all) // target_x)]
    yi = y_idx_all[:: max(1, len(y_idx_all) // target_y)]
    zi = z_idx_all[:: max(1, len(z_idx_all) // target_z)]

    xg, yg, zg = np.meshgrid(x[xi], y[yi], z[zi], indexing="ij")
    ex = solution.ex[np.ix_(xi, yi, zi)]
    ey = solution.ey[np.ix_(xi, yi, zi)]
    ez = solution.ez[np.ix_(xi, yi, zi)]
    magnitude = np.sqrt(ex**2 + ey**2 + ez**2)

    color_scale = float(np.percentile(magnitude, 97.0))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max(color_scale, 1e-30))
    cmap = plt.get_cmap("viridis")
    arrow_colors = cmap(norm(magnitude.ravel()))
    # quiver3D draws each arrow as 3 line segments (shaft + 2 head lines) and
    # expects one color per segment, in [body..., body..., body...] order.
    segment_colors = np.concatenate([arrow_colors, np.repeat(arrow_colors, 2, axis=0)])

    fig = plt.figure(figsize=(9.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(
        xg, yg, zg, ex, ey, ez,
        length=0.55 * (x[xi[1]] - x[xi[0]] if len(xi) > 1 else view_x * 0.05),
        normalize=True, colors=segment_colors, linewidth=1.1, arrow_length_ratio=0.35,
    )

    _draw_plates_3d(ax, solution)
    _set_physical_box_aspect(ax, solution)
    ax.set_xlim(-view_x, view_x)
    ax.set_ylim(-view_y, view_y)
    ax.set_zlim(-view_z, view_z)
    ax.view_init(elev=18, azim=-52)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D electric field around the finite plates")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="|E| (V/m, clipped at the 97th percentile)")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_capacitance_comparison_png(solution: Capacitor3DSolution, path: str, dpi: int = 200) -> None:
    """Export a bar chart comparing the 3D solve's capacitance against the
    ideal (no-fringing) formula and the project's analytic fringing
    heuristic (src/capacitance.py)."""

    from src.capacitance import RectangularCapacitor, effective_area_fringe, ideal_parallel_plate

    summary = solution.summary
    capacitor = RectangularCapacitor(
        length=summary["plate_length_m"], width=summary["plate_width_m"], gap=summary["gap_m"]
    )
    ideal = ideal_parallel_plate(capacitor)
    heuristic = effective_area_fringe(capacitor)
    solved_3d = summary["capacitance_3d_F"]

    labels = ["ideal\n(no fringing)", "analytic heuristic\n(effective-area)", "Mathematica 3D solve\n(this project)"]
    values_pf = [ideal * 1e12, heuristic * 1e12, solved_3d * 1e12]
    colors = ["#9aa3ab", "#e8a33d", "#c0392b"]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    bars = ax.bar(labels, values_pf, color=colors, width=0.55)
    for bar, value in zip(bars, values_pf):
        ax.annotate(
            f"{value:.4f} pF", xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold",
        )
    ax.set_ylabel("Capacitance (pF)")
    ax.set_title(
        f"Capacitance estimates -- {summary['plate_length_m']*1000:.0f}x{summary['plate_width_m']*1000:.0f} mm "
        f"plates, {summary['gap_m']*1000:.1f} mm gap"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_3d_solution_animation_html(
    solution: Capacitor3DSolution,
    path: str,
    max_lines: int = 96,
    points_per_line: int = 120,
) -> None:
    """Export an animated browser view traced from the generated 3D solve.

    The field lines are not hand-shaped. They are streamlines integrated
    through the `Ex,Ey,Ez` values loaded from `field_grid.csv`, using trilinear
    interpolation between grid samples.
    """

    lines = _trace_solution_field_lines(solution, max_lines=max_lines, points_per_line=points_per_line)
    payload = {
        "title": "3D capacitor field from generated finite-difference solve",
        "summary": solution.summary,
        "lines": lines,
        "external_lines": _solution_external_field_lines(solution),
        "plates": {
            "positive": _plate_polygon(solution, solution.summary["electrode_gap_m"] / 2.0).tolist(),
            "negative": _plate_polygon(solution, -solution.summary["electrode_gap_m"] / 2.0).tolist(),
        },
        "bounds": {
            "x": [float(solution.x.min()), float(solution.x.max())],
            "y": [float(solution.y.min()), float(solution.y.max())],
            "z": [float(solution.z.min()), float(solution.z.max())],
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        file.write(_solution_animation_html(payload))


def _trace_solution_field_lines(
    solution: Capacitor3DSolution,
    max_lines: int,
    points_per_line: int,
) -> list[list[list[float]]]:
    summary = solution.summary
    length = float(summary["electrode_length_m"])
    width = float(summary["electrode_width_m"])
    gap = float(summary["electrode_gap_m"])
    x_count = max(5, int(np.sqrt(max_lines * length / max(width, 1e-30))))
    y_count = max(5, int(max_lines / x_count))
    x_values = np.linspace(-0.47 * length, 0.47 * length, x_count)
    y_values = np.linspace(-0.47 * width, 0.47 * width, y_count)

    seeds: list[tuple[float, float]] = []
    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            jitter_x = 0.006 * length * np.sin(1.7 * xi + 0.6 * yi)
            jitter_y = 0.006 * width * np.cos(1.2 * yi - 0.4 * xi)
            seeds.append((float(x + jitter_x), float(y + jitter_y)))

    edge_offsets = np.linspace(-0.5, 0.5, max(8, min(18, max_lines // 8)))
    for side in (-1.0, 1.0):
        for offset in edge_offsets:
            seeds.append((float(side * 0.505 * length), float(offset * width)))
            seeds.append((float(offset * length), float(side * 0.505 * width)))

    lines = []
    start_z = gap / 2.0 - 0.18 * gap
    for x0, y0 in seeds[:max_lines]:
        line = _trace_one_solution_line(
            solution,
            np.array([x0, y0, start_z], dtype=float),
            points_per_line=points_per_line,
        )
        if len(line) >= 4:
            lines.append(line)
    return lines


def _solution_external_field_lines(solution: Capacitor3DSolution) -> list[list[list[float]]]:
    """Simple external/background field guide lines around the capacitor.

    The external field is drawn as a weak oblique perturbation mostly aligned
    with the capacitor's plate-to-plate field direction. A purely transverse
    guide field would look orthogonal to the solved capacitor field and imply a
    different physical setup.
    """

    summary = solution.summary
    length = float(summary["electrode_length_m"])
    width = float(summary["electrode_width_m"])
    gap = float(summary["electrode_gap_m"])
    x_values = np.array([-1.08, -0.78, -0.50, -0.24, 0.24, 0.50, 0.78, 1.08]) * length
    y_values = np.linspace(-0.94 * width, 0.94 * width, 5)
    z_top = min(float(solution.z.max()) * 0.72, 3.0 * gap)
    z_bottom = max(float(solution.z.min()) * 0.72, -3.0 * gap)
    lines: list[list[list[float]]] = []
    for xi, x in enumerate(x_values):
        for yi, y in enumerate(y_values):
            line: list[list[float]] = []
            phase = 0.8 * yi + 1.3 * xi
            for t in np.linspace(0.0, 1.0, 48):
                s = 2.0 * t - 1.0
                near_plate = np.exp(-((abs(s) - 0.18) / 0.56) ** 2)
                convergence = 0.22 * near_plate
                z = z_top * (1.0 - t) + z_bottom * t
                # As the external/background guide approaches the electrode
                # region, bend it toward the finite plate footprint. This
                # visually communicates induction/boundary-condition bending
                # without pretending these gray guides are the solved field.
                x_bent = x * (1.0 - convergence)
                y_bent = y * (1.0 - 0.72 * convergence)
                lateral_drift = 0.10 * gap * s
                wobble = 0.018 * width * np.sin(2.0 * np.pi * t + phase)
                line.append([float(x_bent + lateral_drift), float(y_bent + wobble), float(z)])
            lines.append(line)
    return lines


def _trace_one_solution_line(
    solution: Capacitor3DSolution,
    start: np.ndarray,
    points_per_line: int,
) -> list[list[float]]:
    gap = float(solution.summary["electrode_gap_m"])
    step = 0.38 * min(
        np.min(np.diff(solution.x)),
        np.min(np.diff(solution.y)),
        np.min(np.diff(solution.z)),
    )
    max_steps = max(points_per_line * 8, 300)
    point = start.copy()
    points = [point.copy()]

    for _ in range(max_steps):
        field = _interpolate_solution_field(solution, point)
        norm = float(np.linalg.norm(field))
        if not np.isfinite(norm) or norm < 1e-18:
            break
        point = point + step * field / norm
        if not _inside_solution_box(solution, point):
            break
        points.append(point.copy())
        if point[2] <= -gap / 2.0 + 0.18 * gap:
            break
    if len(points) < 4:
        return []
    return _resample_solution_polyline(points, points_per_line)


def _interpolate_solution_field(solution: Capacitor3DSolution, point: np.ndarray) -> np.ndarray:
    x, y, z = solution.x, solution.y, solution.z
    ix, fx = _axis_index_fraction(x, point[0])
    iy, fy = _axis_index_fraction(y, point[1])
    iz, fz = _axis_index_fraction(z, point[2])
    field = np.empty(3, dtype=float)
    for component_index, grid in enumerate((solution.ex, solution.ey, solution.ez)):
        c000 = grid[ix, iy, iz]
        c100 = grid[ix + 1, iy, iz]
        c010 = grid[ix, iy + 1, iz]
        c110 = grid[ix + 1, iy + 1, iz]
        c001 = grid[ix, iy, iz + 1]
        c101 = grid[ix + 1, iy, iz + 1]
        c011 = grid[ix, iy + 1, iz + 1]
        c111 = grid[ix + 1, iy + 1, iz + 1]
        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        field[component_index] = c0 * (1.0 - fz) + c1 * fz
    return field


def _axis_index_fraction(axis: np.ndarray, value: float) -> tuple[int, float]:
    if value <= axis[0]:
        return 0, 0.0
    if value >= axis[-1]:
        return len(axis) - 2, 1.0
    index = int(np.searchsorted(axis, value) - 1)
    index = max(0, min(index, len(axis) - 2))
    span = axis[index + 1] - axis[index]
    fraction = 0.0 if span == 0 else (value - axis[index]) / span
    return index, float(fraction)


def _inside_solution_box(solution: Capacitor3DSolution, point: np.ndarray) -> bool:
    return (
        solution.x[0] <= point[0] <= solution.x[-1]
        and solution.y[0] <= point[1] <= solution.y[-1]
        and solution.z[0] <= point[2] <= solution.z[-1]
    )


def _resample_solution_polyline(points: list[np.ndarray], count: int) -> list[list[float]]:
    array = np.array(points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-18:
        return np.repeat(array[:1], count, axis=0).tolist()
    targets = np.linspace(0.0, total, count)
    resampled = np.empty((count, 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(targets, cumulative, array[:, dim])
    return resampled.tolist()


def _solution_animation_html(payload: dict) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{payload["title"]}</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #090d12; color: #e6edf3; font-family: Arial, sans-serif; }}
    canvas {{ width: 100vw; height: 100vh; display: block; background: radial-gradient(circle at 48% 42%, #172236 0%, #090d12 68%); }}
    .hud {{ position: fixed; left: 16px; top: 14px; max-width: 430px; padding: 11px 13px; border-radius: 8px; background: rgba(9,13,18,.76); border: 1px solid rgba(230,237,243,.18); font-size: 13px; line-height: 1.35; }}
    .legend {{ position: fixed; right: 16px; bottom: 14px; padding: 10px 12px; border-radius: 8px; background: rgba(9,13,18,.76); border: 1px solid rgba(230,237,243,.18); font-size: 13px; }}
    .gold {{ color: #ffd45c; font-weight: 700; }}
  </style>
</head>
<body>
<canvas id="scene"></canvas>
<div class="hud">
  <strong>3D field from your generated Wolfram finite-difference solve</strong><br>
  <span class="gold">Streamlines are integrated from field_grid.csv, not hand drawn.</span><br>
  C = {(payload["summary"]["capacitance_3d_F"] * 1e12):.4f} pF,
  fringe ratio = {payload["summary"]["fringe_ratio_3d"]:.3f}<br>
  drag to rotate
</div>
<div class="legend">red: +V electrode<br>blue: -V electrode<br>cyan/green: solved capacitor field<br>gray: weak external/background field<br>gold arrows: field direction<br>drag to rotate</div>
<script>
const payload = {data};
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
let yaw = -0.78, pitch = 0.56, dragging = false, lastX = 0, lastY = 0;

function resize() {{ canvas.width = Math.floor(innerWidth * devicePixelRatio); canvas.height = Math.floor(innerHeight * devicePixelRatio); }}
addEventListener("resize", resize); resize();
canvas.addEventListener("pointerdown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); }});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("pointermove", e => {{ if (!dragging) return; yaw += (e.clientX-lastX)*.008; pitch += (e.clientY-lastY)*.008; pitch = Math.max(-1.25, Math.min(1.25, pitch)); lastX=e.clientX; lastY=e.clientY; }});

function rotate(p) {{
  const [x,y,z] = p, cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy*x + sy*z, z1 = -sy*x + cy*z, y1 = cp*y - sp*z1, z2 = sp*y + cp*z1;
  return [x1,y1,z2];
}}
function project(p) {{
  const [x,y,z] = rotate(p);
  const scale = Math.min(canvas.width, canvas.height) * 14.5;
  const depth = 1 / (1 + 9.5*z);
  return [canvas.width*.5 + x*scale*depth, canvas.height*.52 - y*scale*depth, depth];
}}
function drawPoly(poly, fill, stroke, alpha=.75) {{
  const pts = poly.map(project);
  ctx.globalAlpha = alpha; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i=1; i<pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 1.5*devicePixelRatio; ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
}}
function drawLine(line, color, width, alpha) {{
  const first = project(line[0]); ctx.globalAlpha = alpha; ctx.beginPath(); ctx.moveTo(first[0], first[1]);
  for (let i=1; i<line.length; i++) {{ const p = project(line[i]); ctx.lineTo(p[0], p[1]); }}
  ctx.strokeStyle = color; ctx.lineWidth = width*devicePixelRatio; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke(); ctx.globalAlpha = 1;
}}
function sample(line, t) {{
  const raw = t*(line.length-1), i = Math.floor(raw), j = Math.min(line.length-1, i+1), f = raw-i;
  return [line[i][0]*(1-f)+line[j][0]*f, line[i][1]*(1-f)+line[j][1]*f, line[i][2]*(1-f)+line[j][2]*f];
}}
function drawArrowOnLine(line, t, color, alpha) {{
  const p0 = project(sample(line, Math.max(0, t - .012)));
  const p1 = project(sample(line, Math.min(1, t + .012)));
  const dx = p1[0]-p0[0], dy = p1[1]-p0[1], len = Math.hypot(dx, dy);
  if (len < 1e-6) return;
  const ux = dx/len, uy = dy/len, size = 6.0 * devicePixelRatio;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(p1[0], p1[1]);
  ctx.lineTo(p1[0]-ux*size-uy*size*.52, p1[1]-uy*size+ux*size*.52);
  ctx.lineTo(p1[0]-ux*size+uy*size*.52, p1[1]-uy*size-ux*size*.52);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}}
function drawGrid() {{
  const sx = payload.summary.electrode_length_m*.95, sy = payload.summary.electrode_width_m*.95, z = -payload.summary.electrode_gap_m*2.4;
  ctx.globalAlpha=.16; ctx.strokeStyle="rgba(170,190,225,.45)"; ctx.lineWidth=.8*devicePixelRatio;
  for (let i=-5;i<=5;i++) {{
    let a=project([-sx, i*sy/5, z]), b=project([sx, i*sy/5, z]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
    a=project([i*sx/5, -sy, z]); b=project([i*sx/5, sy, z]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
  }}
  ctx.globalAlpha=1;
}}
function renderScene(ms) {{
  ctx.clearRect(0,0,canvas.width,canvas.height); drawGrid();
  drawPoly(payload.plates.negative, "rgba(45,119,255,.42)", "rgba(119,173,255,.95)", .92);
  drawPoly(payload.plates.positive, "rgba(255,76,76,.42)", "rgba(255,145,145,.98)", .92);
  for (let i=0; i<payload.external_lines.length; i++) {{
    const line = payload.external_lines[i];
    drawLine(line, "rgba(175,188,205,.30)", 1.05, 1);
  }}
  for (let i=0; i<payload.lines.length; i++) {{
    const line = payload.lines[i];
    const edge = Math.max(...line.map(p => Math.hypot(p[0], p[1]))) > payload.summary.electrode_length_m*.43;
    const color = edge ? "rgba(102,227,164,.68)" : "rgba(105,208,255,.56)";
    drawLine(line, color, edge ? 1.65 : 1.25, 1);
    drawLine(line, color, edge ? 4.2 : 3.4, .13);
  }}
  const base = ((ms || 0) * .00023) % 1;
  for (let i=0; i<payload.external_lines.length; i++) {{
    drawArrowOnLine(payload.external_lines[i], (base + (i % 9) / 9) % 1, "rgba(210,220,235,.72)", .72);
  }}
  for (let i=0; i<payload.lines.length; i += 3) {{
    drawArrowOnLine(payload.lines[i], (base * 1.35 + (i % 17) / 17) % 1, "rgba(255,214,92,.88)", .86);
  }}
  requestAnimationFrame(renderScene);
}}
requestAnimationFrame(renderScene);
</script>
</body>
</html>
"""
