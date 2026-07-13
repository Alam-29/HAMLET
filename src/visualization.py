"""Matplotlib-based PNG visualization exports for solved capacitor fields."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.laplace2d import SolverResult, electric_field

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.color": "#dcdcdc",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "DejaVu Sans",
    }
)

# One consistent color/line language across every plot in this module:
# unpreconditioned optimizers are cool grays/blues with dashed lines (they
# barely move on the ill-conditioned benchmark loss); metric-preconditioned
# optimizers are warm, thicker, solid lines (they exploit the metric to make
# real progress). Adam sits outside the preconditioning story as its own hue.
OPTIMIZER_COLORS = {
    "sgd": "#9aa3ab",
    "falling_ball": "#4f83b5",
    "adam": "#e8a33d",
    "entropy_descent": "#c0392b",
    "hamiltonian_geometric": "#6a3d9a",
}
OPTIMIZER_STYLES = {
    "sgd": {"linestyle": (0, (5, 2)), "linewidth": 1.6},
    "falling_ball": {"linestyle": (0, (1, 1)), "linewidth": 1.8},
    "adam": {"linestyle": "-", "linewidth": 1.5},
    "entropy_descent": {"linestyle": "-", "linewidth": 2.4},
    "hamiltonian_geometric": {"linestyle": "-", "linewidth": 2.4},
}
PLAIN_COLOR = OPTIMIZER_COLORS["sgd"]
PRECONDITIONED_COLOR = OPTIMIZER_COLORS["entropy_descent"]


def _strip_top_right(ax: "plt.Axes") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def export_optimizer_convergence_png(results, path: str, dpi: int = 200) -> None:
    """Export a log-scale loss-vs-step convergence plot for a list of
    `src.pinn.TrainingResult`, one line per optimizer, colored and styled so
    unpreconditioned vs. metric-preconditioned optimizers are visually
    unmistakable, with each line's final loss labeled directly on the plot."""

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    for result in results:
        steps = range(1, len(result.loss_history) + 1)
        color = OPTIMIZER_COLORS.get(result.optimizer, "#333333")
        style = OPTIMIZER_STYLES.get(result.optimizer, {"linestyle": "-", "linewidth": 1.6})
        ax.plot(steps, result.loss_history, label=result.optimizer, color=color, **style)

    # Stagger end-of-line labels that land within ~1% of each other in log
    # space (e.g. sgd and falling_ball both stall near the initial loss), so
    # every label stays legible instead of overlapping.
    ordered = sorted(results, key=lambda result: result.final_loss)
    last_log_loss = None
    stack_offset = 0.0
    for result in ordered:
        log_loss = np.log10(max(result.final_loss, 1e-300))
        stack_offset = stack_offset + 13.0 if last_log_loss is not None and log_loss - last_log_loss < 0.004 else 0.0
        last_log_loss = log_loss
        color = OPTIMIZER_COLORS.get(result.optimizer, "#333333")
        ax.annotate(
            f"{result.final_loss:.2f}",
            xy=(len(result.loss_history), result.final_loss),
            xytext=(6, stack_offset),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            va="center",
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Optimizer convergence on the capacitor fringing-field PINN loss")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    _strip_top_right(ax)
    ax.legend(loc="lower left", ncols=1)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_normal_mode_spectrum_png(modes, conditioning: dict, path: str, dpi: int = 200) -> None:
    """Export the loss Hessian's eigenvalue spectrum and per-mode contraction
    rates for plain vs. metric-preconditioned descent (`src.normal_modes`),
    with the gap between the two rate curves shaded to make the conditioning-
    independence claim visually immediate."""

    eigenvalues = modes.eigenvalues
    nonzero_mask = eigenvalues > 1e-9 * max(eigenvalues.max(), 1.0)
    order = np.argsort(eigenvalues[nonzero_mask])
    plotted_eigenvalues = eigenvalues[nonzero_mask][order]
    metric_eigenvalues = modes.metric_eigenvalues[nonzero_mask][order]
    index = np.arange(1, plotted_eigenvalues.size + 1)

    eta_plain = conditioning["eta_max_stable_plain"]
    eta_precond = conditioning["learning_rate_preconditioned"]
    plain_rate = np.abs(1.0 - 2.0 * eta_plain * plotted_eigenvalues)
    precond_rate = np.abs(1.0 - 2.0 * eta_precond * plotted_eigenvalues / metric_eigenvalues)

    fig, (ax_spectrum, ax_rate) = plt.subplots(1, 2, figsize=(12.0, 5.2))

    ax_spectrum.semilogy(
        index, plotted_eigenvalues, marker="o", markersize=3.5, linewidth=1.3,
        color="#4f83b5", markerfacecolor="#4f83b5", markeredgecolor="white", markeredgewidth=0.4,
    )
    ax_spectrum.fill_between(index, plotted_eigenvalues, plotted_eigenvalues.min(), color="#4f83b5", alpha=0.08)
    ax_spectrum.set_xlabel("mode index (sorted by curvature)")
    ax_spectrum.set_ylabel(r"eigenvalue of the loss Hessian / 2 (log scale)")
    ax_spectrum.set_title("Loss Hessian spectrum")
    ax_spectrum.text(
        0.03, 0.95,
        f"condition number\n$\\lambda_{{max}}/\\lambda_{{min}}$ = {conditioning['condition_number']:.3g}",
        transform=ax_spectrum.transAxes, fontsize=9, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef3f8", "edgecolor": "#4f83b5", "linewidth": 0.8},
    )
    _strip_top_right(ax_spectrum)

    ax_rate.fill_between(
        index, precond_rate, plain_rate, where=plain_rate >= precond_rate,
        color=PRECONDITIONED_COLOR, alpha=0.08, interpolate=True,
        label="gap closed by preconditioning",
    )
    ax_rate.plot(index, plain_rate, label="plain descent", color=PLAIN_COLOR, linewidth=1.8, linestyle=(0, (5, 2)))
    ax_rate.plot(index, precond_rate, label="metric-preconditioned", color=PRECONDITIONED_COLOR, linewidth=2.2)
    ax_rate.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_rate.text(index[-1], 1.0, "  no progress", fontsize=8, va="center", ha="left", color="#555555")
    ax_rate.set_xlabel("mode index (sorted by curvature)")
    ax_rate.set_ylabel("per-step contraction rate  |1 - eta * (effective curvature)|")
    ax_rate.set_title("Per-mode convergence rate (lower is faster)")
    ax_rate.set_ylim(0.0, 1.08)
    ax_rate.legend(loc="center left")
    _strip_top_right(ax_rate)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_potential_png(result: SolverResult, path: str, dpi: int = 200) -> None:
    """Export a potential heatmap with contours, field arrows, and electrodes."""

    ex, ey = electric_field(result.potential, result.config)
    field_magnitude = np.hypot(ex, ey)
    ux = ex / np.maximum(field_magnitude, 1e-30)
    uy = ey / np.maximum(field_magnitude, 1e-30)

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    mesh = ax.pcolormesh(result.x, result.y, result.potential, cmap="RdBu_r", shading="auto")
    contour_levels = np.linspace(result.potential.min(), result.potential.max(), 21)[1:-1]
    ax.contour(
        result.x,
        result.y,
        result.potential,
        levels=contour_levels,
        colors="#1b1b1b",
        linewidths=0.55,
        alpha=0.62,
    )

    step = max(2, min(result.config.nx, result.config.ny) // 26)
    ax.quiver(
        result.x[::step],
        result.y[::step],
        ux[::step, ::step],
        uy[::step, ::step],
        color="#101820",
        alpha=0.72,
        scale=31,
        width=0.0032,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.3,
        zorder=4,
    )
    fig.colorbar(mesh, ax=ax, label="Potential (V)", shrink=0.9)

    _draw_electrodes(ax, result)
    ax.text(0.0, result.electrode_gap / 2.0 + 2.0 * result.config.dy, "+V plate", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.0, -result.electrode_gap / 2.0 - 2.0 * result.config.dy, "-V plate", ha="center", va="top", fontsize=9, fontweight="bold")

    view_x = min(result.config.domain_width / 2.0, result.electrode_width * 1.85)
    view_y = min(result.config.domain_height / 2.0, max(result.electrode_gap * 8.0, result.electrode_width * 0.52))
    ax.set_xlim(-view_x, view_x)
    ax.set_ylim(-view_y, view_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Potential field with equipotentials and electric-field direction")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_fringing_field_png(result: SolverResult, path: str, dpi: int = 200) -> None:
    """Export a streamline view of the electric field near the capacitor,
    annotated to call out the two regimes the reference PDF's Figure 1
    illustrates: a uniform field in the gap and a fringed, bowing field past
    the plate edges.

    Two deliberate choices fix problems a naive streamplot call has on this
    geometry:

    1. The view is cropped to a region around the plates (like Figure 1 of
       the literature review), not the full simulation domain. A capacitor's
       exterior field lines are real dipole-like loops that eventually close
       far from the plates; showing the whole domain buries the near-plate
       fringing behavior (straight in the middle, bowing at the edges) under
       those large loops.
    2. Streamlines are seeded explicitly along the gap midline and just
       outside each plate tip, not left to streamplot's automatic
       density-based seeding. That automatic seeding uses a coarse internal
       grid that can miss the gap entirely (it is only a few grid rows tall
       out of the full domain height), which silently drops the single most
       important feature of the plot: the straight, uniform field between
       the plates. Seeding is also kept away from the exact electrode tip,
       where the thin-electrode grid representation has a discretization
       singularity that makes integrated streamlines spiral rather than bow
       smoothly.
    """

    ex, ey = electric_field(result.potential, result.config)
    magnitude = np.hypot(ex, ey)
    # Field magnitude blows up at the thin-electrode edge (a grid-scale
    # singularity), which would otherwise saturate the color scale near a
    # handful of pixels and wash out the fringing pattern everywhere else.
    color_max = float(np.percentile(magnitude, 99.0))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=color_max)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    ax.pcolormesh(
        result.x, result.y, result.potential, cmap="RdBu_r", shading="auto", alpha=0.30
    )

    stream = ax.streamplot(
        result.x,
        result.y,
        ex,
        ey,
        start_points=_field_line_seed_points(result),
        color=magnitude,
        cmap="viridis",
        norm=norm,
        linewidth=1.85,
        arrowsize=1.75,
        integration_direction="both",
        minlength=1e-4,
        maxlength=4.0,
        broken_streamlines=False,
    )
    fig.colorbar(stream.lines, ax=ax, label="|E| (V/m, clipped at the 99th percentile)", shrink=0.9)

    _draw_electrodes(ax, result)

    view_x = min(result.config.domain_width / 2.0, result.electrode_width * 1.95)
    view_y = min(result.config.domain_height / 2.0, max(result.electrode_gap * 8.5, result.electrode_width * 0.55))
    _draw_detector_reference(ax, result, view_x, view_y)
    _draw_direction_arrows(ax, result, ex, ey, view_x, view_y)
    _annotate_fringing(ax, result, view_x, view_y)

    ax.set_xlim(-view_x, view_x)
    ax.set_ylim(-view_y, view_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Fringing field lines: aligned gap, bowed edges, detector-side extrema", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _field_line_seed_points(result: SolverResult) -> np.ndarray:
    """Return seed points that guarantee coverage of the gap and the edges.

    Gap seeds sit on the midline strictly between the plates, so every one
    produces a straight vertical line -- the field between an (anti)symmetric
    pair of plates is purely vertical along y=0 (Ex(x, 0) = 0 by the
    potential's antisymmetry about the midline). Edge seeds sit just outside
    each plate tip, close enough to show the bow but far enough to avoid the
    thin-electrode corner singularity.
    """

    width = result.electrode_width
    tip_offset = max(2.0 * result.config.dx, 0.06 * width)

    gap_x = np.linspace(-0.92 * width / 2.0, 0.92 * width / 2.0, 17)
    gap_y = np.array([-0.24, 0.0, 0.24]) * result.electrode_gap
    gap_seeds = [(float(x), float(y)) for y in gap_y for x in gap_x]

    edge_seeds: list[tuple[float, float]] = []
    edge_y_offsets = np.array([-2.2, -1.7, -1.25, -0.85, -0.45, -0.12, 0.12, 0.45, 0.85, 1.25, 1.7, 2.2]) * result.electrode_gap
    for side in (-1.0, 1.0):
        edge_x = side * (width / 2.0 + tip_offset)
        for dy in edge_y_offsets:
            edge_seeds.append((edge_x, float(dy)))

    top_bottom_seeds: list[tuple[float, float]] = []
    edge_x_offsets = np.linspace(-0.54 * width, 0.54 * width, 13)
    for plate_side in (-1.0, 1.0):
        y = plate_side * (result.electrode_gap / 2.0 + 3.0 * result.config.dy)
        for x in edge_x_offsets:
            if abs(x) > 0.38 * width:
                top_bottom_seeds.append((float(x), float(y)))

    return np.array(gap_seeds + edge_seeds + top_bottom_seeds, dtype=float)


def _draw_direction_arrows(
    ax: "plt.Axes",
    result: SolverResult,
    ex: np.ndarray,
    ey: np.ndarray,
    view_x: float,
    view_y: float,
) -> None:
    """Overlay sparse normalized arrows so field direction is unmistakable."""

    magnitude = np.hypot(ex, ey)
    ux = ex / np.maximum(magnitude, 1e-30)
    uy = ey / np.maximum(magnitude, 1e-30)
    x_mask = np.abs(result.x) <= view_x * 0.92
    y_mask = np.abs(result.y) <= view_y * 0.92
    x_indices = np.flatnonzero(x_mask)[:: max(2, max(1, np.count_nonzero(x_mask) // 15))]
    y_indices = np.flatnonzero(y_mask)[:: max(2, max(1, np.count_nonzero(y_mask) // 10))]
    xx, yy = np.meshgrid(result.x[x_indices], result.y[y_indices])
    ax.quiver(
        xx,
        yy,
        ux[np.ix_(y_indices, x_indices)],
        uy[np.ix_(y_indices, x_indices)],
        color="#111111",
        alpha=0.68,
        scale=27,
        width=0.0038,
        headwidth=4.4,
        headlength=5.2,
        headaxislength=4.7,
        zorder=5,
    )


def _draw_detector_reference(ax: "plt.Axes", result: SolverResult, view_x: float, view_y: float) -> None:
    """Draw detector/photo-plate reference strips where fringing extrema form."""

    detector_y = min(view_y * 0.78, result.electrode_gap * 3.8)
    for sign, label, va in ((1.0, "detector/photo-plate sampling line", "bottom"), (-1.0, "opposite detector line", "top")):
        y = sign * detector_y
        ax.plot(
            [-view_x * 0.88, view_x * 0.88],
            [y, y],
            color="#6f2dbd",
            linewidth=1.8,
            linestyle=(0, (4, 3)),
            alpha=0.85,
            zorder=3,
        )
        if sign > 0:
            ax.text(
                -view_x * 0.86,
                y + sign * 0.05 * view_y,
                label,
                color="#4b168c",
                fontsize=8.4,
                ha="left",
                va=va,
                fontweight="bold",
            )


def _annotate_fringing(ax: "plt.Axes", result: SolverResult, view_x: float, view_y: float) -> None:
    """Label the two field regimes and dimension the plate gap, so the plot
    is self-explanatory without relying on an external caption. Labels are
    anchored in axes-fraction space (fixed corners) with short leader arrows
    into data space, so they never collide with the streamlines regardless
    of the crop geometry."""

    width = result.electrode_width
    gap = result.electrode_gap
    tip_offset = max(2.0 * result.config.dx, 0.06 * width)

    # Gap dimension, along the left edge of the cropped view -- clear of the
    # fringing loops, which stay closer to the plate tips.
    dim_x = -view_x * 0.88
    ax.annotate(
        "", xy=(dim_x, gap / 2.0), xytext=(dim_x, -gap / 2.0),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.0},
        annotation_clip=False,
    )
    ax.text(
        dim_x, gap / 2.0 + 0.10 * view_y, f"gap = {gap * 1000:.2g} mm",
        ha="left", va="bottom", fontsize=8, color="#333333",
    )

    ax.annotate(
        "uniform field\n(straight, vertical)",
        xy=(0.0, gap * 0.5), xycoords="data",
        xytext=(0.05, 0.94), textcoords="axes fraction",
        fontsize=8.5, ha="left", va="top", color="#1f4e78",
        arrowprops={
            "arrowstyle": "->", "color": "#1f4e78", "linewidth": 0.9,
            "connectionstyle": "arc3,rad=0.2", "shrinkA": 0, "shrinkB": 4,
        },
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#1f4e78", "linewidth": 0.6, "alpha": 0.92},
    )
    ax.annotate(
        "fringing field\n(bows outward)",
        xy=(width / 2.0 + 1.5 * tip_offset, gap * 1.0), xycoords="data",
        xytext=(0.60, 0.06), textcoords="axes fraction",
        fontsize=8.5, ha="left", va="bottom", color="#8a3b1f",
        arrowprops={
            "arrowstyle": "->", "color": "#8a3b1f", "linewidth": 0.9,
            "connectionstyle": "arc3,rad=-0.2", "shrinkA": 0, "shrinkB": 4,
        },
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#8a3b1f", "linewidth": 0.6, "alpha": 0.92},
    )


def _draw_electrodes(ax: "plt.Axes", result: SolverResult) -> None:
    electrode_rows, electrode_columns = np.nonzero(result.electrode_mask)
    for row in sorted(set(electrode_rows)):
        columns = electrode_columns[electrode_rows == row]
        ax.plot(
            [result.x[columns.min()], result.x[columns.max()]],
            [result.y[row], result.y[row]],
            color="black",
            linewidth=4,
            solid_capstyle="round",
        )
