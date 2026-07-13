"""3D phase-space views of the optimizer's high-dimensional theta trajectory.

theta lives in R^parameter_count (64 dimensions for the default PINN
benchmark config) -- far too high-dimensional to look at directly. PCA finds
the 3 directions that explain the most variance across every optimizer's
recorded path and projects each trajectory onto those shared axes, so the
curved route the Hamiltonian-geometric optimizer takes through parameter
space can be compared against the straighter paths of SGD/Adam/etc. in an
ordinary 3D plot. All optimizers are projected onto the same components (fit
jointly across every recorded theta) so their paths stay directly comparable.
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.pinn import TrainingResult
from src.visualization import OPTIMIZER_COLORS

Array = np.ndarray


def compute_phase_space_projection(
    results: list[TrainingResult], n_components: int = 3
) -> tuple[dict[str, Array], Array]:
    """PCA-project each optimizer's theta trajectory onto shared axes.

    Returns a dict mapping optimizer name to an (steps, n_components) array
    of projected coordinates, and the explained-variance ratio of each
    component.
    """

    recorded = [result for result in results if result.theta_history]
    if not recorded:
        raise ValueError(
            "no TrainingResult has theta_history populated; rerun the "
            "benchmark with record_theta=True"
        )
    stacked = np.vstack([np.asarray(result.theta_history) for result in recorded])
    if n_components > stacked.shape[1]:
        raise ValueError(
            f"n_components={n_components} exceeds the parameter dimension {stacked.shape[1]}"
        )

    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    total_variance = float(np.sum(singular_values**2))
    explained_variance_ratio = (singular_values[:n_components] ** 2) / total_variance

    projections: dict[str, Array] = {}
    for result in recorded:
        theta = np.asarray(result.theta_history) - mean
        projections[result.optimizer] = theta @ components.T
    return projections, explained_variance_ratio


def export_phase_space_trajectories_csv(
    results: list[TrainingResult], projections: dict[str, Array], path: str
) -> None:
    """Write one row per optimizer step: projected PCA coordinates plus loss."""

    loss_by_optimizer = {result.optimizer: result.loss_history for result in results}
    n_components = next(iter(projections.values())).shape[1]
    header = ["step", "optimizer", "loss"] + [f"pc{i + 1}" for i in range(n_components)]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for optimizer, points in projections.items():
            loss_history = loss_by_optimizer[optimizer]
            for step, coordinates in enumerate(points):
                writer.writerow(
                    [step + 1, optimizer, f"{loss_history[step]:.12g}"]
                    + [f"{value:.12g}" for value in coordinates]
                )


def _axis_labels(explained_variance_ratio: Array) -> tuple[str, str, str]:
    return (
        f"PC1 ({explained_variance_ratio[0] * 100:.1f}% var)",
        f"PC2 ({explained_variance_ratio[1] * 100:.1f}% var)",
        f"PC3 ({explained_variance_ratio[2] * 100:.1f}% var)",
    )


def export_phase_space_png(
    projections: dict[str, Array],
    explained_variance_ratio: Array,
    path: str,
    dpi: int = 200,
    elev: float = 22.0,
    azim: float = -60.0,
) -> None:
    """Static 3D PCA phase-space plot: one trajectory line per optimizer."""

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    for optimizer, points in projections.items():
        color = OPTIMIZER_COLORS.get(optimizer, "#333333")
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=color, linewidth=2.0, alpha=0.9, label=optimizer,
        )
        ax.scatter(*points[0], color=color, marker="o", s=45, edgecolor="white", linewidth=0.6)
        ax.scatter(*points[-1], color=color, marker="*", s=150, edgecolor="white", linewidth=0.6)

    x_label, y_label, z_label = _axis_labels(explained_variance_ratio)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title("Hamiltonian-geometric optimizer: theta trajectory in PCA phase space")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_phase_space_rotation_gif(
    projections: dict[str, Array],
    explained_variance_ratio: Array,
    path: str,
    frames: int = 60,
    elev: float = 22.0,
    fps: int = 12,
) -> None:
    """Rotating GIF of the same PCA phase-space plot, one full turn in azimuth."""

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for optimizer, points in projections.items():
        color = OPTIMIZER_COLORS.get(optimizer, "#333333")
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=color, linewidth=2.0, alpha=0.9, label=optimizer,
        )
        ax.scatter(*points[0], color=color, marker="o", s=40, edgecolor="white", linewidth=0.6)
        ax.scatter(*points[-1], color=color, marker="*", s=140, edgecolor="white", linewidth=0.6)

    x_label, y_label, z_label = _axis_labels(explained_variance_ratio)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title("Hamiltonian-geometric optimizer: theta trajectory in PCA phase space")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    azimuths = np.linspace(0.0, 360.0, frames, endpoint=False)

    def draw(frame_index: int):
        ax.view_init(elev=elev, azim=float(azimuths[frame_index]))
        return ()

    animation = FuncAnimation(fig, draw, frames=frames, interval=1000 // fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def export_phase_space_html(
    results: list[TrainingResult],
    projections: dict[str, Array],
    explained_variance_ratio: Array,
    path: str,
    title: str = "Hamiltonian-geometric optimizer phase space",
) -> None:
    """Self-contained, dependency-free interactive HTML: drag to rotate."""

    loss_by_optimizer = {result.optimizer: result.loss_history for result in results}

    scale = 1.0 / max(
        1e-12, max(float(np.abs(points).max()) for points in projections.values())
    )
    trajectories = []
    for optimizer, points in projections.items():
        loss_history = loss_by_optimizer[optimizer]
        min_loss, max_loss = min(loss_history), max(loss_history)
        loss_span = max(1e-12, max_loss - min_loss)
        trajectories.append(
            {
                "name": optimizer,
                "color": OPTIMIZER_COLORS.get(optimizer, "#9aa3ab"),
                "points": (points * scale).tolist(),
                "loss_progress": [
                    (loss - min_loss) / loss_span for loss in loss_history[: len(points)]
                ],
            }
        )

    payload = {
        "title": title,
        "axis_labels": list(_axis_labels(explained_variance_ratio)),
        "trajectories": trajectories,
    }
    with open(path, "w", encoding="utf-8") as file:
        file.write(_phase_space_html(payload))


def _phase_space_html(payload: dict) -> str:
    data = json.dumps(payload)
    legend_rows = "".join(
        f'<div><span class="swatch" style="background:{trajectory["color"]}"></span>{trajectory["name"]}</div>'
        for trajectory in payload["trajectories"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{payload["title"]}</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #090d12; color: #e6edf3; font-family: Arial, sans-serif; }}
    canvas {{ width: 100vw; height: 100vh; display: block; background: radial-gradient(circle at 48% 42%, #172236 0%, #090d12 68%); }}
    .hud {{ position: fixed; left: 16px; top: 14px; max-width: 420px; padding: 11px 13px; border-radius: 8px; background: rgba(9,13,18,.76); border: 1px solid rgba(230,237,243,.18); font-size: 13px; line-height: 1.35; }}
    .legend {{ position: fixed; right: 16px; bottom: 14px; padding: 10px 12px; border-radius: 8px; background: rgba(9,13,18,.76); border: 1px solid rgba(230,237,243,.18); font-size: 13px; }}
    .legend div {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
    .swatch {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
  </style>
</head>
<body>
<canvas id="scene"></canvas>
<div class="hud">
  <strong>PCA projection of theta (parameter-space) trajectory</strong><br>
  {payload["axis_labels"][0]}<br>
  {payload["axis_labels"][1]}<br>
  {payload["axis_labels"][2]}<br>
  circle = step 1, star = final step. Color fades toward loss minimum.<br>
  drag to rotate
</div>
<div class="legend">{legend_rows}</div>
<script>
const payload = {data};
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
let yaw = -0.78, pitch = 0.46, dragging = false, lastX = 0, lastY = 0;

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
  const scale = Math.min(canvas.width, canvas.height) * .34;
  const depth = 1 / (1 + 1.1*z);
  return [canvas.width*.5 + x*scale*depth, canvas.height*.52 - y*scale*depth, depth];
}}
function drawAxes() {{
  const len = 1.15;
  const axes = [[[-len,0,0],[len,0,0],"rgba(154,163,171,.55)"], [[0,-len,0],[0,len,0],"rgba(154,163,171,.55)"], [[0,0,-len],[0,0,len],"rgba(154,163,171,.55)"]];
  ctx.lineWidth = 1.1 * devicePixelRatio;
  for (const [a, b, color] of axes) {{
    const pa = project(a), pb = project(b);
    ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
  }}
}}
function drawTrajectory(trajectory) {{
  const pts = trajectory.points.map(project);
  for (let i = 1; i < pts.length; i++) {{
    const t = trajectory.loss_progress[i] ?? 0;
    ctx.strokeStyle = trajectory.color; ctx.globalAlpha = .35 + .55 * (1 - t);
    ctx.lineWidth = 2.2 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(pts[i-1][0], pts[i-1][1]); ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke();
  }}
  ctx.globalAlpha = 1;
  const start = pts[0], end = pts[pts.length - 1];
  ctx.fillStyle = trajectory.color;
  ctx.beginPath(); ctx.arc(start[0], start[1], 5*devicePixelRatio, 0, 7); ctx.fill();
  ctx.strokeStyle = "white"; ctx.lineWidth = 1*devicePixelRatio; ctx.stroke();
  drawStar(end[0], end[1], 8*devicePixelRatio, trajectory.color);
}}
function drawStar(cx, cy, r, color) {{
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {{
    const rad = i % 2 === 0 ? r : r * .45;
    const angle = Math.PI/2 + i * Math.PI/5;
    const x = cx + rad*Math.cos(angle), y = cy - rad*Math.sin(angle);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }}
  ctx.closePath(); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "white"; ctx.lineWidth = .8*devicePixelRatio; ctx.stroke();
}}
function renderScene() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  drawAxes();
  for (const trajectory of payload.trajectories) drawTrajectory(trajectory);
  requestAnimationFrame(renderScene);
}}
requestAnimationFrame(renderScene);
</script>
</body>
</html>
"""
