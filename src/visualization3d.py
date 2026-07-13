"""Self-contained 3D capacitor field animation exports.

The settled view traces field lines through a finite-plate electrostatic
approximation: each electrode is represented by a weighted grid of surface
charges and field lines are integrated through the resulting 3D electric field.
For a DC capacitor the electrostatic field is static; the animation is a flow
visualization, not a claim that the DC field itself oscillates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np


@dataclass(frozen=True)
class Capacitor3DConfig:
    """Geometry and visual parameters for a finite 3D plate capacitor."""

    plate_length: float = 0.03
    plate_width: float = 0.02
    gap: float = 0.004
    voltage: float = 1.0
    field_line_rows: int = 9
    field_line_columns: int = 11
    points_per_line: int = 72
    fringe_bulge: float = 1.25
    emi_wobble: float = 0.18
    animation_speed: float = 1.05
    insertion_cycle_s: float = 8.0
    chaotic_transient_strength: float = 1.85
    plate_thickness: float = 0.00045
    detector_distance: float = 0.009

    def __post_init__(self) -> None:
        positive_values = {
            "plate_length": self.plate_length,
            "plate_width": self.plate_width,
            "gap": self.gap,
            "voltage": self.voltage,
            "field_line_rows": self.field_line_rows,
            "field_line_columns": self.field_line_columns,
            "points_per_line": self.points_per_line,
            "animation_speed": self.animation_speed,
            "insertion_cycle_s": self.insertion_cycle_s,
            "chaotic_transient_strength": self.chaotic_transient_strength,
            "plate_thickness": self.plate_thickness,
            "detector_distance": self.detector_distance,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        if self.field_line_rows < 2 or self.field_line_columns < 2:
            raise ValueError("field_line_rows and field_line_columns must be at least 2")
        if self.points_per_line < 8:
            raise ValueError("points_per_line must be at least 8")
        if self.fringe_bulge < 0.0:
            raise ValueError("fringe_bulge must be non-negative")
        if self.emi_wobble < 0.0:
            raise ValueError("emi_wobble must be non-negative")
        if self.chaotic_transient_strength < 0.0:
            raise ValueError("chaotic_transient_strength must be non-negative")


def generate_3d_field_lines(config: Capacitor3DConfig) -> list[list[list[float]]]:
    """Return electrostatically traced field-line polylines.

    This is still a visualization-grade model, not a full boundary-element
    solver. But unlike the previous hand-shaped curves, these lines are traced
    through a finite two-plate electric field, so edge lines naturally bow
    outward and interior lines remain comparatively straight.
    """

    positive_cloud, negative_cloud = _plate_charge_cloud(config)
    lines: list[list[list[float]]] = []

    for x_index, y_index, x0, y0 in _plate_seed_points(config):
        line = _trace_electrostatic_field_line(
            float(x0),
            float(y0),
            config,
            positive_cloud,
            negative_cloud,
            seed_phase=x_index * 1.618034 + y_index * 2.399963,
        )
        lines.append(line)

    # Add edge-originating lines from just outside the plate perimeter. These
    # are the lines that make the real finite-capacitor fringing shell legible.
    edge_offsets = np.linspace(-0.48, 0.48, max(config.field_line_rows, config.field_line_columns))
    for side in (-1.0, 1.0):
        x_edge = side * 0.505 * config.plate_length
        for index, offset in enumerate(edge_offsets):
            lines.append(
                _trace_electrostatic_field_line(
                    x_edge,
                    float(offset * config.plate_width),
                    config,
                    positive_cloud,
                    negative_cloud,
                    seed_phase=10.0 + index + side,
                )
            )
        y_edge = side * 0.505 * config.plate_width
        for index, offset in enumerate(edge_offsets):
            lines.append(
                _trace_electrostatic_field_line(
                    float(offset * config.plate_length),
                    y_edge,
                    config,
                    positive_cloud,
                    negative_cloud,
                    seed_phase=20.0 + index - side,
                )
            )
    return lines


def export_3d_field_animation_html(
    config: Capacitor3DConfig,
    path: str,
    title: str = "3D capacitor fringing field animation",
) -> None:
    """Write a standalone animated 3D field-line HTML file."""

    lines = generate_3d_field_lines(config)
    payload = {
        "title": title,
        "config": config.__dict__,
        "lines": lines,
        "incoming_lines": generate_incoming_field_lines(config),
        "chaotic_lines": generate_chaotic_transient_lines(config),
        "plates": _plate_vertices(config),
        "plate_boxes": _plate_box_vertices(config),
        "detector": _detector_plate_vertices(config),
    }
    html = _html_template(payload)
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)


def generate_incoming_field_lines(config: Capacitor3DConfig) -> list[list[list[float]]]:
    """Return external field lines before capacitor boundary conditions dominate."""

    y_values = np.linspace(-0.026, 0.026, config.field_line_rows + 2)
    z_values = np.linspace(-0.012, 0.012, config.field_line_columns)
    x0 = -0.038
    x1 = 0.038
    lines: list[list[list[float]]] = []
    for yi, y in enumerate(y_values):
        for zi, z in enumerate(z_values):
            phase = yi * 1.7 + zi * 2.3
            line: list[list[float]] = []
            for t in np.linspace(0.0, 1.0, config.points_per_line):
                x = x0 * (1.0 - t) + x1 * t
                wobble = config.emi_wobble * config.gap * math.sin(2.0 * math.pi * t + phase)
                line.append([float(x), float(y + 0.25 * wobble), float(z + 0.2 * wobble)])
            lines.append(line)
    return lines


def generate_chaotic_transient_lines(config: Capacitor3DConfig) -> list[list[list[float]]]:
    """Return deterministic transient lines used while plates perturb the field.

    The transient is intentionally visually irregular: it represents external
    interference plus the finite plate boundary condition being imposed during
    insertion. It is not used as the settled electrostatic solution.
    """

    rng = np.random.default_rng(19)
    count = config.field_line_rows * config.field_line_columns + 52
    radius_x = config.plate_length * (0.96 + 0.14 * config.chaotic_transient_strength)
    radius_y = config.plate_width * (0.96 + 0.12 * config.chaotic_transient_strength)
    radius_z = config.gap * (3.2 + 0.85 * config.chaotic_transient_strength)
    lines: list[list[list[float]]] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        phase = rng.uniform(0.0, 2.0 * math.pi)
        twist = rng.uniform(1.6, 4.2) * config.chaotic_transient_strength
        line: list[list[float]] = []
        for t in np.linspace(0.0, 1.0, config.points_per_line):
            s = 2.0 * t - 1.0
            envelope = math.sin(math.pi * t)
            x = radius_x * 0.86 * s
            y = radius_y * math.sin(angle + twist * s + phase) * envelope
            z = radius_z * 0.42 * math.cos(angle * 0.7 + twist * s) * envelope
            x += 0.25 * radius_x * math.sin(6.0 * math.pi * t + phase) * envelope
            y += 0.18 * radius_y * math.cos(5.0 * math.pi * t + phase)
            z += 0.22 * radius_z * math.sin(7.0 * math.pi * t + 0.5 * phase) * envelope
            line.append([float(x), float(y), float(z)])
        lines.append(line)
    return lines


def _plate_seed_points(config: Capacitor3DConfig) -> list[tuple[int, int, float, float]]:
    x_values = np.linspace(
        -0.42 * config.plate_length,
        0.42 * config.plate_length,
        config.field_line_columns,
    )
    y_values = np.linspace(
        -0.42 * config.plate_width,
        0.42 * config.plate_width,
        config.field_line_rows,
    )
    seeds: list[tuple[int, int, float, float]] = []
    for y_index, y0 in enumerate(y_values):
        for x_index, x0 in enumerate(x_values):
            jitter_x = 0.006 * config.plate_length * math.sin(1.91 * x_index + 0.73 * y_index)
            jitter_y = 0.006 * config.plate_width * math.cos(1.37 * y_index - 0.41 * x_index)
            seeds.append((x_index, y_index, float(x0 + jitter_x), float(y0 + jitter_y)))
    return seeds


def _plate_charge_cloud(config: Capacitor3DConfig) -> tuple[np.ndarray, np.ndarray]:
    nx = max(11, min(19, config.field_line_columns + 4))
    ny = max(9, min(17, config.field_line_rows + 4))
    xs = np.linspace(-0.5 * config.plate_length, 0.5 * config.plate_length, nx)
    ys = np.linspace(-0.5 * config.plate_width, 0.5 * config.plate_width, ny)
    hx = config.plate_length / 2.0
    hy = config.plate_width / 2.0
    hz = config.gap / 2.0
    positive: list[list[float]] = []
    negative: list[list[float]] = []
    for x in xs:
        for y in ys:
            edge_x = abs(x) / hx
            edge_y = abs(y) / hy
            edge_weight = max(edge_x, edge_y) ** 3.0
            corner_weight = (edge_x * edge_y) ** 2.0
            weight = 1.0 + 1.9 * edge_weight + 1.2 * corner_weight
            positive.append([float(x), float(y), hz, weight])
            negative.append([float(x), float(y), -hz, -weight])
    return np.array(positive, dtype=float), np.array(negative, dtype=float)


def _trace_electrostatic_field_line(
    x0: float,
    y0: float,
    config: Capacitor3DConfig,
    positive_cloud: np.ndarray,
    negative_cloud: np.ndarray,
    seed_phase: float,
) -> list[list[float]]:
    hz = config.gap / 2.0
    start = np.array([x0, y0, hz - 0.12 * config.gap], dtype=float)
    step = config.gap * 0.055
    max_steps = max(96, config.points_per_line * 5)
    extent_x = config.plate_length * 1.15
    extent_y = config.plate_width * 1.2
    extent_z = config.gap * 2.5
    points = [start.copy()]
    point = start.copy()

    for index in range(max_steps):
        field = _electric_field_at(point, positive_cloud, negative_cloud, config)
        norm = float(np.linalg.norm(field))
        if norm < 1e-18:
            break
        direction = field / norm
        # The EMI term is intentionally tiny in the settled phase: enough to
        # avoid sterile geometry, not enough to fake the fringing structure.
        edge_fraction = min(1.0, max(abs(point[0]) / (config.plate_length / 2.0), abs(point[1]) / (config.plate_width / 2.0)))
        wobble = (
            0.025
            * config.emi_wobble
            * edge_fraction
            * np.array(
                [
                    math.sin(index * 0.37 + seed_phase),
                    math.cos(index * 0.31 + 0.7 * seed_phase),
                    0.0,
                ]
            )
        )
        direction = direction + wobble
        direction = direction / max(float(np.linalg.norm(direction)), 1e-18)
        point = point + step * direction
        points.append(point.copy())

        if point[2] <= -hz + 0.11 * config.gap:
            break
        if abs(point[0]) > extent_x or abs(point[1]) > extent_y or abs(point[2]) > extent_z:
            break

    if len(points) < 4:
        return _fallback_vertical_line(x0, y0, config)
    return _resample_polyline(points, config.points_per_line)


def _electric_field_at(
    point: np.ndarray,
    positive_cloud: np.ndarray,
    negative_cloud: np.ndarray,
    config: Capacitor3DConfig,
) -> np.ndarray:
    charges = np.vstack((positive_cloud, negative_cloud))
    offsets = point[None, :] - charges[:, :3]
    softening = (0.18 * config.gap) ** 2
    radius2 = np.sum(offsets * offsets, axis=1) + softening
    inv_radius3 = 1.0 / (radius2 * np.sqrt(radius2))
    weighted = offsets * (charges[:, 3] * inv_radius3)[:, None]
    return np.sum(weighted, axis=0)


def _resample_polyline(points: list[np.ndarray], count: int) -> list[list[float]]:
    array = np.array(points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-18:
        repeated = np.repeat(array[:1], count, axis=0)
        return repeated.astype(float).tolist()
    targets = np.linspace(0.0, total, count)
    resampled = np.empty((count, 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(targets, cumulative, array[:, dim])
    return resampled.tolist()


def _fallback_vertical_line(x0: float, y0: float, config: Capacitor3DConfig) -> list[list[float]]:
    return [
        [float(x0), float(y0), float(config.gap * (0.5 - t))]
        for t in np.linspace(0.0, 1.0, config.points_per_line)
    ]


def _field_line_from_seed(
    x0: float,
    y0: float,
    x_index: int,
    y_index: int,
    config: Capacitor3DConfig,
    force_edge: bool = False,
) -> list[list[float]]:
    half_length = config.plate_length / 2.0
    half_width = config.plate_width / 2.0
    radial_x = 0.0 if x0 == 0.0 else x0 / max(abs(x0), 1e-30)
    radial_y = 0.0 if y0 == 0.0 else y0 / max(abs(y0), 1e-30)
    if radial_x == 0.0 and radial_y == 0.0:
        radial_y = 1.0

    edge_distance = min(half_length - abs(x0), half_width - abs(y0))
    edge_distance = max(0.0, edge_distance)
    edge_scale = 1.0 - min(1.0, edge_distance / min(half_length, half_width))
    if force_edge:
        edge_scale = 1.0

    phase = x_index * 1.618034 + y_index * 2.399963
    points: list[list[float]] = []
    for t in np.linspace(0.0, 1.0, config.points_per_line):
        z = config.gap * (0.5 - t)
        envelope = math.sin(math.pi * t)
        edge_push = edge_scale**1.35
        bulge = config.fringe_bulge * config.gap * edge_push * envelope
        wobble = config.emi_wobble * config.gap * edge_push * math.sin(2.0 * math.pi * t + phase)
        swirl = 0.22 * config.fringe_bulge * config.gap * edge_push * envelope
        x = x0 + radial_x * bulge - radial_y * swirl * math.sin(math.pi * t + 0.3 * phase) + 0.16 * wobble
        y = y0 + radial_y * bulge + radial_x * swirl * math.cos(math.pi * t + 0.2 * phase) - 0.11 * wobble
        points.append([float(x), float(y), float(z)])
    return points


def _plate_vertices(config: Capacitor3DConfig) -> dict[str, list[list[float]]]:
    hx = config.plate_length / 2.0
    hy = config.plate_width / 2.0
    hz = config.gap / 2.0
    top = [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]
    bottom = [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz]]
    return {"positive": top, "negative": bottom}


def _plate_box_vertices(config: Capacitor3DConfig) -> dict[str, list[list[list[float]]]]:
    hx = config.plate_length / 2.0
    hy = config.plate_width / 2.0
    hz = config.gap / 2.0
    ht = config.plate_thickness / 2.0

    def box(z_center: float) -> list[list[list[float]]]:
        z0 = z_center - ht
        z1 = z_center + ht
        return [
            [[-hx, -hy, z1], [hx, -hy, z1], [hx, hy, z1], [-hx, hy, z1]],
            [[-hx, -hy, z0], [-hx, hy, z0], [hx, hy, z0], [hx, -hy, z0]],
            [[-hx, -hy, z0], [hx, -hy, z0], [hx, -hy, z1], [-hx, -hy, z1]],
            [[hx, -hy, z0], [hx, hy, z0], [hx, hy, z1], [hx, -hy, z1]],
            [[hx, hy, z0], [-hx, hy, z0], [-hx, hy, z1], [hx, hy, z1]],
            [[-hx, hy, z0], [-hx, -hy, z0], [-hx, -hy, z1], [-hx, hy, z1]],
        ]

    return {"positive": box(hz), "negative": box(-hz)}


def _detector_plate_vertices(config: Capacitor3DConfig) -> list[list[float]]:
    hx = config.plate_length / 2.0
    hy = config.plate_width / 2.0
    x = hx + config.detector_distance
    z_margin = 1.7 * config.gap
    y_margin = 0.28 * config.plate_width
    return [
        [x, -hy - y_margin, -z_margin],
        [x, hy + y_margin, -z_margin],
        [x, hy + y_margin, z_margin],
        [x, -hy - y_margin, z_margin],
    ]


def _html_template(payload: dict) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{payload["title"]}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: #0d1117;
      color: #e6edf3;
      font-family: Arial, sans-serif;
      overflow: hidden;
    }}
    canvas {{
      width: 100vw;
      height: 100vh;
      display: block;
      background: radial-gradient(circle at 50% 45%, #172033 0%, #0d1117 62%);
    }}
    .hud {{
      position: fixed;
      left: 16px;
      top: 14px;
      background: rgba(13, 17, 23, 0.72);
      border: 1px solid rgba(230, 237, 243, 0.18);
      border-radius: 8px;
      padding: 10px 12px;
      line-height: 1.35;
      font-size: 13px;
      max-width: 420px;
      backdrop-filter: blur(4px);
    }}
    .phaseTitle {{
      display: block;
      margin: 4px 0 8px;
      color: #ffd45c;
      font-weight: 700;
    }}
    .phaseBar {{
      width: 100%;
      height: 8px;
      background: rgba(230, 237, 243, 0.16);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }}
    .phaseFill {{
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #69d0ff, #ff765c, #ffd45c, #66e3a4);
    }}
    .legend {{
      position: fixed;
      right: 16px;
      bottom: 14px;
      background: rgba(13, 17, 23, 0.72);
      border: 1px solid rgba(230, 237, 243, 0.18);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
    }}
    .legendRow {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 3px 0;
      white-space: nowrap;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
    }}
  </style>
</head>
<body>
<canvas id="scene"></canvas>
<div class="hud">
  <strong>3D capacitor insertion transient</strong><br>
  <span id="phaseLabel" class="phaseTitle">external field before capacitor</span>
  Moving dots are field-direction tracers.<br>
  Phase cycle: free field -> capacitor enters -> chaotic boundary transient -> aligned inside + fringing outside.
  <div class="phaseBar"><div class="phaseFill" id="phaseFill"></div></div>
</div>
<div class="legend">
  <div class="legendRow"><span class="swatch" style="background:#ff8d8d"></span>+V plate</div>
  <div class="legendRow"><span class="swatch" style="background:#77adff"></span>-V plate</div>
  <div class="legendRow"><span class="swatch" style="background:#69d0ff"></span>settled field lines</div>
  <div class="legendRow"><span class="swatch" style="background:#ffd45c"></span>direction tracers</div>
  <div class="legendRow"><span class="swatch" style="background:#b48cff"></span>detector/photo-plate</div>
  drag to rotate
</div>
<script>
const payload = {data};
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const phaseLabel = document.getElementById("phaseLabel");
const phaseFill = document.getElementById("phaseFill");
let yaw = -0.72;
let pitch = 0.58;
let dragging = false;
let lastX = 0;
let lastY = 0;

function resize() {{
  canvas.width = Math.floor(window.innerWidth * devicePixelRatio);
  canvas.height = Math.floor(window.innerHeight * devicePixelRatio);
}}
window.addEventListener("resize", resize);
resize();

canvas.addEventListener("pointerdown", event => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("pointermove", event => {{
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch += (event.clientY - lastY) * 0.008;
  pitch = Math.max(-1.25, Math.min(1.25, pitch));
  lastX = event.clientX;
  lastY = event.clientY;
}});

function rotate(point) {{
  const [x, y, z] = point;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  return [x1, y1, z2];
}}

function project(point) {{
  const [x, y, z] = rotate(point);
  const scaleBase = Math.min(canvas.width, canvas.height) * 15.5;
  const depth = 1.0 / (1.0 + z * 11.0);
  return [
    canvas.width * 0.5 + x * scaleBase * depth,
    canvas.height * 0.52 - y * scaleBase * depth,
    depth
  ];
}}

function alphaColor(color, alpha) {{
  return color.replace("ALPHA", alpha.toFixed(3));
}}

function drawPlate(vertices, fill, stroke, alpha = 1) {{
  const pts = vertices.map(project);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.fill();
  ctx.stroke();
  ctx.globalAlpha = 1;
}}

function drawPolygon(vertices, fill, stroke, alpha = 1, lineWidth = 1.4) {{
  const pts = vertices.map(project);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth * devicePixelRatio;
  ctx.fill();
  ctx.stroke();
  ctx.globalAlpha = 1;
}}

function drawPlateBox(faces, fill, stroke, alpha = 1) {{
  const ordered = faces
    .map(face => {{
      const depth = face.reduce((sum, point) => sum + rotate(point)[2], 0) / face.length;
      return {{face, depth}};
    }})
    .sort((a, b) => a.depth - b.depth);
  for (const item of ordered) drawPolygon(item.face, fill, stroke, alpha, 1.15);
}}

function drawDetector(vertices, phase) {{
  drawPolygon(vertices, "rgba(180, 140, 255, 0.10)", "rgba(190, 160, 255, 0.72)", 0.82, 1.2);
  const pts = vertices.map(project);
  const center = [
    (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
    (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25
  ];
  const pulse = 0.55 + 0.45 * Math.sin(phase * Math.PI * 2);
  ctx.globalAlpha = 0.28 + 0.22 * pulse;
  ctx.fillStyle = "rgba(255, 214, 92, 0.9)";
  for (let i = -2; i <= 2; i++) {{
    const x = center[0] + i * 16 * devicePixelRatio;
    const y = center[1] + Math.sin(i + phase * Math.PI * 2) * 8 * devicePixelRatio;
    ctx.beginPath();
    ctx.arc(x, y, (2.2 + pulse) * devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
  }}
  ctx.globalAlpha = 1;
}}

function drawLine(line, color, width, alpha = 1) {{
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  const first = project(line[0]);
  ctx.moveTo(first[0], first[1]);
  for (let i = 1; i < line.length; i++) {{
    const p = project(line[i]);
    ctx.lineTo(p[0], p[1]);
  }}
  ctx.strokeStyle = color;
  ctx.lineWidth = width * devicePixelRatio;
  ctx.stroke();
  ctx.globalAlpha = 1;
}}

function drawLineGlow(line, color, width, alpha = 1) {{
  drawLine(line, color, width + 2.4, alpha * 0.22);
  drawLine(line, color, width, alpha);
}}

function drawDirectionArrow(line, t, color, alpha = 1) {{
  const p0 = project(sampleLine(line, Math.max(0, t - 0.018)));
  const p1 = project(sampleLine(line, Math.min(1, t + 0.018)));
  const dx = p1[0] - p0[0];
  const dy = p1[1] - p0[1];
  const length = Math.hypot(dx, dy);
  if (length < 1e-6) return;
  const ux = dx / length;
  const uy = dy / length;
  const size = 7.0 * devicePixelRatio;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(p1[0], p1[1]);
  ctx.lineTo(p1[0] - ux * size - uy * size * 0.55, p1[1] - uy * size + ux * size * 0.55);
  ctx.lineTo(p1[0] - ux * size + uy * size * 0.55, p1[1] - uy * size - ux * size * 0.55);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}}

function sampleLine(line, t) {{
  const raw = t * (line.length - 1);
  const i = Math.floor(raw);
  const j = Math.min(line.length - 1, i + 1);
  const f = raw - i;
  return [
    line[i][0] * (1 - f) + line[j][0] * f,
    line[i][1] * (1 - f) + line[j][1] * f,
    line[i][2] * (1 - f) + line[j][2] * f
  ];
}}

function animate(timeMs) {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  drawSceneGrid(timeMs);

  const settled = payload.lines;
  const incoming = payload.incoming_lines;
  const chaotic = payload.chaotic_lines;
  const cycleMs = payload.config.insertion_cycle_s * 1000;
  const phase = (timeMs % cycleMs) / cycleMs;
  const plateAlpha = smoothstep(0.20, 0.36, phase);
  const incomingAlpha = 1.0 - smoothstep(0.18, 0.36, phase);
  const chaoticAlpha = smoothstep(0.30, 0.44, phase) * (1.0 - smoothstep(0.58, 0.76, phase));
  const settledAlpha = smoothstep(0.58, 0.82, phase);
  const insertionShift = (1.0 - plateAlpha) * 0.085;
  phaseFill.style.width = Math.round(phase * 100) + "%";

  if (phase < 0.22) phaseLabel.textContent = "external field moving before the capacitor enters";
  else if (phase < 0.38) phaseLabel.textContent = "capacitor plates entering the field";
  else if (phase < 0.62) phaseLabel.textContent = "chaotic transient from boundary conditions + EMI";
  else phaseLabel.textContent = "settled field: aligned inside, fringing around edges";

  function shiftedPlate(vertices) {{
    return vertices.map(p => [p[0] + insertionShift, p[1], p[2]]);
  }}

  for (let i = 0; i < incoming.length; i++) {{
    drawLineGlow(incoming[i], "rgba(118, 178, 255, 0.78)", 1.35, incomingAlpha);
  }}

  drawDetector(payload.detector, phase);
  drawPlateBox(
    payload.plate_boxes.negative.map(shiftedPlate),
    "rgba(45, 119, 255, 0.36)",
    "rgba(119, 173, 255, 0.95)",
    plateAlpha
  );
  drawPlateBox(
    payload.plate_boxes.positive.map(shiftedPlate),
    "rgba(255, 76, 76, 0.36)",
    "rgba(255, 145, 145, 0.98)",
    plateAlpha
  );

  for (let i = 0; i < chaotic.length; i++) {{
    const chaosColor = i % 3 === 0 ? "rgba(255, 214, 92, 0.74)" : "rgba(255, 118, 92, 0.76)";
    drawLineGlow(chaotic[i], chaosColor, 1.45, chaoticAlpha);
  }}

  for (let i = 0; i < settled.length; i++) {{
    const edgeLine = maxRadialDistance(settled[i]) > payload.config.plate_length * 0.47;
    const settledColor = edgeLine ? "rgba(102, 227, 164, 0.68)" : "rgba(105, 208, 255, 0.54)";
    drawLineGlow(settled[i], settledColor, edgeLine ? 1.65 : 1.25, settledAlpha);
    if (settledAlpha > 0.18 && i % 4 === 0) drawDirectionArrow(settled[i], 0.54, "rgba(255, 244, 185, 0.82)", settledAlpha);
  }}

  const speed = payload.config.animation_speed;
  const base = timeMs * 0.00032 * speed;
  const activeLines = phase < 0.30 ? incoming : (phase < 0.66 ? chaotic : settled);
  const dotAlpha = Math.max(incomingAlpha, chaoticAlpha, settledAlpha);
  ctx.globalAlpha = dotAlpha;
  for (let i = 0; i < activeLines.length; i++) {{
    const line = activeLines[i];
    const t = (base + (i % 17) / 17) % 1.0;
    const p = project(sampleLine(line, t));
    const radius = (3.2 + 2.4 * p[2]) * devicePixelRatio;
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 214, 92, 0.96)";
    ctx.shadowColor = "rgba(255, 214, 92, 0.7)";
    ctx.shadowBlur = 9 * devicePixelRatio;
    ctx.fill();
    ctx.shadowBlur = 0;
  }}
  ctx.globalAlpha = 1;

  requestAnimationFrame(animate);
}}
requestAnimationFrame(animate);

function drawSceneGrid(timeMs) {{
  const span = payload.config.plate_length * 0.9;
  const y = -payload.config.plate_width * 0.78;
  const z = -payload.config.gap * 2.15;
  ctx.globalAlpha = 0.18;
  ctx.strokeStyle = "rgba(150, 172, 210, 0.42)";
  ctx.lineWidth = 0.8 * devicePixelRatio;
  for (let i = -4; i <= 4; i++) {{
    const a = project([-span, y + i * payload.config.plate_width * 0.18, z]);
    const b = project([span, y + i * payload.config.plate_width * 0.18, z]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }}
  for (let i = -4; i <= 4; i++) {{
    const a = project([i * payload.config.plate_length * 0.18, y - payload.config.plate_width * 0.72, z]);
    const b = project([i * payload.config.plate_length * 0.18, y + payload.config.plate_width * 0.72, z]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }}
  ctx.globalAlpha = 1;
}}

function maxRadialDistance(line) {{
  let maxValue = 0;
  for (const p of line) maxValue = Math.max(maxValue, Math.hypot(p[0], p[1]));
  return maxValue;
}}

function smoothstep(edge0, edge1, x) {{
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}}
</script>
</body>
</html>
"""
