"""Standalone HTML dashboards for optimizer benchmark evidence."""

from __future__ import annotations

import json


def export_optimizer_benchmark_html(results, path: str) -> None:
    """Export an animated optimizer convergence dashboard.

    The dashboard is deliberately dependency-free: it reads the optimizer
    histories already computed in Python and animates them on a browser canvas.
    This complements the static PNG and gives an immediate visual sense of
    convergence speed.
    """

    payload = {
        "series": [
            {
                "name": result.optimizer,
                "loss": [float(value) for value in result.loss_history],
                "final_loss": float(result.final_loss),
                "pde_loss": float(result.pde_loss),
                "plate_loss": float(result.plate_loss),
                "spectral_entropy": float(result.spectral_entropy),
            }
            for result in results
        ]
    }
    with open(path, "w", encoding="utf-8") as file:
        file.write(_dashboard_html(payload))


def _dashboard_html(payload: dict) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hamiltonian-Geometric Optimizer Benchmark</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      min-height: 100%;
      background: #101318;
      color: #eef2f6;
      font-family: Arial, sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 22px;
    }}
    canvas {{
      width: 100%;
      height: 560px;
      background: #fbfcfd;
      border-radius: 8px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .card {{
      background: #171c24;
      border: 1px solid rgba(238, 242, 246, 0.12);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .muted {{ color: #aab4c0; }}
  </style>
</head>
<body>
<main>
  <h1>Hamiltonian-Geometric Optimizer Benchmark</h1>
  <p class="muted">Animated log-loss convergence for SGD, Adam, falling-ball, entropy-descent, and Hamiltonian-geometric optimization on the capacitor fringing-field PINN loss.</p>
  <canvas id="chart"></canvas>
  <div id="cards" class="grid"></div>
</main>
<script>
const payload = {data};
const colors = {{
  sgd: "#9aa3ab",
  falling_ball: "#4f83b5",
  adam: "#e8a33d",
  entropy_descent: "#c0392b",
  hamiltonian_geometric: "#6a3d9a"
}};
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const cards = document.getElementById("cards");
let start = performance.now();

function resize() {{
  canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
}}
window.addEventListener("resize", resize);
resize();

const allLoss = payload.series.flatMap(s => s.loss);
const minLog = Math.log10(Math.max(Math.min(...allLoss), 1e-12));
const maxLog = Math.log10(Math.max(...allLoss));
const maxSteps = Math.max(...payload.series.map(s => s.loss.length));

cards.innerHTML = payload.series
  .slice()
  .sort((a, b) => a.final_loss - b.final_loss)
  .map(s => `<div class="card"><strong style="color:${{colors[s.name] || "#ddd"}}">${{s.name}}</strong><br>final loss: ${{s.final_loss.toExponential(3)}}<br>PDE loss: ${{s.pde_loss.toExponential(3)}}<br>spectral entropy: ${{s.spectral_entropy.toFixed(3)}}</div>`)
  .join("");

function xScale(step) {{
  const left = 70 * devicePixelRatio;
  const right = canvas.width - 28 * devicePixelRatio;
  return left + (right - left) * (step / Math.max(1, maxSteps - 1));
}}

function yScale(loss) {{
  const top = 34 * devicePixelRatio;
  const bottom = canvas.height - 58 * devicePixelRatio;
  const value = Math.log10(Math.max(loss, 1e-12));
  const t = (value - minLog) / Math.max(1e-9, maxLog - minLog);
  return bottom - (bottom - top) * t;
}}

function drawAxes() {{
  ctx.strokeStyle = "#d4d9df";
  ctx.lineWidth = devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(xScale(0), 34 * devicePixelRatio);
  ctx.lineTo(xScale(0), canvas.height - 58 * devicePixelRatio);
  ctx.lineTo(canvas.width - 28 * devicePixelRatio, canvas.height - 58 * devicePixelRatio);
  ctx.stroke();
  ctx.fillStyle = "#28313d";
  ctx.font = `${{12 * devicePixelRatio}}px Arial`;
  ctx.fillText("loss (log scale)", 12 * devicePixelRatio, 28 * devicePixelRatio);
  ctx.fillText("training step", canvas.width - 142 * devicePixelRatio, canvas.height - 20 * devicePixelRatio);
}}

function draw(time) {{
  const elapsed = (time - start) / 1000;
  const progress = (elapsed % 8.0) / 8.0;
  const visibleSteps = Math.max(2, Math.floor(1 + progress * (maxSteps - 1)));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawAxes();

  for (const series of payload.series) {{
    ctx.strokeStyle = colors[series.name] || "#333";
    ctx.lineWidth = (series.name.includes("hamiltonian") || series.name.includes("entropy")) ? 3.0 * devicePixelRatio : 1.8 * devicePixelRatio;
    ctx.beginPath();
    for (let i = 0; i < Math.min(visibleSteps, series.loss.length); i++) {{
      const x = xScale(i);
      const y = yScale(series.loss[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    const last = Math.min(visibleSteps - 1, series.loss.length - 1);
    ctx.fillStyle = colors[series.name] || "#333";
    ctx.beginPath();
    ctx.arc(xScale(last), yScale(series.loss[last]), 4.0 * devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
  }}
  requestAnimationFrame(draw);
}}
requestAnimationFrame(draw);
</script>
</body>
</html>
"""
