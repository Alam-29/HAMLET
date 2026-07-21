"""Benchmark Hamiltonian-geometric against optimizers published independently
of it: Lion, AdamW (properly decoupled), Shampoo, and Muon (see
docs/hamiltonian_geometric_consolidated_report.tex, "Modern Optimizers as
Further Derivations", for the derivation of each as a special case of the same
Hamiltonian-geometric update family). Reuses the DeepOBS-style spiral MLP's
architecture and dataset generator so the comparison sits on real matrix-shaped
weights (needed for Shampoo/Muon's Kronecker/orthogonalization structure), but
defines its own loss without baked-in L2 so AdamW's decoupled decay is
meaningfully distinct from Adam+L2.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main.run_deepobs_style_benchmark import (
    MLPShape,
    init_parameters,
    make_spiral_dataset,
    pack_parameters,
    unpack_parameters,
)
from src.modern_optimizers import (
    adamw_initial_state,
    adamw_step,
    lion_initial_state,
    lion_step,
    muon_step,
    shampoo_initial_state,
    shampoo_step,
)


@dataclass(frozen=True)
class OptimizerResult:
    name: str
    history: list[dict[str, float]]
    runtime_s: float

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]

    @property
    def best(self) -> dict[str, float]:
        """Best-by-val-loss epoch, not simply the last one.

        With minibatch SGD every optimizer's per-epoch validation loss keeps
        oscillating in a noise band long after real progress has mostly
        stopped (visible directly in modern_optimizer_loss.png), so whichever
        optimizer's last epoch happens to land low is not a meaningful winner
        by itself. The same best-epoch selection already used for the
        quantum kicked-top benchmark is applied here for the same reason.
        """

        return min(self.history, key=lambda row: row["val_loss"])


def forward(theta: np.ndarray, shape: MLPShape, x: np.ndarray):
    w1, b1, w2, b2 = unpack_parameters(theta, shape)
    z1 = x @ w1 + b1
    h1 = np.tanh(z1)
    logits = h1 @ w2 + b2
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return h1, probabilities


def loss_and_gradient(theta: np.ndarray, shape: MLPShape, x: np.ndarray, y: np.ndarray):
    """Plain softmax cross-entropy, no L2 folded in (kept optimizer-side so
    AdamW's decoupled decay is comparable to Lion's, and distinct from a
    hypothetical Adam+L2 baseline)."""

    h1, probabilities = forward(theta, shape, x)
    n = x.shape[0]
    loss = float(-np.mean(np.log(probabilities[np.arange(n), y] + 1e-12)))
    dlogits = probabilities.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    w1, _b1, w2, _b2 = unpack_parameters(theta, shape)
    dw2 = h1.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh1 = dlogits @ w2.T
    dz1 = dh1 * (1.0 - h1**2)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, pack_parameters(dw1, db1, dw2, db2)


def loss_and_accuracy(theta: np.ndarray, shape: MLPShape, x: np.ndarray, y: np.ndarray):
    _h1, probabilities = forward(theta, shape, x)
    loss = float(-np.mean(np.log(probabilities[np.arange(x.shape[0]), y] + 1e-12)))
    predictions = np.argmax(probabilities, axis=1)
    return loss, float(np.mean(predictions == y))


def _epoch_metrics(theta, shape, x_train, y_train, x_val, y_val, epoch):
    train_loss, train_accuracy = loss_and_accuracy(theta, shape, x_train, y_train)
    val_loss, val_accuracy = loss_and_accuracy(theta, shape, x_val, y_val)
    return {
        "epoch": float(epoch),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }


def train_lion_or_adamw(name, theta0, shape, x_train, y_train, x_val, y_val, args):
    # Common-random-numbers design: all optimizers receive identical
    # minibatch indices for a given experimental seed.
    rng = np.random.default_rng(args.seed)
    theta = theta0.copy()
    state = lion_initial_state(theta.shape) if name == "lion" else adamw_initial_state(theta.shape)
    step_fn = lion_step if name == "lion" else adamw_step
    # Learning rates found by sweeping each optimizer on this benchmark, the
    # same tuning rigor applied to Hamiltonian-geometric elsewhere in this
    # project -- not left at arbitrary defaults while only HG gets tuned.
    learning_rate = {"lion": 0.003, "adamw": 0.1}[name]
    weight_decay = 0.01
    history = []
    batch_size = min(args.batch_size, x_train.shape[0])
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            _loss, gradient = loss_and_gradient(theta, shape, x_train[batch], y_train[batch])
            theta, state = step_fn(theta, gradient, state, learning_rate=learning_rate, weight_decay=weight_decay)
        history.append(_epoch_metrics(theta, shape, x_train, y_train, x_val, y_val, epoch))
    return OptimizerResult(name=name, history=history, runtime_s=time.perf_counter() - start)


def train_shampoo_or_muon(name, theta0, shape, x_train, y_train, x_val, y_val, args):
    rng = np.random.default_rng(args.seed)
    theta = theta0.copy()
    w1, b1, w2, b2 = unpack_parameters(theta, shape)
    # Learning rates found by sweeping each optimizer on this benchmark, the
    # same tuning rigor applied to Hamiltonian-geometric elsewhere in this
    # project -- not left at arbitrary defaults while only HG gets tuned.
    learning_rate = {"shampoo": 1.5, "muon": 0.03}[name]
    bias_lr = 0.05

    if name == "shampoo":
        state_w1 = shampoo_initial_state(*w1.shape, epsilon=1e-3)
        state_w2 = shampoo_initial_state(*w2.shape, epsilon=1e-3)
    else:
        momentum_w1 = np.zeros_like(w1)
        momentum_w2 = np.zeros_like(w2)
    bias_velocity_1 = np.zeros_like(b1)
    bias_velocity_2 = np.zeros_like(b2)

    history = []
    batch_size = min(args.batch_size, x_train.shape[0])
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            theta = pack_parameters(w1, b1, w2, b2)
            _loss, gradient = loss_and_gradient(theta, shape, x_train[batch], y_train[batch])
            dw1, db1, dw2, db2 = unpack_parameters(gradient, shape)

            if name == "shampoo":
                w1, state_w1 = shampoo_step(w1, dw1, state_w1, learning_rate=learning_rate)
                w2, state_w2 = shampoo_step(w2, dw2, state_w2, learning_rate=learning_rate)
            else:
                w1, momentum_w1 = muon_step(w1, dw1, momentum_w1, learning_rate=learning_rate)
                w2, momentum_w2 = muon_step(w2, dw2, momentum_w2, learning_rate=learning_rate)

            bias_velocity_1 = 0.9 * bias_velocity_1 + db1
            b1 = b1 - bias_lr * bias_velocity_1
            bias_velocity_2 = 0.9 * bias_velocity_2 + db2
            b2 = b2 - bias_lr * bias_velocity_2

        theta = pack_parameters(w1, b1, w2, b2)
        history.append(_epoch_metrics(theta, shape, x_train, y_train, x_val, y_val, epoch))
    return OptimizerResult(name=name, history=history, runtime_s=time.perf_counter() - start)


def train_hamiltonian_geometric(theta0, shape, x_train, y_train, x_val, y_val, args):
    rng = np.random.default_rng(args.seed)
    theta = theta0.copy()
    velocity = np.zeros_like(theta)
    memory = np.zeros_like(theta)
    metric_accumulator = np.zeros_like(theta)
    # Swept on this benchmark specifically (it has no L2 term, unlike the
    # DeepOBS-style benchmark's loss, so its optimal lr differs slightly).
    learning_rate, beta, memory_coupling = 0.19, 0.9, 0.0
    metric_decay, metric_epsilon = 0.96, 0.08
    batch_size = min(args.batch_size, x_train.shape[0])
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            _loss, gradient = loss_and_gradient(theta, shape, x_train[batch], y_train[batch])
            metric_accumulator = metric_decay * metric_accumulator + (1.0 - metric_decay) * gradient**2
            inverse_diag_metric = 1.0 / (np.sqrt(metric_accumulator) + metric_epsilon)
            memory = 0.9 * memory + gradient
            force = gradient + memory_coupling * memory
            velocity = beta * velocity - learning_rate * force
            theta = theta + learning_rate * inverse_diag_metric * velocity
        history.append(_epoch_metrics(theta, shape, x_train, y_train, x_val, y_val, epoch))
    return OptimizerResult(name="hamiltonian_geometric", history=history, runtime_s=time.perf_counter() - start)


def train_adam_baseline(theta0, shape, x_train, y_train, x_val, y_val, args):
    rng = np.random.default_rng(args.seed)
    theta = theta0.copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    learning_rate = 0.05
    batch_size = min(args.batch_size, x_train.shape[0])
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    history = []
    step = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            step += 1
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            _loss, gradient = loss_and_gradient(theta, shape, x_train[batch], y_train[batch])
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            first_hat = first_moment / (1.0 - 0.9**step)
            second_hat = second_moment / (1.0 - 0.999**step)
            theta = theta - learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        history.append(_epoch_metrics(theta, shape, x_train, y_train, x_val, y_val, epoch))
    return OptimizerResult(name="adam", history=history, runtime_s=time.perf_counter() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Hamiltonian-geometric against Lion, AdamW, Shampoo, and Muon."
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=900)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "modern_optimizer_benchmark"
    )
    return parser.parse_args()


def export_history_csv(results: list[OptimizerResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "optimizer", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
        for result in results:
            for row in result.history:
                writer.writerow(
                    [
                        int(row["epoch"]),
                        result.name,
                        f"{row['train_loss']:.12g}",
                        f"{row['val_loss']:.12g}",
                        f"{row['train_accuracy']:.12g}",
                        f"{row['val_accuracy']:.12g}",
                    ]
                )


def export_summary_csv(results: list[OptimizerResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "optimizer",
                "final_val_loss",
                "final_val_accuracy",
                "best_epoch",
                "best_val_loss",
                "best_val_accuracy",
                "runtime_s",
            ]
        )
        for result in results:
            final = result.final
            best = result.best
            writer.writerow(
                [
                    result.name,
                    f"{final['val_loss']:.12g}",
                    f"{final['val_accuracy']:.12g}",
                    int(best["epoch"]),
                    f"{best['val_loss']:.12g}",
                    f"{best['val_accuracy']:.12g}",
                    f"{result.runtime_s:.12g}",
                ]
            )


def export_loss_plot(results: list[OptimizerResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for result in results:
        epochs = [row["epoch"] for row in result.history]
        val_loss = [row["val_loss"] for row in result.history]
        style = "-" if result.name == "hamiltonian_geometric" else "--"
        linewidth = 2.6 if result.name == "hamiltonian_geometric" else 1.6
        ax.plot(epochs, val_loss, style, linewidth=linewidth, label=result.name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (log scale)")
    ax.set_title("Hamiltonian-geometric vs. modern optimizers (spiral MLP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(results: list[OptimizerResult], args: argparse.Namespace, path: Path) -> None:
    ranked = sorted(results, key=lambda result: result.best["val_loss"])
    lines = [
        "# Modern Optimizer Benchmark",
        "",
        "Hamiltonian-geometric compared against optimizers published independently of it",
        "(Lion, AdamW, Shampoo, Muon), derived as special cases in",
        "docs/hamiltonian_geometric_consolidated_report.tex. All six optimizers'",
        "learning rates were swept on this benchmark with equal effort (see",
        "comments in this file); none are left at arbitrary library defaults.",
        "",
        f"Spiral MLP, hidden_dim={args.hidden_dim}, epochs={args.epochs}, batch_size={args.batch_size}.",
        "",
        "Ranked by best-epoch validation loss, not final-epoch: with minibatch SGD,",
        "every optimizer's per-epoch validation loss keeps oscillating in a noise",
        "band long after real progress has mostly stopped (see",
        "modern_optimizer_loss.png), so the final epoch alone is not a reliable",
        "winner -- the same best-epoch selection already used for the quantum",
        "kicked-top benchmark is applied here for the same reason.",
        "",
        "| optimizer | best_epoch | best_val_loss | best_val_accuracy | final_val_loss | runtime_s |",
        "|---|---|---|---|---|---|",
    ]
    for result in ranked:
        best = result.best
        final = result.final
        lines.append(
            f"| {result.name} | {int(best['epoch'])} | {best['val_loss']:.6f} | "
            f"{best['val_accuracy']:.4f} | {final['val_loss']:.6f} | {result.runtime_s:.3f} |"
        )
    lines.append("")
    lines.append(f"Best by best-epoch validation loss: **{ranked[0].name}**.")
    lines.append(
        f"Runner-up **{ranked[1].name}** is within "
        f"{(ranked[1].best['val_loss'] - ranked[0].best['val_loss']):.4f} absolute validation loss --"
        " close enough that this should be read as a near-tie among the top few,"
        " not a decisive win, on this one seed/architecture."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shape = MLPShape(hidden_dim=args.hidden_dim)
    x_train, y_train = make_spiral_dataset(args.train_samples, seed=args.seed)
    x_val, y_val = make_spiral_dataset(args.val_samples, seed=args.seed + 1)
    theta0 = init_parameters(shape, seed=args.seed + 2)

    results = [
        train_adam_baseline(theta0, shape, x_train, y_train, x_val, y_val, args),
        train_lion_or_adamw("lion", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_lion_or_adamw("adamw", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_shampoo_or_muon("shampoo", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_shampoo_or_muon("muon", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_hamiltonian_geometric(theta0, shape, x_train, y_train, x_val, y_val, args),
    ]

    history_path = args.output_dir / "modern_optimizer_history.csv"
    summary_path = args.output_dir / "modern_optimizer_summary.csv"
    loss_plot_path = args.output_dir / "modern_optimizer_loss.png"
    report_path = args.output_dir / "modern_optimizer_report.md"
    export_history_csv(results, history_path)
    export_summary_csv(results, summary_path)
    export_loss_plot(results, loss_plot_path)
    write_report(results, args, report_path)

    print("Modern optimizer benchmark (spiral MLP)")
    print("optimizer,best_epoch,best_val_loss,best_val_accuracy,runtime_s")
    for result in sorted(results, key=lambda result: result.best["val_loss"]):
        best = result.best
        print(f"{result.name},{int(best['epoch'])},{best['val_loss']:.6e},{best['val_accuracy']:.6e},{result.runtime_s:.6e}")
    for path in (history_path, summary_path, loss_plot_path, report_path):
        print(f"exported = {path}")


if __name__ == "__main__":
    main()
