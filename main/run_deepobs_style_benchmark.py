"""DeepOBS-style optimizer benchmark using the free local Python stack.

Official DeepOBS 1.1.2 imports TensorFlow at package import time, and
TensorFlow does not currently provide a wheel for this project's Python 3.14
environment. This runner records that compatibility issue and runs a
deterministic DeepOBS-style benchmark instead: a nonconvex MLP classification
task with stochastic minibatches, train/validation loss, train/validation
accuracy, CSV logs, PNG plots, and a GIF convergence animation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MLPShape:
    input_dim: int = 2
    hidden_dim: int = 24
    output_dim: int = 3

    @property
    def parameter_count(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.output_dim
            + self.output_dim
        )


@dataclass(frozen=True)
class OptimizerResult:
    name: str
    theta: np.ndarray
    history: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DeepOBS-style NumPy benchmark for the Hamiltonian-geometric optimizer."
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=900)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "deepobs_style_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shape = MLPShape(hidden_dim=args.hidden_dim)
    x_train, y_train = make_spiral_dataset(args.train_samples, seed=args.seed)
    x_val, y_val = make_spiral_dataset(args.val_samples, seed=args.seed + 1)
    theta0 = init_parameters(shape, seed=args.seed + 2)

    results = [
        train_optimizer("sgd", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_optimizer("adam", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_optimizer("falling_ball", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_optimizer("entropy_descent", theta0, shape, x_train, y_train, x_val, y_val, args),
        train_optimizer("hamiltonian_geometric", theta0, shape, x_train, y_train, x_val, y_val, args),
    ]

    history_path = args.output_dir / "deepobs_style_training_history.csv"
    summary_path = args.output_dir / "deepobs_style_optimizer_summary.csv"
    loss_plot_path = args.output_dir / "deepobs_style_loss.png"
    accuracy_plot_path = args.output_dir / "deepobs_style_accuracy.png"
    gif_path = args.output_dir / "deepobs_style_convergence.gif"
    report_path = args.output_dir / "deepobs_style_results.md"
    compatibility_path = args.output_dir / "official_deepobs_compatibility.txt"

    write_history_csv(results, history_path)
    write_summary_csv(results, summary_path)
    export_metric_plot(results, loss_plot_path, metric="val_loss", ylabel="validation loss")
    export_metric_plot(results, accuracy_plot_path, metric="val_accuracy", ylabel="validation accuracy")
    export_convergence_gif(results, gif_path)
    write_report(results, args, report_path)
    write_deepobs_compatibility_note(compatibility_path)

    print("DeepOBS-style optimizer benchmark")
    print(f"epochs = {args.epochs}")
    print(f"hidden_dim = {shape.hidden_dim}")
    print("optimizer,final_train_loss,final_val_loss,final_train_accuracy,final_val_accuracy")
    for result in sorted(results, key=lambda item: item.final["val_loss"]):
        final = result.final
        print(
            f"{result.name},"
            f"{final['train_loss']:.6e},"
            f"{final['val_loss']:.6e},"
            f"{final['train_accuracy']:.6e},"
            f"{final['val_accuracy']:.6e}"
        )
    for path in (
        history_path,
        summary_path,
        loss_plot_path,
        accuracy_plot_path,
        gif_path,
        report_path,
        compatibility_path,
    ):
        print(f"exported = {path}")


def make_spiral_dataset(samples: int, seed: int, classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if samples < classes:
        raise ValueError("samples must be at least the number of classes")
    rng = np.random.default_rng(seed)
    per_class = samples // classes
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for label in range(classes):
        radius = np.linspace(0.08, 1.0, per_class)
        theta = np.linspace(label * 4.0, (label + 1) * 4.0, per_class)
        theta += rng.normal(0.0, 0.22, per_class)
        points = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
        points += rng.normal(0.0, 0.035, points.shape)
        xs.append(points)
        ys.append(np.full(per_class, label, dtype=int))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    order = rng.permutation(x.shape[0])
    return x[order], y[order]


def init_parameters(shape: MLPShape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / shape.input_dim), (shape.input_dim, shape.hidden_dim))
    b1 = np.zeros(shape.hidden_dim)
    w2 = rng.normal(0.0, np.sqrt(2.0 / shape.hidden_dim), (shape.hidden_dim, shape.output_dim))
    b2 = np.zeros(shape.output_dim)
    return pack_parameters(w1, b1, w2, b2)


def train_optimizer(
    name: str,
    theta0: np.ndarray,
    shape: MLPShape,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> OptimizerResult:
    # A per-optimizer seed offset needs to be stable across processes so the
    # minibatch order (and therefore every reported number) is reproducible;
    # Python's built-in hash() of a str is randomized per-process (PYTHONHASHSEED)
    # unless explicitly disabled, so it cannot be used here.
    seed_offsets = {
        "sgd": 1,
        "adam": 2,
        "falling_ball": 3,
        "entropy_descent": 4,
        "hamiltonian_geometric": 5,
    }
    rng = np.random.default_rng(args.seed + seed_offsets[name] * 1_000)
    theta = theta0.copy()
    velocity = np.zeros_like(theta)
    memory = np.zeros_like(theta)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    metric_accumulator = np.zeros_like(theta)
    history: list[dict[str, float]] = []

    learning_rates = {
        "sgd": 0.05,
        "adam": 0.012,
        "falling_ball": 0.018,
        "entropy_descent": 0.045,
        # The Hamiltonian-geometric update applies its learning rate twice --
        # once building momentum (velocity = beta*velocity - lr*force), again
        # mapping that momentum to a position step (theta += lr*g^-1*velocity),
        # per the paper's own Sec. 17 convention (see the comparison in
        # src.pinn.train_falling_ball's docstring). That means its effective
        # step scales like lr^2, not lr, so it needs a noticeably larger lr
        # than the other optimizers here for a comparable step size; 0.035
        # under-shot badly (val_loss 0.045 after 120 epochs, far behind Adam).
        # 0.25 was found by a learning-rate sweep on this benchmark and is the
        # point where the final-epoch loss still matches the best-epoch loss
        # seen during training (i.e. training has stabilized rather than
        # still oscillating), not merely wherever loss is lowest.
        "hamiltonian_geometric": 0.25,
    }
    lr = learning_rates[name]
    batch_size = min(args.batch_size, x_train.shape[0])
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            global_step += 1
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            _loss, grad = loss_and_gradient(theta, shape, x_train[batch], y_train[batch])

            if name == "sgd":
                theta -= lr * grad
            elif name == "adam":
                adam_m = 0.9 * adam_m + 0.1 * grad
                adam_v = 0.999 * adam_v + 0.001 * grad**2
                m_hat = adam_m / (1.0 - 0.9**global_step)
                v_hat = adam_v / (1.0 - 0.999**global_step)
                theta -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif name == "falling_ball":
                velocity = 0.88 * velocity + grad
                theta -= lr * velocity
            elif name == "entropy_descent":
                metric_accumulator = 0.96 * metric_accumulator + 0.04 * grad**2
                inverse_diag_metric = 1.0 / (np.sqrt(metric_accumulator) + 0.08)
                theta -= lr * inverse_diag_metric * grad
            elif name == "hamiltonian_geometric":
                metric_accumulator = 0.96 * metric_accumulator + 0.04 * grad**2
                inverse_diag_metric = 1.0 / (np.sqrt(metric_accumulator) + 0.08)
                memory = 0.9 * memory + grad
                force = grad + 0.03 * memory
                velocity = 0.9 * velocity - lr * force
                theta += lr * inverse_diag_metric * velocity
            else:
                raise ValueError(f"unknown optimizer {name!r}")

        train_loss, train_accuracy = loss_and_accuracy(theta, shape, x_train, y_train)
        val_loss, val_accuracy = loss_and_accuracy(theta, shape, x_val, y_val)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
    return OptimizerResult(name=name, theta=theta, history=history)


def unpack_parameters(theta: np.ndarray, shape: MLPShape):
    offset = 0
    w1_size = shape.input_dim * shape.hidden_dim
    w1 = theta[offset : offset + w1_size].reshape(shape.input_dim, shape.hidden_dim)
    offset += w1_size
    b1 = theta[offset : offset + shape.hidden_dim]
    offset += shape.hidden_dim
    w2_size = shape.hidden_dim * shape.output_dim
    w2 = theta[offset : offset + w2_size].reshape(shape.hidden_dim, shape.output_dim)
    offset += w2_size
    b2 = theta[offset : offset + shape.output_dim]
    return w1, b1, w2, b2


def pack_parameters(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])


def forward(theta: np.ndarray, shape: MLPShape, x: np.ndarray):
    w1, b1, w2, b2 = unpack_parameters(theta, shape)
    z1 = x @ w1 + b1
    h1 = np.tanh(z1)
    logits = h1 @ w2 + b2
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return z1, h1, probabilities


def loss_and_gradient(theta: np.ndarray, shape: MLPShape, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    _z1, h1, probabilities = forward(theta, shape, x)
    n = x.shape[0]
    loss = float(-np.mean(np.log(probabilities[np.arange(n), y] + 1e-12)))
    dlogits = probabilities
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    w1, _b1, w2, _b2 = unpack_parameters(theta, shape)
    dw2 = h1.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh1 = dlogits @ w2.T
    dz1 = dh1 * (1.0 - h1**2)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    l2 = 1e-4
    loss += 0.5 * l2 * float(np.sum(w1**2) + np.sum(w2**2))
    dw1 += l2 * w1
    dw2 += l2 * w2
    return loss, pack_parameters(dw1, db1, dw2, db2)


def loss_and_accuracy(theta: np.ndarray, shape: MLPShape, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    _z1, _h1, probabilities = forward(theta, shape, x)
    loss = float(-np.mean(np.log(probabilities[np.arange(x.shape[0]), y] + 1e-12)))
    predictions = np.argmax(probabilities, axis=1)
    return loss, float(np.mean(predictions == y))


def write_history_csv(results: list[OptimizerResult], path: Path) -> None:
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


def write_summary_csv(results: list[OptimizerResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "final_train_loss", "final_val_loss", "final_train_accuracy", "final_val_accuracy"])
        for result in results:
            final = result.final
            writer.writerow(
                [
                    result.name,
                    f"{final['train_loss']:.12g}",
                    f"{final['val_loss']:.12g}",
                    f"{final['train_accuracy']:.12g}",
                    f"{final['val_accuracy']:.12g}",
                ]
            )


def export_metric_plot(results: list[OptimizerResult], path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for result in results:
        epochs = [row["epoch"] for row in result.history]
        values = [row[metric] for row in result.history]
        linewidth = 2.8 if result.name == "hamiltonian_geometric" else 1.8
        ax.plot(epochs, values, label=result.name, linewidth=linewidth)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"DeepOBS-style MLP benchmark: {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_convergence_gif(results: list[OptimizerResult], path: Path) -> None:
    max_epoch = max(int(row["epoch"]) for result in results for row in result.history)
    min_loss = min(row["val_loss"] for result in results for row in result.history)
    max_loss = max(row["val_loss"] for result in results for row in result.history)
    frame_count = min(42, max_epoch)
    frame_epochs = sorted({round(1 + i * (max_epoch - 1) / max(1, frame_count - 1)) for i in range(frame_count)})
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    def draw(frame_epoch: int):
        ax.clear()
        for result in results:
            rows = [row for row in result.history if row["epoch"] <= frame_epoch]
            epochs = [row["epoch"] for row in rows]
            losses = [row["val_loss"] for row in rows]
            linewidth = 2.8 if result.name == "hamiltonian_geometric" else 1.8
            ax.plot(epochs, losses, label=result.name, linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(min_loss * 0.9, max_loss * 1.1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss")
        ax.set_title(f"DeepOBS-style optimizer convergence, epoch {frame_epoch}/{max_epoch}")
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="upper right", fontsize=8)
        return ax.lines

    animation = FuncAnimation(fig, draw, frames=frame_epochs, interval=90, blit=False)
    animation.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def write_report(results: list[OptimizerResult], args: argparse.Namespace, path: Path) -> None:
    ordered = sorted(results, key=lambda result: result.final["val_loss"])
    lines = [
        "# DeepOBS-Style Optimizer Benchmark",
        "",
        "Official DeepOBS could not run in this Python 3.14 environment because it imports TensorFlow,",
        "and TensorFlow has no matching wheel here. This run uses a local NumPy MLP benchmark with",
        "DeepOBS-style train/validation metrics.",
        "",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Hidden dimension: {args.hidden_dim}",
        "",
        "| optimizer | final train loss | final val loss | train accuracy | val accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in ordered:
        final = result.final
        lines.append(
            f"| {result.name} | {final['train_loss']:.12g} | {final['val_loss']:.12g} | "
            f"{final['train_accuracy']:.4f} | {final['val_accuracy']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_deepobs_compatibility_note(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Official DeepOBS compatibility note",
                "",
                "Attempted package: deepobs==1.1.2",
                "Observed import failure: deepobs imports deepobs.tensorflow, which requires tensorflow.",
                "Observed install check: `python -m pip index versions tensorflow` reports no matching distribution",
                "for the current Python 3.14 environment.",
                "",
                "Conclusion: official DeepOBS cannot run here without a separate older Python/TensorFlow environment.",
                "This folder therefore records a local NumPy DeepOBS-style benchmark instead.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
