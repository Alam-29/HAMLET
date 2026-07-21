"""AlgoPerf-protocol benchmark for optimizer comparison.

MLCommons AlgoPerf evaluates training algorithms by comparing time/steps to
target across fixed workloads. This benchmark aligns the optimizer comparison
with AlgoPerf's target-setting baseline families -- AdamW, Nesterov Momentum,
and Heavy Ball Momentum -- under a local, reproducible NumPy implementation
of the same evaluation protocol, avoiding the JAX/PyTorch harness dependency
of the reference benchmark.

It also includes this project's Hamiltonian-Geometric optimizer and records a
small external-tuning-style search over workload-agnostic hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main.run_deepobs_style_benchmark import (
    MLPShape,
    OptimizerResult,
    init_parameters,
    loss_and_accuracy,
    loss_and_gradient,
    make_spiral_dataset,
)


@dataclass(frozen=True)
class TrialResult:
    optimizer: str
    trial: int
    hyperparameters: dict[str, float]
    result: OptimizerResult
    runtime_s: float

    @property
    def final_val_loss(self) -> float:
        return self.result.final["val_loss"]

    @property
    def final_val_accuracy(self) -> float:
        return self.result.final["val_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an AlgoPerf-protocol optimizer benchmark."
    )
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=900)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--max-trials-per-optimizer",
        type=int,
        default=6,
        help="Local external-tuning budget per optimizer. Official target setting uses much larger budgets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "algoperf_style_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shape = MLPShape(hidden_dim=args.hidden_dim)
    x_train, y_train = make_spiral_dataset(args.train_samples, seed=args.seed)
    x_val, y_val = make_spiral_dataset(args.val_samples, seed=args.seed + 1)
    theta0 = init_parameters(shape, seed=args.seed + 2)

    all_trials: list[TrialResult] = []
    for optimizer in ("adamw", "nag", "heavy_ball", "hamiltonian_geometric"):
        for trial_index, hyperparameters in enumerate(search_space(optimizer)[: args.max_trials_per_optimizer], 1):
            start = time.perf_counter()
            result = train_optimizer(
                optimizer,
                theta0,
                shape,
                x_train,
                y_train,
                x_val,
                y_val,
                args,
                hyperparameters,
            )
            runtime_s = time.perf_counter() - start
            all_trials.append(
                TrialResult(
                    optimizer=optimizer,
                    trial=trial_index,
                    hyperparameters=hyperparameters,
                    result=result,
                    runtime_s=runtime_s,
                )
            )

    best_trials = [
        min(
            [trial for trial in all_trials if trial.optimizer == optimizer],
            key=lambda trial: trial.final_val_loss,
        )
        for optimizer in ("adamw", "nag", "heavy_ball", "hamiltonian_geometric")
    ]

    trial_path = args.output_dir / "algoperf_style_all_trials.csv"
    summary_path = args.output_dir / "algoperf_style_best_summary.csv"
    history_path = args.output_dir / "algoperf_style_best_histories.csv"
    loss_plot_path = args.output_dir / "algoperf_style_validation_loss.png"
    accuracy_plot_path = args.output_dir / "algoperf_style_validation_accuracy.png"
    adam_hg_plot_path = args.output_dir / "adamw_vs_hamiltonian_geometric.png"
    adam_hg_zoom_plot_path = args.output_dir / "adamw_vs_hamiltonian_geometric_zoom.png"
    gif_path = args.output_dir / "algoperf_style_convergence.gif"
    report_path = args.output_dir / "algoperf_style_results.md"
    implementation_note_path = args.output_dir / "implementation_notes.txt"

    write_trials_csv(all_trials, trial_path)
    write_best_summary_csv(best_trials, summary_path)
    write_best_histories_csv(best_trials, history_path)
    export_best_metric_plot(best_trials, loss_plot_path, "val_loss", "validation loss")
    export_best_metric_plot(best_trials, accuracy_plot_path, "val_accuracy", "validation accuracy")
    export_adamw_hg_comparison_plot(best_trials, adam_hg_plot_path)
    export_adamw_hg_zoom_plot(best_trials, adam_hg_zoom_plot_path)
    export_best_convergence_gif(best_trials, gif_path)
    write_report(args, all_trials, best_trials, report_path)
    write_implementation_note(args, implementation_note_path)

    print("AlgoPerf-protocol optimizer benchmark")
    print(f"epochs = {args.epochs}")
    print(f"trials_per_optimizer = {args.max_trials_per_optimizer}")
    print("optimizer,best_trial,final_val_loss,final_val_accuracy,runtime_s,hyperparameters")
    for trial in sorted(best_trials, key=lambda item: item.final_val_loss):
        print(
            f"{trial.optimizer},"
            f"{trial.trial},"
            f"{trial.final_val_loss:.6e},"
            f"{trial.final_val_accuracy:.6e},"
            f"{trial.runtime_s:.6e},"
            f"{trial.hyperparameters}"
        )
    for path in (
        trial_path,
        summary_path,
        history_path,
        loss_plot_path,
        accuracy_plot_path,
        adam_hg_plot_path,
        adam_hg_zoom_plot_path,
        gif_path,
        report_path,
        implementation_note_path,
    ):
        print(f"exported = {path}")


def search_space(optimizer: str) -> list[dict[str, float]]:
    """Small workload-agnostic search spaces inspired by AlgoPerf baselines."""

    if optimizer == "adamw":
        return [
            {"learning_rate": lr, "beta1": beta1, "beta2": 0.999, "weight_decay": wd}
            for lr, beta1, wd in product((0.003, 0.01, 0.03), (0.9,), (1e-4, 1e-3))
        ]
    if optimizer == "nag":
        return [
            {"learning_rate": lr, "momentum": mom, "weight_decay": wd}
            for lr, mom, wd in product((0.01, 0.03, 0.06), (0.85, 0.9), (1e-4,))
        ]
    if optimizer == "heavy_ball":
        return [
            {"learning_rate": lr, "momentum": mom, "weight_decay": wd}
            for lr, mom, wd in product((0.01, 0.03, 0.06), (0.85, 0.9), (1e-4,))
        ]
    if optimizer == "hamiltonian_geometric":
        return [
            {
                "learning_rate": lr,
                "beta": beta,
                "memory_coupling": mem,
                "metric_decay": metric_decay,
                "metric_epsilon": eps,
                "weight_decay": 1e-4,
            }
            for lr, beta, mem, metric_decay, eps in product(
                (0.03, 0.01, 0.003, 0.001),
                (0.9, 0.88, 0.92),
                (0.001, 0.003, 0.01),
                (0.999, 0.995, 0.99),
                (1e-8,),
            )
        ]
    raise ValueError(f"unknown optimizer {optimizer!r}")


def train_optimizer(
    optimizer: str,
    theta0: np.ndarray,
    shape: MLPShape,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    hyperparameters: dict[str, float],
) -> OptimizerResult:
    # Common-random-numbers design: every optimizer receives exactly the same
    # minibatch sequence for a given experimental seed. This makes paired
    # optimizer differences attributable to the update rule rather than to
    # sampling luck.
    rng = np.random.default_rng(args.seed)
    theta = theta0.copy()
    velocity = np.zeros_like(theta)
    memory = np.zeros_like(theta)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    metric_accumulator = np.zeros_like(theta)
    history: list[dict[str, float]] = []
    batch_size = min(args.batch_size, x_train.shape[0])
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            global_step += 1
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            batch_x = x_train[batch]
            batch_y = y_train[batch]

            if optimizer == "nag":
                lookahead = theta - hyperparameters["momentum"] * velocity
                _loss, grad = loss_and_gradient(lookahead, shape, batch_x, batch_y)
                grad = grad + hyperparameters["weight_decay"] * lookahead
                velocity = hyperparameters["momentum"] * velocity + hyperparameters["learning_rate"] * grad
                theta = theta - velocity
                continue

            _loss, grad = loss_and_gradient(theta, shape, batch_x, batch_y)
            grad = grad + hyperparameters.get("weight_decay", 0.0) * theta

            if optimizer == "adamw":
                beta1 = hyperparameters["beta1"]
                beta2 = hyperparameters["beta2"]
                adam_m = beta1 * adam_m + (1.0 - beta1) * grad
                adam_v = beta2 * adam_v + (1.0 - beta2) * grad**2
                m_hat = adam_m / (1.0 - beta1**global_step)
                v_hat = adam_v / (1.0 - beta2**global_step)
                theta = theta * (1.0 - hyperparameters["learning_rate"] * hyperparameters["weight_decay"])
                theta = theta - hyperparameters["learning_rate"] * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif optimizer == "heavy_ball":
                velocity = hyperparameters["momentum"] * velocity + hyperparameters["learning_rate"] * grad
                theta = theta - velocity
            elif optimizer == "hamiltonian_geometric":
                metric_accumulator = (
                    hyperparameters["metric_decay"] * metric_accumulator
                    + (1.0 - hyperparameters["metric_decay"]) * grad**2
                )
                metric_hat = metric_accumulator / (1.0 - hyperparameters["metric_decay"] ** global_step)
                memory = 0.9 * memory + 0.1 * grad
                force = grad + hyperparameters["memory_coupling"] * (memory - grad)
                velocity = hyperparameters["beta"] * velocity + (1.0 - hyperparameters["beta"]) * force
                velocity_hat = velocity / (1.0 - hyperparameters["beta"] ** global_step)
                theta = theta * (1.0 - hyperparameters["learning_rate"] * hyperparameters["weight_decay"])
                theta = theta - hyperparameters["learning_rate"] * velocity_hat / (
                    np.sqrt(metric_hat) + hyperparameters["metric_epsilon"]
                )
            else:
                raise ValueError(f"unknown optimizer {optimizer!r}")

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
    return OptimizerResult(name=optimizer, theta=theta, history=history)


def write_trials_csv(trials: list[TrialResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "optimizer",
                "trial",
                "final_train_loss",
                "final_val_loss",
                "final_train_accuracy",
                "final_val_accuracy",
                "runtime_s",
                "hyperparameters",
            ]
        )
        for trial in trials:
            final = trial.result.final
            writer.writerow(
                [
                    trial.optimizer,
                    trial.trial,
                    f"{final['train_loss']:.12g}",
                    f"{final['val_loss']:.12g}",
                    f"{final['train_accuracy']:.12g}",
                    f"{final['val_accuracy']:.12g}",
                    f"{trial.runtime_s:.12g}",
                    trial.hyperparameters,
                ]
            )


def write_best_summary_csv(trials: list[TrialResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "optimizer",
                "best_trial",
                "final_train_loss",
                "final_val_loss",
                "final_train_accuracy",
                "final_val_accuracy",
                "runtime_s",
                "hyperparameters",
            ]
        )
        for trial in sorted(trials, key=lambda item: item.final_val_loss):
            final = trial.result.final
            writer.writerow(
                [
                    trial.optimizer,
                    trial.trial,
                    f"{final['train_loss']:.12g}",
                    f"{final['val_loss']:.12g}",
                    f"{final['train_accuracy']:.12g}",
                    f"{final['val_accuracy']:.12g}",
                    f"{trial.runtime_s:.12g}",
                    trial.hyperparameters,
                ]
            )


def write_best_histories_csv(trials: list[TrialResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "optimizer",
                "best_trial",
                "epoch",
                "train_loss",
                "val_loss",
                "train_accuracy",
                "val_accuracy",
            ]
        )
        for trial in sorted(trials, key=lambda item: item.optimizer):
            for row in trial.result.history:
                writer.writerow(
                    [
                        trial.optimizer,
                        trial.trial,
                        f"{row['epoch']:.0f}",
                        f"{row['train_loss']:.12g}",
                        f"{row['val_loss']:.12g}",
                        f"{row['train_accuracy']:.12g}",
                        f"{row['val_accuracy']:.12g}",
                    ]
                )


def export_best_metric_plot(trials: list[TrialResult], path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for trial in trials:
        epochs = [row["epoch"] for row in trial.result.history]
        values = [row[metric] for row in trial.result.history]
        linewidth = 2.8 if trial.optimizer == "hamiltonian_geometric" else 1.8
        ax.plot(epochs, values, label=f"{trial.optimizer} trial {trial.trial}", linewidth=linewidth)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"AlgoPerf-style best trials: {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_adamw_hg_comparison_plot(trials: list[TrialResult], path: Path) -> None:
    selected = {
        trial.optimizer: trial
        for trial in trials
        if trial.optimizer in {"adamw", "hamiltonian_geometric"}
    }
    if set(selected) != {"adamw", "hamiltonian_geometric"}:
        return

    adamw = selected["adamw"]
    hg = selected["hamiltonian_geometric"]
    epochs = np.array([row["epoch"] for row in adamw.result.history])
    adam_loss = np.array([row["val_loss"] for row in adamw.result.history])
    hg_loss = np.array([row["val_loss"] for row in hg.result.history])
    gap = adam_loss - hg_loss

    fig, (ax_loss, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_loss.plot(epochs, adam_loss, label=f"AdamW trial {adamw.trial}", color="#64748b", linewidth=2.3)
    ax_loss.plot(
        epochs,
        hg_loss,
        label=f"Hamiltonian-Geometric trial {hg.trial}",
        color="#2563eb",
        linewidth=2.8,
    )
    ax_loss.set_ylabel("validation loss")
    ax_loss.set_title("AdamW vs Hamiltonian-Geometric tuned validation curve")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend()

    ax_gap.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.55)
    ax_gap.fill_between(
        epochs,
        0.0,
        gap,
        where=gap >= 0.0,
        color="#2563eb",
        alpha=0.25,
        label="HG lower loss",
    )
    ax_gap.fill_between(
        epochs,
        0.0,
        gap,
        where=gap < 0.0,
        color="#64748b",
        alpha=0.25,
        label="AdamW lower loss",
    )
    ax_gap.plot(epochs, gap, color="#0f172a", linewidth=1.6)
    ax_gap.set_xlabel("epoch")
    ax_gap.set_ylabel("AdamW - HG")
    ax_gap.grid(True, alpha=0.25)
    ax_gap.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_adamw_hg_zoom_plot(trials: list[TrialResult], path: Path) -> None:
    selected = {
        trial.optimizer: trial
        for trial in trials
        if trial.optimizer in {"adamw", "hamiltonian_geometric"}
    }
    if set(selected) != {"adamw", "hamiltonian_geometric"}:
        return

    adamw = selected["adamw"]
    hg = selected["hamiltonian_geometric"]
    epochs = np.array([row["epoch"] for row in adamw.result.history])
    adam_loss = np.array([row["val_loss"] for row in adamw.result.history])
    hg_loss = np.array([row["val_loss"] for row in hg.result.history])
    gap = adam_loss - hg_loss
    zoom_mask = epochs >= max(1, epochs.max() * 0.45)

    fig, (ax_zoom, ax_gap) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_zoom.plot(epochs[zoom_mask], adam_loss[zoom_mask], label="AdamW", color="#64748b", linewidth=3.0)
    ax_zoom.plot(
        epochs[zoom_mask],
        hg_loss[zoom_mask],
        label="Hamiltonian-Geometric",
        color="#2563eb",
        linewidth=3.0,
    )
    y_values = np.concatenate([adam_loss[zoom_mask], hg_loss[zoom_mask]])
    padding = max(0.005, 0.12 * (y_values.max() - y_values.min()))
    ax_zoom.set_ylim(y_values.min() - padding, y_values.max() + padding)
    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_ylabel("validation loss")
    ax_zoom.set_title("Zoomed late-training loss")
    ax_zoom.grid(True, alpha=0.28)
    ax_zoom.legend()

    ax_gap.bar(epochs[zoom_mask], gap[zoom_mask], color=np.where(gap[zoom_mask] >= 0, "#2563eb", "#64748b"))
    ax_gap.axhline(0.0, color="#111827", linewidth=1.0)
    ax_gap.set_xlabel("epoch")
    ax_gap.set_ylabel("AdamW loss - HG loss")
    ax_gap.set_title("Positive means HG is lower")
    ax_gap.grid(True, axis="y", alpha=0.28)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_best_convergence_gif(trials: list[TrialResult], path: Path) -> None:
    max_epoch = max(int(row["epoch"]) for trial in trials for row in trial.result.history)
    min_loss = min(row["val_loss"] for trial in trials for row in trial.result.history)
    max_loss = max(row["val_loss"] for trial in trials for row in trial.result.history)
    frame_count = min(42, max_epoch)
    frame_epochs = sorted({round(1 + i * (max_epoch - 1) / max(1, frame_count - 1)) for i in range(frame_count)})
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    def draw(frame_epoch: int):
        ax.clear()
        for trial in trials:
            rows = [row for row in trial.result.history if row["epoch"] <= frame_epoch]
            epochs = [row["epoch"] for row in rows]
            losses = [row["val_loss"] for row in rows]
            linewidth = 2.8 if trial.optimizer == "hamiltonian_geometric" else 1.8
            ax.plot(epochs, losses, label=trial.optimizer, linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(min_loss * 0.9, max_loss * 1.1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss")
        ax.set_title(f"AlgoPerf-style convergence, epoch {frame_epoch}/{max_epoch}")
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="upper right", fontsize=8)
        return ax.lines

    animation = FuncAnimation(fig, draw, frames=frame_epochs, interval=90, blit=False)
    animation.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def write_report(args: argparse.Namespace, all_trials: list[TrialResult], best_trials: list[TrialResult], path: Path) -> None:
    ordered = sorted(best_trials, key=lambda trial: trial.final_val_loss)
    lines = [
        "# AlgoPerf-Protocol Optimizer Benchmark",
        "",
        "Optimizer comparison aligned with the MLCommons AlgoPerf target-setting baseline families,",
        "run under a local, reproducible NumPy implementation of the same evaluation protocol.",
        "",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Hidden dimension: {args.hidden_dim}",
        f"Trials per optimizer: {args.max_trials_per_optimizer}",
        "",
        "| optimizer | best trial | final val loss | final val accuracy | runtime s | hyperparameters |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for trial in ordered:
        lines.append(
            f"| {trial.optimizer} | {trial.trial} | {trial.final_val_loss:.12g} | "
            f"{trial.final_val_accuracy:.4f} | {trial.runtime_s:.4f} | `{trial.hyperparameters}` |"
        )
    lines.extend(
        [
            "",
            "## Tuning Protocol",
            "",
            "Each optimizer receives the same local trial budget under workload-agnostic search spaces,",
            "smaller than the tuning budget used in official AlgoPerf target-setting runs.",
            "",
            f"Total trials recorded: {len(all_trials)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_implementation_note(args: argparse.Namespace, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Implementation notes",
                "",
                "MLCommons AlgoPerf evaluates training algorithms by time-to-target on fixed workloads and",
                "hardware. This benchmark reimplements AlgoPerf's target-setting/reference baseline families --",
                "AdamW, Nesterov Momentum, and Heavy Ball Momentum -- directly in NumPy rather than depending on",
                "the reference `algorithmic-efficiency` harness, and uses a fixed workload-agnostic tuning",
                "budget rather than the substantially larger budget used in official target-setting runs.",
                "",
                f"Local max_trials_per_optimizer = {args.max_trials_per_optimizer}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
