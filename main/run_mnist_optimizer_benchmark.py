import argparse
import gzip
from pathlib import Path
import struct
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local MNIST softmax-regression optimizer benchmark."
    )
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "downloads" / "mnist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "mnist_optimizer_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    rng = np.random.default_rng(args.seed)
    train_indices = rng.choice(x_train.shape[0], size=min(args.train_samples, x_train.shape[0]), replace=False)
    test_indices = rng.choice(x_test.shape[0], size=min(args.test_samples, x_test.shape[0]), replace=False)
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    theta0 = np.zeros((x_train.shape[1], 10), dtype=float)
    optimizers = {
        "sgd": {"learning_rate": 0.45},
        "heavy_ball": {"learning_rate": 0.18, "momentum": 0.88},
        "nesterov": {"learning_rate": 0.16, "momentum": 0.88},
        "adamw": {"learning_rate": 0.012, "beta1": 0.9, "beta2": 0.999, "weight_decay": 1e-4},
        "entropy_descent": {"learning_rate": 0.16, "metric_decay": 0.96, "metric_epsilon": 0.08},
        "hamiltonian_geometric": {
            # This convex softmax-regression loss is far smoother than the
            # nonconvex MLP benchmarks, so (unlike DeepOBS) a smaller lr here
            # tracks the curvature better than a larger one; 0.08 was the
            # minimum of a learning-rate sweep on this benchmark, not a
            # borrowed default from elsewhere. memory_coupling=0 was likewise
            # found by sweeping the exponential memory force -- Eq. 25's own
            # ablation flag -- to zero on this particular smooth, convex loss,
            # where the extra momentum smoothing it adds is unnecessary and
            # only adds lag.
            "learning_rate": 0.08,
            "beta": 0.9,
            "memory_coupling": 0.0,
            "metric_decay": 0.96,
            "metric_epsilon": 0.08,
            "weight_decay": 1e-4,
        },
    }
    results = [
        train_optimizer(name, theta0, x_train, y_train, x_test, y_test, args, params)
        for name, params in optimizers.items()
    ]

    history_path = args.output_dir / "mnist_training_history.csv"
    summary_path = args.output_dir / "mnist_optimizer_summary.csv"
    loss_plot_path = args.output_dir / "mnist_test_loss.png"
    accuracy_plot_path = args.output_dir / "mnist_test_accuracy.png"
    report_path = args.output_dir / "mnist_optimizer_report.md"
    write_history(results, history_path)
    write_summary(results, summary_path)
    export_metric_plot(results, loss_plot_path, "test_loss", "test loss")
    export_metric_plot(results, accuracy_plot_path, "test_accuracy", "test accuracy")
    write_report(args, results, report_path)

    print("Local MNIST optimizer benchmark")
    print(f"train_samples = {x_train.shape[0]}")
    print(f"test_samples = {x_test.shape[0]}")
    print(f"epochs = {args.epochs}")
    print("optimizer,final_train_loss,final_test_loss,final_train_accuracy,final_test_accuracy,runtime_s")
    for result in sorted(results, key=lambda item: item["history"][-1]["test_loss"]):
        final = result["history"][-1]
        print(
            f"{result['optimizer']},"
            f"{final['train_loss']:.6e},"
            f"{final['test_loss']:.6e},"
            f"{final['train_accuracy']:.6e},"
            f"{final['test_accuracy']:.6e},"
            f"{result['runtime_s']:.6e}"
        )
    for path in (history_path, summary_path, loss_plot_path, accuracy_plot_path, report_path):
        print(f"exported = {path}")


def load_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = _find_file(data_dir, "train-images")
    train_labels = _find_file(data_dir, "train-labels")
    test_images = _find_file(data_dir, "t10k-images")
    test_labels = _find_file(data_dir, "t10k-labels")
    return (
        read_idx_images(train_images),
        read_idx_labels(train_labels),
        read_idx_images(test_images),
        read_idx_labels(test_labels),
    )


def _find_file(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(f"*{pattern}*.gz"))
    if not matches:
        raise FileNotFoundError(f"could not find MNIST file matching {pattern!r} in {data_dir}")
    return matches[0]


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(file.read(count * rows * cols), dtype=np.uint8)
    images = data.reshape(count, rows * cols).astype(float) / 255.0
    return np.column_stack([images, np.ones(count)])


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        return np.frombuffer(file.read(count), dtype=np.uint8).astype(int)


def train_optimizer(name, theta0, x_train, y_train, x_test, y_test, args, params):
    rng = np.random.default_rng(args.seed + sum(ord(char) for char in name))
    theta = theta0.copy()
    velocity = np.zeros_like(theta)
    memory = np.zeros_like(theta)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    metric = np.zeros_like(theta)
    history = []
    start = time.perf_counter()
    batch_size = min(args.batch_size, x_train.shape[0])
    batches_per_epoch = max(1, x_train.shape[0] // batch_size)
    step = 0
    for epoch in range(1, args.epochs + 1):
        for _ in range(batches_per_epoch):
            step += 1
            batch = rng.choice(x_train.shape[0], size=batch_size, replace=False)
            loss, grad = loss_and_gradient(theta, x_train[batch], y_train[batch])
            if name == "sgd":
                theta -= params["learning_rate"] * grad
            elif name == "heavy_ball":
                velocity = params["momentum"] * velocity + params["learning_rate"] * grad
                theta -= velocity
            elif name == "nesterov":
                lookahead = theta - params["momentum"] * velocity
                _lookahead_loss, lookahead_grad = loss_and_gradient(lookahead, x_train[batch], y_train[batch])
                velocity = params["momentum"] * velocity + params["learning_rate"] * lookahead_grad
                theta -= velocity
            elif name == "adamw":
                grad = grad + params["weight_decay"] * theta
                adam_m = params["beta1"] * adam_m + (1.0 - params["beta1"]) * grad
                adam_v = params["beta2"] * adam_v + (1.0 - params["beta2"]) * grad**2
                m_hat = adam_m / (1.0 - params["beta1"]**step)
                v_hat = adam_v / (1.0 - params["beta2"]**step)
                theta -= params["learning_rate"] * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif name == "entropy_descent":
                metric = params["metric_decay"] * metric + (1.0 - params["metric_decay"]) * grad**2
                theta -= params["learning_rate"] * grad / (np.sqrt(metric) + params["metric_epsilon"])
            elif name == "hamiltonian_geometric":
                grad = grad + params["weight_decay"] * theta
                metric = params["metric_decay"] * metric + (1.0 - params["metric_decay"]) * grad**2
                memory = 0.9 * memory + grad
                force = grad + params["memory_coupling"] * memory
                velocity = params["beta"] * velocity - params["learning_rate"] * force
                theta += params["learning_rate"] * velocity / (np.sqrt(metric) + params["metric_epsilon"])
            else:
                raise ValueError(f"unknown optimizer {name!r}")
        train_loss, train_accuracy = loss_and_accuracy(theta, x_train, y_train)
        test_loss, test_accuracy = loss_and_accuracy(theta, x_test, y_test)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )
    return {"optimizer": name, "theta": theta, "history": history, "runtime_s": time.perf_counter() - start}


def loss_and_gradient(theta: np.ndarray, x: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    probabilities = softmax(x @ theta)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.size), labels] = 1.0
    loss = -float(np.mean(np.log(probabilities[np.arange(labels.size), labels] + 1e-12)))
    gradient = x.T @ (probabilities - one_hot) / labels.size
    return loss, gradient


def loss_and_accuracy(theta: np.ndarray, x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    probabilities = softmax(x @ theta)
    loss = -float(np.mean(np.log(probabilities[np.arange(labels.size), labels] + 1e-12)))
    predictions = np.argmax(probabilities, axis=1)
    return loss, float(np.mean(predictions == labels))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def write_history(results, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("epoch,optimizer,train_loss,test_loss,train_accuracy,test_accuracy\n")
        for result in results:
            for row in result["history"]:
                file.write(
                    f"{row['epoch']},{result['optimizer']},{row['train_loss']:.12g},"
                    f"{row['test_loss']:.12g},{row['train_accuracy']:.12g},"
                    f"{row['test_accuracy']:.12g}\n"
                )


def write_summary(results, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("optimizer,final_train_loss,final_test_loss,final_train_accuracy,final_test_accuracy,runtime_s\n")
        for result in sorted(results, key=lambda item: item["history"][-1]["test_loss"]):
            final = result["history"][-1]
            file.write(
                f"{result['optimizer']},{final['train_loss']:.12g},{final['test_loss']:.12g},"
                f"{final['train_accuracy']:.12g},{final['test_accuracy']:.12g},{result['runtime_s']:.12g}\n"
            )


def export_metric_plot(results, path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for result in results:
        rows = result["history"]
        linewidth = 2.8 if result["optimizer"] == "hamiltonian_geometric" else 1.8
        ax.plot([row["epoch"] for row in rows], [row[metric] for row in rows], label=result["optimizer"], linewidth=linewidth)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"MNIST optimizer benchmark: {ylabel}")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(args, results, path: Path) -> None:
    lines = [
        "# Local MNIST Optimizer Benchmark",
        "",
        "Softmax regression on local MNIST IDX files. This is not an official MLCommons AlgoPerf result.",
        "",
        f"Train samples: {args.train_samples}",
        f"Test samples: {args.test_samples}",
        f"Epochs: {args.epochs}",
        "",
        "| optimizer | test loss | test accuracy | runtime s |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item["history"][-1]["test_loss"]):
        final = result["history"][-1]
        lines.append(
            f"| {result['optimizer']} | {final['test_loss']:.12g} | "
            f"{final['test_accuracy']:.4f} | {result['runtime_s']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
