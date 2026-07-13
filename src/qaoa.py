"""Exact-statevector QAOA MaxCut optimizer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import time
from pathlib import Path

import numpy as np

from src.hamiltonian_geometric import (
    HamiltonianGeometricConfig,
    hamiltonian_geometric_step,
    initial_state,
    positive_definite_metric,
    spectral_entropy,
)


@dataclass(frozen=True)
class QAOAConfig:
    qubits: int = 6
    depth: int = 2
    seed: int = 19
    finite_difference_step: float = 1e-5

    def __post_init__(self) -> None:
        if self.qubits < 2:
            raise ValueError("qubits must be at least 2")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")


@dataclass(frozen=True)
class QAOAProblem:
    config: QAOAConfig
    edges: list[tuple[int, int]]
    cut_values: np.ndarray
    plus_state: np.ndarray
    max_cut: float


@dataclass(frozen=True)
class QAOAResult:
    optimizer: str
    parameters: np.ndarray
    loss_history: list[float]
    ratio_history: list[float]
    runtime_s: float
    spectral_entropy: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1]

    @property
    def final_ratio(self) -> float:
        return self.ratio_history[-1]


def build_problem(config: QAOAConfig) -> QAOAProblem:
    rng = np.random.default_rng(config.seed)
    edges: set[tuple[int, int]] = set()
    for index in range(config.qubits):
        edges.add((index, (index + 1) % config.qubits))
    while len(edges) < config.qubits + 3:
        a, b = sorted(rng.choice(config.qubits, size=2, replace=False))
        edges.add((int(a), int(b)))
    edge_list = sorted(edges)
    cut_values = maxcut_values(config.qubits, edge_list)
    state = np.ones(2**config.qubits, dtype=complex) / np.sqrt(2**config.qubits)
    return QAOAProblem(
        config=config,
        edges=edge_list,
        cut_values=cut_values,
        plus_state=state,
        max_cut=float(np.max(cut_values)),
    )


def maxcut_values(qubits: int, edges: list[tuple[int, int]]) -> np.ndarray:
    values = np.zeros(2**qubits, dtype=float)
    for basis in range(2**qubits):
        total = 0
        for a, b in edges:
            bit_a = (basis >> a) & 1
            bit_b = (basis >> b) & 1
            total += bit_a != bit_b
        values[basis] = total
    return values


def apply_mixer(state: np.ndarray, beta: float, qubits: int) -> np.ndarray:
    c = np.cos(beta)
    s = -1.0j * np.sin(beta)
    current = state.copy()
    for qubit in range(qubits):
        next_state = current.copy()
        stride = 1 << qubit
        period = stride << 1
        for start in range(0, current.size, period):
            for offset in range(stride):
                i0 = start + offset
                i1 = i0 + stride
                a0 = current[i0]
                a1 = current[i1]
                next_state[i0] = c * a0 + s * a1
                next_state[i1] = s * a0 + c * a1
        current = next_state
    return current


def qaoa_state(problem: QAOAProblem, parameters: np.ndarray) -> np.ndarray:
    gammas = parameters[: problem.config.depth]
    betas = parameters[problem.config.depth :]
    state = problem.plus_state.copy()
    for gamma, beta in zip(gammas, betas):
        state = np.exp(-1.0j * gamma * problem.cut_values) * state
        state = apply_mixer(state, beta, problem.config.qubits)
    return state / np.linalg.norm(state)


def expected_cut(problem: QAOAProblem, parameters: np.ndarray) -> float:
    state = qaoa_state(problem, parameters)
    probabilities = np.abs(state) ** 2
    return float(probabilities @ problem.cut_values)


def loss(problem: QAOAProblem, parameters: np.ndarray) -> float:
    return -expected_cut(problem, parameters) / problem.max_cut


def loss_and_gradient(problem: QAOAProblem, parameters: np.ndarray) -> tuple[float, np.ndarray]:
    base = loss(problem, parameters)
    gradient = np.zeros_like(parameters)
    step = problem.config.finite_difference_step
    for index in range(parameters.size):
        delta = np.zeros_like(parameters)
        delta[index] = step
        gradient[index] = (loss(problem, parameters + delta) - loss(problem, parameters - delta)) / (2.0 * step)
    return base, gradient


def metric_from_gradient(problem: QAOAProblem, parameters: np.ndarray) -> np.ndarray:
    _loss, gradient = loss_and_gradient(problem, parameters)
    return np.diag(0.05 + np.abs(gradient)) + np.outer(gradient, gradient)


def train_qaoa_optimizer(problem: QAOAProblem, optimizer: str, iterations: int, hyperparameters: dict[str, float]) -> QAOAResult:
    rng = np.random.default_rng(problem.config.seed + sum(ord(char) for char in optimizer))
    theta = rng.uniform(-0.05, 0.05, size=2 * problem.config.depth)
    velocity = np.zeros_like(theta)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    losses: list[float] = []
    ratios: list[float] = []
    entropy = 0.0
    start = time.perf_counter()

    if optimizer == "hamiltonian_geometric":
        state = initial_state(theta.size)
        state = state.__class__(
            parameters=theta.copy(),
            momentum=state.momentum,
            memory=state.memory,
            memory_metric=state.memory_metric,
        )
        config = HamiltonianGeometricConfig(
            learning_rate=hyperparameters.get("learning_rate", 0.18),
            beta=hyperparameters.get("beta", 0.45),
            metric_regularization=hyperparameters.get("metric_regularization", 5e-2),
            memory_coupling=hyperparameters.get("memory_coupling", 0.0),
            use_geometric_correction=hyperparameters.get("use_geometric_correction", False),
        )

        def grad_fn(values: np.ndarray) -> np.ndarray:
            return loss_and_gradient(problem, values)[1]

        def metric_fn(values: np.ndarray) -> np.ndarray:
            return metric_from_gradient(problem, values)

        for _ in range(iterations):
            current_loss = loss(problem, state.parameters)
            losses.append(current_loss)
            ratios.append(-current_loss)
            state = hamiltonian_geometric_step(state, grad_fn, metric_fn, config)
            theta = state.parameters
            entropy = state.spectral_entropy
    else:
        for step_index in range(1, iterations + 1):
            current_loss, gradient = loss_and_gradient(problem, theta)
            losses.append(current_loss)
            ratios.append(-current_loss)
            lr = hyperparameters["learning_rate"]
            if optimizer == "sgd":
                theta = theta - lr * gradient
            elif optimizer == "heavy_ball":
                velocity = hyperparameters["momentum"] * velocity + lr * gradient
                theta = theta - velocity
            elif optimizer == "adamw":
                beta1 = hyperparameters.get("beta1", 0.9)
                beta2 = hyperparameters.get("beta2", 0.99)
                adam_m = beta1 * adam_m + (1.0 - beta1) * gradient
                adam_v = beta2 * adam_v + (1.0 - beta2) * gradient**2
                m_hat = adam_m / (1.0 - beta1**step_index)
                v_hat = adam_v / (1.0 - beta2**step_index)
                theta = theta - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif optimizer == "entropy_descent":
                metric = positive_definite_metric(metric_from_gradient(problem, theta), hyperparameters.get("metric_regularization", 5e-2))
                entropy = spectral_entropy(metric)
                theta = theta - lr * (np.linalg.pinv(metric) @ gradient)
            else:
                raise ValueError(f"unknown optimizer {optimizer!r}")

    final_loss = loss(problem, theta)
    if final_loss < losses[-1]:
        losses.append(final_loss)
        ratios.append(-final_loss)
    return QAOAResult(optimizer, theta, losses, ratios, time.perf_counter() - start, entropy)


def run_qaoa_comparison(config: QAOAConfig, iterations: int = 90) -> tuple[QAOAProblem, list[QAOAResult]]:
    problem = build_problem(config)
    settings = [
        ("sgd", {"learning_rate": 0.12}),
        ("heavy_ball", {"learning_rate": 0.08, "momentum": 0.8}),
        ("adamw", {"learning_rate": 0.06, "beta1": 0.9, "beta2": 0.99}),
        ("entropy_descent", {"learning_rate": 0.10, "metric_regularization": 5e-2}),
        ("hamiltonian_geometric", {"learning_rate": 0.18, "beta": 0.45, "metric_regularization": 5e-2}),
    ]
    return problem, [train_qaoa_optimizer(problem, name, iterations, params) for name, params in settings]


def export_qaoa_history(results: list[QAOAResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "optimizer", "loss", "approximation_ratio"])
        for result in results:
            for index, (current_loss, ratio) in enumerate(zip(result.loss_history, result.ratio_history), 1):
                writer.writerow([index, result.optimizer, f"{current_loss:.12g}", f"{ratio:.12g}"])


def export_qaoa_summary(problem: QAOAProblem, results: list[QAOAResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "final_loss", "approximation_ratio", "runtime_s", "spectral_entropy", "max_cut"])
        for result in sorted(results, key=lambda item: item.final_loss):
            writer.writerow([
                result.optimizer,
                f"{result.final_loss:.12g}",
                f"{result.final_ratio:.12g}",
                f"{result.runtime_s:.12g}",
                f"{result.spectral_entropy:.12g}",
                f"{problem.max_cut:.12g}",
            ])
