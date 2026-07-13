"""Synthetic evidence suite for Hamiltonian-geometric optimization.

These experiments are intentionally small and visual.  They test the cases
where a phase-space/geometric optimizer should have an advantage: structured
curvature, coordinate rotations, energy behavior, saddles, and ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.hamiltonian_geometric import (
    HamiltonianGeometricConfig,
    hamiltonian_geometric_step,
    initial_state,
    positive_definite_metric,
)


@dataclass(frozen=True)
class OptimizerTrace:
    name: str
    theta: np.ndarray
    loss: np.ndarray
    energy: np.ndarray
    grad_norm: np.ndarray


def rotated_quadratic(dim: int, condition_number: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(raw)
    eigenvalues = np.geomspace(1.0, condition_number, dim)
    a = q @ np.diag(eigenvalues) @ q.T

    def loss(theta: np.ndarray) -> float:
        return 0.5 * float(theta @ a @ theta)

    def grad(theta: np.ndarray) -> np.ndarray:
        return a @ theta

    def metric(_theta: np.ndarray) -> np.ndarray:
        return a

    return loss, grad, metric, a


def double_well_loss(theta: np.ndarray) -> float:
    x, y = theta
    return float(0.25 * (x * x - 1.0) ** 2 + 0.5 * y * y + 0.15 * x * y)


def double_well_grad(theta: np.ndarray) -> np.ndarray:
    x, y = theta
    return np.array([x * (x * x - 1.0) + 0.15 * y, y + 0.15 * x])


def double_well_metric(theta: np.ndarray) -> np.ndarray:
    x, _y = theta
    hessian = np.array([[3.0 * x * x - 1.0, 0.15], [0.15, 1.0]])
    return hessian


def run_optimizer(
    name: str,
    theta0: np.ndarray,
    loss_fn,
    grad_fn,
    metric_fn,
    steps: int,
    learning_rate: float,
) -> OptimizerTrace:
    theta = theta0.astype(float, copy=True)
    velocity = np.zeros_like(theta)
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    metric_accumulator = np.zeros_like(theta)
    losses: list[float] = []
    energies: list[float] = []
    grad_norms: list[float] = []

    if name == "hamiltonian_geometric":
        state = initial_state(theta.size)
        state = state.__class__(
            parameters=theta.copy(),
            momentum=state.momentum,
            memory=state.memory,
            memory_metric=state.memory_metric,
        )
        config = HamiltonianGeometricConfig(
            learning_rate=learning_rate,
            beta=0.7,
            metric_regularization=1e-3,
            memory_coupling=0.0,
            use_geometric_correction=False,
        )
        for _ in range(steps):
            grad = grad_fn(state.parameters)
            metric = positive_definite_metric(metric_fn(state.parameters), config.metric_regularization)
            losses.append(loss_fn(state.parameters))
            energies.append(0.5 * float(state.momentum @ np.linalg.pinv(metric) @ state.momentum) + losses[-1])
            grad_norms.append(float(np.linalg.norm(grad)))
            state = hamiltonian_geometric_step(state, grad_fn, metric_fn, config)
        theta = state.parameters
    else:
        for step in range(1, steps + 1):
            grad = grad_fn(theta)
            losses.append(loss_fn(theta))
            energies.append(0.5 * float(velocity @ velocity) + losses[-1])
            grad_norms.append(float(np.linalg.norm(grad)))
            if name == "sgd":
                theta = theta - learning_rate * grad
            elif name == "heavy_ball":
                velocity = 0.85 * velocity + learning_rate * grad
                theta = theta - velocity
            elif name == "adam":
                adam_m = 0.9 * adam_m + 0.1 * grad
                adam_v = 0.999 * adam_v + 0.001 * grad**2
                m_hat = adam_m / (1.0 - 0.9**step)
                v_hat = adam_v / (1.0 - 0.999**step)
                theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif name == "entropy_descent":
                metric_accumulator = 0.95 * metric_accumulator + 0.05 * grad**2
                theta = theta - learning_rate * grad / (np.sqrt(metric_accumulator) + 0.05)
            else:
                raise ValueError(f"unknown optimizer {name!r}")

    return OptimizerTrace(
        name=name,
        theta=theta,
        loss=np.array(losses),
        energy=np.array(energies),
        grad_norm=np.array(grad_norms),
    )


def condition_number_experiment(
    condition_numbers=(1e2, 1e4, 1e6, 1e8),
    dim: int = 8,
    steps: int = 160,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for condition in condition_numbers:
        loss_fn, grad_fn, metric_fn, _a = rotated_quadratic(dim, condition, seed=11)
        theta0 = np.linspace(1.0, -1.0, dim)
        for name, lr in (
            ("sgd", 0.8 / condition),
            ("heavy_ball", 0.35 / condition),
            ("adam", 0.05),
            ("entropy_descent", 0.15),
            ("hamiltonian_geometric", 0.18),
        ):
            trace = run_optimizer(name, theta0, loss_fn, grad_fn, metric_fn, steps, lr)
            rows.append(
                {
                    "condition_number": float(condition),
                    "optimizer": name,
                    "final_loss": float(trace.loss[-1]),
                    "loss_reduction": float(trace.loss[0] / max(trace.loss[-1], 1e-300)),
                }
            )
    return rows


def rotation_invariance_experiment(dim: int = 6, condition: float = 1e5, steps: int = 140) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    theta0 = np.linspace(1.0, -0.5, dim)
    for seed in range(12):
        loss_fn, grad_fn, metric_fn, _a = rotated_quadratic(dim, condition, seed=seed)
        for name, lr in (
            ("sgd", 0.5 / condition),
            ("adam", 0.04),
            ("entropy_descent", 0.12),
            ("hamiltonian_geometric", 0.16),
        ):
            trace = run_optimizer(name, theta0, loss_fn, grad_fn, metric_fn, steps, lr)
            rows.append({"rotation_seed": seed, "optimizer": name, "final_loss": float(trace.loss[-1])})
    return rows


def phase_space_experiment(steps: int = 220) -> list[OptimizerTrace]:
    theta0 = np.array([-0.08, 0.55])
    return [
        run_optimizer("sgd", theta0, double_well_loss, double_well_grad, double_well_metric, steps, 0.03),
        run_optimizer("heavy_ball", theta0, double_well_loss, double_well_grad, double_well_metric, steps, 0.035),
        run_optimizer("adam", theta0, double_well_loss, double_well_grad, double_well_metric, steps, 0.025),
        run_optimizer("entropy_descent", theta0, double_well_loss, double_well_grad, double_well_metric, steps, 0.035),
        run_optimizer("hamiltonian_geometric", theta0, double_well_loss, double_well_grad, double_well_metric, steps, 0.08),
    ]


def saddle_escape_experiment(seeds: int = 40, steps: int = 120) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(5)
    for seed in range(seeds):
        theta0 = rng.normal(0.0, 0.04, size=2)
        theta0[1] += 0.2
        for name, lr in (
            ("sgd", 0.03),
            ("heavy_ball", 0.035),
            ("adam", 0.025),
            ("entropy_descent", 0.035),
            ("hamiltonian_geometric", 0.08),
        ):
            trace = run_optimizer(name, theta0, double_well_loss, double_well_grad, double_well_metric, steps, lr)
            escaped = abs(trace.theta[0]) > 0.7
            rows.append(
                {
                    "seed": seed,
                    "optimizer": name,
                    "escaped": float(escaped),
                    "final_loss": float(trace.loss[-1]),
                    "final_x": float(trace.theta[0]),
                }
            )
    return rows
