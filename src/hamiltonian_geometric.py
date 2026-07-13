"""Hamiltonian-geometric optimizer core.

This module implements the mathematical optimizer from the attached
formulation directly:

    p_{t+1} = beta p_t - eta [grad L + F_geo + F_mem + alpha grad S(g)]
    theta_{t+1} = theta_t + eta g^{-1} p_{t+1}

The functions are intentionally NumPy-only and accept callables for the loss
gradient and metric, so the same optimizer can be used by the capacitor PINN
benchmark today and by a fuller autodiff network later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
GradientFn = Callable[[Array], Array]
MetricFn = Callable[[Array], Array]
LossFn = Callable[[Array], float]


@dataclass(frozen=True)
class HamiltonianGeometricConfig:
    """Hyperparameters for the dissipative Hamiltonian-geometric update."""

    learning_rate: float = 0.14
    beta: float = 0.9
    metric_regularization: float = 1e-3
    memory_coupling: float = 0.02
    memory_decay: float = 0.9
    spectral_weight: float = 0.0
    finite_difference_step: float = 1e-5
    use_geometric_correction: bool = True
    use_memory_correction: bool = True
    max_energy_backtracks: int = 0
    energy_backtrack_factor: float = 0.5
    energy_tolerance: float = 0.0
    use_memory_metric: bool = False
    memory_metric_coupling: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0)")
        if self.metric_regularization <= 0.0:
            raise ValueError("metric_regularization must be positive")
        if self.memory_coupling < 0.0:
            raise ValueError("memory_coupling must be non-negative")
        if not 0.0 <= self.memory_decay < 1.0:
            raise ValueError("memory_decay must be in the range [0.0, 1.0)")
        if self.spectral_weight < 0.0:
            raise ValueError("spectral_weight must be non-negative")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        if self.max_energy_backtracks < 0:
            raise ValueError("max_energy_backtracks must be non-negative")
        if not 0.0 < self.energy_backtrack_factor < 1.0:
            raise ValueError("energy_backtrack_factor must be in the range (0.0, 1.0)")
        if self.energy_tolerance < 0.0:
            raise ValueError("energy_tolerance must be non-negative")
        if self.memory_metric_coupling < 0.0:
            raise ValueError("memory_metric_coupling must be non-negative")


@dataclass(frozen=True)
class HamiltonianGeometricState:
    """Optimizer state and diagnostics after one update."""

    parameters: Array
    momentum: Array
    memory: Array
    memory_metric: Array
    step: int = 0
    spectral_entropy: float = 0.0
    geometric_force_norm: float = 0.0
    memory_force_norm: float = 0.0
    spectral_force_norm: float = 0.0


def initial_state(parameter_count: int) -> HamiltonianGeometricState:
    """Return zero-initialized theta, momentum, memory, and memory-metric."""

    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    zeros = np.zeros(parameter_count, dtype=float)
    return HamiltonianGeometricState(
        parameters=zeros.copy(),
        momentum=zeros.copy(),
        memory=zeros.copy(),
        memory_metric=np.zeros((parameter_count, parameter_count), dtype=float),
    )


def hamiltonian_geometric_step(
    state: HamiltonianGeometricState,
    gradient_fn: GradientFn,
    metric_fn: MetricFn,
    config: HamiltonianGeometricConfig,
    loss_fn: LossFn | None = None,
) -> HamiltonianGeometricState:
    """Advance one Euler step of the paper's optimizer equations.

    If `loss_fn` is given and `config.max_energy_backtracks > 0`, the step is
    followed by an energy-dissipation safeguard: a dissipative Hamiltonian
    system's total energy (here, the loss stands in for the potential term
    L(theta)) must not spontaneously increase, but a finite discrete Euler
    step can still overshoot on a rugged or chaotic potential and violate
    that invariant. When it does, the momentum that produced the violating
    step is repeatedly damped by `energy_backtrack_factor` (recomputing the
    resulting position from that same metric and momentum) until the loss no
    longer exceeds its pre-step value by more than `energy_tolerance`, or the
    backtrack budget is exhausted. This is the discrete analogue of adding
    more Rayleigh dissipation exactly when the system needs it to stay
    physical, rather than a fixed damping guess; it changes nothing when the
    step already decreases the loss (backtracking config defaults to 0, so
    existing callers are unaffected unless they opt in).
    """

    theta = state.parameters
    gradient = gradient_fn(theta)

    if config.use_memory_metric:
        # M_ij completion of the paper's g_ij = H_ij + mu*M_ij (Sec. 12): the
        # paper asserts M_ij is "the memory term implied by" the vector F_mem
        # but never builds a matrix from it. Since F_mem is itself an EMA of
        # gradients (Eq. 25), the natural matrix completion with the same
        # decay is the EMA of gradient outer products -- the same
        # uncentered-empirical-Fisher construction already used to motivate
        # Adam's diagonal metric (Sec. 10), just kept as a full matrix instead
        # of truncated to its diagonal. It is symmetric PSD by construction
        # (a nonnegative sum of rank-1 PSD terms), so it can only add
        # curvature, never break positive-definiteness.
        memory_metric = config.memory_decay * state.memory_metric + np.outer(gradient, gradient)
        metric_input = metric_fn(theta) + config.memory_metric_coupling * memory_metric
    else:
        memory_metric = state.memory_metric
        metric_input = metric_fn(theta)

    metric = positive_definite_metric(metric_input, config.metric_regularization)
    inverse_metric = np.linalg.pinv(metric)
    entropy = spectral_entropy(metric)

    if config.use_geometric_correction:
        geometric_force = geometric_force_finite_difference(
            theta,
            state.momentum,
            metric_fn,
            regularization=config.metric_regularization,
            step=config.finite_difference_step,
        )
    else:
        geometric_force = np.zeros_like(theta)

    if config.spectral_weight > 0.0:
        spectral_force = spectral_entropy_gradient_finite_difference(
            theta,
            metric_fn,
            regularization=config.metric_regularization,
            step=config.finite_difference_step,
        )
    else:
        spectral_force = np.zeros_like(theta)

    memory = config.memory_decay * state.memory + gradient
    memory_force = config.memory_coupling * memory if config.use_memory_correction else np.zeros_like(theta)
    force = gradient + geometric_force + memory_force + config.spectral_weight * spectral_force
    momentum = config.beta * state.momentum - config.learning_rate * force
    parameters = theta + config.learning_rate * (inverse_metric @ momentum)

    if loss_fn is not None and config.max_energy_backtracks > 0:
        momentum, parameters = _apply_energy_backtracking(
            theta, momentum, inverse_metric, loss_fn, config
        )

    return HamiltonianGeometricState(
        parameters=parameters,
        momentum=momentum,
        memory=memory,
        memory_metric=memory_metric,
        step=state.step + 1,
        spectral_entropy=entropy,
        geometric_force_norm=float(np.linalg.norm(geometric_force)),
        memory_force_norm=float(np.linalg.norm(memory_force)),
        spectral_force_norm=float(np.linalg.norm(config.spectral_weight * spectral_force)),
    )


def _apply_energy_backtracking(
    theta: Array,
    momentum: Array,
    inverse_metric: Array,
    loss_fn: LossFn,
    config: HamiltonianGeometricConfig,
) -> tuple[Array, Array]:
    """Damp momentum until the step no longer increases the loss (or budget runs out)."""

    current_loss = loss_fn(theta)
    parameters = theta + config.learning_rate * (inverse_metric @ momentum)
    for _ in range(config.max_energy_backtracks):
        if loss_fn(parameters) <= current_loss * (1.0 + config.energy_tolerance) + config.energy_tolerance:
            break
        momentum = config.energy_backtrack_factor * momentum
        parameters = theta + config.learning_rate * (inverse_metric @ momentum)
    return momentum, parameters


def positive_definite_metric(raw_metric: Array, regularization: float) -> Array:
    """Regularize a symmetric metric/Hessian into a positive-definite matrix."""

    symmetric = 0.5 * (raw_metric + raw_metric.T)
    min_eigenvalue = float(np.linalg.eigvalsh(symmetric).min())
    effective_regularization = regularization + max(0.0, -min_eigenvalue)
    return symmetric + effective_regularization * np.eye(symmetric.shape[0])


def spectral_entropy(metric: Array) -> float:
    """Spectral entropy S(g) over normalized eigenvalues of the metric."""

    eigenvalues = np.linalg.eigvalsh(metric)
    eigenvalues = np.clip(eigenvalues, a_min=1e-15, a_max=None)
    normalized = eigenvalues / eigenvalues.sum()
    return float(-np.sum(normalized * np.log(normalized)))


def geometric_force_finite_difference(
    theta: Array,
    momentum: Array,
    metric_fn: MetricFn,
    regularization: float,
    step: float = 1e-5,
) -> Array:
    """Compute F_geo_i = 0.5 p^T (partial_i g^{-1}) p by central differences."""

    force = np.zeros_like(theta)
    for index in range(theta.size):
        delta = np.zeros_like(theta)
        delta[index] = step
        inv_plus = np.linalg.pinv(positive_definite_metric(metric_fn(theta + delta), regularization))
        inv_minus = np.linalg.pinv(positive_definite_metric(metric_fn(theta - delta), regularization))
        derivative = (inv_plus - inv_minus) / (2.0 * step)
        force[index] = 0.5 * float(momentum @ derivative @ momentum)
    return force


def spectral_entropy_gradient_finite_difference(
    theta: Array,
    metric_fn: MetricFn,
    regularization: float,
    step: float = 1e-5,
) -> Array:
    """Compute grad_theta S(g(theta)) by central differences."""

    gradient = np.zeros_like(theta)
    for index in range(theta.size):
        delta = np.zeros_like(theta)
        delta[index] = step
        entropy_plus = spectral_entropy(positive_definite_metric(metric_fn(theta + delta), regularization))
        entropy_minus = spectral_entropy(positive_definite_metric(metric_fn(theta - delta), regularization))
        gradient[index] = (entropy_plus - entropy_minus) / (2.0 * step)
    return gradient
