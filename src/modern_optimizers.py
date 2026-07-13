"""Reference implementations of modern optimizers, matching the derivations in
docs/hamiltonian_geometric_consolidated_report.tex (Sec. "Modern Optimizers as
Further Derivations"): each is a specific choice of metric, kinetic term, or
potential splitting within the Hamiltonian-geometric framework, not a separate
algorithm family. Kept NumPy-only and independent of any particular benchmark's
parameter layout, operating on individual arrays (vectors for Lion/AdamW,
weight matrices for Shampoo/Muon) so callers can apply them per-parameter-group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class LionState:
    """Lion's two independently-decayed momentum estimates (Eq. lion-step)."""

    fast_momentum: Array
    slow_momentum: Array


def lion_initial_state(shape: tuple[int, ...]) -> LionState:
    zeros = np.zeros(shape, dtype=float)
    return LionState(fast_momentum=zeros.copy(), slow_momentum=zeros.copy())


def lion_step(
    parameters: Array,
    gradient: Array,
    state: LionState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
) -> tuple[Array, LionState]:
    """Lion (Chen et al. 2023): flat metric, L1 kinetic term (sign dynamics).

    c_t = beta1*m_slow + (1-beta1)*grad; theta -= eta*(sign(c_t) + weight_decay*theta)
    m_slow_{t+1} = beta2*m_slow_t + (1-beta2)*grad
    """

    fast_momentum = beta1 * state.slow_momentum + (1.0 - beta1) * gradient
    update = np.sign(fast_momentum) + weight_decay * parameters
    new_parameters = parameters - learning_rate * update
    slow_momentum = beta2 * state.slow_momentum + (1.0 - beta2) * gradient
    return new_parameters, LionState(fast_momentum=fast_momentum, slow_momentum=slow_momentum)


@dataclass(frozen=True)
class AdamWState:
    """Adam's first/second moment accumulators, kept separate from decay."""

    first_moment: Array
    second_moment: Array
    step: int


def adamw_initial_state(shape: tuple[int, ...]) -> AdamWState:
    zeros = np.zeros(shape, dtype=float)
    return AdamWState(first_moment=zeros.copy(), second_moment=zeros.copy(), step=0)


def adamw_step(
    parameters: Array,
    gradient: Array,
    state: AdamWState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[Array, AdamWState]:
    """AdamW (Loshchilov & Hutter 2019): Lie-Trotter split of an adaptive
    metric on the loss gradient and a flat metric on the decay term (Eq.
    adamw-split), applied as one combined step (Eq. adamw). `gradient` must
    be the loss gradient alone, with no L2 term folded in, or the split stops
    being meaningfully different from Adam+L2.
    """

    step = state.step + 1
    first_moment = beta1 * state.first_moment + (1.0 - beta1) * gradient
    second_moment = beta2 * state.second_moment + (1.0 - beta2) * gradient**2
    first_hat = first_moment / (1.0 - beta1**step)
    second_hat = second_moment / (1.0 - beta2**step)
    new_parameters = parameters - learning_rate * (
        first_hat / (np.sqrt(second_hat) + epsilon) + weight_decay * parameters
    )
    return new_parameters, AdamWState(first_moment=first_moment, second_moment=second_moment, step=step)


@dataclass(frozen=True)
class ShampooState:
    """Shampoo's Kronecker-factor preconditioner accumulators for one matrix."""

    left_preconditioner: Array
    right_preconditioner: Array


def shampoo_initial_state(rows: int, columns: int, epsilon: float) -> ShampooState:
    return ShampooState(
        left_preconditioner=epsilon * np.eye(rows),
        right_preconditioner=epsilon * np.eye(columns),
    )


def _matrix_inverse_fourth_root(matrix: Array) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, a_min=1e-12, a_max=None)
    inverse_fourth_root = eigenvectors @ np.diag(eigenvalues ** -0.25) @ eigenvectors.T
    return 0.5 * (inverse_fourth_root + inverse_fourth_root.T)


def shampoo_step(
    weight_matrix: Array,
    gradient_matrix: Array,
    state: ShampooState,
    learning_rate: float,
) -> tuple[Array, ShampooState]:
    """Shampoo (Gupta, Koren & Singer 2018): W -= eta * L^(-1/4) G R^(-1/4),
    the Kronecker-factored metric of Eq. shampoo. Uses exact eigendecomposition
    (matrices here are small); large-scale Shampoo instead uses cheaper
    iterative inverse-root approximations, but the update rule is identical.
    """

    left = state.left_preconditioner + gradient_matrix @ gradient_matrix.T
    right = state.right_preconditioner + gradient_matrix.T @ gradient_matrix
    preconditioned = _matrix_inverse_fourth_root(left) @ gradient_matrix @ _matrix_inverse_fourth_root(right)
    new_weight_matrix = weight_matrix - learning_rate * preconditioned
    return new_weight_matrix, ShampooState(left_preconditioner=left, right_preconditioner=right)


def orthogonalize_via_polar_factor(matrix: Array) -> Array:
    """Return M's orthogonal polar factor U V^T from M = U Sigma V^T.

    Real Muon computes this with a Newton-Schulz iteration for GPU efficiency
    at scale; for the small matrices in this benchmark, the exact SVD-based
    polar factor is used directly and is mathematically the same limit.
    """

    left_singular_vectors, _singular_values, right_singular_vectors_t = np.linalg.svd(
        matrix, full_matrices=False
    )
    return left_singular_vectors @ right_singular_vectors_t


def muon_step(
    weight_matrix: Array,
    gradient_matrix: Array,
    momentum: Array,
    learning_rate: float,
    momentum_decay: float = 0.95,
) -> tuple[Array, Array]:
    """Muon (Jordan et al. 2024): momentum orthogonalized to its polar factor,
    the alpha -> infinity (hard-constrained) limit of the spectral-entropy
    regularizer applied to a momentum-built metric (see report Sec. "Muon").
    """

    momentum = momentum_decay * momentum + gradient_matrix
    orthogonalized = orthogonalize_via_polar_factor(momentum)
    new_weight_matrix = weight_matrix - learning_rate * orthogonalized
    return new_weight_matrix, momentum
