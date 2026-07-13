"""Physics-informed fringing-field benchmark from the Hamiltonian paper.

The model represents the electrostatic potential with fixed nonlinear features
and trains only the output weights. That keeps the benchmark lightweight while
still matching the paper's key structure: a parameter vector theta, a PDE loss,
boundary penalties, and optimizer comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.hamiltonian_geometric import (
    HamiltonianGeometricConfig,
    hamiltonian_geometric_step,
    initial_state,
    positive_definite_metric,
    spectral_entropy,
)


@dataclass(frozen=True)
class PINNConfig:
    """Geometry, sampling, and model settings for the PINN benchmark."""

    plate_width: float = 0.02
    gap: float = 0.004
    domain_width: float = 0.08
    domain_height: float = 0.08
    voltage: float = 1.0
    hidden_features: int = 64
    collocation_points: int = 700
    plate_points: int = 80
    outer_boundary_points: int = 100
    boundary_weight: float = 80.0
    outer_boundary_weight: float = 2.0
    seed: int = 7

    def __post_init__(self) -> None:
        positive_values = {
            "plate_width": self.plate_width,
            "gap": self.gap,
            "domain_width": self.domain_width,
            "domain_height": self.domain_height,
            "voltage": self.voltage,
            "hidden_features": self.hidden_features,
            "collocation_points": self.collocation_points,
            "plate_points": self.plate_points,
            "outer_boundary_points": self.outer_boundary_points,
            "boundary_weight": self.boundary_weight,
            "outer_boundary_weight": self.outer_boundary_weight,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        if self.plate_width >= self.domain_width:
            raise ValueError("plate_width must be smaller than domain_width")
        if self.gap >= self.domain_height:
            raise ValueError("gap must be smaller than domain_height")


@dataclass(frozen=True)
class PINNDataset:
    """Feature matrices for the PDE residual and boundary conditions."""

    pde_features: np.ndarray
    pde_targets: np.ndarray
    plate_features: np.ndarray
    plate_targets: np.ndarray
    outer_features: np.ndarray
    outer_targets: np.ndarray
    design_matrix: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    """Training history and final diagnostics for one optimizer."""

    optimizer: str
    parameters: np.ndarray
    loss_history: list[float]
    pde_loss: float
    plate_loss: float
    outer_loss: float
    gradient_norm: float
    spectral_entropy: float = 0.0
    theta_history: list[np.ndarray] | None = None

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1]


class FixedFeaturePotentialModel:
    """Small fixed-feature model for phi_theta(x, y)."""

    def __init__(self, config: PINNConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scales = np.array(
            [2.0 / config.domain_width, 2.0 / config.domain_height],
            dtype=float,
        )
        self.weights = rng.normal(0.0, 1.0, size=(config.hidden_features, 2)) * scales
        self.biases = rng.uniform(-math.pi, math.pi, size=config.hidden_features)

    @property
    def parameter_count(self) -> int:
        return self.config.hidden_features + 1

    def features(self, points: np.ndarray) -> np.ndarray:
        """Return feature matrix for potential values."""

        z = points @ self.weights.T + self.biases
        nonlinear = np.tanh(z)
        return np.column_stack([np.ones(points.shape[0]), nonlinear])

    def laplacian_features(self, points: np.ndarray) -> np.ndarray:
        """Return feature matrix for del^2 phi."""

        z = points @ self.weights.T + self.biases
        tanh_z = np.tanh(z)
        sech2 = 1.0 - tanh_z**2
        weight_norm2 = np.sum(self.weights**2, axis=1)
        laplacian = -2.0 * tanh_z * sech2 * weight_norm2
        return np.column_stack([np.zeros(points.shape[0]), laplacian])

    def predict(self, points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return self.features(points) @ parameters


def build_pinn_dataset(
    model: FixedFeaturePotentialModel,
    config: PINNConfig,
) -> PINNDataset:
    """Sample collocation and boundary points for the capacitor PINN loss."""

    rng = np.random.default_rng(config.seed + 1)
    x = rng.uniform(
        -config.domain_width / 2.0,
        config.domain_width / 2.0,
        size=config.collocation_points,
    )
    y = rng.uniform(
        -config.domain_height / 2.0,
        config.domain_height / 2.0,
        size=config.collocation_points,
    )
    collocation = np.column_stack([x, y])

    plate_x = np.linspace(-config.plate_width / 2.0, config.plate_width / 2.0, config.plate_points)
    upper_plate = np.column_stack([plate_x, np.full(config.plate_points, config.gap / 2.0)])
    lower_plate = np.column_stack([plate_x, np.full(config.plate_points, -config.gap / 2.0)])
    plate_points = np.vstack([upper_plate, lower_plate])
    plate_targets = np.concatenate(
        [
            np.full(config.plate_points, config.voltage / 2.0),
            np.full(config.plate_points, -config.voltage / 2.0),
        ]
    )

    outer_points = _outer_boundary_points(config)
    outer_targets = np.zeros(outer_points.shape[0])

    pde_features = model.laplacian_features(collocation)
    pde_targets = np.zeros(config.collocation_points)
    plate_features = model.features(plate_points)
    outer_features = model.features(outer_points)

    design_parts = [
        pde_features / math.sqrt(config.collocation_points),
        math.sqrt(config.boundary_weight / plate_targets.size) * plate_features,
        math.sqrt(config.outer_boundary_weight / outer_targets.size) * outer_features,
    ]
    target_parts = [
        pde_targets / math.sqrt(config.collocation_points),
        math.sqrt(config.boundary_weight / plate_targets.size) * plate_targets,
        math.sqrt(config.outer_boundary_weight / outer_targets.size) * outer_targets,
    ]
    return PINNDataset(
        pde_features=pde_features,
        pde_targets=pde_targets,
        plate_features=plate_features,
        plate_targets=plate_targets,
        outer_features=outer_features,
        outer_targets=outer_targets,
        design_matrix=np.vstack(design_parts),
        targets=np.concatenate(target_parts),
    )


def train_sgd(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 600,
    learning_rate: float = 1e-7,
    record_theta: bool = False,
) -> TrainingResult:
    """Train with plain gradient descent."""

    parameters = np.zeros(parameter_count)
    history: list[float] = []
    theta_history: list[np.ndarray] | None = [] if record_theta else None
    for _ in range(steps):
        loss, gradient = _loss_and_gradient(dataset.design_matrix, dataset.targets, parameters)
        history.append(loss)
        if theta_history is not None:
            theta_history.append(parameters.copy())
        parameters -= learning_rate * gradient
    return _training_result("sgd", dataset, parameters, history, theta_history=theta_history)


def train_adam(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 600,
    learning_rate: float = 3e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    record_theta: bool = False,
) -> TrainingResult:
    """Train with Adam, the paper's diagonal-metric comparison baseline."""

    parameters = np.zeros(parameter_count)
    first_moment = np.zeros(parameter_count)
    second_moment = np.zeros(parameter_count)
    history: list[float] = []
    theta_history: list[np.ndarray] | None = [] if record_theta else None
    for step in range(1, steps + 1):
        loss, gradient = _loss_and_gradient(dataset.design_matrix, dataset.targets, parameters)
        history.append(loss)
        if theta_history is not None:
            theta_history.append(parameters.copy())
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        parameters -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
    return _training_result("adam", dataset, parameters, history, theta_history=theta_history)


def train_falling_ball(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 600,
    learning_rate: float = 1e-7,
    friction: float = 0.85,
    record_theta: bool = False,
) -> TrainingResult:
    """Train with the literal 'ball rolling downhill with friction' optimizer.

    The flat-Euclidean-metric special case of the dissipative Hamiltonian
    system (Sec. 7-8, Eq. 16-17): a unit-mass point sliding on the loss
    surface L(theta) under gravity (the loss gradient) and linear friction,
    with g = I so F_geo = 0 identically. Unlike train_hamiltonian_geometric
    there is no curvature correction, memory force, or spectral term here --
    only the bare dissipative flow, in the same single-eta convention as
    classical momentum (Sec. 9, Eq. 19: v_{t+1} = beta*v_t + grad L,
    theta_{t+1} = theta_t - eta*v_{t+1}) rather than
    train_hamiltonian_geometric's convention (eta appears in both the
    momentum and position updates, which only stays well-scaled there
    because the metric's inverse divides the accumulated momentum back down;
    with no metric to do that here, a second eta would make the momentum too
    small to move the parameters at all):

        p_{t+1} = friction * p_t + grad L(theta_t)
        theta_{t+1} = theta_t - eta * p_{t+1}

    This is framed as the zero-curvature physical baseline that
    train_hamiltonian_geometric's metric is meant to improve on.
    """

    if not 0.0 <= friction < 1.0:
        raise ValueError("friction must be in the range [0.0, 1.0)")

    parameters = np.zeros(parameter_count)
    momentum = np.zeros(parameter_count)
    history: list[float] = []
    theta_history: list[np.ndarray] | None = [] if record_theta else None
    for _ in range(steps):
        loss, gradient = _loss_and_gradient(dataset.design_matrix, dataset.targets, parameters)
        history.append(loss)
        if theta_history is not None:
            theta_history.append(parameters.copy())
        momentum = friction * momentum + gradient
        parameters = parameters - learning_rate * momentum
    return _training_result("falling_ball", dataset, parameters, history, theta_history=theta_history)


def train_entropy_descent(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 600,
    learning_rate: float = 0.02,
    metric_regularization: float = 1e-3,
    record_theta: bool = False,
) -> TrainingResult:
    """Train with pure metric-preconditioned ('natural gradient') descent.

        theta_{t+1} = theta_t - eta * g^{-1} grad L(theta_t)

    `g` is the same positive-definite Hessian metric as
    train_hamiltonian_geometric (Sec. 12, Eq. 26), and its spectral entropy
    S(g) (Sec. 13-14) is reported alongside the result. This isolates the
    effect of the metric's eigenvalue rescaling alone -- Sec. 13 notes the
    inverse metric "rescales each direction by 1/lambda_i: large-curvature
    directions receive smaller updates, flat directions receive larger
    updates" -- with no momentum and no memory force, so it can be compared
    directly against train_falling_ball (momentum, no preconditioning) and
    train_hamiltonian_geometric (both together).
    """

    parameters = np.zeros(parameter_count)
    raw_hessian = 2.0 * dataset.design_matrix.T @ dataset.design_matrix
    hessian_metric = _positive_definite_metric(raw_hessian, metric_regularization)
    inverse_metric = np.linalg.pinv(hessian_metric)
    spectral_entropy = _spectral_entropy(hessian_metric)

    history: list[float] = []
    theta_history: list[np.ndarray] | None = [] if record_theta else None
    for _ in range(steps):
        loss, gradient = _loss_and_gradient(dataset.design_matrix, dataset.targets, parameters)
        history.append(loss)
        if theta_history is not None:
            theta_history.append(parameters.copy())
        parameters = parameters - learning_rate * (inverse_metric @ gradient)
    return _training_result(
        "entropy_descent",
        dataset,
        parameters,
        history,
        spectral_entropy=spectral_entropy,
        theta_history=theta_history,
    )


def train_hamiltonian_geometric(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 600,
    learning_rate: float = 0.14,
    beta: float = 0.9,
    metric_regularization: float = 1e-3,
    memory_coupling: float = 0.02,
    memory_decay: float = 0.9,
    spectral_weight: float = 0.0,
    use_geometric_correction: bool = True,
    use_memory_correction: bool = True,
    record_theta: bool = False,
) -> TrainingResult:
    """Train with the paper's proposed Hamiltonian-geometric optimizer.

    Implements Sec. 17 (Eq. 33-34) combined with the explicit memory force of
    Sec. 11 (Eq. 25) and the spectral-entropy regularization of Sec. 14
    (Eq. 28-30), since the paper's own novelty summary (Sec. 21) lists the
    memory correction as part of the framework, not just the base update:

        p_{t+1} = beta * p_t - eta * [grad L + F_geo + mu * F_mem + alpha * grad S(g)]
        theta_{t+1} = theta_t + eta * g^{-1} p_{t+1}

    The metric is the regularized Hessian of Sec. 12 (Eq. 26), g = H + lambda*I,
    with lambda boosted above -min_eigenvalue(H) when needed so g is provably
    positive definite (a fixed lambda alone only guarantees this for a convex
    loss; see the regularization code below).

    F_mem is the memory force of Eq. 25: an exponentially weighted sum of past
    gradients. Read as a recursion, `memory_{t} = memory_decay * memory_{t-1}
    + grad L(theta_t)` is the exact forward-Euler discretization of the
    auxiliary ODE `dM/dt = -kappa_hat * M + grad L(theta)`, i.e. F_mem is the
    state of a Markovian variable relaxing toward the current gradient, not an
    arbitrary function of the full gradient history.

    For this fixed-feature benchmark g is constant in theta (the features are
    fixed, only the linear output weights train), so F_geo = 0.5 p^T (grad_theta
    g^-1) p and grad_theta S(g) are both exactly zero, not merely small:
    grad_theta g is identically zero because g = 2 X^T X + lambda*I depends
    only on the fixed feature matrix X, never on theta. That is a deliberate,
    documented limitation of the lightweight experiment (see
    docs/model_notes.md); a full autodiff network would make both terms
    nonzero. `use_geometric_correction` and `use_memory_correction` let a
    caller ablate F_geo and F_mem independently, as proposed in Sec. 19.3.
    """

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in the range [0.0, 1.0)")
    if not 0.0 <= memory_decay < 1.0:
        raise ValueError("memory_decay must be in the range [0.0, 1.0)")
    if metric_regularization <= 0.0:
        raise ValueError("metric_regularization must be positive")

    # Here the loss is an exact convex quadratic in theta (linear model,
    # squared residual), so raw_hessian = 2 X^T X is always PSD and
    # metric_regularization alone would be sufficient -- but
    # _positive_definite_metric computes the correction generally
    # (Levenberg-Marquardt style) rather than assume convexity silently, so
    # this stays correct if the model is ever extended to a non-convex loss.
    raw_hessian = 2.0 * dataset.design_matrix.T @ dataset.design_matrix
    hessian_metric = positive_definite_metric(raw_hessian, metric_regularization)
    spectral_entropy = _spectral_entropy(hessian_metric)
    state = initial_state(parameter_count)
    optimizer_config = HamiltonianGeometricConfig(
        learning_rate=learning_rate,
        beta=beta,
        metric_regularization=metric_regularization,
        memory_coupling=memory_coupling,
        memory_decay=memory_decay,
        spectral_weight=spectral_weight,
        use_geometric_correction=use_geometric_correction,
        use_memory_correction=use_memory_correction,
    )

    def gradient_fn(theta: np.ndarray) -> np.ndarray:
        return _loss_and_gradient(dataset.design_matrix, dataset.targets, theta)[1]

    def metric_fn(_theta: np.ndarray) -> np.ndarray:
        return raw_hessian

    history: list[float] = []
    theta_history: list[np.ndarray] | None = [] if record_theta else None
    for _ in range(steps):
        loss, _gradient = _loss_and_gradient(dataset.design_matrix, dataset.targets, state.parameters)
        history.append(loss)
        if theta_history is not None:
            theta_history.append(state.parameters.copy())
        state = hamiltonian_geometric_step(state, gradient_fn, metric_fn, optimizer_config)
    return _training_result(
        "hamiltonian_geometric",
        dataset,
        state.parameters,
        history,
        spectral_entropy=spectral_entropy,
        theta_history=theta_history,
    )


def run_optimizer_comparison(
    config: PINNConfig,
    steps: int = 600,
    hamiltonian_kwargs: dict | None = None,
    falling_ball_kwargs: dict | None = None,
    entropy_descent_kwargs: dict | None = None,
    record_theta: bool = False,
) -> tuple[FixedFeaturePotentialModel, PINNDataset, list[TrainingResult]]:
    """Train SGD, Adam, falling-ball, entropy-descent, and Hamiltonian-geometric
    optimizers on one dataset.

    The five optimizers span the two design axes the paper's framework
    separates: metric preconditioning (none vs. the Hessian metric g) and
    momentum/memory (none vs. momentum plus the memory force). SGD and
    falling-ball use no metric; entropy-descent and hamiltonian-geometric use
    the same metric g. SGD and entropy-descent use no momentum; falling-ball
    and hamiltonian-geometric use momentum. Adam sits outside this grid as
    the paper's diagonal-metric-approximation baseline (Sec. 10).

    `*_kwargs` dicts are forwarded to the corresponding `train_*` function,
    so callers can run the Sec. 19.3 ablation test (e.g.
    `hamiltonian_kwargs={"use_memory_correction": False}`) without editing
    this module.
    """

    model = FixedFeaturePotentialModel(config)
    dataset = build_pinn_dataset(model, config)
    results = [
        train_sgd(dataset, model.parameter_count, steps=steps, record_theta=record_theta),
        train_adam(dataset, model.parameter_count, steps=steps, record_theta=record_theta),
        train_falling_ball(
            dataset,
            model.parameter_count,
            steps=steps,
            record_theta=record_theta,
            **(falling_ball_kwargs or {}),
        ),
        train_entropy_descent(
            dataset,
            model.parameter_count,
            steps=steps,
            record_theta=record_theta,
            **(entropy_descent_kwargs or {}),
        ),
        train_hamiltonian_geometric(
            dataset,
            model.parameter_count,
            steps=steps,
            record_theta=record_theta,
            **(hamiltonian_kwargs or {}),
        ),
    ]
    return model, dataset, results


def export_training_history(results: list[TrainingResult], path: str) -> None:
    """Write one CSV row per optimizer step."""

    max_length = max(len(result.loss_history) for result in results)
    with open(path, "w", encoding="utf-8") as file:
        file.write("step,optimizer,loss\n")
        for step in range(max_length):
            for result in results:
                if step < len(result.loss_history):
                    file.write(f"{step + 1},{result.optimizer},{result.loss_history[step]:.12g}\n")


def export_optimizer_summary(results: list[TrainingResult], path: str) -> None:
    """Write final optimizer diagnostics."""

    with open(path, "w", encoding="utf-8") as file:
        file.write(
            "optimizer,final_loss,pde_loss,plate_loss,outer_loss,gradient_norm,spectral_entropy\n"
        )
        for result in results:
            file.write(
                f"{result.optimizer},"
                f"{result.final_loss:.12g},"
                f"{result.pde_loss:.12g},"
                f"{result.plate_loss:.12g},"
                f"{result.outer_loss:.12g},"
                f"{result.gradient_norm:.12g},"
                f"{result.spectral_entropy:.12g}\n"
            )


def export_potential_grid(
    model: FixedFeaturePotentialModel,
    parameters: np.ndarray,
    config: PINNConfig,
    path: str,
    nx: int = 81,
    ny: int = 81,
) -> None:
    """Export the learned potential on a regular grid."""

    x_values = np.linspace(-config.domain_width / 2.0, config.domain_width / 2.0, nx)
    y_values = np.linspace(-config.domain_height / 2.0, config.domain_height / 2.0, ny)
    with open(path, "w", encoding="utf-8") as file:
        file.write("x_m,y_m,potential_V\n")
        for y_value in y_values:
            points = np.column_stack([x_values, np.full(nx, y_value)])
            potentials = model.predict(points, parameters)
            for x_value, potential in zip(x_values, potentials):
                file.write(f"{x_value:.12g},{y_value:.12g},{potential:.12g}\n")


def _outer_boundary_points(config: PINNConfig) -> np.ndarray:
    count_per_edge = max(2, config.outer_boundary_points // 4)
    xs = np.linspace(-config.domain_width / 2.0, config.domain_width / 2.0, count_per_edge)
    ys = np.linspace(-config.domain_height / 2.0, config.domain_height / 2.0, count_per_edge)
    top = np.column_stack([xs, np.full(count_per_edge, config.domain_height / 2.0)])
    bottom = np.column_stack([xs, np.full(count_per_edge, -config.domain_height / 2.0)])
    left = np.column_stack([np.full(count_per_edge, -config.domain_width / 2.0), ys])
    right = np.column_stack([np.full(count_per_edge, config.domain_width / 2.0), ys])
    return np.vstack([top, bottom, left, right])


def _positive_definite_metric(raw_hessian: np.ndarray, regularization: float) -> np.ndarray:
    """Regularize a Hessian into a provably positive-definite metric (paper Eq. 26).

    A fixed `regularization` alone only guarantees a positive-definite result
    when `raw_hessian` has no negative eigenvalues (i.e. a convex loss). This
    boosts the regularization above the most negative eigenvalue of
    `raw_hessian` (Levenberg-Marquardt style) so the guarantee holds in
    general, not only in the convex case.
    """

    return positive_definite_metric(raw_hessian, regularization)


def _spectral_entropy(metric: np.ndarray) -> float:
    """Spectral entropy S(g) of the metric's eigenvalues (paper Eq. 28).

    Measures how spread out the metric's curvature directions are: a low
    value means one or a few eigenvalues dominate (sharp curvature along a
    few directions), a high value means curvature is spread evenly.
    """

    return spectral_entropy(metric)


def _loss_and_gradient(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    parameters: np.ndarray,
) -> tuple[float, np.ndarray]:
    residual = design_matrix @ parameters - targets
    return float(residual @ residual), 2.0 * design_matrix.T @ residual


def _training_result(
    optimizer: str,
    dataset: PINNDataset,
    parameters: np.ndarray,
    history: list[float],
    spectral_entropy: float = 0.0,
    theta_history: list[np.ndarray] | None = None,
) -> TrainingResult:
    _, gradient = _loss_and_gradient(dataset.design_matrix, dataset.targets, parameters)
    pde_residual = dataset.pde_features @ parameters - dataset.pde_targets
    plate_residual = dataset.plate_features @ parameters - dataset.plate_targets
    outer_residual = dataset.outer_features @ parameters - dataset.outer_targets
    return TrainingResult(
        optimizer=optimizer,
        parameters=parameters,
        loss_history=history,
        pde_loss=float(np.mean(pde_residual**2)),
        plate_loss=float(np.mean(plate_residual**2)),
        outer_loss=float(np.mean(outer_residual**2)),
        gradient_norm=float(np.linalg.norm(gradient)),
        spectral_entropy=spectral_entropy,
        theta_history=theta_history,
    )
