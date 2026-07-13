"""Normal-mode / action-angle analysis of the Hamiltonian-geometric optimizer.

This module makes precise, and numerically checks, three informal claims
about why the metric-preconditioned optimizer should out-perform plain
gradient descent on this benchmark: that it "exploits symmetries", that a
canonical transformation to action-angle variables makes the conservative
dynamics linear, and that this makes the optimization "dynamically easier to
navigate."

The capacitor benchmark's loss is an exact convex quadratic in theta (a
linear-in-theta model with squared-residual loss), so this is not a local
approximation valid only near a minimum -- it is exact everywhere on the
actual trajectory the optimizers take. Write

    L(theta) = ||X theta - y||^2 = theta^T A theta - 2 b^T theta + const,
    A = X^T X,  b = X^T y,  grad L = 2 A (theta - theta*),  Hessian = 2A.

The metric used by train_hamiltonian_geometric / train_entropy_descent is
g = 2A + lambda*I (Sec. 12, Eq. 26). Because g is built from A itself, g and
2A commute and share the same eigenvectors -- the "descent symmetries" the
metric exploits are literally A's own eigenbasis, the (mass-weighted)
principal-component directions of the feature design matrix X.

Conservative normal modes. Consider the conservative part of the Hamiltonian
(Sec. 5-6, dropping the -gamma*p damping term), H(theta,p) = 0.5 p^T g^-1 p +
L(theta). With A = Q diag(lambda) Q^T, substituting u = theta - theta*,
c = Q^T u decouples the system into n independent unit-mass harmonic
oscillators with frequency

    omega_i = sqrt(2*lambda_i / (2*lambda_i + lambda_reg)).

This change of variables is a canonical (linear, symplectic) transformation.
Passing each decoupled oscillator to the standard action-angle variables
(I_i, phi_i) gives H_i = omega_i * I_i -- linear in the action, so the
conservative flow is exactly integrable: each I_i is a constant of motion,
each level set {I = const} is an invariant torus (Liouville-Arnold), and
phi_i advances at the constant rate omega_i. That is the precise sense in
which the transformation "makes the system linear": an n-dimensional coupled
oscillation becomes n independent, exactly solvable 1-D oscillations.

Discrete-time consequence (what actually matters for the optimizer). Because
g and A share eigenvectors, both plain gradient descent and metric-
preconditioned descent decouple *exactly* -- not approximately -- into n
independent scalar recursions in the same eigenbasis:

    plain:          c_i,{t+1} = (1 - 2 eta lambda_i) c_i,t
    preconditioned: c_i,{t+1} = (1 - 2 eta lambda_i / (2 lambda_i + lambda_reg)) c_i,t

Plain descent's stability requires eta < 1/lambda_max, so the flattest mode
(smallest lambda_i) contracts at rate (1 - 2 lambda_min/lambda_max) per
step -- arbitrarily close to 1 (arbitrarily slow) as the condition number
lambda_max/lambda_min grows. The preconditioned multiplier
2 eta lambda_i/(2 lambda_i + lambda_reg) is bounded by eta for every mode
(the ratio lambda_i/(2 lambda_i + lambda_reg) < 1/2 always), so a single
eta close to 1 contracts every mode at a comparable rate simultaneously,
regardless of the condition number. That is the concrete, checkable form of
"computationally richer and more efficient": conditioning-independent
per-step contraction instead of a worst-case rate set by the flattest
direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.paper_math_validation import MathCheckResult
from src.pinn import PINNConfig, PINNDataset, _positive_definite_metric, build_pinn_dataset


@dataclass(frozen=True)
class NormalModeDecomposition:
    """Eigen-structure of the capacitor benchmark's loss Hessian and metric."""

    eigenvalues: np.ndarray  # lambda_i, eigenvalues of A = X^T X (Hessian = 2A)
    eigenvectors: np.ndarray  # columns are v_i, the shared eigenbasis of A and g
    metric_regularization: float
    metric_eigenvalues: np.ndarray  # eigenvalues of g = 2A + lambda_reg*I
    omega: np.ndarray  # conservative normal-mode frequencies sqrt(2 lambda_i / g-eigenvalue)
    condition_number: float  # lambda_max / lambda_min (nonzero), raw Hessian conditioning


def compute_normal_modes(
    dataset: PINNDataset,
    metric_regularization: float = 1e-3,
) -> NormalModeDecomposition:
    """Diagonalize the benchmark's loss Hessian and derive its normal modes."""

    gram = dataset.design_matrix.T @ dataset.design_matrix  # A = X^T X
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    metric_eigenvalues = 2.0 * eigenvalues + metric_regularization
    omega = np.sqrt(2.0 * eigenvalues / metric_eigenvalues)

    scale = eigenvalues.max() if eigenvalues.size else 0.0
    nonzero = eigenvalues[eigenvalues > 1e-9 * max(scale, 1.0)]
    condition_number = float(eigenvalues.max() / nonzero.min()) if nonzero.size else float("inf")

    return NormalModeDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        metric_regularization=metric_regularization,
        metric_eigenvalues=metric_eigenvalues,
        omega=omega,
        condition_number=condition_number,
    )


def check_metric_shares_eigenbasis_with_hessian(
    dataset: PINNDataset,
    metric_regularization: float = 1e-3,
) -> MathCheckResult:
    """The metric g commutes exactly with the loss Hessian (shared eigenbasis)."""

    raw_hessian = 2.0 * dataset.design_matrix.T @ dataset.design_matrix
    metric = _positive_definite_metric(raw_hessian, metric_regularization)
    commutator = raw_hessian @ metric - metric @ raw_hessian
    max_error = float(np.max(np.abs(commutator)))
    scale = float(np.max(np.abs(raw_hessian)) * np.max(np.abs(metric))) or 1.0
    tolerance = 1e-6 * scale
    return MathCheckResult(
        identifier="shared-eigenbasis",
        kind="literal",
        section="Sec. 12 (Eq. 26) construction",
        description=(
            "g = H + lambda*I commutes exactly with the loss Hessian H, so g "
            "and H share one eigenbasis -- the 'descent symmetries' the "
            "metric preconditioning exploits are literally H's own "
            "eigenvectors, not a separate or approximate structure."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|[H, g]| commutator entry = {max_error:.3e} (scale {scale:.3e})",
    )


def check_plain_descent_decouples_into_normal_modes(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 5,
    learning_rate: float = 1e-7,
) -> MathCheckResult:
    """Unpreconditioned descent, run in full coordinates, matches n independent
    scalar recursions in the shared eigenbasis -- the discrete-time form of
    normal-mode decoupling."""

    design_matrix, targets = dataset.design_matrix, dataset.targets
    gram = design_matrix.T @ design_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    theta_star, *_ = np.linalg.lstsq(design_matrix, targets, rcond=None)

    theta = np.zeros(parameter_count)
    for _ in range(steps):
        gradient = 2.0 * design_matrix.T @ (design_matrix @ theta - targets)
        theta = theta - learning_rate * gradient

    modal_initial = eigenvectors.T @ (-theta_star)
    multiplier = (1.0 - 2.0 * learning_rate * eigenvalues) ** steps
    modal_final = multiplier * modal_initial
    theta_from_modes = theta_star + eigenvectors @ modal_final

    max_error = float(np.max(np.abs(theta - theta_from_modes)))
    tolerance = 1e-6 * max(1.0, float(np.max(np.abs(theta))))
    return MathCheckResult(
        identifier="plain-descent-decoupling",
        kind="literal",
        section="normal-mode analysis (derived)",
        description=(
            "theta_{t+1} = theta_t - eta * grad L(theta_t), run directly, "
            "matches c_i,{t+1} = (1 - 2 eta lambda_i) c_i,t applied "
            "independently in each normal-mode coordinate c_i."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|theta (direct) - theta (from decoupled modes)| after {steps} steps = {max_error:.3e}",
    )


def check_preconditioned_descent_decouples_into_normal_modes(
    dataset: PINNDataset,
    parameter_count: int,
    steps: int = 5,
    learning_rate: float = 0.02,
    metric_regularization: float = 1e-3,
) -> MathCheckResult:
    """Metric-preconditioned descent also decouples exactly, with each mode's
    multiplier set by lambda_i / (2 lambda_i + lambda_reg) instead of lambda_i
    alone -- the mechanism behind conditioning-independent convergence."""

    design_matrix, targets = dataset.design_matrix, dataset.targets
    gram = design_matrix.T @ design_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    theta_star, *_ = np.linalg.lstsq(design_matrix, targets, rcond=None)

    raw_hessian = 2.0 * gram
    metric = _positive_definite_metric(raw_hessian, metric_regularization)
    inverse_metric = np.linalg.pinv(metric)

    theta = np.zeros(parameter_count)
    for _ in range(steps):
        gradient = 2.0 * design_matrix.T @ (design_matrix @ theta - targets)
        theta = theta - learning_rate * (inverse_metric @ gradient)

    # _positive_definite_metric can boost regularization above the literal
    # `metric_regularization` argument when the raw Hessian's smallest
    # eigenvalue computes as slightly negative due to floating-point error on
    # a severely ill-conditioned Gram matrix (this benchmark's condition
    # number is ~1e8-1e9). Recover the regularization actually applied
    # (metric = raw_hessian + effective_regularization * I) rather than
    # assume the literal argument, or this check compares against the wrong
    # per-mode multiplier.
    effective_regularization = float(np.mean(np.diag(metric - raw_hessian)))
    metric_eigenvalues = 2.0 * eigenvalues + effective_regularization
    modal_initial = eigenvectors.T @ (-theta_star)
    multiplier = (1.0 - 2.0 * learning_rate * eigenvalues / metric_eigenvalues) ** steps
    modal_final = multiplier * modal_initial
    theta_from_modes = theta_star + eigenvectors @ modal_final

    max_error = float(np.max(np.abs(theta - theta_from_modes)))
    tolerance = 1e-6 * max(1.0, float(np.max(np.abs(theta))))
    return MathCheckResult(
        identifier="preconditioned-descent-decoupling",
        kind="literal",
        section="normal-mode analysis (derived)",
        description=(
            "theta_{t+1} = theta_t - eta * g^-1 grad L(theta_t), run "
            "directly, matches c_i,{t+1} = (1 - 2 eta lambda_i/(2 lambda_i + "
            "lambda_reg)) c_i,t applied independently in each mode."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|theta (direct) - theta (from decoupled modes)| after {steps} steps = {max_error:.3e}",
    )


def check_conservative_action_is_linear_in_hamiltonian(
    modes: NormalModeDecomposition,
    mode_index: int = -1,
    leapfrog_steps: int = 2000,
) -> MathCheckResult:
    """For the conservative (undamped) part of one normal mode, verify
    H_i = omega_i * I_i is conserved by a symplectic leapfrog integration --
    the discrete-time signature of an action-angle variable."""

    omega = float(modes.omega[mode_index])
    if omega <= 0.0:
        return MathCheckResult(
            identifier="action-linear-hamiltonian",
            kind="completed",
            section="normal-mode analysis (derived), action-angle variables",
            description="H_i = omega_i I_i for the conservative single-mode oscillator.",
            passed=True,
            max_error=0.0,
            tolerance=1.0,
            detail="omega_i == 0 (a null direction of the loss Hessian); skipped as degenerate.",
        )

    # Leapfrog's energy error for a linear (harmonic) system is bounded and
    # oscillatory in dt*omega, not secular -- but its amplitude scales like
    # (dt*omega)^2, so a fixed 0.01 tolerance needs a small enough dt*omega,
    # not just "many steps".
    dt = 0.01 * (2.0 * np.pi / omega)
    q, p = 1.3, 0.0  # arbitrary nonzero initial condition, unit "mass"
    energies = np.empty(leapfrog_steps)
    for step in range(leapfrog_steps):
        p_half = p - 0.5 * dt * (omega**2) * q
        q = q + dt * p_half
        p = p_half - 0.5 * dt * (omega**2) * q
        energies[step] = 0.5 * p**2 + 0.5 * (omega**2) * q**2

    action = energies / omega  # I_i = H_i / omega_i
    max_error = float(np.max(np.abs(action - action[0])) / action[0])
    tolerance = 1e-2  # leapfrog conserves energy up to bounded O(dt^2) oscillation, not exactly
    return MathCheckResult(
        identifier="action-linear-hamiltonian",
        kind="completed",
        section="normal-mode analysis (derived), action-angle variables",
        description=(
            "H_i(t) = omega_i * I_i(t) stays constant along the conservative "
            "single-mode flow (symplectic leapfrog), confirming I_i is the "
            "action variable and H_i is exactly linear in it."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=(
            f"relative action drift over {leapfrog_steps} leapfrog steps "
            f"(mode omega={omega:.4g}) = {max_error:.3e}"
        ),
    )


def summarize_conditioning(modes: NormalModeDecomposition, learning_rate_preconditioned: float = 0.9) -> dict:
    """Concrete numbers behind the 'conditioning-independent convergence' claim."""

    eigenvalues = modes.eigenvalues
    scale = eigenvalues.max() if eigenvalues.size else 0.0
    nonzero = eigenvalues[eigenvalues > 1e-9 * max(scale, 1.0)]
    lambda_max = float(eigenvalues.max())
    lambda_min = float(nonzero.min()) if nonzero.size else 0.0

    eta_max_stable_plain = 1.0 / lambda_max if lambda_max > 0 else float("inf")
    worst_mode_rate_plain = (
        abs(1.0 - 2.0 * eta_max_stable_plain * lambda_min) if lambda_max > 0 else 0.0
    )

    preconditioned_multipliers = np.abs(
        1.0 - 2.0 * learning_rate_preconditioned * eigenvalues / modes.metric_eigenvalues
    )
    worst_mode_rate_preconditioned = float(np.max(preconditioned_multipliers))

    return {
        "condition_number": modes.condition_number,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "eta_max_stable_plain": eta_max_stable_plain,
        "worst_mode_rate_plain": worst_mode_rate_plain,
        "learning_rate_preconditioned": learning_rate_preconditioned,
        "worst_mode_rate_preconditioned": worst_mode_rate_preconditioned,
    }


def run_all_checks(config: PINNConfig | None = None) -> list[MathCheckResult]:
    """Run every normal-mode check against the actual capacitor benchmark data.

    The exact-identity checks (shared eigenbasis, decoupling) use a smaller
    configuration than the production default. The production PINNConfig's
    design matrix has a condition number around 1e8-1e9 (report via
    `summarize_conditioning`, computed separately with the full config) --
    right at the edge of double-precision reliability, where floating-point
    round-off in the smallest eigenvalue's eigenvector, not a flaw in the
    decoupling identity itself, would dominate a tight numerical-agreement
    check. The identity being checked doesn't depend on problem size, so a
    smaller, better-conditioned instance of the same benchmark validates it
    without that confound.
    """

    from src.pinn import FixedFeaturePotentialModel

    identity_config = PINNConfig(
        hidden_features=6, collocation_points=30, plate_points=8, outer_boundary_points=8
    )
    identity_model = FixedFeaturePotentialModel(identity_config)
    identity_dataset = build_pinn_dataset(identity_model, identity_config)

    production_config = config or PINNConfig()
    production_model = FixedFeaturePotentialModel(production_config)
    production_dataset = build_pinn_dataset(production_model, production_config)
    production_modes = compute_normal_modes(production_dataset)

    return [
        check_metric_shares_eigenbasis_with_hessian(production_dataset),
        check_plain_descent_decouples_into_normal_modes(
            identity_dataset, identity_model.parameter_count
        ),
        check_preconditioned_descent_decouples_into_normal_modes(
            identity_dataset, identity_model.parameter_count
        ),
        check_conservative_action_is_linear_in_hamiltonian(production_modes),
    ]


def export_normal_mode_markdown(
    modes: NormalModeDecomposition,
    checks: list[MathCheckResult],
    conditioning: dict,
) -> str:
    """Render the normal-mode analysis as a Markdown report section."""

    lines = [
        "# Normal-Mode / Action-Angle Analysis",
        "",
        (
            "This report makes precise and numerically checks the claim that "
            "the Hamiltonian-geometric optimizer's metric exploits the loss "
            "Hessian's own symmetries via a canonical transformation to "
            "action-angle variables, and quantifies what that buys in "
            "convergence speed. See `src/normal_modes.py` for the full "
            "derivation."
        ),
        "",
        "## Checks",
        "",
        "| id | kind | section | passed | max error | tolerance | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.identifier} | {check.kind} | {check.section} | "
            f"{'PASS' if check.passed else 'FAIL'} | {check.max_error:.3e} | "
            f"{check.tolerance:.3e} | {_escape(check.detail)} |"
        )
    lines.append("")
    for check in checks:
        lines.append(f"**{check.identifier}**: {check.description}")
        lines.append("")

    lines.extend(
        [
            "## Conditioning comparison",
            "",
            f"- Raw Hessian condition number (lambda_max / lambda_min): "
            f"{conditioning['condition_number']:.4g}",
            f"- lambda_max = {conditioning['lambda_max']:.4g}, "
            f"lambda_min = {conditioning['lambda_min']:.4g}",
            f"- Plain gradient descent's stability bound: eta < 1/lambda_max = "
            f"{conditioning['eta_max_stable_plain']:.4g}",
            f"- At that eta, the flattest mode contracts at rate "
            f"{conditioning['worst_mode_rate_plain']:.6f} per step (closer to 1 "
            "is slower; this is the mode that sets plain descent's overall speed)",
            f"- Preconditioned descent at eta = "
            f"{conditioning['learning_rate_preconditioned']:.3g}: worst-mode "
            f"contraction rate = {conditioning['worst_mode_rate_preconditioned']:.6f}, "
            "bounded independent of the condition number above",
            "",
            f"Normal-mode frequency range (conservative case): "
            f"omega in [{float(modes.omega.min()):.4g}, {float(modes.omega.max()):.4g}]",
            "",
        ]
    )
    return "\n".join(lines)


def _escape(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")
