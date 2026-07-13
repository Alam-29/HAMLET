"""Numeric validation of the literature-review PDF's optimizer equations.

`src.paper_validation` records qualitative findings (is a formula well-defined,
does it match another section). This module goes further: it evaluates the
PDF's formulas against finite-difference derivatives of a synthetic,
theta-dependent toy problem and reports the actual numerical agreement, so the
qualitative findings are backed by numbers rather than only prose.

Two kinds of check are distinguished, because not every PDF formula is
directly testable as written:

- literal checks test an equation exactly as printed in the PDF (Legendre
  transform, Hamilton's equations, F_geo, the F_mem recursion, the flat-metric
  reduction).
- completed checks test a formula the PDF uses but never fully derives (the
  Rayleigh dissipation function's behavior under a curved metric, and the
  gradient of the spectral entropy). These validate a specific, documented
  completion of the gap, not a claim that the PDF itself contains the formula.

The Hessian-regularization check is a counterexample check: it demonstrates
that the PDF's literal condition (lambda > 0) fails to guarantee a
positive-definite metric for an indefinite Hessian, then confirms the
project's shipped fix (`src.pinn._positive_definite_metric`) succeeds where
the literal formula fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.pinn import _positive_definite_metric


@dataclass(frozen=True)
class MathCheckResult:
    """Outcome of one numeric validation check."""

    identifier: str
    kind: str  # "literal" | "completed" | "counterexample"
    section: str
    description: str
    passed: bool
    max_error: float
    tolerance: float
    detail: str


# A fixed, seeded toy parameter-space problem used across checks. The metric
# must genuinely depend on theta (unlike the project's fixed-feature capacitor
# benchmark, where g is constant in theta and every gradient term is trivially
# zero) so that F_geo, the Rayleigh force, and grad_theta S(g) are all
# nontrivial and worth checking.
_TOY_DIM = 4
_TOY_SEED = 20260713
_toy_rng = np.random.default_rng(_TOY_SEED)
_TOY_BASE = _toy_rng.normal(size=(_TOY_DIM, _TOY_DIM))
_TOY_BASE_METRIC = _TOY_BASE.T @ _TOY_BASE + 0.5 * np.eye(_TOY_DIM)
_TOY_Q = _toy_rng.normal(size=(_TOY_DIM, _TOY_DIM))
_TOY_Q = 0.5 * (_TOY_Q + _TOY_Q.T)
_TOY_C = _toy_rng.normal(size=_TOY_DIM)
_TOY_THETA = _toy_rng.normal(size=_TOY_DIM) * 0.6
_TOY_P = _toy_rng.normal(size=_TOY_DIM)
_TOY_THETA_DOT = _toy_rng.normal(size=_TOY_DIM)
_H = 1e-6


def toy_metric(theta: np.ndarray) -> np.ndarray:
    """A smooth, always positive-definite, genuinely theta-dependent metric."""

    diagonal_part = 1.0 + 0.5 * np.sin(theta) ** 2 + 0.3 * theta**2
    return _TOY_BASE_METRIC + np.diag(diagonal_part)


def toy_loss(theta: np.ndarray) -> float:
    """A smooth, non-convex scalar loss L(theta) for validation."""

    return float(0.5 * theta @ _TOY_Q @ theta + _TOY_C @ theta + 0.1 * np.sum(theta**4))


def _numerical_gradient(func, point: np.ndarray, h: float = _H) -> np.ndarray:
    """Central-difference gradient of a scalar function at `point`."""

    gradient = np.zeros_like(point)
    for i in range(point.shape[0]):
        step = np.zeros_like(point)
        step[i] = h
        gradient[i] = (func(point + step) - func(point - step)) / (2.0 * h)
    return gradient


def _numerical_matrix_derivative(matrix_func, point: np.ndarray, h: float = _H) -> np.ndarray:
    """Central-difference derivative of a matrix-valued function of `point`.

    Returns an array of shape (n, n, len(point)) where [:, :, a] is
    d(matrix_func(point)) / d(point[a]).
    """

    dim = matrix_func(point).shape[0]
    derivative = np.zeros((dim, dim, point.shape[0]))
    for a in range(point.shape[0]):
        step = np.zeros_like(point)
        step[a] = h
        derivative[:, :, a] = (
            matrix_func(point + step) - matrix_func(point - step)
        ) / (2.0 * h)
    return derivative


def check_legendre_transform() -> MathCheckResult:
    """Eq. 8-11: canonical momentum and the Legendre transform of H."""

    theta, p = _TOY_THETA, _TOY_P
    g = toy_metric(theta)
    theta_dot = np.linalg.solve(g, p)  # Eq. 12: theta_dot = g^{-1} p

    def lagrangian(td: np.ndarray) -> float:
        return float(0.5 * td @ g @ td - toy_loss(theta))

    p_from_lagrangian = _numerical_gradient(lagrangian, theta_dot)  # Eq. 8
    momentum_error = float(np.max(np.abs(p_from_lagrangian - p)))

    hamiltonian_direct = float(p @ theta_dot - lagrangian(theta_dot))  # Eq. 9
    hamiltonian_formula = float(0.5 * p @ np.linalg.solve(g, p) + toy_loss(theta))  # Eq. 11
    hamiltonian_error = abs(hamiltonian_direct - hamiltonian_formula)

    max_error = max(momentum_error, hamiltonian_error)
    tolerance = 1e-6
    return MathCheckResult(
        identifier="legendre-transform",
        kind="literal",
        section="Sec. 4-6 (Eq. 8, 9, 11, 12)",
        description=(
            "p_i = dL/dtheta_dot^i and H = sum(p theta_dot) - L reduce to "
            "H = 0.5 p^T g^{-1} p + L(theta) for the metric kinetic term."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=(
            f"max|numeric dL/dtheta_dot - p| = {momentum_error:.3e}; "
            f"|Legendre H - closed-form H| = {hamiltonian_error:.3e}"
        ),
    )


def check_geometric_force_matches_hamiltons_equation() -> MathCheckResult:
    """Eq. 13 vs Eq. 24: F_geo is exactly Hamilton's curvature term."""

    theta, p = _TOY_THETA, _TOY_P

    def hamiltonian_of_theta(th: np.ndarray) -> float:
        g = toy_metric(th)
        return float(0.5 * p @ np.linalg.solve(g, p) + toy_loss(th))

    dH_dtheta_numeric = _numerical_gradient(hamiltonian_of_theta, theta)  # Eq. 5

    g = toy_metric(theta)
    g_inv = np.linalg.inv(g)
    dg = _numerical_matrix_derivative(toy_metric, theta)
    n = theta.shape[0]
    f_geo = np.zeros(n)
    for i in range(n):
        d_ginv_i = -g_inv @ dg[:, :, i] @ g_inv  # d(g^-1)/d theta_i
        f_geo[i] = 0.5 * p @ d_ginv_i @ p  # Eq. 24

    dL_dtheta = _numerical_gradient(toy_loss, theta)
    predicted_dH_dtheta = dL_dtheta + f_geo  # Eq. 13
    max_error = float(np.max(np.abs(dH_dtheta_numeric - predicted_dH_dtheta)))
    tolerance = 1e-5
    return MathCheckResult(
        identifier="geometric-force",
        kind="literal",
        section="Sec. 6, 11 (Eq. 13, 24)",
        description=(
            "dH/dtheta (finite difference on H directly) equals grad L + F_geo, "
            "confirming F_geo = 0.5 p^T (grad_theta g^-1) p is exactly "
            "Hamilton's curvature term, not an independently-asserted force."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|dH/dtheta (numeric) - (grad L + F_geo)| = {max_error:.3e}",
    )


def check_rayleigh_dissipation_generalizes_to_curved_metric() -> MathCheckResult:
    """Completed Eq. 14: velocity-space Rayleigh function gives gamma*p for any g.

    The PDF's Eq. 14, R(p) = 0.5 * gamma * p^i p_i, contains no metric at all,
    so it cannot be evaluated on a curved g as written -- it is only
    unambiguous in flat coordinates. This checks the natural completion,
    R(theta, theta_dot) = 0.5 * gamma * g_ij(theta) theta_dot^i theta_dot^j
    (the standard textbook Rayleigh dissipation function, built from the same
    metric as the kinetic energy of Eq. 10), and verifies its dissipative
    force dR/d(theta_dot) equals gamma * p exactly, for a genuinely curved g
    -- not merely in the Euclidean case the literal Eq. 14 is confined to.
    """

    theta, theta_dot = _TOY_THETA, _TOY_THETA_DOT
    g = toy_metric(theta)
    p = g @ theta_dot  # Eq. 8/12 canonical momentum

    def rayleigh(td: np.ndarray) -> float:
        gamma = 1.0
        return float(0.5 * gamma * td @ g @ td)

    dR_dtheta_dot = _numerical_gradient(rayleigh, theta_dot)
    predicted = 1.0 * p  # gamma = 1.0
    max_error = float(np.max(np.abs(dR_dtheta_dot - predicted)))
    tolerance = 1e-6
    return MathCheckResult(
        identifier="rayleigh-curved-metric",
        kind="completed",
        section="Sec. 7 (Eq. 14-16)",
        description=(
            "R(theta, theta_dot) = 0.5 gamma g_ij(theta) theta_dot^i theta_dot^j "
            "(velocity-space, using the kinetic-term metric) gives "
            "dR/d(theta_dot) = gamma p exactly for a curved g, resolving the "
            "ambiguity in the PDF's momentum-space Eq. 14."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|dR/d(theta_dot) - gamma*p| = {max_error:.3e} (curved-g test point)",
    )


def check_memory_recursion_matches_closed_form() -> MathCheckResult:
    """Eq. 25: the discrete recursion equals the exponentially weighted sum."""

    kappa = 0.87
    steps = 30
    rng = np.random.default_rng(11)
    gradients = rng.normal(size=(steps, 3))

    memory = np.zeros(3)
    max_error = 0.0
    for t in range(1, steps + 1):
        memory = kappa * memory + gradients[t - 1]
        closed_form = sum(
            kappa ** (t - k) * gradients[k - 1] for k in range(1, t + 1)
        )
        max_error = max(max_error, float(np.max(np.abs(memory - closed_form))))

    tolerance = 1e-9
    return MathCheckResult(
        identifier="memory-recursion",
        kind="literal",
        section="Sec. 11 (Eq. 25)",
        description=(
            "M_t = kappa * M_{t-1} + grad L(theta_t), as used by "
            "src.pinn.train_hamiltonian_geometric, equals Eq. 25's closed-form "
            "sum(kappa^(t-k) grad L(theta_k)) at every step."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max recursion-vs-closed-form error over {steps} steps = {max_error:.3e}",
    )


def _spectral_entropy(metric: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(metric)
    eigenvalues = np.clip(eigenvalues, 1e-15, None)
    normalized = eigenvalues / eigenvalues.sum()
    return float(-np.sum(normalized * np.log(normalized)))


def check_spectral_entropy_gradient() -> MathCheckResult:
    """Completed Eq. 28: a closed-form grad_theta S(g), not given in the PDF.

    Eq. 28 defines S(g) but Sec. 14/17 use grad_theta S(g) (Eq. 30, 33)
    without ever deriving it. This checks a specific completion via
    first-order eigenvalue perturbation theory (the Hellmann-Feynman
    relation): d(lambda_i)/d(theta_a) = u_i^T (d g / d theta_a) u_i, chained
    through S(g) = -sum(lambda_tilde_i * log(lambda_tilde_i)).
    """

    theta = _TOY_THETA
    g = toy_metric(theta)
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    total = float(eigenvalues.sum())
    normalized = eigenvalues / total
    entropy = float(-np.sum(normalized * np.log(normalized)))

    dg = _numerical_matrix_derivative(toy_metric, theta)
    n = theta.shape[0]
    grad_formula = np.zeros(n)
    for a in range(n):
        accumulator = 0.0
        for i in range(n):
            u_i = eigenvectors[:, i]
            perturbation = float(u_i @ dg[:, :, a] @ u_i)
            accumulator += (np.log(normalized[i]) + entropy) * perturbation
        grad_formula[a] = -accumulator / total

    grad_numeric = _numerical_gradient(lambda th: _spectral_entropy(toy_metric(th)), theta)
    max_error = float(np.max(np.abs(grad_formula - grad_numeric)))
    tolerance = 1e-5
    return MathCheckResult(
        identifier="spectral-entropy-gradient",
        kind="completed",
        section="Sec. 14 (Eq. 28, 30)",
        description=(
            "grad_theta S(g) via eigenvalue-perturbation theory matches a "
            "direct finite-difference gradient of S(g(theta)) on a "
            "genuinely theta-dependent metric."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|closed-form grad S - finite-difference grad S| = {max_error:.3e}",
    )


def check_flat_metric_reduces_to_gradient_descent() -> MathCheckResult:
    """Eq. 17: F_geo vanishes identically when g is constant (Euclidean)."""

    theta = _TOY_THETA

    def flat_metric(_theta: np.ndarray) -> np.ndarray:
        return np.eye(theta.shape[0])

    dg = _numerical_matrix_derivative(flat_metric, theta)
    max_error = float(np.max(np.abs(dg)))
    tolerance = 1e-9
    return MathCheckResult(
        identifier="flat-metric-reduction",
        kind="literal",
        section="Sec. 8 (Eq. 17)",
        description=(
            "For the Euclidean metric g = I, d(g^jk)/d(theta^i) = 0 exactly, "
            "so F_geo = 0 and the dissipative system reduces exactly to "
            "damped gradient descent, not merely approximately."
        ),
        passed=max_error < tolerance,
        max_error=max_error,
        tolerance=tolerance,
        detail=f"max|d(g)/d(theta)| for g=I = {max_error:.3e}",
    )


def check_hessian_regularization() -> MathCheckResult:
    """Eq. 26 counterexample + confirmation that the shipped fix repairs it.

    Demonstrates that a literal fixed lambda > 0 fails to guarantee a
    positive-definite metric for an indefinite Hessian (a concrete
    counterexample, not merely an assertion), then confirms
    src.pinn._positive_definite_metric -- the project's actual shipped code --
    succeeds on the same matrix and across many random indefinite trials.
    """

    literal_lambda = 1e-3
    counterexample_hessian = np.diag([-5.0, 0.1, 3.0])
    literal_metric = counterexample_hessian + literal_lambda * np.eye(3)
    literal_min_eigenvalue = float(np.linalg.eigvalsh(literal_metric).min())
    literal_fails = literal_min_eigenvalue <= 0.0

    fixed_metric = _positive_definite_metric(counterexample_hessian, literal_lambda)
    fixed_min_eigenvalue = float(np.linalg.eigvalsh(fixed_metric).min())

    rng = np.random.default_rng(2026)
    worst_min_eigenvalue = float("inf")
    trials = 200
    for _ in range(trials):
        dim = int(rng.integers(2, 6))
        random_symmetric = rng.normal(size=(dim, dim))
        random_symmetric = 0.5 * (random_symmetric + random_symmetric.T) * rng.uniform(0.5, 5.0)
        regularized = _positive_definite_metric(random_symmetric, literal_lambda)
        worst_min_eigenvalue = min(worst_min_eigenvalue, float(np.linalg.eigvalsh(regularized).min()))
    worst_min_eigenvalue = min(worst_min_eigenvalue, fixed_min_eigenvalue)

    # max_error follows the same convention as every other check here (0 is
    # perfect, must be under tolerance): how far the shipped fix's
    # worst-case min eigenvalue falls short of strictly positive, clipped at
    # zero when it never falls short.
    max_error = max(0.0, -worst_min_eigenvalue)
    tolerance = 1e-9
    passed = literal_fails and max_error < tolerance
    return MathCheckResult(
        identifier="hessian-regularization",
        kind="counterexample",
        section="Sec. 12 (Eq. 26)",
        description=(
            "Eq. 26's literal 'lambda > 0' does not guarantee a "
            "positive-definite metric for an indefinite Hessian; the "
            "project's _positive_definite_metric fix does, verified on a "
            "counterexample and 200 random indefinite trials."
        ),
        passed=passed,
        max_error=max_error,
        tolerance=tolerance,
        detail=(
            f"literal Eq. 26 min eigenvalue = {literal_min_eigenvalue:.3e} (fails: {literal_fails}); "
            f"shipped fix min eigenvalue on same matrix = {fixed_min_eigenvalue:.3e}; "
            f"worst min eigenvalue over {trials} random indefinite trials = {worst_min_eigenvalue:.3e}"
        ),
    )


def run_all_checks() -> list[MathCheckResult]:
    """Run every numeric validation check and return the results in order."""

    return [
        check_legendre_transform(),
        check_geometric_force_matches_hamiltons_equation(),
        check_flat_metric_reduces_to_gradient_descent(),
        check_memory_recursion_matches_closed_form(),
        check_hessian_regularization(),
        check_rayleigh_dissipation_generalizes_to_curved_metric(),
        check_spectral_entropy_gradient(),
    ]


def export_numeric_checks_markdown(results: list[MathCheckResult]) -> str:
    """Render the numeric checks as a Markdown section (no file I/O)."""

    lines = [
        "## Numerical Checks",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        (
            "Each row is a finite-difference or direct-computation check against "
            "a fixed, seeded synthetic problem (`src.paper_math_validation`), not "
            "the fixed-feature capacitor benchmark -- the capacitor benchmark's "
            "metric is constant in theta, so F_geo and grad_theta S(g) are "
            "trivially zero there and cannot exercise these formulas."
        ),
        "",
        "- **literal**: tests a PDF equation exactly as printed.",
        (
            "- **completed**: tests a specific, documented completion of a "
            "formula the PDF uses but never fully derives."
        ),
        (
            "- **counterexample**: demonstrates a literal formula fails, then "
            "confirms the shipped code fix succeeds."
        ),
        "",
        "| id | kind | section | passed | max error | tolerance | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.identifier} | {result.kind} | {result.section} | "
            f"{'PASS' if result.passed else 'FAIL'} | "
            f"{result.max_error:.3e} | {result.tolerance:.3e} | "
            f"{_escape_markdown_table(result.detail)} |"
        )
    lines.append("")
    for result in results:
        lines.append(f"**{result.identifier}** ({result.section}): {result.description}")
        lines.append("")
    return "\n".join(lines)


def _escape_markdown_table(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")
