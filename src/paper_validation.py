"""Structured mathematical validation of the literature-review PDF.

The PDF is a conceptual optimizer paper, not a complete numerical method. This
module records the checks needed to turn its equations into defensible code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.paper_math_validation import MathCheckResult, export_numeric_checks_markdown


@dataclass(frozen=True)
class ValidationFinding:
    """One mathematical validation finding for the paper."""

    identifier: str
    severity: str
    section: str
    status: str
    finding: str
    code_resolution: str


def validate_literature_review_math() -> list[ValidationFinding]:
    """Return the mathematical consistency review used by this project."""

    return [
        ValidationFinding(
            identifier="hamiltonian-sign",
            severity="high",
            section="Sec. 6, 11, 14, 17",
            status="corrected-in-code",
            finding=(
                "For H(theta,p) = 0.5 p^T g^{-1}(theta) p + L(theta), "
                "Hamilton's equation gives p_dot = -grad L - 0.5 p^T "
                "(grad_theta g^{-1}) p. The update rules in Sec. 11 and "
                "Sec. 17 match this if F_geo is defined as the positive "
                "0.5 p^T (grad g^{-1}) p and then subtracted inside the "
                "force bracket. Sec. 14, Eq. 30 instead writes '+ F_geo', "
                "which is a sign inconsistency under that definition."
            ),
            code_resolution=(
                "src.pinn.train_hamiltonian_geometric uses the Sec. 11/17 "
                "sign convention: force = grad L + F_geo + memory + spectral."
            ),
        ),
        ValidationFinding(
            identifier="metric-positive-definite",
            severity="high",
            section="Sec. 12",
            status="corrected-in-code",
            finding=(
                "The statement g = H + lambda I, lambda > 0, is not enough "
                "to guarantee a positive-definite metric when H has negative "
                "eigenvalues. It works for the current convex fixed-feature "
                "quadratic loss because H is positive semidefinite, but it is "
                "not generally valid for a nonlinear neural-network loss."
            ),
            code_resolution=(
                "src.pinn._positive_definite_metric boosts regularization "
                "above -lambda_min(H) when needed, making the metric positive "
                "definite even for an indefinite Hessian."
            ),
        ),
        ValidationFinding(
            identifier="memory-vector-vs-matrix",
            severity="medium",
            section="Sec. 11-12",
            status="limited-in-code",
            finding=(
                "F_mem is explicitly defined as an exponentially weighted "
                "sum of gradients, so it is a vector force in parameter space. "
                "Sec. 12 then writes g_ij = H_ij + mu M_ij, where M_ij is "
                "'implied by' the memory term, but no construction maps the "
                "gradient-history vector into a symmetric positive-definite "
                "matrix. That matrix-valued memory metric is underspecified."
            ),
            code_resolution=(
                "The implementation uses memory only as the force F_mem from "
                "Eq. 25. It does not add an undefined M_ij to the metric."
            ),
        ),
        ValidationFinding(
            identifier="spectral-entropy-gradient",
            severity="medium",
            section="Sec. 14",
            status="limited-in-code",
            finding=(
                "The spectral entropy S(g) is well-defined for positive "
                "eigenvalues, but the paper does not derive grad_theta S(g). "
                "That gradient is zero only when the metric is constant; for "
                "a true parameter-dependent neural-network metric it requires "
                "differentiating eigenvalues or using matrix calculus."
            ),
            code_resolution=(
                "The benchmark computes S(g) as a diagnostic. Its gradient is "
                "set to zero because the fixed-feature Hessian metric is "
                "constant in theta."
            ),
        ),
        ValidationFinding(
            identifier="rayleigh-metric-choice",
            severity="medium",
            section="Sec. 7",
            status="documented-assumption",
            finding=(
                "R(p) = 0.5 gamma p_i p_i is Euclidean damping in momentum "
                "coordinates. On a curved parameter manifold a coordinate-"
                "invariant damping law would normally specify whether the "
                "metric, inverse metric, or a separate friction tensor defines "
                "the quadratic form."
            ),
            code_resolution=(
                "The code uses discrete beta damping/momentum and documents "
                "it as an optimizer design choice, not a unique covariant "
                "dissipation law."
            ),
        ),
        ValidationFinding(
            identifier="adam-correspondence",
            severity="low",
            section="Sec. 10",
            status="consistent-with-caveat",
            finding=(
                "The Adam comparison is mathematically only approximate: "
                "Adam's second moment can be interpreted as a local diagonal "
                "preconditioner, but Adam does not include a derivative of "
                "that metric with respect to parameters."
            ),
            code_resolution=(
                "The runner treats Adam as a baseline, not as the same "
                "optimizer as the Hamiltonian-geometric update."
            ),
        ),
        ValidationFinding(
            identifier="fringing-ground-truth",
            severity="medium",
            section="Sec. 19",
            status="not-implemented",
            finding=(
                "The PDF says a conformal Schwarz-Christoffel solution can "
                "serve as ground truth, but does not provide the mapping, "
                "parameters, or reference values. Without those details, the "
                "benchmark can compare optimizer losses but cannot yet claim "
                "field-line accuracy against the promised analytic solution."
            ),
            code_resolution=(
                "The existing finite-difference solver is used as a practical "
                "reference workflow. A conformal-mapping reference remains a "
                "future validation task."
            ),
        ),
    ]


def write_math_validation_report(
    path: str,
    numeric_checks: list[MathCheckResult] | None = None,
) -> None:
    """Write the validation findings, and optional numeric checks, to Markdown."""

    findings = validate_literature_review_math()
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Mathematical Validation of Literature Review\n\n")
        file.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        file.write(
            "This report checks the PDF's Hamiltonian-geometric optimizer equations "
            "before using them in code. The implementation follows the consistent "
            "parts of the paper and makes unresolved assumptions explicit.\n\n"
        )
        file.write("## Qualitative Findings\n\n")
        file.write("| id | severity | section | status | finding | code resolution |\n")
        file.write("|---|---|---|---|---|---|\n")
        for finding in findings:
            file.write(
                f"| {finding.identifier} | "
                f"{finding.severity} | "
                f"{finding.section} | "
                f"{finding.status} | "
                f"{_escape_markdown_table(finding.finding)} | "
                f"{_escape_markdown_table(finding.code_resolution)} |\n"
            )
        if numeric_checks is not None:
            file.write("\n")
            file.write(export_numeric_checks_markdown(numeric_checks))


def _escape_markdown_table(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")
