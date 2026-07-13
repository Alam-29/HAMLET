# Mathematical Validation of Literature Review

Generated: 2026-07-13T21:42:14

This report checks the PDF's Hamiltonian-geometric optimizer equations before using them in code. The implementation follows the consistent parts of the paper and makes unresolved assumptions explicit.

## Qualitative Findings

| id | severity | section | status | finding | code resolution |
|---|---|---|---|---|---|
| hamiltonian-sign | high | Sec. 6, 11, 14, 17 | corrected-in-code | For H(theta,p) = 0.5 p^T g^{-1}(theta) p + L(theta), Hamilton's equation gives p_dot = -grad L - 0.5 p^T (grad_theta g^{-1}) p. The update rules in Sec. 11 and Sec. 17 match this if F_geo is defined as the positive 0.5 p^T (grad g^{-1}) p and then subtracted inside the force bracket. Sec. 14, Eq. 30 instead writes '+ F_geo', which is a sign inconsistency under that definition. | src.pinn.train_hamiltonian_geometric uses the Sec. 11/17 sign convention: force = grad L + F_geo + memory + spectral. |
| metric-positive-definite | high | Sec. 12 | corrected-in-code | The statement g = H + lambda I, lambda > 0, is not enough to guarantee a positive-definite metric when H has negative eigenvalues. It works for the current convex fixed-feature quadratic loss because H is positive semidefinite, but it is not generally valid for a nonlinear neural-network loss. | src.pinn._positive_definite_metric boosts regularization above -lambda_min(H) when needed, making the metric positive definite even for an indefinite Hessian. |
| memory-vector-vs-matrix | medium | Sec. 11-12 | limited-in-code | F_mem is explicitly defined as an exponentially weighted sum of gradients, so it is a vector force in parameter space. Sec. 12 then writes g_ij = H_ij + mu M_ij, where M_ij is 'implied by' the memory term, but no construction maps the gradient-history vector into a symmetric positive-definite matrix. That matrix-valued memory metric is underspecified. | The implementation uses memory only as the force F_mem from Eq. 25. It does not add an undefined M_ij to the metric. |
| spectral-entropy-gradient | medium | Sec. 14 | limited-in-code | The spectral entropy S(g) is well-defined for positive eigenvalues, but the paper does not derive grad_theta S(g). That gradient is zero only when the metric is constant; for a true parameter-dependent neural-network metric it requires differentiating eigenvalues or using matrix calculus. | The benchmark computes S(g) as a diagnostic. Its gradient is set to zero because the fixed-feature Hessian metric is constant in theta. |
| rayleigh-metric-choice | medium | Sec. 7 | documented-assumption | R(p) = 0.5 gamma p_i p_i is Euclidean damping in momentum coordinates. On a curved parameter manifold a coordinate-invariant damping law would normally specify whether the metric, inverse metric, or a separate friction tensor defines the quadratic form. | The code uses discrete beta damping/momentum and documents it as an optimizer design choice, not a unique covariant dissipation law. |
| adam-correspondence | low | Sec. 10 | consistent-with-caveat | The Adam comparison is mathematically only approximate: Adam's second moment can be interpreted as a local diagonal preconditioner, but Adam does not include a derivative of that metric with respect to parameters. | The runner treats Adam as a baseline, not as the same optimizer as the Hamiltonian-geometric update. |
| fringing-ground-truth | medium | Sec. 19 | not-implemented | The PDF says a conformal Schwarz-Christoffel solution can serve as ground truth, but does not provide the mapping, parameters, or reference values. Without those details, the benchmark can compare optimizer losses but cannot yet claim field-line accuracy against the promised analytic solution. | The existing finite-difference solver is used as a practical reference workflow. A conformal-mapping reference remains a future validation task. |

## Numerical Checks

Generated: 2026-07-13T21:42:14

Each row is a finite-difference or direct-computation check against a fixed, seeded synthetic problem (`src.paper_math_validation`), not the fixed-feature capacitor benchmark -- the capacitor benchmark's metric is constant in theta, so F_geo and grad_theta S(g) are trivially zero there and cannot exercise these formulas.

- **literal**: tests a PDF equation exactly as printed.
- **completed**: tests a specific, documented completion of a formula the PDF uses but never fully derives.
- **counterexample**: demonstrates a literal formula fails, then confirms the shipped code fix succeeds.

| id | kind | section | passed | max error | tolerance | detail |
|---|---|---|---|---|---|---|
| legendre-transform | literal | Sec. 4-6 (Eq. 8, 9, 11, 12) | PASS | 2.960e-11 | 1.000e-06 | max\|numeric dL/dtheta_dot - p\| = 2.960e-11; \|Legendre H - closed-form H\| = 0.000e+00 |
| geometric-force | literal | Sec. 6, 11 (Eq. 13, 24) | PASS | 4.598e-11 | 1.000e-05 | max\|dH/dtheta (numeric) - (grad L + F_geo)\| = 4.598e-11 |
| flat-metric-reduction | literal | Sec. 8 (Eq. 17) | PASS | 0.000e+00 | 1.000e-09 | max\|d(g)/d(theta)\| for g=I = 0.000e+00 |
| memory-recursion | literal | Sec. 11 (Eq. 25) | PASS | 8.882e-16 | 1.000e-09 | max recursion-vs-closed-form error over 30 steps = 8.882e-16 |
| hessian-regularization | counterexample | Sec. 12 (Eq. 26) | PASS | 0.000e+00 | 1.000e-09 | literal Eq. 26 min eigenvalue = -4.999e+00 (fails: True); shipped fix min eigenvalue on same matrix = 1.000e-03; worst min eigenvalue over 200 random indefinite trials = 1.000e-03 |
| rayleigh-curved-metric | completed | Sec. 7 (Eq. 14-16) | PASS | 1.640e-09 | 1.000e-06 | max\|dR/d(theta_dot) - gamma*p\| = 1.640e-09 (curved-g test point) |
| spectral-entropy-gradient | completed | Sec. 14 (Eq. 28, 30) | PASS | 1.144e-10 | 1.000e-05 | max\|closed-form grad S - finite-difference grad S\| = 1.144e-10 |

**legendre-transform** (Sec. 4-6 (Eq. 8, 9, 11, 12)): p_i = dL/dtheta_dot^i and H = sum(p theta_dot) - L reduce to H = 0.5 p^T g^{-1} p + L(theta) for the metric kinetic term.

**geometric-force** (Sec. 6, 11 (Eq. 13, 24)): dH/dtheta (finite difference on H directly) equals grad L + F_geo, confirming F_geo = 0.5 p^T (grad_theta g^-1) p is exactly Hamilton's curvature term, not an independently-asserted force.

**flat-metric-reduction** (Sec. 8 (Eq. 17)): For the Euclidean metric g = I, d(g^jk)/d(theta^i) = 0 exactly, so F_geo = 0 and the dissipative system reduces exactly to damped gradient descent, not merely approximately.

**memory-recursion** (Sec. 11 (Eq. 25)): M_t = kappa * M_{t-1} + grad L(theta_t), as used by src.pinn.train_hamiltonian_geometric, equals Eq. 25's closed-form sum(kappa^(t-k) grad L(theta_k)) at every step.

**hessian-regularization** (Sec. 12 (Eq. 26)): Eq. 26's literal 'lambda > 0' does not guarantee a positive-definite metric for an indefinite Hessian; the project's _positive_definite_metric fix does, verified on a counterexample and 200 random indefinite trials.

**rayleigh-curved-metric** (Sec. 7 (Eq. 14-16)): R(theta, theta_dot) = 0.5 gamma g_ij(theta) theta_dot^i theta_dot^j (velocity-space, using the kinetic-term metric) gives dR/d(theta_dot) = gamma p exactly for a curved g, resolving the ambiguity in the PDF's momentum-space Eq. 14.

**spectral-entropy-gradient** (Sec. 14 (Eq. 28, 30)): grad_theta S(g) via eigenvalue-perturbation theory matches a direct finite-difference gradient of S(g(theta)) on a genuinely theta-dependent metric.
