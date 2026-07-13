# Capacitor Fringing Model Notes

These notes explain what the current project models, what it does not model
yet, and which outputs should be trusted cautiously.

## Physical Problem

The ideal parallel-plate capacitance formula is

```text
C = epsilon A / d 
```

where `epsilon` is the permittivity, `A` is plate area, and `d` is plate
separation.

That formula assumes the electric field is uniform and vertical everywhere
between the plates. A real finite capacitor violates that assumption near its
edges: field lines bend outward, increasing stored field energy and changing
the effective capacitance. That edge bending is the fringing effect.

## Analytical Baseline

The analytical baseline in `src/capacitance.py` contains two models:

- Ideal parallel plate: assumes no fringing.
- Effective-area fringe approximation: expands plate dimensions by a tunable
  fraction of the gap.

The effective-area model is not a law of nature. It is a simple comparison
model. Its value is that it gives a quick expected trend: fringing becomes more
important when the gap is larger compared with the plate dimensions.

## Numerical Model

The numerical model in `src/laplace2d.py` solves a 2D cross-section. It treats
the plates as thin electrode line segments in a rectangular computational box.
The solved equation is Laplace's equation:

```text
del^2(phi) = 0
```

The upper electrode is fixed at `+V/2`, and the lower electrode is fixed at
`-V/2`. The outside boundary uses a zero-normal-gradient approximation, which
means the boundary is intended to behave like a distant open boundary rather
than a grounded conductor.

This is why domain-size studies matter: if the box is too small, the artificial
boundary can still affect the field.

The default numerical iteration is red-black successive over-relaxation (SOR),
which is a Gauss-Seidel style update split into alternating checkerboard
subsets. This is more credible and faster than the original plain relaxation
scheme because newly updated neighboring values are used within the same
iteration. The CLI still supports `--method jacobi` when a simpler baseline is
useful for comparison.

The solver reports both `max_delta` and `residual_norm`. `max_delta` measures
how much the potential changed during the final iteration. `residual_norm`
checks the actual discrete Laplace equation on non-electrode interior nodes,
normalized by `V / h^2`, where `h` is the smaller grid spacing. For numerical
validation, residual error is the more direct equation-level diagnostic.

## Capacitance Estimates

The solver reports two capacitance estimates per unit depth.

Energy method:

```text
U' = 0.5 epsilon integral(|E|^2 dA)
C' = 2 U' / V^2
```

Charge method:

```text
C' = Q' / V
```

The charge method estimates `Q'` from electric flux near the positive
electrode. In a perfect continuum solution, both methods should agree. In this
finite-difference model, disagreement between the two is a numerical uncertainty
signal.

## Physical System Layer

`src/physical_system.py` adds non-ideal operating parameters on top of the
field and capacitance models. It does not replace the Laplace solver. Instead,
it computes observables that depend on the built device and its environment:

- effective relative permittivity from temperature and humidity;
- effective capacitance after applying a fringing multiplier;
- stored energy, charge, electric field, and breakdown margin;
- humidity-adjusted leakage resistance and leakage current;
- AC dielectric loss from frequency and loss tangent;
- generator conversion loss, shaft/friction loss, and source-resistor loss;
- heat dissipation through a lumped thermal conductance plus radiation;
- net heat and instantaneous temperature rate;
- RC time constant and dielectric quality factor.

The formulas are deliberately explicit engineering models. The default
coefficients are placeholders for ordinary trend studies, not measured material
constants. For a real device, choose `DielectricMaterial` coefficients from a
datasheet or experiment, especially permittivity temperature coefficient,
humidity coefficient, resistivity, loss tangent, and dielectric strength.

The main runner writes these values to `physical_observables.csv`. The
environmental and build parameters are independent of the field solve except
for the fringing multiplier, which is taken from the latest numerical solve.

## Detector Plate And Interference Layer

`src/measurement.py` turns the solved field into a measurement prototype. A
`DetectorPlate` samples the field near the capacitor, like a photo plate or
detector plate placed around the fringing region. Each sample records:

- `Ex`, `Ey`, field magnitude, and direction angle;
- normal and tangential components relative to the detector plate;
- local maxima and minima along the detector;
- edge-to-center field contrast;
- direction spread and tangential-to-normal fringing ratio.

`ExternalInterference` adds measurement-side perturbations without changing the
underlying electrostatic solution: uniform external fields, sinusoidal spatial
EMI, and deterministic stochastic noise. The random component is scaled by
humidity and temperature so environmental conditions can broaden the observed
field extrema. This gives the optimization framework two useful targets: the
clean simulated field and the statistically perturbed detector field.

Even with interference, field-line crossing is not inserted into the physical
Laplace solution. Crossed lines would mean two electric-field directions at one
point. The detector layer instead models real-life observation complexity as
external perturbation, noise, and shifted maxima/minima.

## 3D Moving Field View

`src/visualization3d.py` exports a standalone HTML animation of a finite 3D
parallel-plate capacitor. The animation is staged: first an external field
moves through the empty scene, then the capacitor plates enter that field, then
an intentionally irregular transient shows the field responding to the imposed
plate boundary conditions plus optional EMI, and finally the settled field
lines align inside the capacitor and fringe around all plate edges.

The moving dots in `visualizations/3d_projections/capacitor_3d_field_animation.html`
are tracer particles.
Before insertion they follow the external field. During the transient they
follow irregular perturbation paths. In the settled phase they travel along the
field-line paths from the positive plate to the negative plate. In a DC
electrostatic solve the final field is static; the animation shows direction
and relaxation behavior for the prototype, not a full Maxwell time-domain
solver.

## What The Studies Check

`grid_convergence.csv` checks whether the result stabilizes as the grid is
refined. Prefer rows where:

- `converged` is `1`.
- `gap_error_m` is close to zero.
- `plate_width_error_m` is close to zero.
- `relative_change_from_previous` decreases as the grid gets finer.
- `capacitance_estimate_relative_difference` decreases as the grid gets finer.

`domain_size_study.csv` checks whether moving the artificial outer boundary
changes the result. If the final relative change is small, the chosen domain is
probably large enough for that geometry.

`numerical_gap_sweep.csv` compares the numerical field solver against the
simple effective-area trend across several gaps. This is mostly a qualitative
check for now: the energy and charge methods should tell a consistent story
before the numerical ratios are treated as final.

`adaptive_gap_sweep.csv` repeats the same comparison after choosing the grid
spacing from the smallest gap. This is usually the fairer sweep when small gaps
are included, because every gap has a reasonable number of grid cells across
the electrode separation.

The gap sweeps include `gap_to_width`. This dimensionless geometry ratio is
more useful than the raw gap alone when comparing fringing strength across
capacitor sizes. A larger `gap_to_width` generally produces stronger fringing
relative to the ideal uniform-field estimate.

If the energy-based numerical fringe ratio falls below `1` for a small gap,
that should be read as a numerical warning, not as physical evidence that
fringing reduced capacitance. For a finite capacitor in this setup, fringing is
expected to increase capacitance relative to the ideal no-edge-field estimate.

## Current Limitations

The current model is useful for learning and trend studies, but it is not yet a
high-accuracy reference solver.

Known limitations:

- It is 2D, so capacitance is per unit depth rather than full 3D capacitance.
- Electrodes are represented on grid nodes as thin line segments.
- The charge estimate is sensitive to how the electrode surface is represented.
- The open boundary is approximate, not truly infinite space.
- The dielectric is uniform; layered or nonlinear dielectrics are not modeled.
- The physical-system layer uses lumped scalar coefficients for temperature,
  humidity, leakage, and heat flow; it is not a spatial thermal or moisture
  diffusion model.
- There is no comparison against an external analytical conformal-mapping
  solution or finite-element package yet.

## Hamiltonian-Geometric PINN Benchmark

`src/hamiltonian_geometric.py` implements the optimizer equations from the
literature review, and `src/pinn.py` applies them to the benchmark proposed in
the paper: learning the capacitor potential `phi(x, y)` by minimizing a
physics-informed loss,

```text
L(theta) = mean((del^2 phi_theta)^2) + boundary penalties.
```

The current implementation uses fixed nonlinear features and trains only the
linear output weights. This is intentionally smaller than a full neural network,
but it keeps the parameter vector, PDE residual, boundary penalties, and Adam
baseline explicit.

The reusable `hamiltonian_geometric_step` implements the paper's "Proposed
Optimizer" (Sec. 17, Eq. 33-34) combined with the explicit memory force
(Sec. 11, Eq. 25) and the spectral-entropy regularizer (Sec. 14, Eq. 28-30):

```text
p_{t+1} = beta * p_t - eta * [grad L + F_geo + mu * F_mem + alpha * grad S(g)]
theta_{t+1} = theta_t + eta * g^{-1} p_{t+1}
```

The metric `g` is the regularized Hessian of the loss (Sec. 12, Eq. 26). The
reusable optimizer accepts a general `metric_fn(theta)` and computes
`F_geo = 0.5 p^T (partial_theta g^-1) p` plus `grad S(g)` by central finite
differences, so theta-dependent metrics are now represented in code. In this
fixed-feature benchmark specifically, `g` does not vary with `theta`, so both
the curvature force `F_geo` and the spectral-entropy gradient `grad S(g)` are
exactly zero -- not merely small -- and the run is mathematically equivalent
to a plain Hessian-metric optimizer plus the memory force. The spectral entropy
`S(g)` itself is still computed and reported (as `spectral_entropy` in the
exported summary) because it is a cheap, informative diagnostic of how
concentrated the metric's curvature is (Sec. 13-14), even when its gradient
doesn't move the trajectory here.

`train_hamiltonian_geometric` accepts `use_geometric_correction` and
`use_memory_correction` flags so `F_geo` and `F_mem` can be ablated
independently, per the test proposed in Sec. 19.3, item 4. `run_optimizer_comparison`
forwards a `hamiltonian_kwargs` dict for this, and
`main/run_pinn_benchmark.py` exposes `--disable-geometric-correction`,
`--disable-memory-correction`, and `--spectral-weight` on the command line.
Because `F_geo` is exactly zero here, `--disable-geometric-correction` is a
no-op check (the loss trajectory is identical with or without it) rather than
a meaningful ablation -- that only becomes meaningful once the metric depends
on `theta` (see Next Research Improvements, item 4).

The runner `main/run_pinn_benchmark.py` exports optimizer histories and final
diagnostics. Use those CSV files as a falsifiable comparison, not as proof that
one optimizer is universally better.

The paper's equations are not all mutually complete as written. The structured
review in `src/paper_validation.py` and `docs/mathematical_validation.md`
records the main issues before implementation:

- `F_geo` has a sign inconsistency between Sec. 14 and the Sec. 11/17 update
  rules; the code follows the Sec. 11/17 convention.
- `g = H + lambda I` is positive definite only when `lambda` is large enough
  relative to the most negative Hessian eigenvalue; the code adapts the
  regularization instead of assuming convexity.
- `F_mem` is a vector force, while the paper's matrix-valued `M_ij` is not
  constructed; the code uses the explicit vector memory force only.
- The promised conformal-mapping ground truth is not specified in enough detail
  to implement yet.

## Next Research Improvements

Good next steps:

1. Improve electrode treatment, for example by using a control-volume style
   charge estimate or thicker electrode representation.
2. Compare against a published conformal-mapping result or an independently
   computed finite-element reference case.
3. Add Richardson-style extrapolation for grid-converged capacitance estimates.
4. Replace the fixed-feature PINN with a full autodiff neural network and pass
   its theta-dependent metric into `hamiltonian_geometric_step`, so `F_geo` and
   `grad S(g)` become nonzero in the actual capacitor benchmark rather than
   only in the standalone optimizer-core tests.
5. Extend to 3D only after the 2D solver is well understood.
