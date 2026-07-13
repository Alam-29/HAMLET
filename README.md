# Capacitor Fringing Effects Project

This project is a small research workspace for modeling how a real finite
parallel-plate capacitor differs from the ideal infinite-plate result.

The first version keeps the physics assumptions visible:

- `ideal_parallel_plate`: textbook capacitance, `C = epsilon A / d`.
- `effective_area_fringe`: a simple engineering approximation that pretends the
  plate edges expand by a fraction of the gap.
- `sweep_gap`: helper for comparing the ideal and approximate fringing models
  across several plate spacings.
- `solve_parallel_plate_2d`: finite-difference Laplace solver for a 2D
  cross-section with finite electrodes. The default iteration method is
  red-black SOR; Jacobi is still available for comparison.
- `grid_convergence_study`: repeats the numerical model at multiple grid
  resolutions so the result can be checked for stability.
- `domain_size_study`: repeats the numerical model with larger outer
  simulation boxes so boundary effects can be checked.
- `evaluate_physical_state`: adds system-level operating conditions such as
  temperature, humidity, dielectric drift, leakage, generator/friction losses,
  and heat dissipation.

The key research warning is that fringing fields are not captured exactly by a
single universal correction factor. The correction depends on geometry,
boundary conditions, dielectric environment, and nearby conductors. Treat the
effective-area model as a baseline to compare against better numerical models.

The numerical model estimates capacitance per unit depth from stored field
energy and from electrode charge. Its result depends on grid resolution and how
large the simulation domain is compared with the capacitor. Since electrodes
must sit on grid rows, the solver reports both requested dimensions and actual
grid electrode dimensions. It also reports a dimensionless Laplace residual so
the solved field can be judged by equation error, not only by iteration-to-
iteration change.

The literature-review optimizer architecture is implemented in
`src/hamiltonian_geometric.py` and used by the PINN benchmark in `src/pinn.py`
and `main/run_pinn_benchmark.py`. It trains a lightweight physics-informed
model for the same finite-plate fringing-field problem and compares plain SGD,
Adam, and a Hessian-metric Hamiltonian optimizer with geometric, memory, and
spectral-entropy forces. Because the benchmark uses fixed nonlinear features,
the benchmark metric is exact and constant; the reusable optimizer core also
supports theta-dependent metrics through finite-difference `F_geo` and
`grad S(g)` terms for later full-network experiments.

## Suggested Roadmap

1. Build the analytical baseline.
2. Add a 2D numerical Laplace solver for a capacitor cross-section.
3. Estimate capacitance from stored electrostatic energy.
4. Compare numerical results against the analytical baseline.
5. Add parameter sweeps and visualizations.

For more detail on the equations, assumptions, and limitations, read
`docs/model_notes.md`. For the mathematical consistency review of the PDF's
Hamiltonian-geometric optimizer, read `docs/mathematical_validation.md`.

## Run

```powershell
python .\main\run_models.py
```

You can change the experiment without editing the Python file:

```powershell
python .\main\run_models.py --plate-width 0.015 --gap 0.003 --domain-width 0.08 --domain-height 0.08 --grid 61 --study-grid 61 --output-dir visualizations\custom_demo
```

For a more publication-style visual run with stronger fringing:

```powershell
python .\main\run_models.py --plate-width 0.015 --gap 0.008 --baseline-gap 0.002 --domain-width 0.08 --domain-height 0.08 --grid 161 --study-grid 101 --output-dir visualizations\research_demo
```

For a cleaner small-gap adaptive sweep, increase the number of cells across the
smallest swept gap:

```powershell
python .\main\run_models.py --gap 0.004 --study-grid 101 --adaptive-cells 4
```

To run the Hamiltonian-geometric optimizer benchmark proposed by the literature
review:

```powershell
python .\main\run_pinn_benchmark.py
```

To run the same benchmark with only free/open-source Python tools and export
CSV, PNG, HTML, GIF, and a symbolic derivation note:

```powershell
python .\main\run_free_optimizer_benchmark.py
```

To compare against AlgoPerf-style reference baseline families (`AdamW`,
`Nesterov Momentum`, and `Heavy Ball Momentum`) with a local fixed tuning
budget:

```powershell
python .\main\run_algoperf_style_benchmark.py
```

To regenerate the mathematical validation report for the PDF:

```powershell
python .\main\validate_paper_math.py
```

This writes `visualizations\pinn_benchmark\pinn_training_history.csv`,
`visualizations\pinn_benchmark\pinn_optimizer_summary.csv` (now including a
`spectral_entropy` diagnostic column), `optimizer_convergence.png`,
`optimizer_convergence_dashboard.html`, and a learned potential grid for the
best optimizer.

If Wolfram Mathematica or Wolfram Engine is installed, the companion
Wolfram-language derivation and plotting script can be run with:

```powershell
wolframscript -file .\wolfram\hamiltonian_geometric_benchmark.wl
```

That script symbolically derives Hamilton's equations for
`H(theta,p) = 1/2 p^T g^-1(theta) p + L(theta)` and imports the benchmark CSV
to make a Wolfram `ListLogPlot`. Mathematica itself is licensed software and
is not bundled with this project.

To run the paper's Sec. 19.3 ablation test, disabling the curvature force
`F_geo` and/or the memory force `F_mem`, or to sweep the spectral-entropy
coupling `alpha`:

```powershell
python .\main\run_pinn_benchmark.py --disable-geometric-correction --disable-memory-correction --spectral-weight 0.0
```

See `docs\model_notes.md` for why `--disable-geometric-correction` is
currently a no-op check rather than a meaningful ablation in this
fixed-feature benchmark.

This also writes `visualizations/potential_grid.csv`, which contains the
solved potential field in columns:

```text
x_m,y_m,potential_V,is_electrode
```

It writes `visualizations/potential_field.png` (a matplotlib heatmap of the
solved potential; black bars show the electrode locations).

It also writes `visualizations/fringing_field_lines.png`, a matplotlib
streamline plot of the electric field. This view is usually the clearest way
to see fringing: the field lines run straight and parallel between the
plates, then visibly bow outward and thin out near the plate ends instead of
staying straight everywhere.

It also writes `visualizations/3d_projections/capacitor_3d_field_animation.html`, a
self-contained 3D animation. It cycles through an external field moving before
the capacitor enters, the plates appearing inside that field, a chaotic
boundary-condition/EMI transient, and finally the settled field lining up
inside the capacitor while fringing around the edges. For a DC capacitor the
final field is static; the moving dots are a direction/flow visualization.
You can also generate only the 3D view:

```powershell
python .\main\run_3d_fields.py --fringe-bulge 0.9 --emi-wobble 0.12 --chaotic-transient-strength 1.4
```

It also writes `visualizations/grid_convergence.csv`, which tracks how the
estimated capacitance per unit depth changes as the grid is refined. Prefer
rows where `gap_error_m` and `plate_width_error_m` are close to zero when
comparing resolutions.

Finally, it writes `visualizations/domain_size_study.csv`, which checks whether
moving the artificial outer boundary changes the capacitance estimate.

It writes `visualizations/numerical_gap_sweep.csv`, which compares the 2D
solver's gap trend against the simple effective-area fringing trend. The sweep
includes `gap_to_width`, because fringing strength is best compared through
dimensionless geometry ratios.

It also writes `visualizations/adaptive_gap_sweep.csv`, which repeats that
comparison on a grid chosen from the smallest gap.

It also writes `visualizations/validation_report.md`, a compact Markdown report
of the latest numerical evidence.

It also writes `visualizations/physical_observables.csv`, which contains
system-level observables such as effective relative permittivity, effective
capacitance, stored energy, leakage current, dielectric loss power, generator
heat, heat dissipation, temperature rate, RC time constant, quality factor, and
breakdown margin.

It also writes `visualizations/detector_field_observations.csv` and
`visualizations/detector_fringing_summary.csv`. These sample a detector/photo
plate near the capacitor and report field direction, normal/tangential
components, local maxima/minima, edge-to-center contrast, and the effect of
external DC fields, EMI, humidity-scaled noise, and temperature-scaled noise.

Physical operating parameters can be changed from the CLI:

```powershell
python .\main\run_models.py --temperature-c 45 --relative-humidity 0.8 --frequency-hz 1000 --voltage 100 --mechanical-power 2 --mechanical-efficiency 0.75 --friction-coefficient 0.2 --normal-force 4 --friction-radius 0.01 --shaft-speed 200
```

To make the detector observations include outside interference:

```powershell
python .\main\run_models.py --external-field-x 100 --external-field-y -40 --emi-amplitude 75 --emi-spatial-frequency 120 --field-noise-std 20 --relative-humidity 0.8 --temperature-c 40
```

## Test

```powershell
python -m unittest discover -s tests
```

## How To Read The Numerical Result

For a credible numerical estimate, look for:

- `converged = True` from the solver.
- A small `residual_norm`, which means the discrete Laplace equation is being
  satisfied on free grid nodes.
- Grid electrode dimensions close to the requested physical dimensions.
- A fringe ratio that changes less as the grid becomes finer.
- Energy-based and charge-based capacitance estimates that move toward each
  other as the grid is refined.
- Similar results when the simulation domain is made larger.

The current thin-electrode grid model still shows a noticeable difference
between energy-based and charge-based capacitance. Treat that difference as a
numerical uncertainty indicator until a finer grid, improved electrode
treatment, or independent reference solution reduces it.
