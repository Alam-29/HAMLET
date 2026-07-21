# Hamiltonian Metric Learning and Energy-Based Training

This project is a research workspace for a single, generalized energy-based
optimizer that aims to unify the common gradient-descent family (SGD,
momentum/Heavy Ball, Nesterov, Adam, AdamW, ...) as special cases of one
Hamiltonian dynamic, rather than treating each as a separate hand-tuned
update rule:

```
p_{t+1}     = beta p_t - eta [grad L + F_geo + F_mem + alpha grad S(g)]
theta_{t+1} = theta_t + eta g^-1(theta) p_{t+1}
```

`theta` and `p` are treated as Hamiltonian position/momentum coordinates, so
each step conserves and dissipates energy the way a physical system would
rather than following a plain gradient. Setting the metric `g` to the
identity and turning off `F_geo`, `F_mem`, and the spectral term recovers
plain (momentum) SGD; a diagonal, curvature-adaptive `g` moves the family
toward Adam-style preconditioning. In general `g(theta)` is a *learned*
parameter-space metric that rescales and reorients each step (metric
learning); `F_geo` is a curvature correction derived from that metric,
`F_mem` is a decaying memory force over past updates, and `alpha grad S(g)`
couples training to the spectral entropy of the metric so the optimizer
favors better-conditioned regions of parameter space (energy-based
training). The reusable core is implemented in `src/hamiltonian_geometric.py`
as plain NumPy, accepting callables for the loss gradient and the metric so
it works with fixed features today and a theta-dependent metric (via
finite-difference `F_geo` and `grad S(g)`) for full-network experiments
later.

Because the optimizer is meant to generalize, it is validated against
plain SGD, Adam, and AlgoPerf-style baselines on more than one workload:

- A physics-informed neural network (PINN), `main/run_pinn_benchmark.py`,
  trained on a finite-plate capacitor fringing-field problem. This is one
  physical benchmark among several, not the subject of the project — it
  exists to give the optimizer a nonconvex problem with a known, checkable
  physical baseline.
- A DeepOBS-style nonconvex MLP classification task,
  `main/run_deepobs_style_benchmark.py`.
- AlgoPerf-style baseline families (`AdamW`, `Nesterov Momentum`, `Heavy Ball
  Momentum`) with a local fixed tuning budget,
  `main/run_algoperf_style_benchmark.py`.

The physics assumptions behind the capacitor benchmark problem are kept
visible for anyone extending or auditing that particular workload:

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

The PINN model itself lives in `src/pinn.py`. Because this benchmark uses
fixed nonlinear features, its metric `g` is exact and constant; the reusable
optimizer core still supports theta-dependent metrics for the full-network
experiments described above.

## Suggested Roadmap

1. Build the analytical baseline for the capacitor benchmark problem.
2. Add a 2D numerical Laplace solver for a capacitor cross-section.
3. Estimate capacitance from stored electrostatic energy.
4. Compare numerical results against the analytical baseline.
5. Add parameter sweeps and visualizations.
6. Extend the Hamiltonian-geometric optimizer to a theta-dependent metric and
   a full-network (non-fixed-feature) benchmark.

For more detail on the equations, assumptions, and limitations, read
`docs/model_notes.md`. For the mathematical consistency review of the PDF's
Hamiltonian-geometric optimizer, read `docs/mathematical_validation.md`.

## Run

### Reproduce the submission evidence

The authoritative audit-grade results can be verified without rerunning the
long experiments. The command also refreshes the SHA-256 manifest and rebuilds
the privacy-checked release archive, preventing those deliverables from
drifting behind the checked evidence. ZIP member timestamps and permissions
are normalized, so identical inputs produce the same archive checksum:

```powershell
.\scripts\reproduce_submission_evidence.ps1 -Mode verify
```

To regenerate the full factorial ablation (including CUDA), paired classical
and PINN replications, equal-budget held-out tuning, modern-optimizer
robustness, the synchronized three-seed WikiText-2 compute audit, theorem
checks, runtime-normalized evidence, tests, and SHA-256 artifact manifest:

```powershell
.\scripts\reproduce_submission_evidence.ps1 -Mode full
```

Use Python 3.11 with `requirements-repro.txt`. The full run takes several
hours and requires CUDA. `results/submission_artifact_manifest.json` identifies
the authoritative artifacts; older unpaired `classical_multiseed_*` files are
retained for audit history but are superseded and must not be cited.

The separate official DeepOBS `mnist_mlp` comparison uses DeepOBS's own
`StandardRunner` and data pipeline. The checked-in result is a 20-epoch,
single-seed scope check, not a broad or statistically conclusive benchmark:

```powershell
python .\main\run_official_deepobs_benchmark.py --epochs 20 --seed 42
```

Its downloaded MNIST data are intentionally ignored; runner JSON, tuning
notes, and the consolidated CSV remain under `results/official_deepobs/`.

An official-API AlgoPerf submission adapter and execution runbook are retained
under `algoperf_submissions/` and `docs/algoperf_official_runbook.md`. Current
AlgoPerf smoke runs validate integration only; they contain no completed,
matched evaluation and are not manuscript evidence.

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

The small GPT-2 benchmark runners use PyTorch and can run on CUDA. By default
they use `--device auto`, which selects CUDA when PyTorch can see a GPU and
falls back to CPU otherwise. To require CUDA and fail fast if the installed
PyTorch build cannot access it:

```powershell
python .\main\run_llm_benchmark.py --device cuda
python .\main\run_industry_llm_benchmark.py --device cuda
```

You can also target a specific GPU, for example `--device cuda:0`, or force a
CPU run with `--device cpu`. The NumPy-only benchmarks remain CPU benchmarks.

To benchmark the optimizer on a chaotic quantum-control workload, run the
quantum kicked-top state-transfer comparison. This compares SGD, AdamW,
Nesterov Momentum, Heavy Ball Momentum, entropy descent, and the
Hamiltonian-geometric optimizer:

```powershell
python .\main\run_quantum_chaos_benchmark.py
```

To see the optimizer's theta trajectory as a 3D PCA phase-space plot (theta
lives in a 64-dimensional space by default; this projects every optimizer's
recorded path onto the top 3 shared principal components so their routes
through parameter space can be compared directly):

```powershell
python .\main\run_phase_space_visualization.py
```

This writes `visualizations\phase_space\phase_space_trajectories.csv` (PCA
coordinates and loss per step per optimizer), `phase_space.png` (a static 3D
view), `phase_space_rotation.gif` (a rotating 3D view), and `phase_space.html`
(a self-contained, dependency-free interactive view -- drag to rotate).

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
