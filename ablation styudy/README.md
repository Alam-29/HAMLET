# Extended Ablation Study

This directory is a reproducible, journal-oriented ablation package for the
Hamiltonian-geometric optimizer.  It deliberately separates three kinds of
evidence:

1. **Full-architecture factorial:** the exact NumPy implementation is tested
   on a non-convex quartic with a genuinely parameter-dependent metric.  A
   full `2^3` design crosses geometric, memory, and spectral corrections under
   both a curvature-derived Hessian metric and an unrelated positive-definite
   control metric.
2. **Chaotic quantum replication:** the same `2^3` component design is run on
   the kicked-top control problem with paired seeds, alongside AdamW, heavy
   ball, and entropy-descent references.
3. **CUDA scaling ablation:** the repository's documented diagonal-metric
   Torch reduction is compared with its metric and momentum components
   switched on/off, plus SGD and AdamW.  This is explicitly *not* described as
   a full-matrix geometric-force experiment.

All trials retain raw per-seed observations.  Summaries contain medians,
bootstrap 95% confidence intervals, divergence counts, paired effect sizes,
factorial OLS effects, corrected tests, runtime, and (for CUDA) peak GPU
memory.

## Run

From the repository root:

```powershell
.\.venv-ablation\Scripts\python.exe ".\ablation styudy\run_ablation_study.py" --mode full
```

For a quick validation:

```powershell
.\.venv-ablation\Scripts\python.exe ".\ablation styudy\run_ablation_study.py" --mode smoke
```

If CUDA PyTorch is unavailable, the script completes the exact NumPy studies,
records the GPU study as unavailable, and still produces the supplement.  Use
`--require-cuda` to fail instead.  Output is written below `results/` and
`figures/`; `supplementary_ablation.tex` is regenerated from measured values.
Long sections can be run independently with `--sections architecture`,
`--sections quantum`, or `--sections gpu`; omitted sections reuse their
existing raw CSV when summaries and the supplement are regenerated.

## Design sizes

| mode | architecture | quantum | CUDA neural |
|---|---:|---:|---:|
| smoke | 3 seeds, 40 steps | 2 seeds, 12 iterations | 2 seeds, 40 steps |
| full | 40 seeds, 300 steps | 20 seeds, 80 iterations | 10 seeds, 600 steps |

The full configuration is intentionally substantial but remains realistic for
a 4 GB GTX 1050.  Increase individual sizes with command-line flags only after
checking available runtime and memory.
