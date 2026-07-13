# Compiling this report

No LaTeX toolchain (pdflatex/xelatex/tectonic/MiKTeX) is installed on the machine this
was written on, so `main.tex` has not been compiled here. Every figure path, table
column count, brace, and `\begin`/`\end` pair has been checked by hand/script, but a
first real compile may still surface something a static check can't catch.

## Fastest option: Overleaf

1. Go to https://overleaf.com and create a new project ("Upload Project").
2. Zip this `report/` folder (`main.tex` + `figures/`) and upload it.
3. Set the main document to `main.tex` and click Recompile.

## Local option

Requires a TeX distribution (MiKTeX on Windows, TeX Live on Linux/Mac, or the
lightweight `tectonic` engine).

```powershell
cd report
pdflatex main.tex
pdflatex main.tex   # run twice so the table of contents and references resolve
```

or, with tectonic (no separate package installation needed):

```powershell
cd report
tectonic main.tex
```

## Contents

- `main.tex` -- the report source.
- `figures/` -- all 13 figures referenced by the report, copied from `visualizations/`
  at the time this report was written. Regenerate any of them from the project root with:
  - `python main/run_models.py` (2D capacitor potential/fringing field)
  - `python main/run_pinn_benchmark.py` (5-way PINN optimizer benchmark)
  - `python main/analyze_normal_modes.py` (normal-mode/action-angle spectrum)
  - `python main/run_3d_capacitor_solve.py` (3D Mathematica capacitor solve)
  - `.venv_algoperf/Scripts/python.exe main/run_mnist_benchmark.py --epochs 5 --train-samples 10000 --val-samples 2000 --data-source streaming`
  - `.venv_algoperf/Scripts/python.exe main/run_cifar10_benchmark.py --epochs 5 --train-samples 10000 --val-samples 2000`
  - The official AlgoPerf figures (`algoperf_*`) come from
    `visualizations/official_algoperf_runs/`, produced via the official harness --
    see `docs/algoperf_official_runbook.md`.
