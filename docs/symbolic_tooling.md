# Symbolic Tooling

This project uses SymPy directly from Python and supports Cadabra as an
optional external symbolic engine.

## SymPy

SymPy is installed through `requirements.txt` and is used by:

```text
src/symbolic_checks.py
main/run_symbolic_checks.py
```

Run:

```powershell
.\.venv\Scripts\python.exe -u main\run_symbolic_checks.py
```

Outputs:

```text
visualizations/symbolic_checks/symbolic_checks.json
visualizations/symbolic_checks/symbolic_checks.md
```

## Cadabra

Cadabra is not available as a normal `pip install cadabra2` package in this
Windows Python environment. Install the standalone Windows package from the
official Cadabra download page, then make sure one of these commands is on
`PATH`:

```text
cadabra2-cli
cadabra2
cadabra2-gtk
```

The project Cadabra example lives at:

```text
cadabra/maxwell_field_symmetry.cdb
```

Once Cadabra is installed, `main/run_symbolic_checks.py` will automatically
run that script and include Cadabra output in the symbolic report.
