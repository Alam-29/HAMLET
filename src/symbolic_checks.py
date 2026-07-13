"""Symbolic checks for the capacitor and optimizer equations.

SymPy is used for executable algebra inside the Python test suite. Cadabra is
supported as an optional external tool through the scripts in `cadabra/`; the
runner detects `cadabra2-cli`/`cadabra2` when the standalone Cadabra installer
is available on the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from pathlib import Path

import sympy as sp


@dataclass(frozen=True)
class SymbolicCheck:
    name: str
    expression: str
    simplified: str
    passed: bool


@dataclass(frozen=True)
class CadabraRun:
    available: bool
    executable: str | None
    returncode: int | None
    stdout: str
    stderr: str


def capacitor_energy_identity() -> SymbolicCheck:
    eps, length, width, gap, voltage = sp.symbols(
        "epsilon L W d V", positive=True, finite=True
    )
    capacitance = eps * length * width / gap
    electric_field = voltage / gap
    volume = length * width * gap
    energy_from_field = sp.Rational(1, 2) * eps * electric_field**2 * volume
    energy_from_capacitance = sp.Rational(1, 2) * capacitance * voltage**2
    residual = sp.simplify(energy_from_field - energy_from_capacitance)
    return SymbolicCheck(
        name="parallel_plate_energy_identity",
        expression=str(energy_from_field - energy_from_capacitance),
        simplified=str(residual),
        passed=residual == 0,
    )


def capacitor_force_identity() -> SymbolicCheck:
    eps, length, width, gap, voltage = sp.symbols(
        "epsilon L W d V", positive=True, finite=True
    )
    capacitance = eps * length * width / gap
    stored_energy = sp.Rational(1, 2) * capacitance * voltage**2
    attractive_force_magnitude = -sp.diff(stored_energy, gap)
    expected = eps * length * width * voltage**2 / (2 * gap**2)
    residual = sp.simplify(attractive_force_magnitude - expected)
    return SymbolicCheck(
        name="parallel_plate_force_identity",
        expression=str(attractive_force_magnitude),
        simplified=str(residual),
        passed=residual == 0,
    )


def hamiltonian_metric_positive_identity() -> SymbolicCheck:
    beta2, epsilon, grad = sp.symbols("beta_2 epsilon g", positive=True, finite=True)
    beta2 = sp.Symbol("beta_2", positive=True)
    metric = (1 - beta2) * grad**2
    preconditioner = 1 / (sp.sqrt(metric / (1 - beta2)) + epsilon)
    expected = 1 / (sp.Abs(grad) + epsilon)
    residual = sp.simplify(preconditioner - expected)
    return SymbolicCheck(
        name="one_step_adam_metric_limit",
        expression=str(preconditioner),
        simplified=str(residual),
        passed=residual == 0,
    )


def run_sympy_checks() -> list[SymbolicCheck]:
    return [
        capacitor_energy_identity(),
        capacitor_force_identity(),
        hamiltonian_metric_positive_identity(),
    ]


def find_cadabra_executable() -> str | None:
    for name in ("cadabra2-cli", "cadabra2", "cadabra2-gtk"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run_cadabra_script(script_path: str | Path, timeout_s: int = 30) -> CadabraRun:
    executable = find_cadabra_executable()
    if executable is None:
        return CadabraRun(
            available=False,
            executable=None,
            returncode=None,
            stdout="",
            stderr="Cadabra executable not found on PATH.",
        )
    completed = subprocess.run(
        [executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return CadabraRun(
        available=True,
        executable=executable,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
