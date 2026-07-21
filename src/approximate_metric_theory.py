"""Executable checks for the constant approximate-metric quadratic theorem."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def modal_update_matrix(relative_eigenvalue: float, learning_rate: float, momentum: float) -> np.ndarray:
    """Return the two-state update block for one generalized eigenmode."""
    lam = float(relative_eigenvalue)
    eta = float(learning_rate)
    beta = float(momentum)
    return np.array([[1.0 - eta * eta * lam, eta * beta], [-eta * lam, beta]])


def modal_spectral_radius(relative_eigenvalue: float, learning_rate: float, momentum: float) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(modal_update_matrix(relative_eigenvalue, learning_rate, momentum)))))


def approximate_metric_stability_bound(delta: float, learning_rate: float, momentum: float) -> bool:
    """The theorem's strict sufficient-and-necessary interval condition."""
    if not (0.0 <= delta < 1.0 and 0.0 <= momentum < 1.0 and learning_rate > 0.0):
        return False
    return learning_rate**2 * (1.0 + delta) < 2.0 * (1.0 + momentum)


def verify_relative_spectrum(
    relative_eigenvalues: Iterable[float], delta: float, learning_rate: float, momentum: float
) -> dict[str, float | bool]:
    values = np.asarray(tuple(relative_eigenvalues), dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0.0):
        raise ValueError("relative_eigenvalues must be a nonempty positive sequence")
    radii = np.array([modal_spectral_radius(value, learning_rate, momentum) for value in values])
    interval_ok = bool(np.all(values >= 1.0 - delta) and np.all(values <= 1.0 + delta))
    theorem_condition = approximate_metric_stability_bound(delta, learning_rate, momentum)
    return {
        "delta": float(delta),
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
        "min_relative_eigenvalue": float(values.min()),
        "max_relative_eigenvalue": float(values.max()),
        "max_spectral_radius": float(radii.max()),
        "interval_ok": interval_ok,
        "theorem_condition": theorem_condition,
        "empirically_stable": bool(radii.max() < 1.0),
        "complex_mode_radius": math.sqrt(momentum),
    }
