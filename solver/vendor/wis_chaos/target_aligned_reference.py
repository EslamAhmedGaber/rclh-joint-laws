"""Target-aligned hybrid quadrature for full-curve Gjessing references.

At each requested time, an orthogonal Householder map aligns the first
standard Gaussian coordinate with that target's loading vector.  Gauss--
Hermite then resolves the exact active ``B_t`` direction, while scrambled
Sobol integrates its orthogonal bridge.  This reduces intermediate-time
variance compared with integrating only the terminal ``B_T`` coordinate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .fbm import GaussianPathFactor
from .reference import translated_grid_target_rk4_samples


ArrayFunction = Callable[[float, np.ndarray], np.ndarray]


def target_loading_rotation(loading: np.ndarray) -> np.ndarray:
    """Return orthogonal ``Q`` whose first column is ``loading/||loading||``."""

    vector = np.asarray(loading, dtype=float)
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError("loading must be a nonempty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("loading must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("loading must have positive norm")
    direction = vector / norm
    first = np.zeros(vector.size, dtype=float)
    first[0] = 1.0
    difference = first - direction
    squared = float(np.dot(difference, difference))
    if squared <= 64.0 * np.finfo(float).eps:
        return np.eye(vector.size)
    rotation = np.eye(vector.size) - 2.0 * np.outer(
        difference,
        difference,
    ) / squared
    if float(np.dot(rotation[:, 0], direction)) < 1.0 - 2.0e-13:
        raise FloatingPointError("Householder target alignment failed")
    return rotation


def target_aligned_hybrid_moment_curves(
    factor: GaussianPathFactor,
    initial_value: float,
    *,
    linear_drift_rate: float,
    beta: float,
    nonlinear_drift: ArrayFunction,
    core_order: int,
    tail_exponent: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One scrambled hybrid replicate of target-correct moment curves."""

    order = int(core_order)
    exponent = int(tail_exponent)
    if order < 1 or exponent < 0:
        raise ValueError("core_order must be positive and tail_exponent nonnegative")
    core_nodes, core_weights = np.polynomial.hermite_e.hermegauss(order)
    core_weights = core_weights / np.sqrt(2.0 * np.pi)
    tail_count = 2**exponent
    if factor.modes == 1:
        tail = np.empty((1, 0), dtype=float)
        tail_count = 1
    else:
        engine = qmc.Sobol(
            d=factor.modes - 1,
            scramble=True,
            seed=int(seed),
        )
        tiny = np.finfo(float).eps
        tail = ndtri(
            np.clip(
                engine.random_base2(exponent),
                tiny,
                1.0 - tiny,
            )
        )

    eta = np.column_stack(
        (
            np.tile(core_nodes, tail_count),
            np.repeat(tail, order, axis=0),
        )
    )
    weights = np.tile(core_weights, tail_count) / tail_count
    means = np.empty(factor.times.size + 1, dtype=float)
    seconds = np.empty_like(means)
    means[0] = float(initial_value)
    seconds[0] = float(initial_value) ** 2

    for target_index, loading in enumerate(factor.loadings):
        rotation = target_loading_rotation(loading)
        nodes = eta @ rotation.T
        samples = translated_grid_target_rk4_samples(
            factor,
            nodes,
            initial_value,
            target_index=target_index,
            linear_drift_rate=linear_drift_rate,
            beta=beta,
            nonlinear_drift=nonlinear_drift,
        )
        means[target_index + 1] = float(np.dot(weights, samples))
        seconds[target_index + 1] = float(
            np.dot(weights, samples * samples)
        )
    return means, seconds


__all__ = [
    "target_aligned_hybrid_moment_curves",
    "target_loading_rotation",
]
