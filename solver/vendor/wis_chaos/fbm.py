"""Positive covariance representations of standard fractional Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianPathFactor:
    """A finite path representation B(t_i) = loadings[i] @ xi."""

    times: np.ndarray
    hurst: float
    loadings: np.ndarray
    covariance: np.ndarray
    residual: np.ndarray
    method: str

    @property
    def modes(self) -> int:
        return int(self.loadings.shape[1])

    @property
    def increments(self) -> np.ndarray:
        zero = np.zeros((1, self.modes), dtype=float)
        return np.diff(np.vstack((zero, self.loadings)), axis=0)

    @property
    def covariance_reconstruction_error(self) -> float:
        scale = max(float(np.linalg.norm(self.covariance, ord="fro")), 1.0)
        return float(
            np.linalg.norm(
                self.covariance - self.loadings @ self.loadings.T - self.residual,
                ord="fro",
            )
            / scale
        )

    @property
    def residual_trace_fraction(self) -> float:
        denom = max(float(np.trace(self.covariance)), np.finfo(float).tiny)
        return max(0.0, float(np.trace(self.residual)) / denom)

    @property
    def residual_max_variance_fraction(self) -> float:
        diagonal = np.diag(self.covariance)
        mask = diagonal > np.finfo(float).tiny
        if not np.any(mask):
            return 0.0
        ratios = np.diag(self.residual)[mask] / diagonal[mask]
        return max(0.0, float(np.max(ratios)))


def _validate_times(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if values[0] <= 0.0 or np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be strictly increasing and positive")
    return values


def fbm_covariance(times: np.ndarray, hurst: float) -> np.ndarray:
    """Return R_H(t_i,t_j) for standard Type-I fractional Brownian motion."""

    values = _validate_times(times)
    if not 0.0 < hurst < 1.0:
        raise ValueError("standard fBM requires 0 < hurst < 1")
    left = values[:, None]
    right = values[None, :]
    covariance = 0.5 * (
        left ** (2.0 * hurst)
        + right ** (2.0 * hurst)
        - np.abs(left - right) ** (2.0 * hurst)
    )
    return 0.5 * (covariance + covariance.T)


def canonical_feature_interpolation_error(
    left: float,
    evaluation_time: float,
    right: float,
    hurst: float,
) -> float:
    """Return the canonical-Hilbert error of linear feature interpolation.

    If ``h_t`` is the canonical fBM feature with
    ``<h_t, h_s> = R_H(t, s)``, this evaluates

    ``||h_t - ((1-theta) h_left + theta h_right)||``.

    Only the positive covariance formula is used, so the calculation is
    valid for every ``0 < H < 1`` and when ``left == 0``.
    """

    start = float(left)
    time = float(evaluation_time)
    stop = float(right)
    exponent = float(hurst)
    if not 0.0 < exponent < 1.0:
        raise ValueError("standard fBM requires 0 < hurst < 1")
    if not 0.0 <= start <= time <= stop:
        raise ValueError(
            "times must satisfy 0 <= left <= evaluation_time <= right"
        )
    if stop <= start:
        raise ValueError("right must be strictly larger than left")

    theta = (time - start) / (stop - start)
    points = np.asarray((time, start, stop), dtype=float)
    lhs = points[:, None]
    rhs = points[None, :]
    covariance = 0.5 * (
        lhs ** (2.0 * exponent)
        + rhs ** (2.0 * exponent)
        - np.abs(lhs - rhs) ** (2.0 * exponent)
    )
    coefficients = np.asarray(
        (1.0, -(1.0 - theta), -theta),
        dtype=float,
    )
    squared = float(coefficients @ covariance @ coefficients)
    scale = max(float(np.max(np.diag(covariance))), 1.0)
    if squared < -100.0 * np.finfo(float).eps * scale:
        raise FloatingPointError(
            "canonical interpolation variance became negative"
        )
    return float(np.sqrt(max(squared, 0.0)))


def _ordered_psd_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -2.0e-11 * scale:
        raise np.linalg.LinAlgError(
            f"covariance is not PSD: min eigenvalue={np.min(eigenvalues):.3e}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def path_kl_factor(
    times: np.ndarray, hurst: float, modes: int | None = None
) -> GaussianPathFactor:
    """Optimal discrete path-covariance KL factor in Frobenius/trace energy."""

    values = _validate_times(times)
    covariance = fbm_covariance(values, hurst)
    eigenvalues, eigenvectors = _ordered_psd_eigensystem(covariance)
    maximum = values.size
    retained = maximum if modes is None else int(modes)
    if not 1 <= retained <= maximum:
        raise ValueError(f"modes must lie in [1, {maximum}]")
    loadings = eigenvectors[:, :retained] * np.sqrt(eigenvalues[:retained])
    residual = 0.5 * (
        covariance - loadings @ loadings.T
        + (covariance - loadings @ loadings.T).T
    )
    return GaussianPathFactor(
        values, hurst, loadings, covariance, residual, "path_kl"
    )


def endpoint_conditioned_factor(
    times: np.ndarray, hurst: float, modes: int
) -> GaussianPathFactor:
    """Represent B_T exactly, followed by KL modes of the fBM bridge.

    The first coordinate is xi_0 = B_T / T**H.  This is a goal-conditioned
    representation for terminal outputs; it is not claimed to be an optimal
    full-path approximation.
    """

    values = _validate_times(times)
    retained = int(modes)
    if not 1 <= retained <= values.size:
        raise ValueError(f"modes must lie in [1, {values.size}]")
    covariance = fbm_covariance(values, hurst)
    terminal_sd = values[-1] ** hurst
    terminal_loading = covariance[:, -1] / terminal_sd
    bridge = covariance - np.outer(terminal_loading, terminal_loading)
    bridge = 0.5 * (bridge + bridge.T)
    eigenvalues, eigenvectors = _ordered_psd_eigensystem(bridge)
    bridge_modes = retained - 1
    if bridge_modes:
        bridge_factor = (
            eigenvectors[:, :bridge_modes] * np.sqrt(eigenvalues[:bridge_modes])
        )
        loadings = np.column_stack((terminal_loading, bridge_factor))
    else:
        loadings = terminal_loading[:, None]
    residual = 0.5 * (
        covariance - loadings @ loadings.T
        + (covariance - loadings @ loadings.T).T
    )
    return GaussianPathFactor(
        values,
        hurst,
        loadings,
        covariance,
        residual,
        "endpoint_conditioned_bridge_kl",
    )


def multi_anchor_conditioned_factor(
    times: np.ndarray,
    hurst: float,
    anchor_indices: tuple[int, ...],
    modes: int,
) -> GaussianPathFactor:
    """Represent selected path values exactly, then KL-expand the bridge.

    Anchor order is meaningful.  Passing the terminal index first makes the
    first independent coordinate equal to the normalized terminal value.
    """

    values = _validate_times(times)
    anchors = tuple(int(index) for index in anchor_indices)
    if not anchors:
        raise ValueError("anchor_indices must be nonempty")
    if len(set(anchors)) != len(anchors):
        raise ValueError("anchor_indices must be unique")
    if any(index < 0 or index >= values.size for index in anchors):
        raise ValueError("anchor index is outside the time grid")
    retained = int(modes)
    if not len(anchors) <= retained <= values.size:
        raise ValueError(
            "modes must lie between anchor count and grid size"
        )

    covariance = fbm_covariance(values, hurst)
    anchor_covariance = covariance[np.ix_(anchors, anchors)]
    anchor_cholesky = np.linalg.cholesky(anchor_covariance)
    anchor_loadings = np.linalg.solve(
        anchor_cholesky,
        covariance[:, anchors].T,
    ).T
    bridge = covariance - anchor_loadings @ anchor_loadings.T
    bridge = 0.5 * (bridge + bridge.T)
    eigenvalues, eigenvectors = _ordered_psd_eigensystem(bridge)
    bridge_modes = retained - len(anchors)
    if bridge_modes:
        positive = eigenvalues > (
            100.0
            * np.finfo(float).eps
            * max(float(eigenvalues[0]), 1.0)
        )
        available = int(np.sum(positive))
        if bridge_modes > available:
            raise np.linalg.LinAlgError(
                "requested more bridge modes than numerical bridge rank"
            )
        bridge_factor = (
            eigenvectors[:, :bridge_modes]
            * np.sqrt(eigenvalues[:bridge_modes])
        )
        loadings = np.column_stack(
            (anchor_loadings, bridge_factor)
        )
    else:
        loadings = anchor_loadings
    residual = covariance - loadings @ loadings.T
    residual = 0.5 * (residual + residual.T)
    anchor_label = "_".join(str(index) for index in anchors)
    return GaussianPathFactor(
        values,
        hurst,
        loadings,
        covariance,
        residual,
        f"multi_anchor_{anchor_label}_conditioned_bridge_kl",
    )


def pivoted_cholesky_factor(
    times: np.ndarray,
    hurst: float,
    modes: int | None = None,
) -> GaussianPathFactor:
    """Greedy maximum-residual-variance factor of the path covariance."""

    values = _validate_times(times)
    covariance = fbm_covariance(values, hurst)
    maximum = values.size
    retained = maximum if modes is None else int(modes)
    if not 1 <= retained <= maximum:
        raise ValueError(f"modes must lie in [1, {maximum}]")

    loadings = np.zeros((maximum, retained), dtype=float)
    residual_diagonal = np.diag(covariance).copy()
    scale = max(float(np.max(residual_diagonal)), 1.0)
    for column in range(retained):
        pivot = int(np.argmax(residual_diagonal))
        pivot_variance = float(residual_diagonal[pivot])
        if pivot_variance <= 100.0 * np.finfo(float).eps * scale:
            raise np.linalg.LinAlgError(
                "pivoted Cholesky exhausted numerical covariance rank"
            )
        if column:
            correction = loadings[:, :column] @ loadings[pivot, :column]
        else:
            correction = np.zeros(maximum, dtype=float)
        new_column = (
            covariance[:, pivot] - correction
        ) / np.sqrt(pivot_variance)
        loadings[:, column] = new_column
        residual_diagonal -= new_column * new_column
        residual_diagonal = np.maximum(residual_diagonal, 0.0)

    residual = covariance - loadings @ loadings.T
    residual = 0.5 * (residual + residual.T)
    return GaussianPathFactor(
        values,
        hurst,
        loadings,
        covariance,
        residual,
        "pivoted_cholesky_max_residual_variance",
    )


def rotate_factor(
    factor: GaussianPathFactor,
    rotation: np.ndarray,
    *,
    method: str,
) -> GaussianPathFactor:
    """Orthogonally rotate retained independent Gaussian coordinates."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (factor.modes, factor.modes):
        raise ValueError("rotation has incompatible shape")
    error = float(
        np.linalg.norm(
            matrix.T @ matrix - np.eye(factor.modes),
            ord=2,
        )
    )
    if error > 2.0e-10:
        raise ValueError(
            f"rotation must be orthogonal; error={error:.3e}"
        )
    return GaussianPathFactor(
        times=factor.times,
        hurst=factor.hurst,
        loadings=factor.loadings @ matrix,
        covariance=factor.covariance,
        residual=factor.residual,
        method=method,
    )
