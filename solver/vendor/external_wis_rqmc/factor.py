"""Independent nested full-covariance factors for standard fBM.

At level ``L`` the positive grid is ``j*T/2**L``.  The covariance is
factorized after ordering the points hierarchically: ``T`` first, followed by
the newly inserted odd dyadic points at every refinement level.  Consequently
the level-``L`` Gaussian coordinates are a prefix of the level-``L+1``
coordinates, while every grid covariance is represented to Cholesky accuracy.

No deterministic-chaos factor, basis, or coefficient is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def _validate_hurst(hurst: float) -> float:
    value = float(hurst)
    if not 0.0 < value < 1.0:
        raise ValueError("standard fBM requires 0 < hurst < 1")
    return value


def fbm_covariance(
    left: np.ndarray | float,
    right: np.ndarray | float,
    hurst: float,
) -> np.ndarray:
    """Return the standard fBM covariance ``R_H(left, right)``."""

    exponent = _validate_hurst(hurst)
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if np.any(lhs < 0.0) or np.any(rhs < 0.0):
        raise ValueError("fBM covariance times must be nonnegative")
    return 0.5 * (
        lhs ** (2.0 * exponent)
        + rhs ** (2.0 * exponent)
        - np.abs(lhs - rhs) ** (2.0 * exponent)
    )


def hierarchical_dyadic_times(level: int, horizon: float = 1.0) -> np.ndarray:
    """Return dyadic positive times in a prefix-preserving hierarchy."""

    selected_level = int(level)
    final_time = float(horizon)
    if selected_level < 0:
        raise ValueError("level must be nonnegative")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("horizon must be finite and positive")
    values = [final_time]
    for refinement in range(1, selected_level + 1):
        denominator = 2**refinement
        values.extend(
            final_time * odd / denominator
            for odd in range(1, denominator, 2)
        )
    return np.asarray(values, dtype=float)


def _readonly(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class NestedFbmFactor:
    """A finite Gaussian feature ``B_L(t)=g_L(t)^T xi``."""

    hurst: float
    level: int
    horizon: float
    times: np.ndarray
    loadings: np.ndarray
    hierarchical_times: np.ndarray
    hierarchical_cholesky: np.ndarray
    reconstruction_relative_error: float
    fingerprint: str

    @property
    def modes(self) -> int:
        return int(self.loadings.shape[1])

    @property
    def mesh(self) -> float:
        return float(self.horizon / self.modes)

    def loadings_at(self, evaluation_times: np.ndarray | float) -> np.ndarray:
        """Piecewise-linearly interpolate the finite Gaussian feature."""

        values = np.asarray(evaluation_times, dtype=float)
        scalar = values.ndim == 0
        flat = values.reshape(-1)
        if np.any(~np.isfinite(flat)):
            raise ValueError("evaluation times must be finite")
        tolerance = 32.0 * np.finfo(float).eps * max(self.horizon, 1.0)
        if np.any(flat < -tolerance) or np.any(flat > self.horizon + tolerance):
            raise ValueError("evaluation time is outside [0, horizon]")
        clipped = np.clip(flat, 0.0, self.horizon)
        extended_times = np.concatenate(([0.0], self.times))
        extended_loadings = np.vstack(
            (np.zeros((1, self.modes), dtype=float), self.loadings)
        )
        upper = np.searchsorted(extended_times, clipped, side="left")
        upper = np.clip(upper, 1, self.times.size)
        left_index = upper - 1
        left_time = extended_times[left_index]
        right_time = extended_times[upper]
        fraction = (clipped - left_time) / (right_time - left_time)
        interpolated = (
            (1.0 - fraction)[:, None] * extended_loadings[left_index]
            + fraction[:, None] * extended_loadings[upper]
        )
        zero = clipped == 0.0
        if np.any(zero):
            interpolated[zero] = 0.0
        if scalar:
            return interpolated[0]
        return interpolated.reshape(values.shape + (self.modes,))

    def covariance_at(self, left: float, right: float) -> float:
        """Return the finite-factor covariance ``C_L(left, right)``."""

        lhs = self.loadings_at(float(left))
        rhs = self.loadings_at(float(right))
        return float(np.dot(lhs, rhs))


def nested_fbm_factor(
    hurst: float,
    level: int,
    *,
    horizon: float = 1.0,
) -> NestedFbmFactor:
    """Construct a deterministic nested Cholesky factor of fBM covariance."""

    exponent = _validate_hurst(hurst)
    selected_level = int(level)
    final_time = float(horizon)
    hierarchical = hierarchical_dyadic_times(selected_level, final_time)
    covariance = fbm_covariance(
        hierarchical[:, None],
        hierarchical[None, :],
        exponent,
    )
    covariance = 0.5 * (covariance + covariance.T)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "full fBM covariance Cholesky failed; no jitter is permitted"
        ) from error

    chronological_order = np.argsort(hierarchical, kind="stable")
    times = hierarchical[chronological_order]
    loadings = cholesky[chronological_order]
    chronological_covariance = fbm_covariance(
        times[:, None], times[None, :], exponent
    )
    residual = chronological_covariance - loadings @ loadings.T
    scale = max(float(np.linalg.norm(chronological_covariance, ord="fro")), 1.0)
    relative_error = float(np.linalg.norm(residual, ord="fro") / scale)
    if relative_error > 2.0e-12:
        raise FloatingPointError(
            "full-covariance factor reconstruction exceeded 2e-12"
        )

    digest = hashlib.sha256()
    metadata = {
        "method": "hierarchical_dyadic_full_covariance_cholesky_v1",
        "hurst": exponent,
        "level": selected_level,
        "horizon": final_time,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(np.asarray(times, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(loadings, dtype="<f8").tobytes(order="C"))

    return NestedFbmFactor(
        hurst=exponent,
        level=selected_level,
        horizon=final_time,
        times=_readonly(times),
        loadings=_readonly(loadings),
        hierarchical_times=_readonly(hierarchical),
        hierarchical_cholesky=_readonly(cholesky),
        reconstruction_relative_error=relative_error,
        fingerprint=digest.hexdigest(),
    )


__all__ = [
    "NestedFbmFactor",
    "fbm_covariance",
    "hierarchical_dyadic_times",
    "nested_fbm_factor",
]
