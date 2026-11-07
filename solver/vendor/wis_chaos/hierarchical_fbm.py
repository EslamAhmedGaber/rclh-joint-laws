"""Endpoint-first hierarchical Gaussian factors on dyadic fBM grids.

Independent path-KL factorizations rotate their Gaussian coordinates whenever
the time grid changes.  Sparse Hermite supports are not rotation invariant, so
that construction cannot support coefficient-level Cauchy comparisons across
levels.  This module orders canonical fBM features hierarchically and applies
one Gram--Schmidt/Cholesky factorization.  Every coarse coordinate and loading
is then preserved exactly as the dyadic grid is refined.
"""

from __future__ import annotations

import numpy as np

from .fbm import GaussianPathFactor, fbm_covariance


def _positive_power_of_two(value: int, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer")
    integer = int(value)
    if integer < 1 or integer & (integer - 1):
        raise ValueError(f"{label} must be a positive power of two")
    return integer


def dyadic_endpoint_first_order(steps: int) -> np.ndarray:
    """Return natural-grid row indices in endpoint-first hierarchical order."""

    count = _positive_power_of_two(steps, "steps")
    order = [count - 1]
    level_count = int(np.log2(count))
    for level in range(1, level_count + 1):
        denominator = 1 << level
        stride = count // denominator
        for numerator in range(1, denominator, 2):
            order.append(numerator * stride - 1)
    result = np.asarray(order, dtype=np.int64)
    if result.size != count or set(result.tolist()) != set(range(count)):
        raise RuntimeError("dyadic hierarchy did not enumerate the grid")
    return result


def endpoint_hierarchical_dyadic_factor(
    steps: int,
    hurst: float,
    *,
    modes: int | None = None,
) -> GaussianPathFactor:
    """Build a nested endpoint-first factor on ``i/steps``, ``i=1..steps``.

    With all modes retained the factor reproduces the discrete fBM covariance.
    Prefix columns are orthogonal projections onto successively refined
    canonical-feature spaces.  The first column is always the normalized
    terminal fBM value.
    """

    count = _positive_power_of_two(steps, "steps")
    if modes is not None and (
        isinstance(modes, (bool, np.bool_))
        or not isinstance(modes, (int, np.integer))
    ):
        raise TypeError("modes must be an integer")
    retained = count if modes is None else int(modes)
    if not 1 <= retained <= count:
        raise ValueError(f"modes must lie in [1, {count}]")

    times = np.linspace(1.0 / count, 1.0, count)
    covariance = fbm_covariance(times, float(hurst))
    order = dyadic_endpoint_first_order(count)
    ordered_covariance = covariance[np.ix_(order, order)]
    ordered_loadings = np.linalg.cholesky(ordered_covariance)
    full_loadings = np.empty_like(ordered_loadings)
    full_loadings[order] = ordered_loadings
    loadings = full_loadings[:, :retained].copy()
    residual = covariance - loadings @ loadings.T
    residual = 0.5 * (residual + residual.T)
    return GaussianPathFactor(
        times=times,
        hurst=float(hurst),
        loadings=loadings,
        covariance=covariance,
        residual=residual,
        method="endpoint_first_hierarchical_dyadic_cholesky",
    )


def hierarchical_dyadic_subfactor(
    fine_factor: GaussianPathFactor,
    steps: int,
    *,
    modes: int | None = None,
) -> GaussianPathFactor:
    """Restrict a hierarchical fine factor to a nested dyadic subgrid/prefix."""

    if not isinstance(fine_factor, GaussianPathFactor):
        raise TypeError("fine_factor must be a GaussianPathFactor")
    fine_steps = _positive_power_of_two(
        int(fine_factor.times.size), "fine grid size"
    )
    coarse_steps = _positive_power_of_two(steps, "steps")
    if coarse_steps > fine_steps or fine_steps % coarse_steps:
        raise ValueError("steps must define a nested subgrid of the fine grid")
    if modes is not None and (
        isinstance(modes, (bool, np.bool_))
        or not isinstance(modes, (int, np.integer))
    ):
        raise TypeError("modes must be an integer")
    retained = fine_factor.modes if modes is None else int(modes)
    if not 1 <= retained <= fine_factor.modes:
        raise ValueError(
            f"modes must lie in [1, {fine_factor.modes}]"
        )
    stride = fine_steps // coarse_steps
    indices = np.arange(stride - 1, fine_steps, stride, dtype=np.int64)
    times = fine_factor.times[indices].copy()
    expected_times = np.linspace(1.0 / coarse_steps, 1.0, coarse_steps)
    if not np.allclose(times, expected_times, rtol=0.0, atol=2.0e-15):
        raise ValueError("fine factor does not use the expected unit dyadic grid")
    loadings = fine_factor.loadings[indices, :retained].copy()
    covariance = fbm_covariance(times, fine_factor.hurst)
    residual = covariance - loadings @ loadings.T
    residual = 0.5 * (residual + residual.T)
    return GaussianPathFactor(
        times=times,
        hurst=fine_factor.hurst,
        loadings=loadings,
        covariance=covariance,
        residual=residual,
        method=(
            "endpoint_first_hierarchical_dyadic_cholesky_nested_subfactor"
        ),
    )


__all__ = [
    "dyadic_endpoint_first_order",
    "endpoint_hierarchical_dyadic_factor",
    "hierarchical_dyadic_subfactor",
]
