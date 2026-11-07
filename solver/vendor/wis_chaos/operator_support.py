"""Target-free sparse supports derived from the discrete path operator.

This module deliberately has no dependency on solver moments, reference
solutions, or observed errors.  A support built here is therefore frozen by
the Gaussian path representation before any output of the stochastic solver
is inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import sqrt

import numpy as np

from .basis import quadratic_tail_core_enriched_index_set
from .fbm import GaussianPathFactor


@dataclass(frozen=True)
class OperatorGeometryTailPair:
    """One tail pair ranked by a full-time cubic loading product."""

    left: int
    right: int
    score_squared: float

    @property
    def score(self) -> float:
        """Return the discrete :math:`L^2` loading-product norm."""

        return sqrt(self.score_squared)


def operator_geometry_tail_pair_ranking(
    factor: GaussianPathFactor,
    *,
    core_modes: int,
    triad_core_coordinate: int = 0,
) -> tuple[OperatorGeometryTailPair, ...]:
    r"""Rank all ``i <= j`` tail pairs without using a solver output.

    For the positive time grid ``t_1 < ... < t_N`` and loadings
    ``g_k(t_n)``, the squared score is the right-endpoint approximation

    .. math::

       w_{ij}^2 = \sum_{n=1}^N (t_n-t_{n-1})
       [g_c(t_n)g_i(t_n)g_j(t_n)]^2, \qquad t_0=0.

    The rule uses the same time grid as the finite Gaussian path factor.
    Candidates are ordered by decreasing ``w_ij``.  Exact ties are resolved
    lexicographically by ``(i, j)``.  Diagonal pairs are included.
    """

    times = np.asarray(factor.times, dtype=float)
    loadings = np.asarray(factor.loadings, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("factor times must be a nonempty vector")
    if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("factor times must be positive and increasing")
    if loadings.shape != (times.size, factor.modes):
        raise ValueError("factor loadings have incompatible shape")

    selected_core_modes = int(core_modes)
    if not 1 <= selected_core_modes < factor.modes:
        raise ValueError("core_modes must lie in [1, factor.modes)")
    core_coordinate = int(triad_core_coordinate)
    if not 0 <= core_coordinate < selected_core_modes:
        raise ValueError(
            "triad_core_coordinate must identify a core coordinate"
        )

    time_widths = np.diff(np.concatenate(([0.0], times)))
    core_loading = loadings[:, core_coordinate]
    ranked: list[OperatorGeometryTailPair] = []
    for left in range(selected_core_modes, factor.modes):
        core_left = core_loading * loadings[:, left]
        for right in range(left, factor.modes):
            product = core_left * loadings[:, right]
            score_squared = float(
                np.dot(time_widths, product * product)
            )
            if not np.isfinite(score_squared) or score_squared < 0.0:
                raise FloatingPointError(
                    "operator-geometry score is not finite and nonnegative"
                )
            ranked.append(
                OperatorGeometryTailPair(
                    left=left,
                    right=right,
                    score_squared=score_squared,
                )
            )

    ranked.sort(
        key=lambda candidate: (
            -candidate.score_squared,
            candidate.left,
            candidate.right,
        )
    )
    return tuple(ranked)


def operator_geometry_triad_index_set(
    factor: GaussianPathFactor,
    *,
    core_modes: int,
    core_degree: int,
    triad_count: int,
    triad_core_coordinate: int = 0,
) -> tuple[tuple[int, ...], ...]:
    r"""Add the leading target-free cubic triads to the ``X2`` support.

    The baseline contains a complete core polynomial layer, every tail-linear
    term times a core polynomial of degree at most two, and every core-free
    quadratic tail pair.  The first ``triad_count`` candidates from
    :func:`operator_geometry_tail_pair_ranking` add

    .. math:: e_c + e_i + e_j, \qquad i \le j.

    With normalized probabilists' Hermites, an off-diagonal term represents
    ``xi_c xi_i xi_j``.  A diagonal term represents
    ``xi_c (xi_i**2 - 1) / sqrt(2)``.  The normalized-Hermite scaling is
    handled by :class:`~wis_chaos.basis.HermiteBasis`; this function only
    constructs multi-indices.

    Every predecessor of an added triad already belongs to the ``X2``
    baseline, so any ranked prefix remains downward closed.
    """

    selected_count = int(triad_count)
    if selected_count < 0:
        raise ValueError("triad_count must be nonnegative")
    selected_core_degree = int(core_degree)
    if selected_core_degree <= 2:
        raise ValueError("core_degree must exceed the fixed cross degree 2")

    baseline = quadratic_tail_core_enriched_index_set(
        factor.modes,
        core_modes=int(core_modes),
        core_degree=selected_core_degree,
        cross_degree=2,
    )
    ranking = operator_geometry_tail_pair_ranking(
        factor,
        core_modes=int(core_modes),
        triad_core_coordinate=int(triad_core_coordinate),
    )
    if selected_count > len(ranking):
        raise ValueError(
            "triad_count exceeds the number of diagonal-inclusive tail pairs"
        )

    support = set(baseline)
    for candidate in ranking[:selected_count]:
        alpha = [0] * factor.modes
        alpha[int(triad_core_coordinate)] = 1
        alpha[candidate.left] += 1
        alpha[candidate.right] += 1
        support.add(tuple(alpha))

    expected_size = len(baseline) + selected_count
    if len(support) != expected_size:
        raise RuntimeError(
            "ranked cubic triads unexpectedly overlap the X2 baseline"
        )
    return tuple(sorted(support, key=lambda alpha: (sum(alpha), alpha)))


def operator_geometry_pair_fingerprint(
    ranking: tuple[OperatorGeometryTailPair, ...],
    count: int,
) -> str:
    """Hash a ranked pair prefix independently of floating-point scores."""

    selected_count = int(count)
    if not 0 <= selected_count <= len(ranking):
        raise ValueError("count must index a prefix of the ranking")
    payload = ";".join(
        f"{candidate.left}:{candidate.right}"
        for candidate in ranking[:selected_count]
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "OperatorGeometryTailPair",
    "operator_geometry_pair_fingerprint",
    "operator_geometry_tail_pair_ranking",
    "operator_geometry_triad_index_set",
]
