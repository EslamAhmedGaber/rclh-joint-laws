"""Low-cost Gaussian-coordinate rotations for sparse chaos solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .basis import HermiteBasis
from .fbm import GaussianPathFactor


class ProjectionDesign(Protocol):
    basis_matrix: np.ndarray
    projection_operator: np.ndarray


@dataclass(frozen=True)
class HistoryRotation:
    rotation: np.ndarray
    eigenvalues: np.ndarray
    cumulative_fraction: np.ndarray
    gram: np.ndarray


@dataclass(frozen=True)
class TransformedIndicatorScores:
    rhs: np.ndarray
    state: np.ndarray
    snapshot_steps: np.ndarray


@dataclass(frozen=True)
class NestedOperatorIndicatorScores:
    rhs: np.ndarray
    operator_defect: np.ndarray
    inverse_closure: np.ndarray
    forward_closure: np.ndarray
    snapshot_steps: np.ndarray


def first_chaos_history(
    basis: HermiteBasis,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Extract first-chaos coefficient vectors at every output time."""

    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or values.shape[1] != basis.size:
        raise ValueError("coefficients have incompatible shape")
    history = np.empty((values.shape[0], basis.dimension), dtype=float)
    for coordinate in range(basis.dimension):
        alpha = [0] * basis.dimension
        alpha[coordinate] = 1
        position = basis.lookup.get(tuple(alpha))
        if position is None:
            raise ValueError("basis does not contain every first-order mode")
        history[:, coordinate] = values[:, position]
    return history


def history_sensitivity_rotation(
    sensitivity_history: np.ndarray,
    *,
    output_scales: np.ndarray | None = None,
    time_weights: np.ndarray | None = None,
    path_loadings: np.ndarray | None = None,
    path_regularization: float = 1.0e-8,
) -> HistoryRotation:
    """Rotate Gaussian coordinates using a full-history sensitivity Gramian.

    The input can be obtained from a cheap degree-one chaos solve.  The
    optional path-energy term only orders directions left unresolved by the
    solution sensitivity; it is scaled relative to the Gramian trace.
    """

    sensitivities = np.asarray(sensitivity_history, dtype=float)
    if sensitivities.ndim != 2 or sensitivities.shape[1] < 1:
        raise ValueError(
            "sensitivity_history must have shape (n_times, dimension)"
        )
    rows, dimension = sensitivities.shape

    if output_scales is None:
        scales = np.ones(rows, dtype=float)
    else:
        scales = np.asarray(output_scales, dtype=float)
        if scales.shape != (rows,):
            raise ValueError("output_scales have incompatible shape")
        scales = np.maximum(scales, np.finfo(float).tiny)

    if time_weights is None:
        weights = np.ones(rows, dtype=float) / rows
    else:
        weights = np.asarray(time_weights, dtype=float)
        if weights.shape != (rows,):
            raise ValueError("time_weights have incompatible shape")
        if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("time_weights must be nonnegative and nonzero")
        weights = weights / np.sum(weights)

    normalized = sensitivities / scales[:, None]
    gram = normalized.T @ (weights[:, None] * normalized)
    gram = 0.5 * (gram + gram.T)

    if path_loadings is not None and path_regularization > 0.0:
        loadings = np.asarray(path_loadings, dtype=float)
        if loadings.ndim != 2 or loadings.shape[1] != dimension:
            raise ValueError("path_loadings have incompatible shape")
        path_gram = loadings.T @ loadings
        path_trace = float(np.trace(path_gram))
        gram_trace = float(np.trace(gram))
        if path_trace > np.finfo(float).tiny:
            scale = (
                max(gram_trace, np.finfo(float).tiny)
                / path_trace
            )
            gram = gram + float(path_regularization) * scale * path_gram

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    rotation = eigenvectors[:, order]

    # Fix arbitrary eigenvector signs so that frozen runs remain reproducible.
    for column in range(dimension):
        pivot = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[pivot, column] < 0.0:
            rotation[:, column] *= -1.0

    total = float(np.sum(eigenvalues))
    if total > np.finfo(float).tiny:
        cumulative = np.cumsum(eigenvalues) / total
    else:
        cumulative = np.zeros_like(eigenvalues)
    return HistoryRotation(rotation, eigenvalues, cumulative, gram)


def transformed_candidate_indicators(
    factor: GaussianPathFactor,
    pilot_basis: HermiteBasis,
    pilot_coefficients: np.ndarray,
    candidate_basis: HermiteBasis,
    projection: ProjectionDesign,
    drift: Callable[[float, np.ndarray], np.ndarray],
    *,
    beta: float = 1.0,
    snapshot_steps: np.ndarray | None = None,
) -> TransformedIndicatorScores:
    """Score candidate coefficients in the Lawson-transformed drift.

    The theorem-aligned score is ``rhs``: a time-weighted coefficient-energy
    approximation to the omitted transformed right-hand side

    ``D_-g P[b(t, D_g y)]``.

    ``state`` is returned separately as a diagnostic.  Keeping the two
    quantities separate lets experiments test whether adding state energy
    helps prediction without silently calling it a residual estimator.
    """

    history = np.asarray(pilot_coefficients, dtype=float)
    if pilot_basis.dimension != factor.modes:
        raise ValueError("pilot basis and factor dimensions differ")
    if candidate_basis.dimension != factor.modes:
        raise ValueError("candidate basis and factor dimensions differ")
    if history.shape != (factor.times.size + 1, pilot_basis.size):
        raise ValueError("pilot_coefficients have incompatible shape")
    if projection.basis_matrix.shape[1] != candidate_basis.size:
        raise ValueError("projection and candidate basis differ")
    if projection.projection_operator.shape[0] != candidate_basis.size:
        raise ValueError("projection and candidate basis differ")

    if snapshot_steps is None:
        snapshots = np.unique(
            np.linspace(1, factor.times.size, 12, dtype=int)
        )
    else:
        snapshots = np.unique(
            np.asarray(snapshot_steps, dtype=int)
        )
    if (
        snapshots.ndim != 1
        or snapshots.size == 0
        or np.any(snapshots < 1)
        or np.any(snapshots > factor.times.size)
    ):
        raise ValueError("snapshot_steps must index positive solver steps")

    positions = []
    for alpha in pilot_basis.indices:
        position = candidate_basis.lookup.get(
            tuple(int(value) for value in alpha)
        )
        if position is None:
            raise ValueError(
                "candidate basis must contain the complete pilot basis"
            )
        positions.append(position)

    rhs_scores = np.zeros(candidate_basis.size, dtype=float)
    state_scores = np.zeros(candidate_basis.size, dtype=float)
    previous_time = 0.0
    for step in snapshots:
        time_value = float(factor.times[step - 1])
        embedded = np.zeros(candidate_basis.size, dtype=float)
        embedded[positions] = history[step]
        values = projection.basis_matrix @ embedded
        drift_coefficients = projection.projection_operator @ (
            drift(time_value, values)
        )
        loading = float(beta) * factor.loadings[step - 1]
        transformed_state = candidate_basis.apply_wick_exponential(
            embedded,
            -loading,
        )
        transformed_rhs = candidate_basis.apply_wick_exponential(
            drift_coefficients,
            -loading,
        )
        weight = max(
            time_value - previous_time,
            1.0 / factor.times.size,
        )
        rhs_scores += weight * transformed_rhs * transformed_rhs
        state_scores += weight * transformed_state * transformed_state
        previous_time = time_value

    return TransformedIndicatorScores(
        rhs=rhs_scores,
        state=state_scores,
        snapshot_steps=snapshots,
    )


def nested_lawson_operator_indicators(
    factor: GaussianPathFactor,
    pilot_basis: HermiteBasis,
    pilot_coefficients: np.ndarray,
    candidate_basis: HermiteBasis,
    projection: ProjectionDesign,
    drift: Callable[[float, np.ndarray], np.ndarray],
    *,
    beta: float = 1.0,
    snapshot_steps: np.ndarray | None = None,
) -> NestedOperatorIndicatorScores:
    """Evaluate nested-space defects for the transformed finite ODE.

    Let ``E`` embed the pilot space into the downward-closed candidate space
    and let ``D_g`` denote the exact finite Wick exponential.  At every
    snapshot this routine evaluates the candidate operator

    ``F_Gamma(E y) = D_-g^Gamma P_Gamma b(D_g^Gamma E y)``.

    It returns coefficient-energy indicators for this right-hand side, its
    difference from the embedded reduced operator, and the forward/inverse
    Wick-closure defects.  Candidate coefficients outside the pilot support
    have identical ``rhs`` and ``operator_defect`` scores; the latter also
    records the non-nested feedback on already active coefficients.
    """

    history = np.asarray(pilot_coefficients, dtype=float)
    if pilot_basis.dimension != factor.modes:
        raise ValueError("pilot basis and factor dimensions differ")
    if candidate_basis.dimension != factor.modes:
        raise ValueError("candidate basis and factor dimensions differ")
    if history.shape != (factor.times.size + 1, pilot_basis.size):
        raise ValueError("pilot_coefficients have incompatible shape")
    if projection.basis_matrix.shape[1] != candidate_basis.size:
        raise ValueError("projection and candidate basis differ")
    if projection.projection_operator.shape[0] != candidate_basis.size:
        raise ValueError("projection and candidate basis differ")

    if snapshot_steps is None:
        snapshots = np.unique(
            np.linspace(1, factor.times.size, 12, dtype=int)
        )
    else:
        snapshots = np.unique(
            np.asarray(snapshot_steps, dtype=int)
        )
    if (
        snapshots.ndim != 1
        or snapshots.size == 0
        or np.any(snapshots < 1)
        or np.any(snapshots > factor.times.size)
    ):
        raise ValueError("snapshot_steps must index positive solver steps")

    positions = []
    for alpha in pilot_basis.indices:
        position = candidate_basis.lookup.get(
            tuple(int(value) for value in alpha)
        )
        if position is None:
            raise ValueError(
                "candidate basis must contain the complete pilot basis"
            )
        positions.append(position)
    positions_array = np.asarray(positions, dtype=np.int64)

    rhs_scores = np.zeros(candidate_basis.size, dtype=float)
    operator_scores = np.zeros(candidate_basis.size, dtype=float)
    inverse_scores = np.zeros(candidate_basis.size, dtype=float)
    forward_scores = np.zeros(candidate_basis.size, dtype=float)
    previous_time = 0.0

    for step in snapshots:
        time_value = float(factor.times[step - 1])
        loading = float(beta) * factor.loadings[step - 1]
        pilot_physical = history[step]
        pilot_transformed = pilot_basis.apply_wick_exponential(
            pilot_physical,
            -loading,
        )

        embedded_physical = np.zeros(candidate_basis.size, dtype=float)
        embedded_physical[positions_array] = pilot_physical
        embedded_transformed = np.zeros(
            candidate_basis.size,
            dtype=float,
        )
        embedded_transformed[positions_array] = pilot_transformed

        candidate_physical = (
            candidate_basis.apply_wick_exponential(
                embedded_transformed,
                loading,
            )
        )
        candidate_values = (
            projection.basis_matrix @ candidate_physical
        )
        candidate_drift = projection.projection_operator @ (
            drift(time_value, candidate_values)
        )
        candidate_rhs = candidate_basis.apply_wick_exponential(
            candidate_drift,
            -loading,
        )

        reduced_values = (
            projection.basis_matrix @ embedded_physical
        )
        reduced_drift_candidate = projection.projection_operator @ (
            drift(time_value, reduced_values)
        )
        reduced_drift = reduced_drift_candidate[positions_array]
        reduced_rhs = pilot_basis.apply_wick_exponential(
            reduced_drift,
            -loading,
        )
        embedded_reduced_rhs = np.zeros(
            candidate_basis.size,
            dtype=float,
        )
        embedded_reduced_rhs[positions_array] = reduced_rhs

        inverse_lift = candidate_basis.apply_wick_exponential(
            embedded_physical,
            -loading,
        )
        inverse_closure = inverse_lift - embedded_transformed
        forward_closure = candidate_physical - embedded_physical
        operator_defect = candidate_rhs - embedded_reduced_rhs

        weight = max(
            time_value - previous_time,
            1.0 / factor.times.size,
        )
        rhs_scores += weight * candidate_rhs * candidate_rhs
        operator_scores += weight * operator_defect * operator_defect
        inverse_scores += weight * inverse_closure * inverse_closure
        forward_scores += weight * forward_closure * forward_closure
        previous_time = time_value

    return NestedOperatorIndicatorScores(
        rhs=rhs_scores,
        operator_defect=operator_scores,
        inverse_closure=inverse_scores,
        forward_closure=forward_scores,
        snapshot_steps=snapshots,
    )


def greedy_downward_closed_index_set(
    candidate_indices: tuple[tuple[int, ...], ...],
    initial_indices: tuple[tuple[int, ...], ...],
    scores: np.ndarray,
    *,
    target_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Grow a downward-closed support using frozen nonnegative indicators."""

    candidates = tuple(
        tuple(int(value) for value in alpha)
        for alpha in candidate_indices
    )
    initial = {
        tuple(int(value) for value in alpha)
        for alpha in initial_indices
    }
    indicator = np.asarray(scores, dtype=float)
    if indicator.shape != (len(candidates),):
        raise ValueError("scores have incompatible shape")
    if np.any(~np.isfinite(indicator)) or np.any(indicator < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    candidate_set = set(candidates)
    if not initial.issubset(candidate_set):
        raise ValueError("initial_indices must be a candidate subset")
    if not len(initial) <= target_size <= len(candidates):
        raise ValueError(
            "target_size must lie between initial and candidate sizes"
        )
    score_lookup = {
        alpha: float(indicator[position])
        for position, alpha in enumerate(candidates)
    }

    selected = set(initial)
    while len(selected) < target_size:
        admissible = []
        for alpha in candidates:
            if alpha in selected:
                continue
            valid = True
            for coordinate, value in enumerate(alpha):
                if value == 0:
                    continue
                predecessor = list(alpha)
                predecessor[coordinate] -= 1
                if tuple(predecessor) not in selected:
                    valid = False
                    break
            if valid:
                admissible.append(alpha)
        if not admissible:
            raise RuntimeError(
                "candidate set cannot produce requested downward closure"
            )
        chosen = max(
            admissible,
            key=lambda alpha: (
                score_lookup[alpha],
                -sum(alpha),
                tuple(-value for value in alpha),
            ),
        )
        selected.add(chosen)

    return tuple(sorted(selected, key=lambda alpha: (sum(alpha), alpha)))
