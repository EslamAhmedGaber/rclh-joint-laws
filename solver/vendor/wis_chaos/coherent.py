"""Algebra-only coherent reconstruction for finite Lawson states.

This module post-processes a finite physical coefficient vector.  It does
not alter the solver's nonlinear ODE, time integrator, projection backend,
or stored trajectory, and it does not estimate the physical Lawson defect.
All identities are exact in the finite nilpotent Hermite algebra, up to
floating-point roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import HermiteBasis


_DEFAULT_ROUNDOFF_TOLERANCE = 512.0 * np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class CoherentReconstruction:
    """Finite Lawson data and exact moments of its coherent output."""

    lawson_coefficients: np.ndarray
    shifted_coefficients: np.ndarray
    coherent_mean: float
    coherent_variance: float
    coherent_second_moment: float
    current_second_moment: float
    tau_squared: float
    loading_norm_squared: float

    def __post_init__(self) -> None:
        for field_name in (
            "lawson_coefficients",
            "shifted_coefficients",
        ):
            values = np.asarray(getattr(self, field_name), dtype=float).copy()
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


def recover_lawson_coefficients(
    basis: HermiteBasis,
    physical_coefficients: np.ndarray,
    loading: np.ndarray,
) -> np.ndarray:
    """Recover ``y = D_Lambda(-v)x`` in the finite Hermite algebra."""

    vector = np.asarray(loading, dtype=float)
    return basis.apply_wick_exponential(
        np.asarray(physical_coefficients, dtype=float),
        -vector,
    )


def shifted_polynomial_coefficients(
    basis: HermiteBasis,
    lawson_coefficients: np.ndarray,
    loading: np.ndarray,
) -> np.ndarray:
    """Return normalized-Hermite coefficients of ``y(xi + v)``.

    This applies the exponential of the transposed projected creation
    operator.  The recurrence traverses the reverse first-raising edges and
    costs ``O(P E)`` without constructing a dense translation matrix.
    """

    state = np.asarray(lawson_coefficients, dtype=float)
    vector = np.asarray(loading, dtype=float)
    if state.shape != (basis.size,):
        raise ValueError("coefficient vector has wrong size")
    if vector.shape != (basis.dimension,):
        raise ValueError("loading has wrong dimension")
    if not np.any(vector):
        return state.copy()

    (
        sources,
        targets,
        coordinates,
        factors,
        degree_offsets,
    ) = basis.first_raising_edges
    if not sources.size:
        return state.copy()

    edge_weights = factors * vector[coordinates]
    result = state.copy()
    term = state.copy()
    for order in range(1, len(degree_offsets) + 1):
        term = np.bincount(
            sources,
            weights=edge_weights * term[targets],
            minlength=basis.size,
        ) / order
        result += term
    return result


def _clip_nonnegative_roundoff(
    value: float,
    *scales: float,
    tolerance: float,
    quantity: str,
) -> float:
    scale = max(1.0, *(abs(float(item)) for item in scales))
    if value < -tolerance * scale:
        raise FloatingPointError(
            f"{quantity} is negative beyond roundoff tolerance: {value!r}"
        )
    return max(0.0, float(value))


def coherent_reconstruction(
    basis: HermiteBasis,
    physical_coefficients: np.ndarray,
    loading: np.ndarray,
    *,
    roundoff_tolerance: float = _DEFAULT_ROUNDOFF_TOLERANCE,
) -> CoherentReconstruction:
    """Reconstruct the coherent output and its exact algebraic moments.

    The coherent random variable is represented compactly by the recovered
    finite Lawson polynomial and ``v``; it is not truncated back onto the
    basis.  ``tau_squared`` is the exact coherent energy missing from the
    supplied finite physical coefficient vector.
    """

    tolerance = float(roundoff_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("roundoff_tolerance must be finite and nonnegative")

    physical = np.asarray(physical_coefficients, dtype=float)
    vector = np.asarray(loading, dtype=float)
    if physical.shape != (basis.size,):
        raise ValueError("coefficient vector has wrong size")
    if vector.shape != (basis.dimension,):
        raise ValueError("loading has wrong dimension")
    if not np.all(np.isfinite(physical)):
        raise ValueError("physical_coefficients must be finite")
    if not np.all(np.isfinite(vector)):
        raise ValueError("loading must be finite")

    lawson = recover_lawson_coefficients(basis, physical, vector)
    shifted = shifted_polynomial_coefficients(basis, lawson, vector)
    zero_position = basis.lookup[(0,) * basis.dimension]
    coherent_mean = float(lawson[zero_position])
    loading_norm_squared = float(np.dot(vector, vector))
    exponential_weight = float(np.exp(loading_norm_squared))
    if not np.isfinite(exponential_weight):
        raise OverflowError("coherent exponential weight exceeds float range")

    current_second_moment = float(np.dot(physical, physical))
    coherent_second_moment = float(
        exponential_weight * np.dot(shifted, shifted)
    )
    coherent_variance = _clip_nonnegative_roundoff(
        coherent_second_moment - coherent_mean * coherent_mean,
        coherent_second_moment,
        coherent_mean * coherent_mean,
        tolerance=tolerance,
        quantity="coherent variance",
    )
    tau_squared = _clip_nonnegative_roundoff(
        coherent_second_moment - current_second_moment,
        coherent_second_moment,
        current_second_moment,
        tolerance=tolerance,
        quantity="tau_squared",
    )
    return CoherentReconstruction(
        lawson_coefficients=lawson,
        shifted_coefficients=shifted,
        coherent_mean=coherent_mean,
        coherent_variance=coherent_variance,
        coherent_second_moment=coherent_second_moment,
        current_second_moment=current_second_moment,
        tau_squared=tau_squared,
        loading_norm_squared=loading_norm_squared,
    )
