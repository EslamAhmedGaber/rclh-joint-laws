"""Independent physical-Lawson defect evaluation for a stored trajectory.

The routines in this module do not reuse nonlinear-projector nodes.  They
evaluate a declared piecewise-affine reconstruction of the finite Lawson
state at supplied joint time--Gaussian points.  The resulting quantities are
resolved numerical indicators unless the supplied cubature has a separate
rigorous error bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import HermiteBasis
from .coherent import recover_lawson_coefficients


def _readonly_vector(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class JointPhysicalLawsonDefect:
    """Physical defect values at supplied joint time--Gaussian points."""

    evaluation_times: np.ndarray
    residual_values: np.ndarray
    lawson_coefficients: np.ndarray

    def __post_init__(self) -> None:
        times = _readonly_vector(self.evaluation_times)
        residuals = _readonly_vector(self.residual_values)
        lawson = np.asarray(self.lawson_coefficients, dtype=float).copy()
        lawson.setflags(write=False)
        if times.ndim != 1 or residuals.shape != times.shape:
            raise ValueError("defect times and values must be equal vectors")
        if lawson.ndim != 2:
            raise ValueError("lawson_coefficients must be a matrix")
        object.__setattr__(self, "evaluation_times", times)
        object.__setattr__(self, "residual_values", residuals)
        object.__setattr__(self, "lawson_coefficients", lawson)


@dataclass(frozen=True, slots=True)
class JointDefectIndicator:
    """Space--time Lp defect and terminal Gronwall indicators."""

    power: float
    sample_count: int
    duration: float
    power_integral_estimate: float
    space_time_lp_estimate: float
    integrated_time_lp_upper_estimate: float
    translation_factor: float
    lawson_terminal_indicator: float
    coherent_terminal_indicator: float
    tau_terminal: float
    current_terminal_indicator: float


@dataclass(frozen=True, slots=True)
class DeterministicDefectEnvelope:
    """Guaranteed analytic envelope for the declared reconstruction.

    The bound uses Gaussian hypercontractivity and the pointwise rational
    bound.  It is intentionally separated from the tighter cubature-based
    indicator because it can be very conservative.
    """

    power: float
    integrated_time_lp_upper_bound: float
    translation_factor: float
    lawson_terminal_bound: float
    coherent_terminal_bound: float
    tau_terminal: float
    current_terminal_bound: float


def recover_lawson_trajectory(
    basis: HermiteBasis,
    physical_coefficients: np.ndarray,
    loading_trajectory: np.ndarray,
) -> np.ndarray:
    """Recover every nodal Lawson coefficient vector from physical output."""

    physical = np.asarray(physical_coefficients, dtype=float)
    loadings = np.asarray(loading_trajectory, dtype=float)
    if physical.ndim != 2 or physical.shape[1] != basis.size:
        raise ValueError("physical coefficient trajectory has wrong shape")
    if loadings.shape != (physical.shape[0], basis.dimension):
        raise ValueError("loading trajectory has wrong shape")
    if not np.all(np.isfinite(physical)):
        raise ValueError("physical coefficient trajectory must be finite")
    if not np.all(np.isfinite(loadings)):
        raise ValueError("loading trajectory must be finite")

    result = np.empty_like(physical)
    for row in range(physical.shape[0]):
        result[row] = recover_lawson_coefficients(
            basis,
            physical[row],
            loadings[row],
        )
    return result


def stable_affine_rational_lawson_field(
    lawson_values: np.ndarray,
    log_j_squared: np.ndarray,
    *,
    intercept: float = 0.0,
    rate: float = 1.0,
    scale: float = 4.0,
) -> np.ndarray:
    """Evaluate ``J*b(J^-1*y)`` without forming ``J^-1``.

    Here ``b(x) = intercept + rate*x + scale*x/(1+x*x)``.  The
    rational ratio is evaluated through a log-domain logistic expression.
    """

    values = np.asarray(lawson_values, dtype=float)
    log_j2 = np.asarray(log_j_squared, dtype=float)
    if values.shape != log_j2.shape:
        raise ValueError("lawson_values and log_j_squared must have one shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(log_j2)):
        raise ValueError("Lawson field inputs must be finite")
    parameters = np.asarray((intercept, rate, scale), dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("drift parameters must be finite")

    with np.errstate(divide="ignore"):
        log_y_squared = 2.0 * np.log(np.abs(values))
    log_fraction = log_j2 - np.logaddexp(log_j2, log_y_squared)
    rational = float(scale) * values * np.exp(log_fraction)
    result = float(rate) * values + rational
    if float(intercept) != 0.0:
        with np.errstate(over="ignore"):
            result = result + float(intercept) * np.exp(0.5 * log_j2)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("physical Lawson field exceeds float range")
    return result


def physical_lawson_defect_samples(
    basis: HermiteBasis,
    time_nodes: np.ndarray,
    physical_coefficients: np.ndarray,
    loading_trajectory: np.ndarray,
    evaluation_times: np.ndarray,
    gaussian_nodes: np.ndarray,
    *,
    intercept: float = 0.0,
    rate: float = 1.0,
    scale: float = 4.0,
    batch_size: int = 256,
) -> JointPhysicalLawsonDefect:
    """Evaluate the piecewise-affine physical defect at supplied points.

    ``loading_trajectory`` already includes the noise coefficient beta.  The
    first row should therefore be the zero loading when the factor starts at
    time zero.
    """

    times = np.asarray(time_nodes, dtype=float)
    physical = np.asarray(physical_coefficients, dtype=float)
    loadings = np.asarray(loading_trajectory, dtype=float)
    query_times = np.asarray(evaluation_times, dtype=float)
    points = np.asarray(gaussian_nodes, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time_nodes must contain at least two nodes")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_nodes must be finite and strictly increasing")
    if physical.shape != (times.size, basis.size):
        raise ValueError("physical coefficient trajectory has wrong shape")
    if loadings.shape != (times.size, basis.dimension):
        raise ValueError("loading trajectory has wrong shape")
    if query_times.ndim != 1:
        raise ValueError("evaluation_times must be a vector")
    if points.shape != (query_times.size, basis.dimension):
        raise ValueError("gaussian_nodes have wrong shape")
    if query_times.size < 1:
        raise ValueError("at least one evaluation point is required")
    if not np.all(np.isfinite(query_times)) or not np.all(np.isfinite(points)):
        raise ValueError("evaluation points must be finite")
    if np.any(query_times < times[0]) or np.any(query_times > times[-1]):
        raise ValueError("evaluation time lies outside the trajectory")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size:
        raise ValueError("batch_size must be a positive integer")
    batch = int(batch_size)
    if batch < 1:
        raise ValueError("batch_size must be a positive integer")

    lawson = recover_lawson_trajectory(basis, physical, loadings)
    cell = np.searchsorted(times, query_times, side="right") - 1
    cell = np.clip(cell, 0, times.size - 2)
    widths = times[cell + 1] - times[cell]
    theta = (query_times - times[cell]) / widths
    residuals = np.empty(query_times.size, dtype=float)

    for start in range(0, query_times.size, batch):
        stop = min(query_times.size, start + batch)
        local_points = points[start:stop]
        matrix = basis.evaluate(local_points)
        local_cells = cell[start:stop]
        local_theta = theta[start:stop]
        values = np.empty(stop - start, dtype=float)
        derivatives = np.empty(stop - start, dtype=float)

        for interval in np.unique(local_cells):
            positions = np.flatnonzero(local_cells == interval)
            left_values = matrix[positions] @ lawson[interval]
            right_values = matrix[positions] @ lawson[interval + 1]
            fractions = local_theta[positions]
            values[positions] = (
                (1.0 - fractions) * left_values
                + fractions * right_values
            )
            derivatives[positions] = (
                right_values - left_values
            ) / (times[interval + 1] - times[interval])

        left_loading = loadings[local_cells]
        right_loading = loadings[local_cells + 1]
        local_loading = (
            (1.0 - local_theta[:, None]) * left_loading
            + local_theta[:, None] * right_loading
        )
        loading_dot = np.einsum(
            "ij,ij->i", local_loading, local_points, optimize=True
        )
        loading_norm_squared = np.einsum(
            "ij,ij->i", local_loading, local_loading, optimize=True
        )
        log_j_squared = -2.0 * loading_dot - loading_norm_squared
        field = stable_affine_rational_lawson_field(
            values,
            log_j_squared,
            intercept=intercept,
            rate=rate,
            scale=scale,
        )
        residuals[start:stop] = derivatives - field

    if not np.all(np.isfinite(residuals)):
        raise FloatingPointError("physical defect contains nonfinite values")
    return JointPhysicalLawsonDefect(
        evaluation_times=query_times,
        residual_values=residuals,
        lawson_coefficients=lawson,
    )


def _translation_factor(power: float, loading_norm_squared: float) -> float:
    if power <= 2.0:
        raise ValueError("power must exceed two for the Lp-to-L2 translation")
    exponent = (
        0.5 * loading_norm_squared
        + loading_norm_squared / (2.0 * (power - 2.0))
    )
    value = float(np.exp(exponent))
    if not np.isfinite(value):
        raise OverflowError("translation factor exceeds float range")
    return value


def joint_defect_indicator(
    residual_values: np.ndarray,
    *,
    duration: float,
    power: float,
    lipschitz_constant: float,
    terminal_loading_norm_squared: float,
    tau_squared: float = 0.0,
    initial_lawson_error_lp: float = 0.0,
) -> JointDefectIndicator:
    """Convert joint defect observations to terminal numerical indicators."""

    residuals = np.asarray(residual_values, dtype=float)
    scalars = np.asarray(
        (
            duration,
            power,
            lipschitz_constant,
            terminal_loading_norm_squared,
            tau_squared,
            initial_lawson_error_lp,
        ),
        dtype=float,
    )
    if residuals.ndim != 1 or residuals.size < 1:
        raise ValueError("residual_values must be a nonempty vector")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(scalars)):
        raise ValueError("indicator inputs must be finite")
    if duration <= 0.0 or power <= 2.0 or lipschitz_constant < 0.0:
        raise ValueError("invalid duration, power, or Lipschitz constant")
    if terminal_loading_norm_squared < 0.0:
        raise ValueError("loading norm squared must be nonnegative")
    if tau_squared < 0.0 or initial_lawson_error_lp < 0.0:
        raise ValueError("error inputs must be nonnegative")

    with np.errstate(over="ignore"):
        power_values = np.abs(residuals) ** float(power)
    if not np.all(np.isfinite(power_values)):
        raise FloatingPointError("defect power exceeds float range")
    power_integral = float(duration * np.mean(power_values))
    space_time_lp = float(power_integral ** (1.0 / power))
    integrated_lp = float(duration ** (1.0 - 1.0 / power) * space_time_lp)
    growth = float(np.exp(lipschitz_constant * duration))
    if not np.isfinite(growth):
        raise OverflowError("Gronwall growth factor exceeds float range")
    lawson_indicator = growth * (
        float(initial_lawson_error_lp) + integrated_lp
    )
    translation = _translation_factor(power, terminal_loading_norm_squared)
    coherent_indicator = translation * lawson_indicator
    tau = float(np.sqrt(tau_squared))
    current_indicator = coherent_indicator + tau
    return JointDefectIndicator(
        power=float(power),
        sample_count=int(residuals.size),
        duration=float(duration),
        power_integral_estimate=power_integral,
        space_time_lp_estimate=space_time_lp,
        integrated_time_lp_upper_estimate=integrated_lp,
        translation_factor=translation,
        lawson_terminal_indicator=float(lawson_indicator),
        coherent_terminal_indicator=float(coherent_indicator),
        tau_terminal=tau,
        current_terminal_indicator=float(current_indicator),
    )


def deterministic_piecewise_affine_defect_envelope(
    basis: HermiteBasis,
    time_nodes: np.ndarray,
    lawson_coefficients: np.ndarray,
    loading_trajectory: np.ndarray,
    *,
    power: float,
    lipschitz_constant: float,
    intercept: float = 0.0,
    rate: float = 1.0,
    scale: float = 4.0,
    tau_squared: float = 0.0,
) -> DeterministicDefectEnvelope:
    """Return a deterministic hypercontractive physical-defect envelope."""

    times = np.asarray(time_nodes, dtype=float)
    lawson = np.asarray(lawson_coefficients, dtype=float)
    loadings = np.asarray(loading_trajectory, dtype=float)
    parameters = np.asarray(
        (power, lipschitz_constant, intercept, rate, scale, tau_squared),
        dtype=float,
    )
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_nodes must be strictly increasing")
    if lawson.shape != (times.size, basis.size):
        raise ValueError("lawson coefficient trajectory has wrong shape")
    if loadings.shape != (times.size, basis.dimension):
        raise ValueError("loading trajectory has wrong shape")
    if not (
        np.all(np.isfinite(times))
        and np.all(np.isfinite(lawson))
        and np.all(np.isfinite(loadings))
        and np.all(np.isfinite(parameters))
    ):
        raise ValueError("envelope inputs must be finite")
    if power <= 2.0 or lipschitz_constant < 0.0 or tau_squared < 0.0:
        raise ValueError("invalid envelope parameters")

    hypercontractive = float((power - 1.0) ** (0.5 * basis.degree))
    j_multiplier = abs(float(intercept)) + 0.5 * abs(float(scale))
    integrated = 0.0
    for cell in range(times.size - 1):
        width = float(times[cell + 1] - times[cell])
        derivative = (lawson[cell + 1] - lawson[cell]) / width
        polynomial_left = derivative - float(rate) * lawson[cell]
        polynomial_right = derivative - float(rate) * lawson[cell + 1]
        polynomial_bound = hypercontractive * max(
            float(np.linalg.norm(polynomial_left)),
            float(np.linalg.norm(polynomial_right)),
        )
        maximum_loading_norm_squared = max(
            float(np.dot(loadings[cell], loadings[cell])),
            float(np.dot(loadings[cell + 1], loadings[cell + 1])),
        )
        j_lp_bound = float(
            np.exp(0.5 * (power - 1.0) * maximum_loading_norm_squared)
        )
        if not np.isfinite(j_lp_bound):
            raise OverflowError("J_v Lp bound exceeds float range")
        integrated += width * (
            polynomial_bound + j_multiplier * j_lp_bound
        )

    duration = float(times[-1] - times[0])
    growth = float(np.exp(lipschitz_constant * duration))
    terminal_norm_squared = float(np.dot(loadings[-1], loadings[-1]))
    translation = _translation_factor(power, terminal_norm_squared)
    lawson_bound = growth * integrated
    coherent_bound = translation * lawson_bound
    tau = float(np.sqrt(tau_squared))
    return DeterministicDefectEnvelope(
        power=float(power),
        integrated_time_lp_upper_bound=float(integrated),
        translation_factor=translation,
        lawson_terminal_bound=float(lawson_bound),
        coherent_terminal_bound=float(coherent_bound),
        tau_terminal=tau,
        current_terminal_bound=float(coherent_bound + tau),
    )
