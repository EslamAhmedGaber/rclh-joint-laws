"""Target-specific translated-ODE evaluation for finite Gaussian fBM.

For each requested target ``t`` this module solves a new ordinary
differential equation on ``[0,t]``.  Intermediate states of the final-target
solve are deliberately never reused as path samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .cases import BenchmarkCase
from .factor import NestedFbmFactor


@dataclass(frozen=True)
class OdeControls:
    """Frozen adaptive high-order ODE controls."""

    rtol: float = 1.0e-9
    atol: float = 1.0e-11
    max_step_fraction: float = 0.25
    method: str = "DOP853"

    def __post_init__(self) -> None:
        if self.method != "DOP853":
            raise ValueError("the governed reference method is DOP853")
        if not 0.0 < float(self.rtol) < 1.0:
            raise ValueError("rtol must lie in (0, 1)")
        if not 0.0 < float(self.atol) < 1.0:
            raise ValueError("atol must lie in (0, 1)")
        if not 0.0 < float(self.max_step_fraction) <= 1.0:
            raise ValueError("max_step_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class OdeDiagnostics:
    target_solves: int
    right_hand_side_evaluations: int
    geometric_shortcuts: int

    def __add__(self, other: "OdeDiagnostics") -> "OdeDiagnostics":
        return OdeDiagnostics(
            self.target_solves + other.target_solves,
            self.right_hand_side_evaluations
            + other.right_hand_side_evaluations,
            self.geometric_shortcuts + other.geometric_shortcuts,
        )


@dataclass(frozen=True)
class GeometricAnchorAudit:
    """Roundoff-level audit of exact G paths and reporting-node moments."""

    maximum_path_relative_error: float
    maximum_moment_relative_error: float
    tolerance: float
    classification: str


def translated_multiplier(
    factor: NestedFbmFactor,
    gaussian_nodes: np.ndarray,
    case: BenchmarkCase,
    *,
    target_time: float,
    evaluation_time: float,
) -> np.ndarray:
    """Evaluate the finite-factor Gjessing multiplier ``J_L^t(s)``."""

    nodes = np.asarray(gaussian_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != factor.modes:
        raise ValueError("gaussian_nodes must have shape (n, factor.modes)")
    target = float(target_time)
    evaluation = float(evaluation_time)
    if not 0.0 <= evaluation <= target <= factor.horizon:
        raise ValueError("times must satisfy 0 <= evaluation <= target <= T")
    loading = factor.loadings_at(evaluation)
    target_loading = factor.loadings_at(target)
    path = nodes @ loading
    beta_squared = case.beta * case.beta
    exponent = (
        -case.alpha * evaluation
        - 0.5 * beta_squared * float(np.dot(loading, loading))
        + beta_squared * float(np.dot(target_loading, loading))
        - case.beta * path
    )
    multiplier = np.exp(exponent)
    if np.any(~np.isfinite(multiplier)) or np.any(multiplier <= 0.0):
        raise FloatingPointError("translated multiplier is non-finite")
    return multiplier


def geometric_samples(
    factor: NestedFbmFactor,
    gaussian_nodes: np.ndarray,
    case: BenchmarkCase,
    target_time: float,
) -> np.ndarray:
    """Return exact finite-factor geometric WIS samples."""

    if case.code != "G":
        raise ValueError("geometric_samples requires the frozen G case")
    nodes = np.asarray(gaussian_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != factor.modes:
        raise ValueError("gaussian_nodes must have shape (n, factor.modes)")
    target = float(target_time)
    if not 0.0 <= target <= factor.horizon:
        raise ValueError("target_time is outside [0, T]")
    loading = factor.loadings_at(target)
    path = nodes @ loading
    samples = case.initial_value * np.exp(
        case.alpha * target
        - 0.5 * case.beta**2 * float(np.dot(loading, loading))
        + case.beta * path
    )
    if np.any(~np.isfinite(samples)):
        raise FloatingPointError("geometric samples are non-finite")
    return samples


def geometric_moments(
    case: BenchmarkCase,
    times: np.ndarray,
    *,
    factor: NestedFbmFactor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact geometric mean, second moment, and variance curves.

    With ``factor=None`` the standard continuum covariance ``t**(2H)`` is
    required through ``times`` only for the mean; callers needing the
    continuum second moment should use :func:`continuum_geometric_moments`.
    Supplying a factor returns exact moments of its finite Gaussian feature.
    """

    if case.code != "G":
        raise ValueError("geometric_moments requires the frozen G case")
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or np.any(values < 0.0):
        raise ValueError("times must be a nonnegative one-dimensional array")
    if factor is None:
        raise ValueError(
            "a finite factor is required; use continuum_geometric_moments "
            "for standard fBM"
        )
    loadings = factor.loadings_at(values)
    variances = np.einsum("ij,ij->i", loadings, loadings)
    mean = case.initial_value * np.exp(case.alpha * values)
    second = case.initial_value**2 * np.exp(
        2.0 * case.alpha * values + case.beta**2 * variances
    )
    return mean, second, second - mean * mean


def continuum_geometric_moments(
    case: BenchmarkCase,
    times: np.ndarray,
    hurst: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact standard-fBM geometric moments."""

    if case.code != "G":
        raise ValueError("continuum_geometric_moments requires G")
    values = np.asarray(times, dtype=float)
    exponent = float(hurst)
    if values.ndim != 1 or np.any(values < 0.0):
        raise ValueError("times must be a nonnegative one-dimensional array")
    if not 0.0 < exponent < 1.0:
        raise ValueError("standard fBM requires 0 < hurst < 1")
    mean = case.initial_value * np.exp(case.alpha * values)
    second = case.initial_value**2 * np.exp(
        2.0 * case.alpha * values
        + case.beta**2 * values ** (2.0 * exponent)
    )
    return mean, second, second - mean * mean


def audit_geometric_anchor(
    factor: NestedFbmFactor,
    reporting_times: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-10,
) -> GeometricAnchorAudit:
    """Check exact finite-factor G samples and continuum moments at knots."""

    tolerance = float(relative_tolerance)
    if tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    times = np.asarray(reporting_times, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or np.any(np.diff(times) <= 0.0)
        or times[0] != 0.0
        or times[-1] > factor.horizon
    ):
        raise ValueError("reporting_times must increase from zero inside [0, T]")
    case = BenchmarkCase("G", 1.0, 1.0, 1.0, "zero")
    nodes = np.zeros((4, factor.modes), dtype=float)
    nodes[1, 0] = 0.375
    active = min(8, factor.modes)
    nodes[2, :active] = np.linspace(-0.4, 0.5, active)
    nodes[3] = -nodes[2]
    path_error = 0.0
    for target in times[1:]:
        loading = factor.loadings_at(float(target))
        expected = case.initial_value * np.exp(
            case.alpha * target
            - 0.5 * case.beta**2 * float(np.dot(loading, loading))
            + case.beta * (nodes @ loading)
        )
        observed = geometric_samples(factor, nodes, case, float(target))
        scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
        path_error = max(
            path_error,
            float(np.max(np.abs(observed - expected) / scale)),
        )
    finite = geometric_moments(case, times, factor=factor)
    continuum = continuum_geometric_moments(case, times, factor.hurst)
    moment_error = 0.0
    for observed, expected in zip(finite, continuum):
        scale = np.maximum(np.abs(expected), 1.0)
        moment_error = max(
            moment_error,
            float(np.max(np.abs(observed - expected) / scale)),
        )
    classification = (
        "PASS_GEOMETRIC_ANCHOR"
        if path_error <= tolerance and moment_error <= tolerance
        else "STOP_GEOMETRIC_ANCHOR"
    )
    return GeometricAnchorAudit(
        maximum_path_relative_error=path_error,
        maximum_moment_relative_error=moment_error,
        tolerance=tolerance,
        classification=classification,
    )


def solve_target_samples(
    factor: NestedFbmFactor,
    gaussian_nodes: np.ndarray,
    case: BenchmarkCase,
    target_time: float,
    controls: OdeControls,
) -> tuple[np.ndarray, OdeDiagnostics]:
    """Solve the independently translated ODE for one target time."""

    nodes = np.asarray(gaussian_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[0] < 1 or nodes.shape[1] != factor.modes:
        raise ValueError("gaussian_nodes must have shape (n, factor.modes)")
    if np.any(~np.isfinite(nodes)):
        raise ValueError("gaussian_nodes must be finite")
    target = float(target_time)
    if not 0.0 <= target <= factor.horizon:
        raise ValueError("target_time is outside [0, T]")
    if target == 0.0:
        return (
            np.full(nodes.shape[0], case.initial_value, dtype=float),
            OdeDiagnostics(0, 0, 0),
        )
    if case.code == "G":
        return (
            geometric_samples(factor, nodes, case, target),
            OdeDiagnostics(1, 0, 1),
        )

    target_loading = factor.loadings_at(target)
    beta_squared = case.beta * case.beta

    def right_hand_side(time_value: float, state: np.ndarray) -> np.ndarray:
        loading = factor.loadings_at(time_value)
        path = nodes @ loading
        exponent = (
            -case.alpha * time_value
            - 0.5 * beta_squared * float(np.dot(loading, loading))
            + beta_squared * float(np.dot(target_loading, loading))
            - case.beta * path
        )
        multiplier = np.exp(exponent)
        if np.any(~np.isfinite(multiplier)) or np.any(multiplier <= 0.0):
            raise FloatingPointError("translated multiplier is non-finite")
        physical = state / multiplier
        drift = case.drift(time_value, physical)
        output = multiplier * drift
        if np.any(~np.isfinite(output)):
            raise FloatingPointError("translated ODE right-hand side is non-finite")
        return output

    solution = solve_ivp(
        right_hand_side,
        (0.0, target),
        np.full(nodes.shape[0], case.initial_value, dtype=float),
        method=controls.method,
        rtol=controls.rtol,
        atol=controls.atol,
        max_step=factor.mesh * controls.max_step_fraction,
        t_eval=(target,),
    )
    if not solution.success or solution.y.shape != (nodes.shape[0], 1):
        raise RuntimeError(
            f"translated ODE failed at t={target}: {solution.message}"
        )
    terminal_multiplier = translated_multiplier(
        factor,
        nodes,
        case,
        target_time=target,
        evaluation_time=target,
    )
    samples = solution.y[:, -1] / terminal_multiplier
    if np.any(~np.isfinite(samples)):
        raise FloatingPointError("translated ODE samples are non-finite")
    return samples, OdeDiagnostics(1, int(solution.nfev), 0)


def solve_curve_samples(
    factor: NestedFbmFactor,
    gaussian_nodes: np.ndarray,
    case: BenchmarkCase,
    reporting_times: np.ndarray,
    controls: OdeControls,
) -> tuple[np.ndarray, OdeDiagnostics]:
    """Return target-correct samples at all requested reporting times."""

    times = np.asarray(reporting_times, dtype=float)
    if (
        times.ndim != 1
        or times.size < 1
        or np.any(np.diff(times) <= 0.0)
        or times[0] < 0.0
        or times[-1] > factor.horizon
    ):
        raise ValueError("reporting_times must increase inside [0, T]")
    nodes = np.asarray(gaussian_nodes, dtype=float)
    curves = np.empty((nodes.shape[0], times.size), dtype=float)
    diagnostics = OdeDiagnostics(0, 0, 0)
    for index, target in enumerate(times):
        values, target_diagnostics = solve_target_samples(
            factor, nodes, case, float(target), controls
        )
        curves[:, index] = values
        diagnostics = diagnostics + target_diagnostics
    return curves, diagnostics


def published_gfbm_euler_terminal(
    path: np.ndarray,
    hurst: float,
    horizon: float,
    case: BenchmarkCase,
) -> float:
    """Algorithm 1 from Erdo\u011fan--Lord--Schieven arXiv v2.

    This small conformance kernel follows the continuum-covariance formula in
    the paper and pinned official script.  It is not the high-order reference
    used for accepted moment curves.
    """

    values = np.asarray(path, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("path must be a finite vector including B(0)")
    if abs(float(values[0])) > 4.0 * np.finfo(float).eps:
        raise ValueError("path[0] must be zero")
    exponent = float(hurst)
    final_time = float(horizon)
    if not 0.0 < exponent < 1.0 or final_time <= 0.0:
        raise ValueError("invalid hurst or horizon")
    steps = values.size - 1
    dt = final_time / steps
    transformed = float(case.initial_value)
    for index in range(steps):
        time_value = index * dt
        multiplier = np.exp(
            -case.alpha * time_value
            + 0.5
            * case.beta**2
            * (final_time ** (2.0 * exponent)
               - (final_time - time_value) ** (2.0 * exponent))
            - case.beta * values[index]
        )
        physical = transformed / multiplier
        drift = float(case.drift(time_value, np.asarray([physical]))[0])
        transformed += dt * multiplier * drift
    terminal_multiplier = np.exp(
        -case.alpha * final_time
        + 0.5 * case.beta**2 * final_time ** (2.0 * exponent)
        - case.beta * values[-1]
    )
    return float(transformed / terminal_multiplier)


def published_gfbm_euler_path(
    path: np.ndarray,
    hurst: float,
    horizon: float,
    case: BenchmarkCase,
) -> np.ndarray:
    """Algorithm 2: rerun Algorithm 1 at each requested grid target."""

    values = np.asarray(path, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("path must be one-dimensional and include B(0)")
    dt = float(horizon) / (values.size - 1)
    output = np.empty_like(values)
    output[0] = case.initial_value
    for target_index in range(1, values.size):
        output[target_index] = published_gfbm_euler_terminal(
            values[: target_index + 1],
            hurst,
            target_index * dt,
            case,
        )
    return output


__all__ = [
    "GeometricAnchorAudit",
    "OdeControls",
    "OdeDiagnostics",
    "continuum_geometric_moments",
    "audit_geometric_anchor",
    "geometric_moments",
    "geometric_samples",
    "published_gfbm_euler_path",
    "published_gfbm_euler_terminal",
    "solve_curve_samples",
    "solve_target_samples",
    "translated_multiplier",
]
