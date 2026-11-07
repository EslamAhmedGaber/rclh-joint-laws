"""Separated reference-resolution and target-solver error accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .rqmc import MomentEstimate


QUANTITIES = ("mean", "second_moment", "variance")
TARGET_REFINEMENT_AXES = ("modal", "time", "support_chaos", "projector")


@dataclass(frozen=True)
class CurveError:
    infinity: float
    time_l2: float

    @property
    def maximum(self) -> float:
        return max(float(self.infinity), float(self.time_l2))


@dataclass(frozen=True)
class ReferenceErrorBudget:
    sampling_half_width: float
    rqmc_doubling_shift: float
    ode_tightening_shift: float
    factor_refinement_shift: float
    classification: str
    thresholds: Mapping[str, float]

    @property
    def target_reference_allowance(self) -> float:
        """Conservative reference allowance used in target accuracy bounds."""

        return (
            float(self.sampling_half_width)
            + float(self.ode_tightening_shift)
            + float(self.factor_refinement_shift)
        )


@dataclass(frozen=True)
class TargetAccuracyAssessment:
    quantity_errors: Mapping[str, CurveError]
    quantity_bounds: Mapping[str, float]
    target_refinement_shifts: Mapping[str, float]
    reference_allowance: float
    method_allowance: float
    tolerance: float
    classification: str


def normalized_curve_error(
    times: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> CurveError:
    """Return normalized time-infinity and trapezoidal time-L2 errors."""

    nodes = np.asarray(times, dtype=float)
    truth = np.asarray(reference, dtype=float)
    approximation = np.asarray(candidate, dtype=float)
    if (
        nodes.ndim != 1
        or truth.shape != nodes.shape
        or approximation.shape != nodes.shape
        or nodes.size < 2
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise ValueError("times and curves must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(approximation)):
        return CurveError(float("inf"), float("inf"))
    scale = max(1.0, float(np.max(np.abs(truth))))
    error = approximation - truth
    infinity = float(np.max(np.abs(error)) / scale)
    duration = float(nodes[-1] - nodes[0])
    if duration <= 0.0:
        raise ValueError("reporting interval must have positive duration")
    time_l2 = float(
        np.sqrt(max(float(np.trapezoid(error * error, nodes)) / duration, 0.0))
        / scale
    )
    return CurveError(infinity, time_l2)


def _maximum_quantity_shift(
    reference: MomentEstimate,
    candidate: MomentEstimate,
) -> float:
    if not np.array_equal(reference.reporting_times, candidate.reporting_times):
        raise ValueError("moment estimates use different reporting nodes")
    shifts = [
        normalized_curve_error(
            reference.reporting_times,
            reference.quantity(name),
            candidate.quantity(name),
        ).maximum
        for name in QUANTITIES
    ]
    return float(max(shifts))


def _maximum_sampling_half_width(reference: MomentEstimate) -> float:
    maxima = []
    for name in QUANTITIES:
        scale = max(1.0, float(np.max(np.abs(reference.quantity(name)))))
        pointwise = reference.half_width(name) / scale
        infinity = float(np.max(pointwise))
        duration = float(
            reference.reporting_times[-1] - reference.reporting_times[0]
        )
        time_l2 = float(
            np.sqrt(
                max(
                    float(
                        np.trapezoid(
                            pointwise * pointwise,
                            reference.reporting_times,
                        )
                    )
                    / duration,
                    0.0,
                )
            )
        )
        maxima.append(max(infinity, time_l2))
    return float(max(maxima))


def reference_error_budget(
    primary_fine: MomentEstimate,
    primary_coarse: MomentEstimate,
    tightened_ode: MomentEstimate,
    refined_factor: MomentEstimate,
    *,
    tolerance: float,
) -> ReferenceErrorBudget:
    """Separate RQMC, ODE-time, and factor-interpolation resolution errors."""

    tau = float(tolerance)
    if tau <= 0.0:
        raise ValueError("tolerance must be positive")
    sampling = _maximum_sampling_half_width(primary_fine)
    rqmc_shift = _maximum_quantity_shift(primary_fine, primary_coarse)
    ode_shift = _maximum_quantity_shift(primary_fine, tightened_ode)
    factor_shift = _maximum_quantity_shift(primary_fine, refined_factor)
    thresholds = {
        "sampling_half_width": 0.10 * tau,
        "rqmc_doubling_shift": 0.10 * tau,
        "ode_tightening_shift": 0.05 * tau,
        "factor_refinement_shift": 0.10 * tau,
    }
    values = (sampling, rqmc_shift, ode_shift, factor_shift)
    if any(not np.isfinite(value) for value in values):
        classification = "STOP_REFERENCE_NONFINITE"
    elif (
        sampling <= thresholds["sampling_half_width"]
        and rqmc_shift <= thresholds["rqmc_doubling_shift"]
        and ode_shift <= thresholds["ode_tightening_shift"]
        and factor_shift <= thresholds["factor_refinement_shift"]
    ):
        classification = "PASS_REFERENCE"
    else:
        classification = "INCONCLUSIVE_REFERENCE"
    return ReferenceErrorBudget(
        sampling_half_width=sampling,
        rqmc_doubling_shift=rqmc_shift,
        ode_tightening_shift=ode_shift,
        factor_refinement_shift=factor_shift,
        classification=classification,
        thresholds=thresholds,
    )


def assess_target_solver(
    reference: MomentEstimate,
    reference_budget: ReferenceErrorBudget,
    target_curves: Mapping[str, np.ndarray],
    target_refinement_shifts: Mapping[str, float],
    *,
    tolerance: float,
    method_allowance: float = 0.0,
) -> TargetAccuracyAssessment:
    """Assess target moment curves without importing target coefficients."""

    tau = float(tolerance)
    method_uncertainty = float(method_allowance)
    if tau <= 0.0 or method_uncertainty < 0.0:
        raise ValueError("invalid tolerance or method allowance")
    missing_quantities = [name for name in QUANTITIES if name not in target_curves]
    if missing_quantities:
        raise ValueError(f"missing target curves: {missing_quantities}")
    missing_axes = [
        name for name in TARGET_REFINEMENT_AXES if name not in target_refinement_shifts
    ]
    if missing_axes:
        raise ValueError(f"missing target refinement shifts: {missing_axes}")
    refinement = {
        name: float(target_refinement_shifts[name])
        for name in TARGET_REFINEMENT_AXES
    }
    if any(value < 0.0 for value in refinement.values()):
        raise ValueError("target refinement shifts must be nonnegative")

    errors = {
        name: normalized_curve_error(
            reference.reporting_times,
            reference.quantity(name),
            np.asarray(target_curves[name], dtype=float),
        )
        for name in QUANTITIES
    }
    reference_allowance = reference_budget.target_reference_allowance
    bounds = {
        name: error.maximum + reference_allowance + method_uncertainty
        for name, error in errors.items()
    }
    lower_bounds = {
        name: max(
            0.0,
            error.maximum - reference_allowance - method_uncertainty,
        )
        for name, error in errors.items()
    }
    any_nonfinite = any(
        not np.isfinite(value)
        for error in errors.values()
        for value in (error.infinity, error.time_l2)
    )
    refinement_passes = all(value <= 0.20 * tau for value in refinement.values())
    if reference_budget.classification != "PASS_REFERENCE":
        classification = "INCONCLUSIVE_UNRESOLVED_REFERENCE"
    elif any_nonfinite:
        classification = "STOP_ACCURACY_NONFINITE"
    elif all(value <= tau for value in bounds.values()) and refinement_passes:
        classification = "PASS_ACCURACY"
    elif any(value > tau for value in lower_bounds.values()):
        classification = "STOP_ACCURACY"
    else:
        classification = "INCONCLUSIVE_ACCURACY"
    return TargetAccuracyAssessment(
        quantity_errors=errors,
        quantity_bounds=bounds,
        target_refinement_shifts=refinement,
        reference_allowance=reference_allowance,
        method_allowance=method_uncertainty,
        tolerance=tau,
        classification=classification,
    )


__all__ = [
    "CurveError",
    "QUANTITIES",
    "ReferenceErrorBudget",
    "TARGET_REFINEMENT_AXES",
    "TargetAccuracyAssessment",
    "assess_target_solver",
    "normalized_curve_error",
    "reference_error_budget",
]
