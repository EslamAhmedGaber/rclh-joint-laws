"""Reference-blind refinement certificates for governed solver studies.

The routines in this module are algebraic post-processing only.  They do not
run a solver, construct a stochastic reference, measure execution time, or
claim a rigorous continuum bound.  A certificate becomes empirical evidence
only after a separately governed reference package confirms coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from typing import Iterable, Mapping

import numpy as np


class CertificateError(ValueError):
    """Raised when a certificate input violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class RefinementCertificate:
    """Geometric-tail envelope inferred from three nested approximations."""

    first_difference: float
    second_difference: float
    observed_contraction: float
    effective_contraction: float
    observed_order: float | None
    safety_factor: float
    bound: float | None
    resolved: bool
    classification: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReferenceInterval:
    """Conservative interval for an error measured against an uncertain oracle."""

    central_error: float
    reference_uncertainty: float
    lower_error: float
    upper_error: float

    def to_jsonable(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricAssessment:
    """Accuracy, coverage, and usefulness status for one frozen metric."""

    metric_id: str
    scale: float
    pass_relative_limit: float
    stop_relative_limit: float
    interval: ReferenceInterval
    certificate_bound: float | None
    normalized_lower_error: float
    normalized_upper_error: float
    normalized_certificate_width: float | None
    accuracy_status: str
    coverage_status: str
    effectivity: float | None

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["interval"] = self.interval.to_jsonable()
        return payload


def _finite_nonnegative(value: float, label: str) -> float:
    observed = float(value)
    if not np.isfinite(observed) or observed < 0.0:
        raise CertificateError(f"{label} must be finite and nonnegative")
    return observed


def _finite_scale(value: float) -> float:
    observed = float(value)
    if not np.isfinite(observed) or observed <= 0.0:
        raise CertificateError("scale must be finite and positive")
    return observed


def normalized_l2_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    scale: float = 1.0,
) -> float:
    """Return a finite normalized Euclidean difference for equal arrays."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape:
        raise CertificateError("refinement arrays must have identical shapes")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise CertificateError("refinement arrays must be finite")
    denominator = _finite_scale(scale)
    return float(np.linalg.norm((first - second).ravel()) / denominator)


def observed_order(first_difference: float, second_difference: float) -> float | None:
    """Return ``log2(d0/d1)`` when both differences are strictly positive."""

    first = _finite_nonnegative(first_difference, "first_difference")
    second = _finite_nonnegative(second_difference, "second_difference")
    if first == 0.0 or second == 0.0:
        return None
    return float(log2(first / second))


def refinement_certificate(
    coarse: np.ndarray,
    middle: np.ndarray,
    fine: np.ndarray,
    *,
    scale: float,
    contraction_floor: float,
    contraction_limit: float,
    safety_factor: float = 1.25,
    roundoff_floor: float | None = None,
) -> RefinementCertificate:
    """Build a conservative geometric-tail envelope from three levels.

    The returned quantity is a resolved *numerical indicator*, not a theorem.
    If contraction is absent, the routine returns an explicit inconclusive
    classification and no bound instead of clipping the observation.
    """

    denominator = _finite_scale(scale)
    floor = _finite_nonnegative(contraction_floor, "contraction_floor")
    limit = _finite_nonnegative(contraction_limit, "contraction_limit")
    safety = _finite_scale(safety_factor)
    if floor >= limit or limit >= 1.0:
        raise CertificateError(
            "contractions must satisfy 0 <= floor < limit < 1"
        )
    machine_floor = (
        256.0 * np.finfo(float).eps
        if roundoff_floor is None
        else _finite_nonnegative(roundoff_floor, "roundoff_floor")
    )

    first = normalized_l2_difference(coarse, middle, scale=denominator)
    second = normalized_l2_difference(middle, fine, scale=denominator)
    if first <= machine_floor and second <= machine_floor:
        effective = floor
        bound = safety * machine_floor / (1.0 - effective)
        return RefinementCertificate(
            first_difference=first,
            second_difference=second,
            observed_contraction=0.0,
            effective_contraction=effective,
            observed_order=None,
            safety_factor=safety,
            bound=float(bound * denominator),
            resolved=True,
            classification="RESOLVED_AT_ROUNDOFF",
        )
    if first <= machine_floor:
        return RefinementCertificate(
            first_difference=first,
            second_difference=second,
            observed_contraction=float("inf"),
            effective_contraction=float("inf"),
            observed_order=None,
            safety_factor=safety,
            bound=None,
            resolved=False,
            classification="INCONCLUSIVE_ZERO_FIRST_DIFFERENCE",
        )

    contraction = second / first
    effective = max(contraction, floor)
    order = observed_order(first, second)
    if not np.isfinite(effective) or effective >= limit:
        return RefinementCertificate(
            first_difference=first,
            second_difference=second,
            observed_contraction=float(contraction),
            effective_contraction=float(effective),
            observed_order=order,
            safety_factor=safety,
            bound=None,
            resolved=False,
            classification="INCONCLUSIVE_NO_SATURATION",
        )
    normalized_bound = safety * first / (1.0 - effective)
    return RefinementCertificate(
        first_difference=first,
        second_difference=second,
        observed_contraction=float(contraction),
        effective_contraction=float(effective),
        observed_order=order,
        safety_factor=safety,
        bound=float(normalized_bound * denominator),
        resolved=True,
        classification="RESOLVED_NUMERICAL_INDICATOR",
    )


def combine_component_bounds(
    components: Mapping[str, RefinementCertificate],
    *,
    floating_point_allowance: float = 0.0,
) -> tuple[float | None, tuple[str, ...]]:
    """Add resolved component envelopes using the triangle inequality."""

    allowance = _finite_nonnegative(
        floating_point_allowance, "floating_point_allowance"
    )
    unresolved = tuple(
        sorted(name for name, item in components.items() if not item.resolved)
    )
    if unresolved:
        return None, unresolved
    total = allowance
    for item in components.values():
        if item.bound is None:
            raise CertificateError("resolved component is missing its bound")
        total += _finite_nonnegative(item.bound, "component bound")
    return float(total), ()


def reference_error_interval(
    central_error: float,
    reference_uncertainty: float,
) -> ReferenceInterval:
    """Return the frozen lower/upper error interval used for coverage."""

    error = _finite_nonnegative(central_error, "central_error")
    uncertainty = _finite_nonnegative(
        reference_uncertainty, "reference_uncertainty"
    )
    return ReferenceInterval(
        central_error=error,
        reference_uncertainty=uncertainty,
        lower_error=max(0.0, error - uncertainty),
        upper_error=error + uncertainty,
    )


def assess_metric(
    metric_id: str,
    *,
    central_error: float,
    reference_uncertainty: float,
    certificate_bound: float | None,
    scale: float,
    pass_relative_limit: float,
    stop_relative_limit: float,
) -> MetricAssessment:
    """Classify one metric without turning uncertainty into a false failure."""

    if not metric_id:
        raise CertificateError("metric_id must be nonempty")
    denominator = _finite_scale(scale)
    pass_limit = _finite_nonnegative(pass_relative_limit, "pass limit")
    stop_limit = _finite_nonnegative(stop_relative_limit, "stop limit")
    if pass_limit >= stop_limit:
        raise CertificateError("pass limit must be smaller than stop limit")
    interval = reference_error_interval(central_error, reference_uncertainty)
    lower = interval.lower_error / denominator
    upper = interval.upper_error / denominator
    if upper <= pass_limit:
        accuracy = "PASS_ACCURACY"
    elif lower > stop_limit:
        accuracy = "STOP_ACCURACY"
    else:
        accuracy = "INCONCLUSIVE_ACCURACY"

    if certificate_bound is None:
        width = None
        coverage = "INCONCLUSIVE_CERTIFICATE"
    else:
        bound = _finite_nonnegative(certificate_bound, "certificate_bound")
        width = bound / denominator
        if interval.upper_error <= bound:
            coverage = "PASS_COVERAGE"
        elif interval.lower_error > bound:
            coverage = "STOP_UNDERCOVERAGE"
        else:
            coverage = "INCONCLUSIVE_COVERAGE"

    effectivity = None
    if (
        certificate_bound is not None
        and interval.central_error >= 3.0 * interval.reference_uncertainty
        and interval.central_error > 0.0
    ):
        effectivity = float(certificate_bound / interval.central_error)
    return MetricAssessment(
        metric_id=metric_id,
        scale=denominator,
        pass_relative_limit=pass_limit,
        stop_relative_limit=stop_limit,
        interval=interval,
        certificate_bound=(
            None if certificate_bound is None else float(certificate_bound)
        ),
        normalized_lower_error=float(lower),
        normalized_upper_error=float(upper),
        normalized_certificate_width=width,
        accuracy_status=accuracy,
        coverage_status=coverage,
        effectivity=effectivity,
    )


def effectivity_summary(
    assessments: Iterable[MetricAssessment],
) -> dict[str, float | int | None]:
    """Return frozen distribution summaries for resolved effectivities."""

    values = np.asarray(
        [
            item.effectivity
            for item in assessments
            if item.effectivity is not None and np.isfinite(item.effectivity)
        ],
        dtype=float,
    )
    if values.size == 0:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(np.max(values)),
    }


__all__ = [
    "CertificateError",
    "MetricAssessment",
    "ReferenceInterval",
    "RefinementCertificate",
    "assess_metric",
    "combine_component_bounds",
    "effectivity_summary",
    "normalized_l2_difference",
    "observed_order",
    "reference_error_interval",
    "refinement_certificate",
]
