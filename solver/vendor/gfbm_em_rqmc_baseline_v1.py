"""Independent GfBm_EM plus randomized-QMC comparison baseline.

This additive module implements Algorithms 1--2 of
Erdoğan--Lord--Schieven (arXiv:2404.07013v2) from the published formula.  It
does not copy the authors' unlicensed research script and it does not import
the PAPER IX deterministic coefficient solver.

The native numerical method is the target-dependent translated explicit
Euler recursion.  Scrambled Sobol integration is only an estimator wrapper
used to turn its trajectory outputs into moment and covariance estimates.
The wrapper never changes the GfBm_EM state update.

The rational case lies inside the published global-Lipschitz/linear-growth
convergence assumptions.  The cubic Landau case is deliberately labelled an
out-of-theorem stress test.  No clipping, taming, or silent repair is allowed;
non-finite arithmetic is a terminal STOP for that configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t


ARXIV_ID = "2404.07013v2"
OFFICIAL_COMMIT = "720a14982845c7cbd7fdea4c8566645736cc47d4"
OFFICIAL_RAW_SHA256 = (
    "3cf305e3dee0b7c921b71496751e7cfe4c43271742d7338959be99f4ac1c07e0"
)
TIME_STEP_LADDER = (16, 32, 64, 128, 256, 512)
RQMC_EXPONENT_LADDER = (9, 11, 13)
SCRAMBLE_SEEDS = (
    1701,
    1709,
    1721,
    1733,
    1741,
    1753,
    1777,
    1789,
    1801,
    1811,
    1823,
    1831,
)
FINITE_TARGET_TOLERANCE = 5.0e-3


def _readonly(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class GfbmEmCase:
    """One scalar quasilinear WIS problem in the GfBm_EM decomposition."""

    code: str
    alpha: float
    beta: float
    initial_value: float
    nonlinear_kind: str
    nonlinear_scale: float
    label: str

    def __post_init__(self) -> None:
        if self.code not in {"G", "R", "LANDAU"}:
            raise ValueError("unsupported GfBm_EM case code")
        for value in (self.alpha, self.beta, self.initial_value, self.nonlinear_scale):
            if not np.isfinite(float(value)):
                raise ValueError("case parameters must be finite")
        expected = {"G": "zero", "R": "rational", "LANDAU": "cubic"}
        if self.nonlinear_kind != expected[self.code]:
            raise ValueError("case code and nonlinear kind differ")
        if self.code == "LANDAU" and self.nonlinear_scale <= 0.0:
            raise ValueError("the Landau cubic coefficient must be positive")

    @property
    def theorem_supported(self) -> bool:
        """Whether the published global-Lipschitz theorem covers this drift."""

        return self.code in {"G", "R"}

    @property
    def convergence_scope(self) -> str:
        if self.code == "LANDAU":
            return (
                "OUTSIDE_ELS_GLOBAL_LIPSCHITZ_AND_LINEAR_GROWTH_THEOREM;"
                "EMPIRICAL_STRESS_TEST_ONLY"
            )
        return "ELS_STRONG_L2_ORDER_H_FOR_AUTONOMOUS_DRIFT"

    def drift(self, time_value: float, state: np.ndarray) -> np.ndarray:
        """Evaluate only the nonlinear term ``a(t,x)`` in the ELS split."""

        del time_value
        values = np.asarray(state, dtype=float)
        if self.code == "G":
            return np.zeros_like(values)
        if self.code == "R":
            magnitude = np.abs(values)
            output = np.empty_like(values)
            small = magnitude <= 1.0
            output[small] = (
                self.nonlinear_scale
                * values[small]
                / (1.0 + values[small] * values[small])
            )
            large = ~small
            output[large] = self.nonlinear_scale / (
                values[large] + 1.0 / values[large]
            )
            return output
        return -self.nonlinear_scale * values * values * values


def geometric_case() -> GfbmEmCase:
    return GfbmEmCase("G", 1.0, 1.0, 1.0, "zero", 0.0, "geometric_control")


def rational_case() -> GfbmEmCase:
    return GfbmEmCase(
        "R", 1.0, 1.0, 1.0, "rational", 4.0, "4x/(1+x^2)"
    )


def landau_case() -> GfbmEmCase:
    return GfbmEmCase(
        "LANDAU", 0.7, 0.55, 0.35, "cubic", 1.0, "0.7x-x^3"
    )


@dataclass(frozen=True, slots=True)
class GfbmEmBatchDiagnostics:
    trajectories: int
    target_solves: int
    drift_evaluations: int
    state_updates: int
    multiplier_exponentials: int
    gaussian_coordinates: int
    cross_moment_accumulations: int
    nonfinite_internal_events: int
    nonfinite_output_count: int
    negative_output_count: int
    maximum_cubic_stability_number: float

    @property
    def classification(self) -> str:
        if self.nonfinite_internal_events or self.nonfinite_output_count:
            return "STOP_GFBM_EM_NONFINITE"
        return "PASS_GFBM_EM_FINITE"

    def __add__(self, other: "GfbmEmBatchDiagnostics") -> "GfbmEmBatchDiagnostics":
        return GfbmEmBatchDiagnostics(
            trajectories=self.trajectories + other.trajectories,
            target_solves=self.target_solves + other.target_solves,
            drift_evaluations=self.drift_evaluations + other.drift_evaluations,
            state_updates=self.state_updates + other.state_updates,
            multiplier_exponentials=(
                self.multiplier_exponentials + other.multiplier_exponentials
            ),
            gaussian_coordinates=(
                self.gaussian_coordinates + other.gaussian_coordinates
            ),
            cross_moment_accumulations=(
                self.cross_moment_accumulations
                + other.cross_moment_accumulations
            ),
            nonfinite_internal_events=(
                self.nonfinite_internal_events + other.nonfinite_internal_events
            ),
            nonfinite_output_count=(
                self.nonfinite_output_count + other.nonfinite_output_count
            ),
            negative_output_count=(
                self.negative_output_count + other.negative_output_count
            ),
            maximum_cubic_stability_number=max(
                self.maximum_cubic_stability_number,
                other.maximum_cubic_stability_number,
            ),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "target_solves": self.target_solves,
            "drift_evaluations": self.drift_evaluations,
            "state_updates": self.state_updates,
            "multiplier_exponentials": self.multiplier_exponentials,
            "gaussian_coordinates": self.gaussian_coordinates,
            "cross_moment_accumulations": self.cross_moment_accumulations,
            "nonfinite_internal_events": self.nonfinite_internal_events,
            "nonfinite_output_count": self.nonfinite_output_count,
            "negative_output_count": self.negative_output_count,
            "maximum_cubic_stability_number": (
                self.maximum_cubic_stability_number
            ),
            "classification": self.classification,
        }


def _zero_diagnostics() -> GfbmEmBatchDiagnostics:
    return GfbmEmBatchDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)


class GfbmEmNumericalFailure(FloatingPointError):
    """Fail-closed non-finite GfBm_EM configuration."""

    def __init__(self, diagnostics: GfbmEmBatchDiagnostics) -> None:
        super().__init__(
            "GfBm_EM produced non-finite arithmetic; clipping/taming is forbidden"
        )
        self.diagnostics = diagnostics


def _reporting_indices(values: np.ndarray, steps: int) -> np.ndarray:
    indices = np.asarray(values, dtype=int)
    if (
        indices.ndim != 1
        or indices.size < 2
        or indices[0] != 0
        or indices[-1] != steps
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "reporting indices must increase strictly from zero to the terminal step"
        )
    return indices


def gfbm_em_curve_batch(
    paths: np.ndarray,
    hurst: float,
    horizon: float,
    case: GfbmEmCase,
    reporting_indices: np.ndarray,
) -> tuple[np.ndarray, GfbmEmBatchDiagnostics]:
    """Vectorized Algorithms 1--2 on a batch of standard-fBM grid paths.

    Every nonzero requested target is recomputed from ``Z(0)=x0``.  An
    intermediate state from the terminal-target solve is never reused as an
    earlier sample.
    """

    values = np.asarray(paths, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("paths must have shape (batch, steps+1)")
    if np.any(~np.isfinite(values)):
        raise ValueError("input fBM paths must be finite")
    zero_tolerance = 16.0 * np.finfo(float).eps
    if np.any(np.abs(values[:, 0]) > zero_tolerance):
        raise ValueError("every fBM path must start at zero")
    exponent = float(hurst)
    final_time = float(horizon)
    if not 0.0 < exponent < 1.0 or not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("require 0 < H < 1 and a positive finite horizon")

    batch, path_width = values.shape
    steps = path_width - 1
    selected = _reporting_indices(reporting_indices, steps)
    dt = final_time / steps
    curves = np.empty((batch, selected.size), dtype=float)
    curves[:, 0] = case.initial_value
    nonfinite_internal = 0
    negative_outputs = 0
    maximum_stability = 0.0

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for output_index, target_index in enumerate(selected[1:], start=1):
            target_time = target_index * dt
            transformed = np.full(batch, case.initial_value, dtype=float)
            for time_index in range(int(target_index)):
                time_value = time_index * dt
                multiplier = np.exp(
                    -case.alpha * time_value
                    + 0.5
                    * case.beta**2
                    * (
                        target_time ** (2.0 * exponent)
                        - (target_time - time_value) ** (2.0 * exponent)
                    )
                    - case.beta * values[:, time_index]
                )
                invalid_multiplier = (~np.isfinite(multiplier)) | (multiplier <= 0.0)
                nonfinite_internal += int(np.count_nonzero(invalid_multiplier))
                physical = transformed / multiplier
                if case.code == "LANDAU":
                    finite_physical = np.isfinite(physical)
                    if np.any(finite_physical):
                        maximum_stability = max(
                            maximum_stability,
                            float(
                                np.max(
                                    dt
                                    * case.nonlinear_scale
                                    * physical[finite_physical]
                                    * physical[finite_physical],
                                    initial=0.0,
                                )
                            ),
                        )
                update = dt * multiplier * case.drift(time_value, physical)
                transformed = transformed + update
                nonfinite_internal += int(
                    np.count_nonzero(~np.isfinite(physical))
                    + np.count_nonzero(~np.isfinite(update))
                    + np.count_nonzero(~np.isfinite(transformed))
                )
            terminal_multiplier = np.exp(
                -case.alpha * target_time
                + 0.5 * case.beta**2 * target_time ** (2.0 * exponent)
                - case.beta * values[:, target_index]
            )
            nonfinite_internal += int(
                np.count_nonzero(
                    (~np.isfinite(terminal_multiplier))
                    | (terminal_multiplier <= 0.0)
                )
            )
            output = transformed / terminal_multiplier
            curves[:, output_index] = output
            finite_output = np.isfinite(output)
            negative_outputs += int(np.count_nonzero(finite_output & (output < 0.0)))

    nonfinite_outputs = int(np.count_nonzero(~np.isfinite(curves)))
    nonzero_targets = selected.size - 1
    summed_target_steps = int(np.sum(selected[1:], dtype=np.int64))
    diagnostics = GfbmEmBatchDiagnostics(
        trajectories=batch,
        target_solves=batch * nonzero_targets,
        drift_evaluations=batch * summed_target_steps,
        state_updates=batch * summed_target_steps,
        multiplier_exponentials=batch * (summed_target_steps + nonzero_targets),
        gaussian_coordinates=batch * steps,
        cross_moment_accumulations=(
            batch * selected.size * (selected.size + 1) // 2
        ),
        nonfinite_internal_events=nonfinite_internal,
        nonfinite_output_count=nonfinite_outputs,
        negative_output_count=negative_outputs,
        maximum_cubic_stability_number=maximum_stability,
    )
    return curves, diagnostics


@dataclass(frozen=True, slots=True)
class GfbmEmRqmcControls:
    fine_exponent: int
    source_dimension: int
    scramble_seeds: tuple[int, ...]
    batch_size: int
    familywise_confidence: float
    family_size: int

    def __post_init__(self) -> None:
        if self.fine_exponent < 2:
            raise ValueError("fine_exponent must be at least two")
        if self.source_dimension < 1:
            raise ValueError("source_dimension must be positive")
        if len(self.scramble_seeds) < 2 or len(set(self.scramble_seeds)) != len(
            self.scramble_seeds
        ):
            raise ValueError("at least two distinct scramble seeds are required")
        if any(seed < 0 or seed >= 2**32 for seed in self.scramble_seeds):
            raise ValueError("scramble seeds must lie in [0,2**32)")
        if (
            self.batch_size < 1
            or self.batch_size & (self.batch_size - 1)
            or self.batch_size > self.coarse_sample_count
        ):
            raise ValueError("batch_size must be a bounded positive power of two")
        if not 0.0 < self.familywise_confidence < 1.0:
            raise ValueError("familywise confidence must lie in (0,1)")
        if self.family_size < 1:
            raise ValueError("family_size must be positive")

    @property
    def coarse_sample_count(self) -> int:
        return 2 ** (self.fine_exponent - 1)

    @property
    def fine_sample_count(self) -> int:
        return 2**self.fine_exponent


@dataclass(frozen=True, slots=True)
class GfbmEmScalarFunctionalEstimate:
    """Moment estimate for one declared scalar path functional."""

    label: str
    mean: float
    second_moment: float
    variance: float
    mean_half_width: float
    second_half_width: float
    variance_half_width: float
    mean_replicates: np.ndarray
    second_replicates: np.ndarray

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mean": self.mean,
            "second_moment": self.second_moment,
            "variance": self.variance,
            "mean_half_width": self.mean_half_width,
            "second_half_width": self.second_half_width,
            "variance_half_width": self.variance_half_width,
            "mean_replicates": self.mean_replicates.tolist(),
            "second_replicates": self.second_replicates.tolist(),
        }


@dataclass(frozen=True, slots=True)
class GfbmEmMomentEstimate:
    reporting_times: np.ndarray
    sample_count_per_scramble: int
    scramble_seeds: tuple[int, ...]
    mean: np.ndarray
    second_moment: np.ndarray
    variance: np.ndarray
    cross_moment: np.ndarray
    covariance: np.ndarray
    mean_half_width: np.ndarray
    second_half_width: np.ndarray
    variance_half_width: np.ndarray
    cross_half_width: np.ndarray
    covariance_half_width: np.ndarray
    mean_replicates: np.ndarray
    second_replicates: np.ndarray
    cross_replicates: np.ndarray
    critical_value: float
    familywise_confidence: float
    family_size: int
    terminal: GfbmEmScalarFunctionalEstimate
    trapezoidal_time_average: GfbmEmScalarFunctionalEstimate
    fingerprint: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "reporting_times": self.reporting_times.tolist(),
            "sample_count_per_scramble": self.sample_count_per_scramble,
            "scramble_seeds": list(self.scramble_seeds),
            "mean": self.mean.tolist(),
            "second_moment": self.second_moment.tolist(),
            "variance": self.variance.tolist(),
            "cross_moment": self.cross_moment.tolist(),
            "covariance": self.covariance.tolist(),
            "mean_half_width": self.mean_half_width.tolist(),
            "second_half_width": self.second_half_width.tolist(),
            "variance_half_width": self.variance_half_width.tolist(),
            "cross_half_width": self.cross_half_width.tolist(),
            "covariance_half_width": self.covariance_half_width.tolist(),
            "mean_replicates": self.mean_replicates.tolist(),
            "second_replicates": self.second_replicates.tolist(),
            "cross_replicates": self.cross_replicates.tolist(),
            "critical_value": self.critical_value,
            "familywise_confidence": self.familywise_confidence,
            "family_size": self.family_size,
            "terminal": self.terminal.to_jsonable(),
            "trapezoidal_time_average": (
                self.trapezoidal_time_average.to_jsonable()
            ),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class GfbmEmRuntimeBreakdown:
    total_seconds: float
    normal_node_seconds: float
    path_generation_seconds: float
    euler_seconds: float
    reduction_seconds: float
    peak_batch_working_bytes: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "normal_node_seconds": self.normal_node_seconds,
            "path_generation_seconds": self.path_generation_seconds,
            "euler_seconds": self.euler_seconds,
            "reduction_seconds": self.reduction_seconds,
            "peak_batch_working_bytes": self.peak_batch_working_bytes,
            "scope": "ONE_ESTIMATOR_EXECUTION;NOT_THE_SEVEN_REPEAT_COST_GATE",
        }


@dataclass(frozen=True, slots=True)
class GfbmEmNestedRqmcResult:
    coarse: GfbmEmMomentEstimate
    fine: GfbmEmMomentEstimate
    diagnostics: GfbmEmBatchDiagnostics
    runtime: GfbmEmRuntimeBreakdown
    factor_fingerprint: str
    source_dimension: int
    sobol_coordinates_generated: int
    case_code: str
    convergence_scope: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "method": "GfBm_EM_explicit_Euler_plus_scrambled_Sobol_RQMC",
            "arxiv_id": ARXIV_ID,
            "official_commit": OFFICIAL_COMMIT,
            "official_raw_sha256": OFFICIAL_RAW_SHA256,
            "case_code": self.case_code,
            "convergence_scope": self.convergence_scope,
            "factor_fingerprint": self.factor_fingerprint,
            "source_dimension": self.source_dimension,
            "sobol_coordinates_generated": self.sobol_coordinates_generated,
            "coarse": self.coarse.to_jsonable(),
            "fine": self.fine.to_jsonable(),
            "diagnostics": self.diagnostics.to_jsonable(),
            "runtime": self.runtime.to_jsonable(),
        }


def _normal_nodes(uniforms: np.ndarray) -> np.ndarray:
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    output = ndtri(np.clip(np.asarray(uniforms, dtype=float), lower, upper))
    if np.any(~np.isfinite(output)):
        raise FloatingPointError("Sobol-to-normal transform was non-finite")
    return output


def _standard_grid_from_factor(factor: Any) -> tuple[int, float]:
    modes = int(factor.modes)
    horizon = float(factor.horizon)
    expected = np.linspace(horizon / modes, horizon, modes)
    observed = np.asarray(factor.times, dtype=float)
    if observed.shape != expected.shape or not np.allclose(
        observed, expected, rtol=0.0, atol=32.0 * np.finfo(float).eps
    ):
        raise ValueError("GfBm_EM requires a full standard grid factor")
    if np.asarray(factor.loadings).shape != (modes, modes):
        raise ValueError("GfBm_EM requires the full-rank grid covariance factor")
    return modes, horizon


def _simultaneous_standard_error(values: np.ndarray) -> np.ndarray:
    replicates = values.shape[0]
    return np.std(values, axis=0, ddof=1) / np.sqrt(replicates)


def _scalar_functional_estimate(
    label: str,
    weights: np.ndarray,
    mean_replicates: np.ndarray,
    cross_replicates: np.ndarray,
    critical: float,
) -> GfbmEmScalarFunctionalEstimate:
    selected_weights = np.asarray(weights, dtype=float)
    functional_means = mean_replicates @ selected_weights
    functional_seconds = np.einsum(
        "i,sij,j->s",
        selected_weights,
        cross_replicates,
        selected_weights,
        optimize=True,
    )
    mean = float(np.mean(functional_means))
    second = float(np.mean(functional_seconds))
    variance_raw = second - mean * mean
    scale = max(abs(second), 1.0)
    if variance_raw < -1.0e-12 * scale:
        raise FloatingPointError(
            f"GfBm_EM estimated {label} variance is materially negative"
        )
    variance = max(variance_raw, 0.0)
    mean_se = float(_simultaneous_standard_error(functional_means))
    second_se = float(_simultaneous_standard_error(functional_seconds))
    variance_influence = functional_seconds - 2.0 * mean * functional_means
    variance_se = float(_simultaneous_standard_error(variance_influence))
    return GfbmEmScalarFunctionalEstimate(
        label=label,
        mean=mean,
        second_moment=second,
        variance=variance,
        mean_half_width=critical * mean_se,
        second_half_width=critical * second_se,
        variance_half_width=critical * variance_se,
        mean_replicates=_readonly(functional_means),
        second_replicates=_readonly(functional_seconds),
    )


def _moment_estimate(
    reporting_times: np.ndarray,
    sample_count: int,
    seeds: tuple[int, ...],
    mean_replicates: np.ndarray,
    second_replicates: np.ndarray,
    cross_replicates: np.ndarray,
    controls: GfbmEmRqmcControls,
) -> GfbmEmMomentEstimate:
    mean = np.mean(mean_replicates, axis=0)
    second = np.mean(second_replicates, axis=0)
    cross = np.mean(cross_replicates, axis=0)
    variance_raw = second - mean * mean
    scale = np.maximum(np.abs(second), 1.0)
    if np.any(variance_raw < -1.0e-12 * scale):
        raise FloatingPointError("GfBm_EM estimated variance is materially negative")
    variance = np.maximum(variance_raw, 0.0)
    covariance = cross - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    np.fill_diagonal(covariance, variance)

    alpha = 1.0 - controls.familywise_confidence
    critical = float(
        student_t.ppf(
            1.0 - alpha / (2.0 * controls.family_size),
            df=len(seeds) - 1,
        )
    )
    if not np.isfinite(critical):
        raise FloatingPointError("simultaneous critical value is non-finite")
    mean_se = _simultaneous_standard_error(mean_replicates)
    second_se = _simultaneous_standard_error(second_replicates)
    cross_se = _simultaneous_standard_error(cross_replicates)
    variance_influence = second_replicates - 2.0 * mean * mean_replicates
    covariance_influence = (
        cross_replicates
        - mean_replicates[:, :, None] * mean[None, None, :]
        - mean[None, :, None] * mean_replicates[:, None, :]
    )
    variance_se = _simultaneous_standard_error(variance_influence)
    covariance_se = _simultaneous_standard_error(covariance_influence)

    terminal_weights = np.zeros(reporting_times.size, dtype=float)
    terminal_weights[-1] = 1.0
    intervals = np.diff(reporting_times)
    duration = float(reporting_times[-1] - reporting_times[0])
    if (
        reporting_times.ndim != 1
        or reporting_times.size < 2
        or reporting_times[0] != 0.0
        or duration <= 0.0
        or np.any(intervals <= 0.0)
    ):
        raise ValueError("reporting times must increase from zero")
    time_average_weights = np.zeros(reporting_times.size, dtype=float)
    time_average_weights[:-1] += 0.5 * intervals / duration
    time_average_weights[1:] += 0.5 * intervals / duration
    terminal = _scalar_functional_estimate(
        "terminal_value",
        terminal_weights,
        mean_replicates,
        cross_replicates,
        critical,
    )
    time_average = _scalar_functional_estimate(
        "trapezoidal_time_average",
        time_average_weights,
        mean_replicates,
        cross_replicates,
        critical,
    )

    digest = hashlib.sha256()
    metadata = {
        "method": "gfbm_em_explicit_euler_owen_scrambled_sobol_v1",
        "sample_count": int(sample_count),
        "seeds": list(seeds),
        "confidence": controls.familywise_confidence,
        "family_size": controls.family_size,
        "source_dimension": controls.source_dimension,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for array in (
        reporting_times,
        mean_replicates,
        second_replicates,
        cross_replicates,
    ):
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))

    return GfbmEmMomentEstimate(
        reporting_times=_readonly(reporting_times),
        sample_count_per_scramble=int(sample_count),
        scramble_seeds=seeds,
        mean=_readonly(mean),
        second_moment=_readonly(second),
        variance=_readonly(variance),
        cross_moment=_readonly(cross),
        covariance=_readonly(covariance),
        mean_half_width=_readonly(critical * mean_se),
        second_half_width=_readonly(critical * second_se),
        variance_half_width=_readonly(critical * variance_se),
        cross_half_width=_readonly(critical * cross_se),
        covariance_half_width=_readonly(critical * covariance_se),
        mean_replicates=_readonly(mean_replicates),
        second_replicates=_readonly(second_replicates),
        cross_replicates=_readonly(cross_replicates),
        critical_value=critical,
        familywise_confidence=controls.familywise_confidence,
        family_size=controls.family_size,
        terminal=terminal,
        trapezoidal_time_average=time_average,
        fingerprint=digest.hexdigest(),
    )


def estimate_gfbm_em_rqmc(
    factor: Any,
    case: GfbmEmCase,
    reporting_indices: np.ndarray,
    controls: GfbmEmRqmcControls,
) -> GfbmEmNestedRqmcResult:
    """Estimate nested-Q moment and covariance outputs for GfBm_EM.

    The factor must be a full-rank standard-fBM covariance factor on a
    uniform grid.  Factor construction time is caller-owned and must be
    charged separately in the end-to-end cost gate.
    """

    steps, horizon = _standard_grid_from_factor(factor)
    if controls.source_dimension < steps:
        raise ValueError("source_dimension must cover every fBM factor coordinate")
    selected = _reporting_indices(reporting_indices, steps)
    times = selected * (horizon / steps)
    seeds = tuple(int(seed) for seed in controls.scramble_seeds)
    replicates = len(seeds)
    reports = selected.size
    coarse_mean = np.empty((replicates, reports), dtype=float)
    coarse_second = np.empty_like(coarse_mean)
    coarse_cross = np.empty((replicates, reports, reports), dtype=float)
    fine_mean = np.empty_like(coarse_mean)
    fine_second = np.empty_like(coarse_mean)
    fine_cross = np.empty_like(coarse_cross)

    diagnostics = _zero_diagnostics()
    normal_seconds = 0.0
    path_seconds = 0.0
    euler_seconds = 0.0
    reduction_seconds = 0.0
    peak_batch_bytes = 0
    total_started = perf_counter()

    for replicate, seed in enumerate(seeds):
        engine = qmc.Sobol(
            d=controls.source_dimension,
            scramble=True,
            seed=seed,
        )
        sum_first = np.zeros(reports, dtype=float)
        sum_second = np.zeros(reports, dtype=float)
        sum_cross = np.zeros((reports, reports), dtype=float)
        completed = 0
        while completed < controls.fine_sample_count:
            boundary = (
                controls.coarse_sample_count
                if completed < controls.coarse_sample_count
                else controls.fine_sample_count
            )
            count = min(controls.batch_size, boundary - completed)

            started = perf_counter()
            source_nodes = _normal_nodes(engine.random(count))
            normal_seconds += perf_counter() - started

            started = perf_counter()
            nodes = source_nodes[:, :steps]
            positive_path = nodes @ np.asarray(factor.loadings, dtype=float).T
            paths = np.concatenate(
                (np.zeros((count, 1), dtype=float), positive_path), axis=1
            )
            path_seconds += perf_counter() - started

            started = perf_counter()
            curves, batch_diagnostics = gfbm_em_curve_batch(
                paths, factor.hurst, horizon, case, selected
            )
            euler_seconds += perf_counter() - started
            diagnostics = diagnostics + batch_diagnostics
            if not batch_diagnostics.classification.startswith("PASS"):
                raise GfbmEmNumericalFailure(diagnostics)

            started = perf_counter()
            sum_first += np.sum(curves, axis=0)
            sum_second += np.sum(curves * curves, axis=0)
            sum_cross += curves.T @ curves
            reduction_seconds += perf_counter() - started
            completed += count
            peak_batch_bytes = max(
                peak_batch_bytes,
                int(
                    source_nodes.nbytes
                    + positive_path.nbytes
                    + paths.nbytes
                    + curves.nbytes
                ),
            )

            if completed == controls.coarse_sample_count:
                coarse_mean[replicate] = sum_first / completed
                coarse_second[replicate] = sum_second / completed
                coarse_cross[replicate] = sum_cross / completed

        fine_mean[replicate] = sum_first / completed
        fine_second[replicate] = sum_second / completed
        fine_cross[replicate] = sum_cross / completed

    started = perf_counter()
    coarse = _moment_estimate(
        times,
        controls.coarse_sample_count,
        seeds,
        coarse_mean,
        coarse_second,
        coarse_cross,
        controls,
    )
    fine = _moment_estimate(
        times,
        controls.fine_sample_count,
        seeds,
        fine_mean,
        fine_second,
        fine_cross,
        controls,
    )
    reduction_seconds += perf_counter() - started
    runtime = GfbmEmRuntimeBreakdown(
        total_seconds=float(perf_counter() - total_started),
        normal_node_seconds=float(normal_seconds),
        path_generation_seconds=float(path_seconds),
        euler_seconds=float(euler_seconds),
        reduction_seconds=float(reduction_seconds),
        peak_batch_working_bytes=peak_batch_bytes,
    )
    return GfbmEmNestedRqmcResult(
        coarse=coarse,
        fine=fine,
        diagnostics=diagnostics,
        runtime=runtime,
        factor_fingerprint=str(factor.fingerprint),
        source_dimension=controls.source_dimension,
        sobol_coordinates_generated=(
            replicates * controls.fine_sample_count * controls.source_dimension
        ),
        case_code=case.code,
        convergence_scope=case.convergence_scope,
    )


def baseline_freeze() -> dict[str, Any]:
    """Return the exact prospective ladder and claim boundary."""

    return {
        "protocol": "paper_ix_gfbm_em_rqmc_baseline_v1",
        "source": {
            "arxiv_id": ARXIV_ID,
            "official_commit": OFFICIAL_COMMIT,
            "official_raw_sha256": OFFICIAL_RAW_SHA256,
        },
        "time_step_ladder": list(TIME_STEP_LADDER),
        "rqmc_exponent_ladder": list(RQMC_EXPONENT_LADDER),
        "sobol_source_dimension": max(TIME_STEP_LADDER),
        "coupling_rule": (
            "FIX_SOURCE_DIMENSION_AT_512;USE_THE_FIRST_n_COORDINATES;"
            "SAME_SEEDS_AND_NESTED_Q_PREFIXES_AT_EVERY_n"
        ),
        "scramble_seeds": list(SCRAMBLE_SEEDS),
        "reporting_count": 9,
        "mandatory_outputs": [
            "mean_curve",
            "second_moment_curve",
            "variance_curve",
            "full_cross_moment_matrix",
            "full_covariance_matrix",
            "terminal_moments",
            "trapezoidal_time_average_moments",
        ],
        "finite_target_tolerance": FINITE_TARGET_TOLERANCE,
        "selection_rule": (
            "FIRST_JOINT_N_AND_Q_CONFIGURATION_PASSING_ACCURACY;"
            "CAP_EXHAUSTION_IS_INCONCLUSIVE"
        ),
        "timing_rule": (
            "ONLY_ACCURACY_PASSING_CONFIGURATIONS;ONE_THREAD;"
            "TWO_WARMUPS;SEVEN_TIMED_REPETITIONS"
        ),
        "memory_rule": (
            "REPORT_PEAK_PROCESS_RSS_AND_PEAK_BATCH_WORKING_BYTES;"
            "FACTOR_CONSTRUCTION_CHARGED_TO_END_TO_END_TOTAL"
        ),
        "cost_counter_law_per_scramble": {
            "trajectories": "Q",
            "target_solves": "Q*R",
            "drift_evaluations": "Q*SUM(k_j,j=1..R)",
            "state_updates": "Q*SUM(k_j,j=1..R)",
            "multiplier_exponentials": "Q*(SUM(k_j,j=1..R)+R)",
            "gaussian_coordinates": "Q*n",
            "sobol_coordinates_generated": "Q*d_source",
            "unique_cross_moment_accumulations": "Q*(R+1)*(R+2)/2",
            "symbols": (
                "Q=paths_per_scramble;n=time_steps;d_source=512;"
                "k_j=nonzero_reporting_indices;R=nonzero_reporting_count"
            ),
        },
        "claim_boundary": {
            "rational_published_strong_order": "H",
            "landau_published_strong_order": None,
            "empirical_rate_is_not_a_theorem": True,
            "runtime_superiority_requires_matched_accuracy": True,
        },
    }


__all__ = [
    "ARXIV_ID",
    "FINITE_TARGET_TOLERANCE",
    "GfbmEmBatchDiagnostics",
    "GfbmEmCase",
    "GfbmEmMomentEstimate",
    "GfbmEmNestedRqmcResult",
    "GfbmEmNumericalFailure",
    "GfbmEmRqmcControls",
    "GfbmEmRuntimeBreakdown",
    "GfbmEmScalarFunctionalEstimate",
    "OFFICIAL_COMMIT",
    "OFFICIAL_RAW_SHA256",
    "RQMC_EXPONENT_LADDER",
    "SCRAMBLE_SEEDS",
    "TIME_STEP_LADDER",
    "baseline_freeze",
    "estimate_gfbm_em_rqmc",
    "geometric_case",
    "gfbm_em_curve_batch",
    "landau_case",
    "rational_case",
]
