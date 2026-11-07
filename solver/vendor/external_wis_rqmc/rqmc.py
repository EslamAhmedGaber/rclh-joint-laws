"""Deterministic scrambled-Sobol integration for translated WIS trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t

from .cases import BenchmarkCase
from .factor import NestedFbmFactor
from .translated_ode import (
    OdeControls,
    OdeDiagnostics,
    solve_curve_samples,
)


def _readonly(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class RqmcControls:
    """Nested two-size RQMC controls for independent scrambles."""

    fine_exponent: int
    scramble_seeds: tuple[int, ...]
    batch_size: int
    source_dimension: int
    familywise_confidence: float = 0.99
    family_size: int = 1

    def __post_init__(self) -> None:
        if int(self.fine_exponent) < 1:
            raise ValueError("fine_exponent must be at least one")
        if len(self.scramble_seeds) < 2:
            raise ValueError("at least two independent scrambles are required")
        if len(set(int(seed) for seed in self.scramble_seeds)) != len(
            self.scramble_seeds
        ):
            raise ValueError("scramble seeds must be distinct")
        if any(int(seed) < 0 or int(seed) >= 2**32 for seed in self.scramble_seeds):
            raise ValueError("scramble seeds must lie in [0, 2**32)")
        selected_batch = int(self.batch_size)
        if selected_batch < 1 or selected_batch & (selected_batch - 1):
            raise ValueError("batch_size must be a positive power of two")
        if selected_batch > 2 ** (int(self.fine_exponent) - 1):
            raise ValueError("batch_size may not exceed the coarse sample size")
        if int(self.source_dimension) < 1:
            raise ValueError("source_dimension must be positive")
        if not 0.0 < float(self.familywise_confidence) < 1.0:
            raise ValueError("familywise_confidence must lie in (0, 1)")
        if int(self.family_size) < 1:
            raise ValueError("family_size must be positive")

    @property
    def coarse_sample_count(self) -> int:
        return 2 ** (int(self.fine_exponent) - 1)

    @property
    def fine_sample_count(self) -> int:
        return 2 ** int(self.fine_exponent)


@dataclass(frozen=True)
class MomentEstimate:
    """Moment curves and replicate-based simultaneous uncertainty."""

    reporting_times: np.ndarray
    sample_count_per_scramble: int
    scramble_seeds: tuple[int, ...]
    mean: np.ndarray
    second_moment: np.ndarray
    variance: np.ndarray
    mean_standard_error: np.ndarray
    second_standard_error: np.ndarray
    variance_standard_error: np.ndarray
    mean_half_width: np.ndarray
    second_half_width: np.ndarray
    variance_half_width: np.ndarray
    mean_replicates: np.ndarray
    second_replicates: np.ndarray
    critical_value: float
    familywise_confidence: float
    family_size: int
    fingerprint: str

    def quantity(self, name: str) -> np.ndarray:
        if name == "mean":
            return self.mean
        if name == "second_moment":
            return self.second_moment
        if name == "variance":
            return self.variance
        raise ValueError("quantity must be mean, second_moment, or variance")

    def half_width(self, name: str) -> np.ndarray:
        if name == "mean":
            return self.mean_half_width
        if name == "second_moment":
            return self.second_half_width
        if name == "variance":
            return self.variance_half_width
        raise ValueError("quantity must be mean, second_moment, or variance")

    def to_jsonable(self) -> dict:
        return {
            "reporting_times": self.reporting_times.tolist(),
            "sample_count_per_scramble": self.sample_count_per_scramble,
            "scramble_seeds": list(self.scramble_seeds),
            "mean": self.mean.tolist(),
            "second_moment": self.second_moment.tolist(),
            "variance": self.variance.tolist(),
            "mean_standard_error": self.mean_standard_error.tolist(),
            "second_standard_error": self.second_standard_error.tolist(),
            "variance_standard_error": self.variance_standard_error.tolist(),
            "mean_half_width": self.mean_half_width.tolist(),
            "second_half_width": self.second_half_width.tolist(),
            "variance_half_width": self.variance_half_width.tolist(),
            "mean_replicates": self.mean_replicates.tolist(),
            "second_replicates": self.second_replicates.tolist(),
            "critical_value": self.critical_value,
            "familywise_confidence": self.familywise_confidence,
            "family_size": self.family_size,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class NestedRqmcResult:
    coarse: MomentEstimate
    fine: MomentEstimate
    diagnostics: OdeDiagnostics
    factor_fingerprint: str
    source_dimension: int


def moment_estimate_from_jsonable(payload: dict) -> MomentEstimate:
    """Reconstruct and fingerprint-check a serialized moment estimate."""

    required = {
        "reporting_times",
        "sample_count_per_scramble",
        "scramble_seeds",
        "mean",
        "second_moment",
        "variance",
        "mean_standard_error",
        "second_standard_error",
        "variance_standard_error",
        "mean_half_width",
        "second_half_width",
        "variance_half_width",
        "mean_replicates",
        "second_replicates",
        "critical_value",
        "familywise_confidence",
        "family_size",
        "fingerprint",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            f"serialized moment estimate schema mismatch; missing={missing}, extra={extra}"
        )
    times = np.asarray(payload["reporting_times"], dtype=float)
    seeds = tuple(int(seed) for seed in payload["scramble_seeds"])
    reconstructed = _moment_estimate(
        times,
        int(payload["sample_count_per_scramble"]),
        seeds,
        np.asarray(payload["mean_replicates"], dtype=float),
        np.asarray(payload["second_replicates"], dtype=float),
        confidence=float(payload["familywise_confidence"]),
        family_size=int(payload["family_size"]),
    )
    if reconstructed.fingerprint != payload["fingerprint"]:
        raise ValueError("serialized moment estimate fingerprint mismatch")
    checks = {
        "mean": reconstructed.mean,
        "second_moment": reconstructed.second_moment,
        "variance": reconstructed.variance,
        "mean_standard_error": reconstructed.mean_standard_error,
        "second_standard_error": reconstructed.second_standard_error,
        "variance_standard_error": reconstructed.variance_standard_error,
        "mean_half_width": reconstructed.mean_half_width,
        "second_half_width": reconstructed.second_half_width,
        "variance_half_width": reconstructed.variance_half_width,
    }
    for name, expected in checks.items():
        observed = np.asarray(payload[name], dtype=float)
        if not np.array_equal(observed, expected):
            raise ValueError(f"serialized moment field does not reproduce: {name}")
    if float(payload["critical_value"]) != reconstructed.critical_value:
        raise ValueError("serialized critical value does not reproduce")
    return reconstructed


def _moment_estimate(
    reporting_times: np.ndarray,
    sample_count: int,
    seeds: tuple[int, ...],
    mean_replicates: np.ndarray,
    second_replicates: np.ndarray,
    *,
    confidence: float,
    family_size: int,
) -> MomentEstimate:
    replicates = len(seeds)
    if mean_replicates.shape != second_replicates.shape:
        raise ValueError("replicate arrays must have identical shape")
    if mean_replicates.shape != (replicates, reporting_times.size):
        raise ValueError("replicate arrays have incompatible shape")
    mean = np.mean(mean_replicates, axis=0)
    second = np.mean(second_replicates, axis=0)
    variance = second - mean * mean
    if np.any(variance < -1.0e-12 * np.maximum(np.abs(second), 1.0)):
        raise FloatingPointError("estimated variance is materially negative")
    variance = np.maximum(variance, 0.0)
    mean_se = np.std(mean_replicates, axis=0, ddof=1) / np.sqrt(replicates)
    second_se = np.std(second_replicates, axis=0, ddof=1) / np.sqrt(
        replicates
    )
    variance_influence = second_replicates - 2.0 * mean * mean_replicates
    variance_se = np.std(variance_influence, axis=0, ddof=1) / np.sqrt(
        replicates
    )
    alpha = 1.0 - confidence
    critical = float(
        student_t.ppf(
            1.0 - alpha / (2.0 * family_size),
            df=replicates - 1,
        )
    )
    if not np.isfinite(critical):
        raise FloatingPointError("simultaneous critical value is non-finite")

    digest = hashlib.sha256()
    metadata = {
        "method": "owen_scrambled_sobol_normal_v1",
        "sample_count": int(sample_count),
        "seeds": list(seeds),
        "confidence": float(confidence),
        "family_size": int(family_size),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for values in (reporting_times, mean_replicates, second_replicates):
        digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))

    return MomentEstimate(
        reporting_times=_readonly(reporting_times),
        sample_count_per_scramble=int(sample_count),
        scramble_seeds=tuple(int(seed) for seed in seeds),
        mean=_readonly(mean),
        second_moment=_readonly(second),
        variance=_readonly(variance),
        mean_standard_error=_readonly(mean_se),
        second_standard_error=_readonly(second_se),
        variance_standard_error=_readonly(variance_se),
        mean_half_width=_readonly(critical * mean_se),
        second_half_width=_readonly(critical * second_se),
        variance_half_width=_readonly(critical * variance_se),
        mean_replicates=_readonly(mean_replicates),
        second_replicates=_readonly(second_replicates),
        critical_value=critical,
        familywise_confidence=float(confidence),
        family_size=int(family_size),
        fingerprint=digest.hexdigest(),
    )


def _normal_nodes(uniforms: np.ndarray) -> np.ndarray:
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    nodes = ndtri(np.clip(uniforms, lower, upper))
    if np.any(~np.isfinite(nodes)):
        raise FloatingPointError("Sobol-to-normal transform produced non-finite nodes")
    return nodes


def estimate_nested_moment_curves(
    factor: NestedFbmFactor,
    case: BenchmarkCase,
    reporting_times: np.ndarray,
    ode_controls: OdeControls,
    rqmc_controls: RqmcControls,
) -> NestedRqmcResult:
    """Estimate moment curves at nested ``2**(m-1)`` and ``2**m`` sizes.

    A Sobol engine is built in ``source_dimension`` and only the first
    ``factor.modes`` coordinates are consumed.  Using the same source
    dimension and seeds for two nested factors therefore supplies explicit
    common-random-number coupling across factor refinements.
    """

    times = np.asarray(reporting_times, dtype=float)
    if (
        times.ndim != 1
        or times.size < 1
        or np.any(np.diff(times) <= 0.0)
        or times[0] < 0.0
        or times[-1] > factor.horizon
    ):
        raise ValueError("reporting_times must increase inside [0, T]")
    if rqmc_controls.source_dimension < factor.modes:
        raise ValueError("source_dimension must cover all factor modes")

    seeds = tuple(int(seed) for seed in rqmc_controls.scramble_seeds)
    coarse_means = np.empty((len(seeds), times.size), dtype=float)
    coarse_seconds = np.empty_like(coarse_means)
    fine_means = np.empty_like(coarse_means)
    fine_seconds = np.empty_like(coarse_means)
    total_diagnostics = OdeDiagnostics(0, 0, 0)

    for replicate, seed in enumerate(seeds):
        engine = qmc.Sobol(
            d=int(rqmc_controls.source_dimension),
            scramble=True,
            seed=seed,
        )
        sum_first = np.zeros(times.size, dtype=float)
        sum_second = np.zeros(times.size, dtype=float)
        completed = 0
        while completed < rqmc_controls.fine_sample_count:
            next_boundary = (
                rqmc_controls.coarse_sample_count
                if completed < rqmc_controls.coarse_sample_count
                else rqmc_controls.fine_sample_count
            )
            count = min(
                int(rqmc_controls.batch_size),
                next_boundary - completed,
            )
            uniforms = engine.random(count)
            nodes = _normal_nodes(uniforms)[:, : factor.modes]
            curves, diagnostics = solve_curve_samples(
                factor, nodes, case, times, ode_controls
            )
            total_diagnostics = total_diagnostics + diagnostics
            sum_first += np.sum(curves, axis=0)
            sum_second += np.sum(curves * curves, axis=0)
            completed += count
            if completed == rqmc_controls.coarse_sample_count:
                coarse_means[replicate] = sum_first / completed
                coarse_seconds[replicate] = sum_second / completed
        fine_means[replicate] = sum_first / completed
        fine_seconds[replicate] = sum_second / completed

    coarse = _moment_estimate(
        times,
        rqmc_controls.coarse_sample_count,
        seeds,
        coarse_means,
        coarse_seconds,
        confidence=rqmc_controls.familywise_confidence,
        family_size=rqmc_controls.family_size,
    )
    fine = _moment_estimate(
        times,
        rqmc_controls.fine_sample_count,
        seeds,
        fine_means,
        fine_seconds,
        confidence=rqmc_controls.familywise_confidence,
        family_size=rqmc_controls.family_size,
    )
    return NestedRqmcResult(
        coarse=coarse,
        fine=fine,
        diagnostics=total_diagnostics,
        factor_fingerprint=factor.fingerprint,
        source_dimension=int(rqmc_controls.source_dimension),
    )


__all__ = [
    "MomentEstimate",
    "NestedRqmcResult",
    "RqmcControls",
    "estimate_nested_moment_curves",
    "moment_estimate_from_jsonable",
]
