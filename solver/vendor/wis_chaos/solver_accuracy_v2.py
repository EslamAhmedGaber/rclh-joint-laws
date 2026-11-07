"""Numerical producer primitives for the solver-accuracy V2 gate.

This module is an additive layer over the officially equivalent fused solver.
It does not alter any fingerprinted solver source.  Its central distinction is
between the retained physical chaos vector and the untruncated coherent output
of the finite Lawson polynomial.  External moment comparisons must use the
latter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from .basis import HermiteBasis, quadratic_tail_core_enriched_index_set
from .coherent import coherent_reconstruction
from .fbm import GaussianPathFactor, endpoint_conditioned_factor
from .fused_spectral_solver import FusedSpectralDeterministicWISChaosSolver
from .operator_support import operator_geometry_triad_index_set
from .solver import SolverResult
from .spectral_projection import (
    AffineSpectralDrift,
    CosineTransform,
    LogQuadraticCosineTransform,
    RationalSineTransform,
)


REPORTING_INTERVALS = 32
MOMENT_SEMANTICS = "coherent_full_output_v1"
SUPPORT_PROFILES = (
    "quadratic_core4",
    "quadratic_core6",
    "operator_triad_core6",
)


@dataclass(frozen=True, slots=True)
class SolverProblemV2:
    code: str
    initial_value: float
    beta: float
    drift: AffineSpectralDrift


@dataclass(frozen=True, slots=True)
class ProjectorControlsV2:
    core_order: int
    cutoff: float
    absolute_tolerance: float
    relative_tolerance: float
    maximum_panels: int
    frequency_chunk_size: int = 32
    refinement_batch_size: int = 1

    def __post_init__(self) -> None:
        integer_values = (
            int(self.core_order),
            int(self.maximum_panels),
            int(self.frequency_chunk_size),
            int(self.refinement_batch_size),
        )
        floating_values = np.asarray(
            (
                self.cutoff,
                self.absolute_tolerance,
                self.relative_tolerance,
            ),
            dtype=float,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("projector integer controls must be positive")
        if not np.all(np.isfinite(floating_values)) or np.any(floating_values <= 0.0):
            raise ValueError("projector floating controls must be finite and positive")

    def to_jsonable(self) -> dict[str, int | float]:
        return {
            "core_order": int(self.core_order),
            "cutoff": float(self.cutoff),
            "absolute_tolerance": float(self.absolute_tolerance),
            "relative_tolerance": float(self.relative_tolerance),
            "maximum_panels": int(self.maximum_panels),
            "frequency_chunk_size": int(self.frequency_chunk_size),
            "refinement_batch_size": int(self.refinement_batch_size),
        }


@dataclass(frozen=True, slots=True)
class SolverAccuracyRunV2:
    problem_code: str
    hurst: float
    dimension: int
    support_profile: str
    support_fingerprint: str
    factor_fingerprint: str
    projector_controls: Mapping[str, int | float]
    drift_substeps: int
    times: np.ndarray
    support: np.ndarray
    coefficients: np.ndarray
    current_means: np.ndarray
    current_second_moments: np.ndarray
    coherent_means: np.ndarray
    coherent_second_moments: np.ndarray
    coherent_variances: np.ndarray
    tau_squared: np.ndarray
    projection_residuals: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "times",
            "support",
            "coefficients",
            "current_means",
            "current_second_moments",
            "coherent_means",
            "coherent_second_moments",
            "coherent_variances",
            "tau_squared",
            "projection_residuals",
        ):
            values = np.asarray(getattr(self, name)).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def coefficient_count(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def grid_steps(self) -> int:
        """Number of piecewise-linear fBM/time cells in this run."""

        return int(self.times.size - 1)

    @property
    def reporting_indices(self) -> np.ndarray:
        if self.dimension % REPORTING_INTERVALS:
            raise ValueError("dimension must be divisible by 32 for V2 reporting")
        stride = self.dimension // REPORTING_INTERVALS
        return np.arange(0, self.dimension + 1, stride, dtype=np.int64)

    def reporting_moments(self) -> dict[str, np.ndarray]:
        indices = self.reporting_indices
        return {
            "reporting_times": self.times[indices],
            "mean": self.coherent_means[indices],
            "second_moment": self.coherent_second_moments[indices],
            "variance": self.coherent_variances[indices],
        }

    def moment_curves_at(self, reporting_times: np.ndarray) -> dict[str, np.ndarray]:
        """Return coherent moment curves on fixed nodes by linear interpolation.

        Official target output uses a grid that contains all 33 reporting
        nodes, so no interpolation is used for the final payload.  This
        helper exists for the coarser members of a factor/grid refinement
        ladder.  Their interpolation error is deliberately part of the
        observed coarse-to-fine shift.
        """

        nodes = np.asarray(reporting_times, dtype=float)
        if (
            nodes.ndim != 1
            or nodes.size < 2
            or not np.all(np.isfinite(nodes))
            or np.any(np.diff(nodes) <= 0.0)
            or nodes[0] < self.times[0]
            or nodes[-1] > self.times[-1]
        ):
            raise ValueError("reporting nodes must be finite, ordered, and in range")
        return {
            "reporting_times": nodes.copy(),
            "mean": np.interp(nodes, self.times, self.coherent_means),
            "second_moment": np.interp(
                nodes, self.times, self.coherent_second_moments
            ),
            "variance": np.interp(nodes, self.times, self.coherent_variances),
        }


def external_problem_v2(code: str) -> SolverProblemV2:
    """Return an exact R/C/L/L1 problem used by the external V2 comparator."""

    selected = str(code)
    if selected == "R":
        return SolverProblemV2(
            "R",
            1.0,
            1.0,
            AffineSpectralDrift(
                intercept=0.0,
                rate=1.0,
                transforms=(RationalSineTransform(4.0, width=1.0),),
            ),
        )
    if selected == "C":
        return SolverProblemV2(
            "C",
            10.0,
            0.5,
            AffineSpectralDrift(
                intercept=0.0,
                rate=-1.0,
                transforms=(CosineTransform(1.0, frequency=1.0),),
            ),
        )
    if selected == "L":
        return SolverProblemV2(
            "L",
            25.0,
            5.0,
            AffineSpectralDrift(
                intercept=0.0,
                rate=0.0,
                transforms=(
                    LogQuadraticCosineTransform(25.0, width=1.0),
                ),
            ),
        )
    if selected == "L1":
        return SolverProblemV2(
            "L1",
            1.0,
            1.0,
            AffineSpectralDrift(
                intercept=0.0,
                rate=0.0,
                transforms=(LogQuadraticCosineTransform(1.0, width=1.0),),
            ),
        )
    raise ValueError("external V2 problem code must be R, C, L, or L1")


def factor_fingerprint_v2(factor: GaussianPathFactor) -> str:
    digest = hashlib.sha256()
    metadata = {
        "hurst": float(factor.hurst),
        "method": factor.method,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (
        ("times", factor.times),
        ("loadings", factor.loadings),
        ("covariance", factor.covariance),
        ("residual", factor.residual),
    ):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def support_fingerprint_v2(support: tuple[tuple[int, ...], ...]) -> str:
    array = np.ascontiguousarray(support, dtype="<i2")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_factor_v2(dimension: int, hurst: float) -> GaussianPathFactor:
    selected_dimension = int(dimension)
    exponent = float(hurst)
    if selected_dimension < 4:
        raise ValueError("solver-accuracy dimension must be at least four")
    if not 0.0 < exponent < 1.0:
        raise ValueError("standard fBM requires 0 < H < 1")
    times = np.linspace(1.0 / selected_dimension, 1.0, selected_dimension)
    factor = endpoint_conditioned_factor(
        times,
        exponent,
        modes=selected_dimension,
    )
    if factor.residual_max_variance_fraction > 2e-10:
        raise FloatingPointError("full solver factor retains material residual covariance")
    return factor


def build_support_v2(
    factor: GaussianPathFactor,
    profile: str,
    *,
    core_modes: int = 3,
    triad_multiplier: int = 2,
) -> tuple[tuple[int, ...], ...]:
    selected = str(profile)
    if selected not in SUPPORT_PROFILES:
        raise ValueError("unknown V2 support profile")
    core = int(core_modes)
    if not 1 <= core < factor.modes:
        raise ValueError("core_modes must lie inside the factor dimension")
    core_degree = 4 if selected == "quadratic_core4" else 6
    if selected == "operator_triad_core6":
        return operator_geometry_triad_index_set(
            factor,
            core_modes=core,
            core_degree=core_degree,
            triad_count=int(triad_multiplier) * factor.modes,
            triad_core_coordinate=0,
        )
    return quadratic_tail_core_enriched_index_set(
        dimension=factor.modes,
        core_modes=core,
        core_degree=core_degree,
        cross_degree=2,
        tail_pair_core_modes=0,
        tail_pair_core_degree=0,
    )


def _coherent_curves(
    basis: HermiteBasis,
    factor: GaussianPathFactor,
    beta: float,
    result: SolverResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = result.times.size
    means = np.empty(size, dtype=float)
    seconds = np.empty(size, dtype=float)
    variances = np.empty(size, dtype=float)
    omitted = np.empty(size, dtype=float)
    zero = np.zeros(factor.modes, dtype=float)
    for index in range(size):
        loading = zero if index == 0 else float(beta) * factor.loadings[index - 1]
        reconstruction = coherent_reconstruction(
            basis,
            result.coefficients[index],
            loading,
        )
        means[index] = reconstruction.coherent_mean
        seconds[index] = reconstruction.coherent_second_moment
        variances[index] = reconstruction.coherent_variance
        omitted[index] = reconstruction.tau_squared
    return means, seconds, variances, omitted


def solve_accuracy_run_v2(
    problem: SolverProblemV2,
    hurst: float,
    dimension: int,
    *,
    support_profile: str,
    projector: ProjectorControlsV2,
    drift_substeps: int,
    core_modes: int = 3,
    triad_multiplier: int = 2,
) -> SolverAccuracyRunV2:
    """Run one complete coefficient trajectory and reconstruct coherent moments."""

    if not isinstance(problem, SolverProblemV2):
        raise TypeError("problem must be SolverProblemV2")
    substeps = int(drift_substeps)
    if substeps < 1:
        raise ValueError("drift_substeps must be positive")
    factor = build_factor_v2(dimension, hurst)
    support = build_support_v2(
        factor,
        support_profile,
        core_modes=core_modes,
        triad_multiplier=triad_multiplier,
    )
    solver = FusedSpectralDeterministicWISChaosSolver(
        factor,
        degree=6,
        beta=problem.beta,
        drift=problem.drift,
        structured_core_dimensions=int(core_modes),
        core_quadrature_order=int(projector.core_order),
        transform_cutoff=float(projector.cutoff),
        transform_absolute_tolerance=float(projector.absolute_tolerance),
        transform_relative_tolerance=float(projector.relative_tolerance),
        transform_max_panels=int(projector.maximum_panels),
        drift_substeps=substeps,
        nonlinear_propagator="integrating_factor",
        index_set=support,
        refinement_batch_size=int(projector.refinement_batch_size),
        frequency_chunk_size=int(projector.frequency_chunk_size),
    )
    result = solver.solve(problem.initial_value)
    coherent_mean, coherent_second, coherent_variance, omitted = _coherent_curves(
        solver.basis,
        factor,
        problem.beta,
        result,
    )
    expected_times = np.linspace(0.0, 1.0, int(dimension) + 1)
    arrays = (
        result.coefficients,
        result.means,
        result.second_moments,
        coherent_mean,
        coherent_second,
        coherent_variance,
        omitted,
        result.projection_residuals,
    )
    if not np.array_equal(result.times, expected_times):
        raise RuntimeError("solver returned an unexpected time grid")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise FloatingPointError("solver-accuracy run produced nonfinite data")
    if not np.allclose(result.means, coherent_mean, rtol=2e-12, atol=2e-12):
        raise FloatingPointError("current and coherent means are inconsistent")
    if not np.allclose(
        result.second_moments + omitted,
        coherent_second,
        rtol=5e-12,
        atol=5e-12,
    ):
        raise FloatingPointError("coherent energy identity failed")
    if np.any(coherent_variance < 0.0) or np.any(omitted < 0.0):
        raise FloatingPointError("coherent variance or omitted energy is negative")
    return SolverAccuracyRunV2(
        problem_code=problem.code,
        hurst=float(hurst),
        dimension=int(dimension),
        support_profile=str(support_profile),
        support_fingerprint=support_fingerprint_v2(support),
        factor_fingerprint=factor_fingerprint_v2(factor),
        projector_controls=projector.to_jsonable(),
        drift_substeps=substeps,
        times=result.times,
        support=np.asarray(support, dtype=np.int16),
        coefficients=result.coefficients,
        current_means=result.means,
        current_second_moments=result.second_moments,
        coherent_means=coherent_mean,
        coherent_second_moments=coherent_second,
        coherent_variances=coherent_variance,
        tau_squared=omitted,
        projection_residuals=result.projection_residuals,
    )


__all__ = [
    "MOMENT_SEMANTICS",
    "ProjectorControlsV2",
    "REPORTING_INTERVALS",
    "SUPPORT_PROFILES",
    "SolverAccuracyRunV2",
    "SolverProblemV2",
    "build_factor_v2",
    "build_support_v2",
    "external_problem_v2",
    "factor_fingerprint_v2",
    "solve_accuracy_run_v2",
    "support_fingerprint_v2",
]
