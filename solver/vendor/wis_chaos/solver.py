"""Deterministic finite-mode, finite-chaos WIS coefficient propagator."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable

import numpy as np

from .basis import (
    HermiteBasis,
    HybridCoreGaussHermiteSobol,
    MatrixFreeSobolHermiteProjection,
    SobolHermiteLeastSquares,
    StructuredQuadraticTailSobolProjection,
    StructuredRationalLaguerreProjection,
    TensorGaussHermite,
    gaussian_moments,
)
from .fbm import GaussianPathFactor
from .modulated_rational import CoreModulatedRationalLaguerreProjection


ArrayFunction = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class AffineRationalDrift:
    """Immutable drift specification for the rational projection backends.

    The callable evaluates
    ``intercept + rate*x + scale*x/(1+x*x)``.  Its named parameters are the
    source of truth used to validate the legacy projector controls, so the
    specialized backends cannot silently solve a different drift.
    """

    intercept: float
    rate: float
    scale: float

    def __post_init__(self) -> None:
        for name in ("intercept", "rate", "scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def __call__(
        self, _time: float, values: np.ndarray
    ) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        return (
            self.intercept
            + self.rate * state
            + self.scale * state / (1.0 + state * state)
        )


@dataclass(frozen=True)
class SolverResult:
    times: np.ndarray
    coefficients: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    second_moments: np.ndarray
    projection_residuals: np.ndarray


class DeterministicWISChaosSolver:
    """Strang propagator with an exact finite-chaos Wick diffusion cell.

    The solver is theorem-safe as a finite deterministic algorithm.  It does
    not by itself prove convergence of the Gaussian path representation as
    its dimension tends to infinity.
    """

    def __init__(
        self,
        factor: GaussianPathFactor,
        degree: int,
        beta: float,
        *,
        linear_drift_rate: float | None = None,
        drift: ArrayFunction | None = None,
        quadrature_order: int | None = None,
        drift_substeps: int = 1,
        nonlinear_propagator: str = "integrating_factor",
        projection_backend: str = "tensor",
        regression_nodes: int | None = None,
        regression_seed: int = 1729,
        index_set: tuple[tuple[int, ...], ...] | None = None,
        core_dimensions: int | None = None,
        core_quadrature_order: int | None = None,
        tail_regression_nodes: int | None = None,
        projection_control_variate_rate: float | None = None,
        projection_control_variate_intercept: float = 0.0,
        projection_block_size: int = 256,
        structured_core_dimensions: int | None = None,
        rational_residual_scale: float | None = None,
        laguerre_order: int = 24,
        modulation_core_coordinate: int = 0,
    ) -> None:
        if (linear_drift_rate is None) == (drift is None):
            raise ValueError(
                "provide exactly one of linear_drift_rate or nonlinear drift"
            )
        if drift_substeps < 1:
            raise ValueError("drift_substeps must be positive")
        if nonlinear_propagator not in {"integrating_factor", "strang"}:
            raise ValueError(
                "nonlinear_propagator must be 'integrating_factor' or 'strang'"
            )
        if projection_backend not in {
            "tensor",
            "sobol_ls",
            "sobol_cv_matrix_free",
            "structured_cv_matrix_free",
            "structured_rational_laguerre",
            "structured_modulated_rational_laguerre",
            "hybrid_core",
        }:
            raise ValueError(
                "projection_backend must be 'tensor', 'sobol_ls', "
                "'sobol_cv_matrix_free', 'structured_cv_matrix_free', "
                "'structured_rational_laguerre', "
                "'structured_modulated_rational_laguerre', or "
                "'hybrid_core'"
            )
        if (
            projection_backend
            != "structured_modulated_rational_laguerre"
            and int(modulation_core_coordinate) != 0
        ):
            raise ValueError(
                "nonzero modulation_core_coordinate is only used by "
                "structured_modulated_rational_laguerre"
            )
        if projection_backend in {
            "sobol_cv_matrix_free",
            "structured_cv_matrix_free",
        }:
            if projection_control_variate_rate is None:
                raise ValueError(
                    "matrix-free CV projection requires "
                    "projection_control_variate_rate"
                )
            if regression_nodes is None:
                raise ValueError(
                    "matrix-free CV projection requires regression_nodes"
                )
            if (
                projection_backend == "structured_cv_matrix_free"
                and structured_core_dimensions is None
            ):
                raise ValueError(
                    "structured_cv_matrix_free requires "
                    "structured_core_dimensions"
                )
        elif projection_backend in {
            "structured_rational_laguerre",
            "structured_modulated_rational_laguerre",
        }:
            if not isinstance(drift, AffineRationalDrift):
                raise ValueError(
                    "structured rational projection requires "
                    "drift=AffineRationalDrift(...)"
                )
            if projection_control_variate_rate is None:
                raise ValueError(
                    "structured rational projection requires "
                    "projection_control_variate_rate"
                )
            if structured_core_dimensions is None:
                raise ValueError(
                    "structured rational projection requires "
                    "structured_core_dimensions"
                )
            if core_quadrature_order is None:
                raise ValueError(
                    "structured rational projection requires "
                    "core_quadrature_order"
                )
            if rational_residual_scale is None:
                raise ValueError(
                    "structured rational projection requires "
                    "rational_residual_scale"
                )
            if laguerre_order < 1:
                raise ValueError("laguerre_order must be positive")
            if not np.isfinite(float(projection_control_variate_rate)):
                raise ValueError(
                    "projection_control_variate_rate must be finite"
                )
            if not np.isfinite(float(projection_control_variate_intercept)):
                raise ValueError(
                    "projection_control_variate_intercept must be finite"
                )
            if not np.isfinite(float(rational_residual_scale)):
                raise ValueError("rational_residual_scale must be finite")
            legacy_parameters = (
                (
                    "projection_control_variate_intercept",
                    float(projection_control_variate_intercept),
                    "intercept",
                    drift.intercept,
                ),
                (
                    "projection_control_variate_rate",
                    float(projection_control_variate_rate),
                    "rate",
                    drift.rate,
                ),
                (
                    "rational_residual_scale",
                    float(rational_residual_scale),
                    "scale",
                    drift.scale,
                ),
            )
            for legacy_name, legacy_value, field_name, field_value in (
                legacy_parameters
            ):
                if legacy_value != field_value:
                    raise ValueError(
                        f"{legacy_name} must exactly match "
                        f"drift.{field_name}: {legacy_value!r} != "
                        f"{field_value!r}"
                    )
        elif projection_control_variate_rate is not None:
            raise ValueError(
                "projection_control_variate_rate is only used by a "
                "matrix-free CV or structured rational backend"
            )
        elif float(projection_control_variate_intercept) != 0.0:
            raise ValueError(
                "projection_control_variate_intercept is only used by "
                "a matrix-free CV or structured rational backend"
            )
        self.factor = factor
        self.basis = HermiteBasis(
            factor.modes,
            int(degree),
            index_set=index_set,
        )
        self.beta = float(beta)
        self.linear_drift_rate = linear_drift_rate
        self.drift = drift
        self.drift_substeps = int(drift_substeps)
        self.nonlinear_propagator = nonlinear_propagator
        self.projection_backend = projection_backend
        self.projection_control_variate_rate = (
            None
            if projection_control_variate_rate is None
            else float(projection_control_variate_rate)
        )
        self.projection_control_variate_intercept = float(
            projection_control_variate_intercept
        )
        self.rational_residual_scale = (
            None
            if rational_residual_scale is None
            else float(rational_residual_scale)
        )
        self.modulation_core_coordinate = int(
            modulation_core_coordinate
        )

        self._nodes: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._basis_matrix: np.ndarray | None = None
        self._projection_operator: np.ndarray | None = None
        self._matrix_free_projection: (
            MatrixFreeSobolHermiteProjection
            | StructuredQuadraticTailSobolProjection
            | None
        ) = None
        self._rational_projection: (
            StructuredRationalLaguerreProjection
            | CoreModulatedRationalLaguerreProjection
            | None
        ) = None
        self.projection_gram_error = 0.0
        self.projection_condition_number = 1.0
        if drift is not None:
            if projection_backend == "tensor":
                if quadrature_order is None:
                    raise ValueError(
                        "tensor projection requires quadrature_order"
                    )
                quadrature = TensorGaussHermite(
                    self.basis.dimension, int(quadrature_order)
                )
                self._nodes, self._weights = quadrature.nodes_and_weights
                self._basis_matrix = self.basis.evaluate(self._nodes)
                self._projection_operator = (
                    self._basis_matrix.T * self._weights
                )
                gram = (
                    self._projection_operator @ self._basis_matrix
                )
                self.projection_gram_error = float(
                    np.linalg.norm(
                        gram - np.eye(self.basis.size), ord=2
                    )
                )
                self.projection_condition_number = float(
                    np.linalg.cond(gram)
                )
            elif projection_backend == "sobol_ls":
                if regression_nodes is None:
                    raise ValueError(
                        "sobol_ls projection requires regression_nodes"
                    )
                regression = SobolHermiteLeastSquares(
                    self.basis,
                    int(regression_nodes),
                    int(regression_seed),
                )
                self._nodes = regression.nodes
                self._weights = np.full(
                    regression.node_count,
                    1.0 / regression.node_count,
                    dtype=float,
                )
                self._basis_matrix = regression.basis_matrix
                self._projection_operator = regression.projection_operator
                self.projection_gram_error = regression.gram_identity_error
                self.projection_condition_number = (
                    regression.gram_condition_number
                )
            elif projection_backend in {
                "sobol_cv_matrix_free",
                "structured_cv_matrix_free",
            }:
                assert regression_nodes is not None
                if projection_backend == "sobol_cv_matrix_free":
                    matrix_free = MatrixFreeSobolHermiteProjection(
                        self.basis,
                        int(regression_nodes),
                        int(regression_seed),
                        int(projection_block_size),
                    )
                else:
                    assert structured_core_dimensions is not None
                    matrix_free = StructuredQuadraticTailSobolProjection(
                        self.basis,
                        int(structured_core_dimensions),
                        int(regression_nodes),
                        int(regression_seed),
                        int(projection_block_size),
                    )
                self._matrix_free_projection = matrix_free
                self._nodes = matrix_free.nodes
                self.projection_gram_error = float("nan")
                self.projection_condition_number = float("nan")
            elif projection_backend in {
                "structured_rational_laguerre",
                "structured_modulated_rational_laguerre",
            }:
                assert structured_core_dimensions is not None
                assert core_quadrature_order is not None
                assert rational_residual_scale is not None
                assert projection_control_variate_rate is not None
                if projection_backend == "structured_rational_laguerre":
                    self._rational_projection = (
                        StructuredRationalLaguerreProjection(
                            self.basis,
                            int(structured_core_dimensions),
                            int(core_quadrature_order),
                            int(laguerre_order),
                        )
                    )
                else:
                    try:
                        self._rational_projection = (
                            CoreModulatedRationalLaguerreProjection(
                                self.basis,
                                int(structured_core_dimensions),
                                modulation_core_coordinate=(
                                    self.modulation_core_coordinate
                                ),
                                core_order=int(core_quadrature_order),
                                laguerre_order=int(laguerre_order),
                            )
                        )
                    except ValueError as error:
                        raise ValueError(
                            "structured modulated rational support "
                            "validation failed: the basis must contain "
                            "complete core, tail-linear, and core-free "
                            "quadratic-tail layers, while every retained "
                            "core-modulated tail pair must use the selected "
                            "degree-one modulation_core_coordinate; "
                            f"details: {error}"
                        ) from error
                self.projection_gram_error = 0.0
                self.projection_condition_number = 1.0
            else:
                if (
                    core_dimensions is None
                    or core_quadrature_order is None
                    or tail_regression_nodes is None
                ):
                    raise ValueError(
                        "hybrid_core projection requires "
                        "core_dimensions, core_quadrature_order, and "
                        "tail_regression_nodes"
                    )
                hybrid = HybridCoreGaussHermiteSobol(
                    self.basis,
                    int(core_dimensions),
                    int(core_quadrature_order),
                    int(tail_regression_nodes),
                    int(regression_seed),
                )
                self._nodes = hybrid.nodes
                self._weights = hybrid.weights
                self._basis_matrix = hybrid.basis_matrix
                self._projection_operator = (
                    hybrid.projection_operator
                )
                self.projection_gram_error = (
                    hybrid.gram_identity_error
                )
                self.projection_condition_number = (
                    hybrid.gram_condition_number
                )

    def _linear_drift_step(
        self, coefficients: np.ndarray, duration: float
    ) -> tuple[np.ndarray, float]:
        return (
            np.exp(float(self.linear_drift_rate) * duration) * coefficients,
            0.0,
        )

    def _projected_drift_rhs(
        self, time: float, coefficients: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Galerkin projection of b(t, X_P), plus a discrete closure residual."""

        assert self.drift is not None
        if self._rational_projection is not None:
            assert self.projection_control_variate_rate is not None
            assert self.rational_residual_scale is not None
            return self._rational_projection.project_with_linear_control(
                coefficients,
                self.projection_control_variate_rate,
                self.rational_residual_scale,
                self.projection_control_variate_intercept,
            )
        if self._matrix_free_projection is not None:
            assert self.projection_control_variate_rate is not None
            return (
                self._matrix_free_projection.project_with_linear_control(
                    time,
                    coefficients,
                    self.drift,
                    self.projection_control_variate_rate,
                    self.projection_control_variate_intercept,
                )
            )
        assert self._basis_matrix is not None
        assert self._weights is not None
        assert self._projection_operator is not None
        values = self._basis_matrix @ coefficients
        raw_rhs = self.drift(time, values)
        projected_rhs = self._projection_operator @ raw_rhs
        reconstructed_rhs = self._basis_matrix @ projected_rhs
        residual = float(
            np.sqrt(
                max(
                    np.dot(
                        self._weights,
                        (raw_rhs - reconstructed_rhs) ** 2,
                    ),
                    0.0,
                )
            )
        )
        return projected_rhs, residual

    def _nonlinear_drift_step(
        self, coefficients: np.ndarray, start_time: float, duration: float
    ) -> tuple[np.ndarray, float]:
        """RK4 integration of the deterministic Galerkin drift ODE."""

        state = np.asarray(coefficients, dtype=float).copy()
        substep = duration / self.drift_substeps
        current_time = start_time
        integrated_residual = 0.0
        for _ in range(self.drift_substeps):
            k1, r1 = self._projected_drift_rhs(current_time, state)
            k2, r2 = self._projected_drift_rhs(
                current_time + 0.5 * substep,
                state + 0.5 * substep * k1,
            )
            k3, r3 = self._projected_drift_rhs(
                current_time + 0.5 * substep,
                state + 0.5 * substep * k2,
            )
            k4, r4 = self._projected_drift_rhs(
                current_time + substep, state + substep * k3
            )
            state = state + (substep / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            integrated_residual += (substep / 6.0) * (
                r1 + 2.0 * r2 + 2.0 * r3 + r4
            )
            current_time += substep
        return state, integrated_residual

    def _drift_step(
        self, coefficients: np.ndarray, start_time: float, duration: float
    ) -> tuple[np.ndarray, float]:
        if self.linear_drift_rate is not None:
            return self._linear_drift_step(coefficients, duration)
        return self._nonlinear_drift_step(
            coefficients, start_time, duration
        )

    def _assemble_result(
        self,
        all_times: np.ndarray,
        coefficients: np.ndarray,
        projection_residuals: np.ndarray,
    ) -> SolverResult:
        moments = np.asarray(
            [gaussian_moments(row) for row in coefficients], dtype=float
        )
        return SolverResult(
            all_times,
            coefficients,
            moments[:, 0],
            moments[:, 1],
            moments[:, 2],
            projection_residuals,
        )

    def _solve_strang(self, initial_value: float) -> SolverResult:
        node_times = self.factor.times
        all_times = np.concatenate(([0.0], node_times))
        coefficients = np.zeros(
            (node_times.size + 1, self.basis.size), dtype=float
        )
        coefficients[0, 0] = float(initial_value)
        projection_residuals = np.zeros(node_times.size, dtype=float)

        increments = self.factor.increments
        for step in range(node_times.size):
            left = all_times[step]
            right = all_times[step + 1]
            half = 0.5 * (right - left)
            state, first_residual = self._drift_step(
                coefficients[step], left, half
            )
            state = self.basis.apply_wick_exponential(
                state, self.beta * increments[step]
            )
            state, second_residual = self._drift_step(
                state, left + half, half
            )
            coefficients[step + 1] = state
            projection_residuals[step] = first_residual + second_residual

        return self._assemble_result(
            all_times, coefficients, projection_residuals
        )

    def _solve_integrating_factor(self, initial_value: float) -> SolverResult:
        """Integrate y' = D_-g B_P(t, D_g y), then reconstruct x = D_g y."""

        if self.drift is None:
            raise ValueError("integrating-factor path is for nonlinear drift")
        node_times = self.factor.times
        all_times = np.concatenate(([0.0], node_times))
        coefficients = np.zeros(
            (node_times.size + 1, self.basis.size), dtype=float
        )
        coefficients[0, 0] = float(initial_value)
        transformed = coefficients[0].copy()
        projection_residuals = np.zeros(node_times.size, dtype=float)
        zero_loading = np.zeros(self.factor.modes, dtype=float)

        for step in range(node_times.size):
            left_time = all_times[step]
            right_time = all_times[step + 1]
            left_loading = (
                zero_loading
                if step == 0
                else self.factor.loadings[step - 1]
            )
            right_loading = self.factor.loadings[step]
            cell_length = right_time - left_time
            substep = cell_length / self.drift_substeps
            integrated_residual = 0.0

            def transformed_rhs(
                evaluation_time: float, state: np.ndarray
            ) -> tuple[np.ndarray, float]:
                fraction = (evaluation_time - left_time) / cell_length
                fraction = min(1.0, max(0.0, fraction))
                loading = (
                    left_loading
                    + fraction * (right_loading - left_loading)
                )
                physical = self.basis.apply_wick_exponential(
                    state, self.beta * loading
                )
                drift_rhs, residual = self._projected_drift_rhs(
                    evaluation_time, physical
                )
                pulled_back = self.basis.apply_wick_exponential(
                    drift_rhs, -self.beta * loading
                )
                return pulled_back, residual

            current_time = left_time
            for _ in range(self.drift_substeps):
                k1, r1 = transformed_rhs(current_time, transformed)
                k2, r2 = transformed_rhs(
                    current_time + 0.5 * substep,
                    transformed + 0.5 * substep * k1,
                )
                k3, r3 = transformed_rhs(
                    current_time + 0.5 * substep,
                    transformed + 0.5 * substep * k2,
                )
                k4, r4 = transformed_rhs(
                    current_time + substep,
                    transformed + substep * k3,
                )
                transformed = transformed + (substep / 6.0) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4
                )
                integrated_residual += (substep / 6.0) * (
                    r1 + 2.0 * r2 + 2.0 * r3 + r4
                )
                current_time += substep

            coefficients[step + 1] = self.basis.apply_wick_exponential(
                transformed, self.beta * right_loading
            )
            projection_residuals[step] = integrated_residual

        return self._assemble_result(
            all_times, coefficients, projection_residuals
        )

    def solve(self, initial_value: float) -> SolverResult:
        if self.drift is not None and self.nonlinear_propagator == "integrating_factor":
            return self._solve_integrating_factor(initial_value)
        return self._solve_strang(initial_value)


def exact_geometric_coefficients(
    basis: HermiteBasis,
    loading: np.ndarray,
    time: float,
    initial_value: float,
    drift_rate: float,
    beta: float,
) -> np.ndarray:
    scale = float(initial_value) * np.exp(float(drift_rate) * float(time))
    return scale * basis.exact_wick_exponential_coefficients(
        float(beta) * np.asarray(loading, dtype=float)
    )


def exact_geometric_chaos_tail_squared(
    time: float,
    hurst: float,
    degree: int,
    initial_value: float,
    drift_rate: float,
    beta: float,
) -> float:
    intensity = float(beta) ** 2 * float(time) ** (2.0 * float(hurst))
    partial = sum(intensity**order / factorial(order)
                  for order in range(int(degree) + 1))
    scale = float(initial_value) ** 2 * np.exp(
        2.0 * float(drift_rate) * float(time)
    )
    return float(scale * (np.exp(intensity) - partial))
