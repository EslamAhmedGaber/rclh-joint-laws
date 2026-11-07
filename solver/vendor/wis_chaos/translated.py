"""Deterministic chaos projection of a terminal-target WIS translation scheme."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .basis import (
    HermiteBasis,
    HybridCoreGaussHermiteSobol,
    SobolHermiteLeastSquares,
    TensorGaussHermite,
    gaussian_moments,
)
from .fbm import GaussianPathFactor


ArrayFunction = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TranslatedEulerResult:
    terminal_coefficients: np.ndarray
    terminal_mean: float
    terminal_variance: float
    terminal_second_moment: float
    transformed_coefficients: np.ndarray
    stage_projection_residuals: np.ndarray
    terminal_projection_residual: float
    terminal_quadrature_mean: float
    terminal_quadrature_second_moment: float


class DeterministicTranslatedEulerChaosSolver:
    """Project the published terminal-target translated Euler recurrence.

    This is a deterministic polynomial-chaos counterpart of the path
    recurrence for

        dX = [alpha X + a(t, X)] dt + beta X diamond dB^H.

    It is useful for an equal-discretisation comparison.  Unlike the global
    Lawson coefficient solver, a new target terminal time generally requires
    a new run.
    """

    def __init__(
        self,
        factor: GaussianPathFactor,
        degree: int,
        *,
        linear_drift_rate: float,
        beta: float,
        nonlinear_drift: ArrayFunction,
        projection_backend: str = "sobol_ls",
        quadrature_order: int | None = None,
        regression_nodes: int | None = None,
        regression_seed: int = 1729,
        index_set: tuple[tuple[int, ...], ...] | None = None,
        core_dimensions: int | None = None,
        core_quadrature_order: int | None = None,
        tail_regression_nodes: int | None = None,
    ) -> None:
        if projection_backend not in {
            "tensor",
            "sobol_ls",
            "hybrid_core",
        }:
            raise ValueError(
                "projection_backend must be 'tensor', 'sobol_ls', "
                "or 'hybrid_core'"
            )
        self.factor = factor
        self.basis = HermiteBasis(
            factor.modes,
            int(degree),
            index_set=index_set,
        )
        self.linear_drift_rate = float(linear_drift_rate)
        self.beta = float(beta)
        self.nonlinear_drift = nonlinear_drift
        self.projection_backend = projection_backend

        if projection_backend == "tensor":
            if quadrature_order is None:
                raise ValueError(
                    "tensor projection requires quadrature_order"
                )
            quadrature = TensorGaussHermite(
                self.basis.dimension,
                int(quadrature_order),
            )
            self.nodes, self.weights = quadrature.nodes_and_weights
            self.basis_matrix = self.basis.evaluate(self.nodes)
            self.projection_operator = (
                self.basis_matrix.T * self.weights
            )
            gram = self.projection_operator @ self.basis_matrix
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
            self.nodes = regression.nodes
            self.weights = np.full(
                regression.node_count,
                1.0 / regression.node_count,
                dtype=float,
            )
            self.basis_matrix = regression.basis_matrix
            self.projection_operator = regression.projection_operator
            gram = regression.gram
        else:
            if (
                core_dimensions is None
                or core_quadrature_order is None
                or tail_regression_nodes is None
            ):
                raise ValueError(
                    "hybrid_core projection requires core_dimensions, "
                    "core_quadrature_order, and tail_regression_nodes"
                )
            hybrid = HybridCoreGaussHermiteSobol(
                self.basis,
                int(core_dimensions),
                int(core_quadrature_order),
                int(tail_regression_nodes),
                int(regression_seed),
            )
            self.nodes = hybrid.nodes
            self.weights = hybrid.weights
            self.basis_matrix = hybrid.basis_matrix
            self.projection_operator = hybrid.projection_operator
            gram = hybrid.gram

        self.projection_gram_error = float(
            np.linalg.norm(
                gram - np.eye(self.basis.size),
                ord=2,
            )
        )
        self.projection_condition_number = float(np.linalg.cond(gram))

    def _project(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        coefficients = self.projection_operator @ values
        residual_values = values - self.basis_matrix @ coefficients
        residual = float(
            np.sqrt(
                max(
                    np.dot(
                        self.weights,
                        residual_values * residual_values,
                    ),
                    0.0,
                )
            )
        )
        return coefficients, residual

    def _translation_values(
        self,
        evaluation_time: float,
        loading: np.ndarray,
    ) -> np.ndarray:
        terminal_time = float(self.factor.times[-1])
        hurst = float(self.factor.hurst)
        covariance_shift = 0.5 * self.beta**2 * (
            terminal_time ** (2.0 * hurst)
            - (terminal_time - evaluation_time) ** (2.0 * hurst)
        )
        path_values = self.nodes @ loading
        return np.exp(
            -self.linear_drift_rate * evaluation_time
            + covariance_shift
            - self.beta * path_values
        )

    def solve(self, initial_value: float) -> TranslatedEulerResult:
        node_times = self.factor.times
        transformed = np.zeros(self.basis.size, dtype=float)
        transformed[0] = float(initial_value)
        residuals = np.zeros(node_times.size, dtype=float)
        zero_loading = np.zeros(self.factor.modes, dtype=float)
        left_time = 0.0

        for step, right_time in enumerate(node_times):
            loading = (
                zero_loading
                if step == 0
                else self.factor.loadings[step - 1]
            )
            translation = self._translation_values(left_time, loading)
            transformed_values = self.basis_matrix @ transformed
            physical_values = transformed_values / translation
            rhs_values = translation * self.nonlinear_drift(
                left_time,
                physical_values,
            )
            rhs_coefficients, residuals[step] = self._project(rhs_values)
            transformed = transformed + (
                float(right_time) - left_time
            ) * rhs_coefficients
            left_time = float(right_time)

        terminal_translation = self._translation_values(
            float(node_times[-1]),
            self.factor.loadings[-1],
        )
        terminal_values = (
            self.basis_matrix @ transformed
        ) / terminal_translation
        terminal_coefficients, terminal_residual = self._project(
            terminal_values
        )
        quadrature_mean = float(
            np.dot(self.weights, terminal_values)
        )
        quadrature_second = float(
            np.dot(
                self.weights,
                terminal_values * terminal_values,
            )
        )
        mean, variance, second = gaussian_moments(
            terminal_coefficients
        )
        return TranslatedEulerResult(
            terminal_coefficients=terminal_coefficients,
            terminal_mean=mean,
            terminal_variance=variance,
            terminal_second_moment=second,
            transformed_coefficients=transformed,
            stage_projection_residuals=residuals,
            terminal_projection_residual=terminal_residual,
            terminal_quadrature_mean=quadrature_mean,
            terminal_quadrature_second_moment=quadrature_second,
        )
