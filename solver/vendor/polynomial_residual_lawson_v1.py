"""Exact polynomial residual closure for the PAPER IX application chain.

This module is additive.  It reuses the frozen finite Gaussian factor,
normalised Hermite basis, Lawson time stepper, and coherent reconstruction
from the PAPER IX research backend, but it does not modify those historical
sources.  For a polynomial nonlinearity the one-dimensional residual Gaussian
expectation is evaluated in closed form.

The resulting solver is deterministic after the finite design and quadrature
rule have been fixed.  It solves the declared finite active-manifold Galerkin
system; it is not an exact marginal solver for discarded solution-chaos
coefficients and it does not establish an N/M/P continuum limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Mapping

import numpy as np
from scipy.special import roots_hermitenorm

from wis_chaos.basis import HermiteBasis
from wis_chaos.residual_coherent_lawson import (
    ResidualClosureDiagnostic,
    ResidualCoherentLawsonSolver,
)
from wis_chaos.spectral_projection import (
    AffineSpectralDrift,
    CosineTransform,
)


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        selected = int(value)
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if not np.isfinite(scalar) or scalar != selected or selected < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return selected


def _normal_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(_integer(order, "order", 1))
    weights = weights / np.sqrt(2.0 * np.pi)
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
        and abs(float(np.sum(weights)) - 1.0) <= 2.0e-14
    ):
        raise FloatingPointError("normal quadrature construction failed")
    return nodes, weights


def _tensor_normal_rule(
    dimension: int,
    order: int,
    *,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_dimension = _integer(dimension, "dimension", 1)
    selected_order = _integer(order, "active_quadrature_order", 1)
    selected_maximum = _integer(max_nodes, "max_active_nodes", 1)
    count = selected_order**selected_dimension
    if count > selected_maximum:
        raise MemoryError(
            f"active tensor rule would have {count:,} nodes; "
            f"frozen maximum is {selected_maximum:,}"
        )
    one_nodes, one_weights = _normal_rule(selected_order)
    node_mesh = np.meshgrid(
        *([one_nodes] * selected_dimension), indexing="ij"
    )
    weight_mesh = np.meshgrid(
        *([one_weights] * selected_dimension), indexing="ij"
    )
    nodes = np.column_stack([values.ravel() for values in node_mesh])
    weights = np.ones(prod((selected_order,) * selected_dimension))
    for values in weight_mesh:
        weights *= values.ravel()
    if not (
        nodes.shape == (count, selected_dimension)
        and weights.shape == (count,)
        and np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
        and abs(float(np.sum(weights)) - 1.0) <= 5.0e-13
    ):
        raise FloatingPointError("active tensor rule construction failed")
    return nodes, weights


@dataclass(frozen=True, slots=True)
class PolynomialDrift:
    """Autonomous polynomial drift split into an exact linear part.

    ``linear_rate`` is the coefficient of x and is projected without stochastic
    quadrature error at the coefficient level.  Time integration remains the
    inherited RK4 discretisation.  ``coefficients`` contains the remaining
    monomials; degree one is deliberately forbidden to prevent double counting.
    """

    linear_rate: float
    coefficients: tuple[tuple[int, float], ...]
    label: str = "polynomial_drift"

    def __post_init__(self) -> None:
        rate = float(self.linear_rate)
        if not np.isfinite(rate):
            raise ValueError("linear_rate must be finite")
        raw = tuple(self.coefficients)
        if not raw:
            raise ValueError("at least one non-linear or constant monomial is required")
        normalised: list[tuple[int, float]] = []
        seen: set[int] = set()
        for raw_degree, raw_coefficient in raw:
            degree = _integer(raw_degree, "polynomial degree", 0)
            coefficient = float(raw_coefficient)
            if degree == 1:
                raise ValueError("degree-one drift belongs in linear_rate")
            if degree in seen:
                raise ValueError("polynomial degrees must be unique")
            if not np.isfinite(coefficient):
                raise ValueError("polynomial coefficients must be finite")
            seen.add(degree)
            normalised.append((degree, coefficient))
        normalised.sort(key=lambda item: item[0])
        label = str(self.label).strip()
        if not label:
            raise ValueError("label must be nonempty")
        object.__setattr__(self, "linear_rate", rate)
        object.__setattr__(self, "coefficients", tuple(normalised))
        object.__setattr__(self, "label", label)

    @classmethod
    def from_mapping(
        cls,
        linear_rate: float,
        coefficients: Mapping[int, float],
        *,
        label: str = "polynomial_drift",
    ) -> "PolynomialDrift":
        return cls(
            float(linear_rate),
            tuple((int(degree), float(value)) for degree, value in coefficients.items()),
            label,
        )

    @property
    def maximum_degree(self) -> int:
        return max(degree for degree, _coefficient in self.coefficients)

    def __call__(self, _time: float, values: np.ndarray) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        output = self.linear_rate * state
        for degree, coefficient in self.coefficients:
            output = output + coefficient * np.power(state, degree)
        return output


def polynomial_residual_multiplier(degree: int, residual_variance: float) -> float:
    """Return exp(p(p-1) rho / 2) with fail-closed overflow checks."""

    selected_degree = _integer(degree, "degree", 0)
    variance = float(residual_variance)
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("residual_variance must be finite and nonnegative")
    exponent = 0.5 * selected_degree * (selected_degree - 1) * variance
    if exponent > np.log(np.finfo(float).max):
        raise FloatingPointError("polynomial residual multiplier overflows")
    return float(np.exp(exponent))


class PolynomialResidualProjection:
    """Positive-weight active projection with exact residual elimination."""

    residual_method = "closed_form_polynomial_gaussian_moment"

    def __init__(
        self,
        basis: HermiteBasis,
        drift: PolynomialDrift,
        *,
        active_quadrature_order: int,
        max_active_nodes: int = 2_000_000,
    ) -> None:
        if not isinstance(basis, HermiteBasis):
            raise TypeError("basis must be HermiteBasis")
        if not isinstance(drift, PolynomialDrift):
            raise TypeError("drift must be PolynomialDrift")
        selected_order = _integer(
            active_quadrature_order, "active_quadrature_order", 1
        )
        if selected_order < basis.degree + 1:
            raise ValueError(
                "active quadrature is below the exact Hermite Gram floor P+1"
            )
        active_nodes, active_weights = _tensor_normal_rule(
            basis.dimension,
            selected_order,
            max_nodes=max_active_nodes,
        )
        basis_matrix = basis.evaluate(active_nodes)
        projection = basis_matrix.T * active_weights
        gram = projection @ basis_matrix
        gram_error = float(np.linalg.norm(gram - np.eye(basis.size), ord=2))
        if not np.isfinite(gram_error) or gram_error > 5.0e-11:
            raise FloatingPointError("active quadrature does not resolve the basis")
        self.basis = basis
        self.drift = drift
        self.active_quadrature_order = selected_order
        self.active_nodes = active_nodes
        self.active_weights = active_weights
        self.basis_matrix = basis_matrix
        self.projection = projection
        self.gram_error = gram_error

    def conditional_values(
        self,
        coefficients: np.ndarray,
        active_loading: np.ndarray,
        residual_variance: float,
    ) -> np.ndarray:
        state = np.asarray(coefficients, dtype=float)
        loading = np.asarray(active_loading, dtype=float)
        variance = float(residual_variance)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has the wrong shape")
        if loading.shape != (self.basis.dimension,):
            raise ValueError("active loading has the wrong shape")
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(loading))
            or not np.isfinite(variance)
            or variance < 0.0
        ):
            raise ValueError("projection inputs must be finite/nonnegative")
        values = self.basis_matrix @ state
        q_zero = self.active_nodes @ loading + 0.5 * float(np.dot(loading, loading))
        conditional = np.zeros(values.shape, dtype=float)
        for degree, coefficient in self.drift.coefficients:
            exponent = (degree - 1) * q_zero
            exponent = exponent + (
                0.5 * degree * (degree - 1) * variance
            )
            if float(np.max(exponent, initial=-np.inf)) > np.log(
                np.finfo(float).max
            ):
                raise FloatingPointError("polynomial conditional field overflows")
            conditional = conditional + (
                coefficient * np.power(values, degree) * np.exp(exponent)
            )
        if not np.all(np.isfinite(conditional)):
            raise FloatingPointError("polynomial conditional field is nonfinite")
        return conditional

    def project(
        self,
        coefficients: np.ndarray,
        active_loading: np.ndarray,
        residual_variance: float,
    ) -> tuple[np.ndarray, ResidualClosureDiagnostic]:
        state = np.asarray(coefficients, dtype=float)
        conditional = self.conditional_values(
            state, active_loading, residual_variance
        )
        rhs = self.drift.linear_rate * state + self.projection @ conditional
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError("polynomial projected RHS is nonfinite")
        return rhs, ResidualClosureDiagnostic(
            active_node_count=int(self.active_nodes.shape[0]),
            residual_node_count=0,
            residual_variance=float(residual_variance),
            projected_rhs_norm=float(np.linalg.norm(rhs, ord=np.inf)),
            cosine_residual_method=self.residual_method,
            maximum_cosine_phase=0.0,
        )

    def discrete_energy_rate(
        self,
        coefficients: np.ndarray,
        active_loading: np.ndarray,
        residual_variance: float,
    ) -> float:
        """Return the semidiscrete rate y dot F(y).

        For a negative cubic coefficient and an exact active Gram matrix this
        rate is bounded above by ``linear_rate * ||y||^2`` because every
        nonlinear quadrature contribution is a positive weight multiplying a
        non-positive fourth power.  This is an RHS inequality, not a claim that
        an arbitrary RK4 step is energy diminishing.
        """

        state = np.asarray(coefficients, dtype=float)
        rhs, _diagnostic = self.project(
            state, active_loading, residual_variance
        )
        return float(np.dot(state, rhs))


class PolynomialResidualLawsonSolver(ResidualCoherentLawsonSolver):
    """Finite deterministic Lawson solver for autonomous polynomial drifts."""

    implementation_label = "polynomial_residual_coherent_lawson_v1_additive"

    def __init__(
        self,
        factor,
        degree: int,
        beta: float,
        *,
        drift: PolynomialDrift,
        active_quadrature_order: int,
        drift_substeps: int = 1,
        max_active_nodes: int = 2_000_000,
    ) -> None:
        if not isinstance(drift, PolynomialDrift):
            raise TypeError("drift must be PolynomialDrift")
        # Initialise and validate the historical finite factor and coherent
        # output machinery through the existing additive backend.  A zero
        # spectral atom is used only during construction; it is replaced by
        # the exact polynomial projector before any solve is allowed.
        zero_drift = AffineSpectralDrift(
            intercept=0.0,
            rate=0.0,
            transforms=(CosineTransform(0.0, frequency=0.0),),
        )
        super().__init__(
            factor,
            degree,
            beta,
            drift=zero_drift,
            active_quadrature_order=active_quadrature_order,
            residual_quadrature_order=1,
            drift_substeps=drift_substeps,
            max_active_nodes=max_active_nodes,
        )
        self.polynomial_drift = drift
        self.drift = drift
        self.projection = PolynomialResidualProjection(
            self.basis,
            drift,
            active_quadrature_order=active_quadrature_order,
            max_active_nodes=max_active_nodes,
        )


__all__ = [
    "PolynomialDrift",
    "PolynomialResidualLawsonSolver",
    "PolynomialResidualProjection",
    "polynomial_residual_multiplier",
]
