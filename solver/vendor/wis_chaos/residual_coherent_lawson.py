"""Residual-covariance coherent Lawson Galerkin prototype.

The unresolved orthogonal Gaussian bridge is integrated inside every
projected RHS through one scalar normal expectation.  This is an additive
research backend and is not part of the officially fingerprinted fused
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

import numpy as np
from scipy.special import roots_hermitenorm, roots_legendre

from .basis import HermiteBasis, gaussian_moments
from .coherent import (
    coherent_reconstruction,
    shifted_polynomial_coefficients,
)
from .contour_shifted_lognormal import ContourShiftedLognormalRule
from .fbm import GaussianPathFactor
from .spectral_projection import (
    AffineSpectralDrift,
    CosineTransform,
    LogQuadraticCosineTransform,
    RationalSineTransform,
)


def _integer_control(value: object, name: str, *, minimum: int) -> int:
    """Validate integer-valued public controls without silent truncation."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(value, (int, np.integer)):
        selected = int(value)
        if selected < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return selected
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if (
        not np.isfinite(scalar)
        or scalar != np.floor(scalar)
        or scalar < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(scalar)


def _normal_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    selected = _integer_control(order, "Gaussian quadrature order", minimum=1)
    nodes, weights = roots_hermitenorm(selected)
    weights = weights / np.sqrt(2.0 * np.pi)
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
        and abs(float(np.sum(weights)) - 1.0) <= 2e-14
    ):
        raise FloatingPointError("normal quadrature construction failed")
    return nodes, weights


def _active_order_control(
    value: object,
    dimension: int,
) -> int | tuple[int, ...]:
    """Validate scalar or coordinate-wise active Gaussian orders."""

    if np.isscalar(value):
        return _integer_control(value, "active_quadrature_order", minimum=1)
    try:
        raw_orders = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            "active_quadrature_order must be an integer or one per coordinate"
        ) from error
    if len(raw_orders) != int(dimension):
        raise ValueError(
            "anisotropic active_quadrature_order has the wrong dimension"
        )
    return tuple(
        _integer_control(item, "active_quadrature_order", minimum=1)
        for item in raw_orders
    )


def _active_rule_control(
    value: object,
    dimension: int,
) -> tuple[str, ...]:
    """Validate one active rule name or one rule name per coordinate."""

    if isinstance(value, str):
        raw_rules = (value,) * int(dimension)
    else:
        try:
            raw_rules = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                "active_quadrature_rule must be a string or one per coordinate"
            ) from error
        if len(raw_rules) != int(dimension):
            raise ValueError(
                "anisotropic active_quadrature_rule has the wrong dimension"
            )
    aliases = {
        "gh": "gauss_hermite",
        "gauss_hermite": "gauss_hermite",
        "gl": "truncated_gauss_legendre",
        "truncated_gauss_legendre": "truncated_gauss_legendre",
    }
    selected = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, str):
            raise ValueError("active quadrature rule names must be strings")
        canonical = aliases.get(raw_rule.strip().lower())
        if canonical is None:
            raise ValueError("unsupported active quadrature rule")
        selected.append(canonical)
    return tuple(selected)


def _active_truncation_control(
    value: object | None,
    rules: tuple[str, ...],
) -> tuple[float | None, ...]:
    """Validate coordinate-wise truncations required by active GL axes."""

    dimension = len(rules)
    if value is None:
        raw_values: tuple[object | None, ...] = (None,) * dimension
    elif np.isscalar(value):
        raw_values = (value,) * dimension
    else:
        try:
            raw_values = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                "active_truncation must be scalar or one per coordinate"
            ) from error
        if len(raw_values) != dimension:
            raise ValueError("anisotropic active_truncation has the wrong dimension")
    selected: list[float | None] = []
    for rule, raw_value in zip(rules, raw_values, strict=True):
        if rule == "gauss_hermite":
            selected.append(None)
            continue
        if raw_value is None:
            raise ValueError("active_truncation is required for every GL axis")
        truncation = float(raw_value)
        if not np.isfinite(truncation) or truncation <= 0.0:
            raise ValueError("active_truncation must be finite and positive")
        selected.append(truncation)
    return tuple(selected)


def _active_normal_rule(
    dimension: int,
    control: int | tuple[int, ...],
    *,
    rules: tuple[str, ...] | None = None,
    truncations: tuple[float | None, ...] | None = None,
    max_nodes: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    selected_rules = (
        ("gauss_hermite",) * int(dimension) if rules is None else tuple(rules)
    )
    selected_truncations = (
        (None,) * int(dimension)
        if truncations is None
        else tuple(truncations)
    )
    if len(selected_rules) != int(dimension) or len(selected_truncations) != int(
        dimension
    ):
        raise ValueError("active rule controls have the wrong dimension")
    orders = (
        (control,) * int(dimension) if isinstance(control, int) else tuple(control)
    )
    selected_max_nodes = _integer_control(
        max_nodes,
        "max_active_nodes",
        minimum=1,
    )
    node_count = prod(orders)
    if node_count > selected_max_nodes:
        raise MemoryError(
            f"anisotropic tensor rule would have {node_count:,} nodes"
        )
    one_dimensional_rules = []
    for order, rule_name, truncation in zip(
        orders,
        selected_rules,
        selected_truncations,
        strict=True,
    ):
        if rule_name == "gauss_hermite":
            one_dimensional_rules.append(_normal_rule(order))
            continue
        if rule_name != "truncated_gauss_legendre" or truncation is None:
            raise ValueError("invalid active quadrature rule configuration")
        base_nodes, base_weights = roots_legendre(order)
        normal_nodes = truncation * base_nodes
        normal_weights = (
            truncation
            * base_weights
            * np.exp(-0.5 * normal_nodes * normal_nodes)
            / np.sqrt(2.0 * np.pi)
        )
        if not (
            np.all(np.isfinite(normal_nodes))
            and np.all(np.isfinite(normal_weights))
            and np.all(normal_weights > 0.0)
        ):
            raise FloatingPointError("active GL rule construction failed")
        one_dimensional_rules.append((normal_nodes, normal_weights))
    node_mesh = np.meshgrid(
        *[nodes for nodes, _weights in one_dimensional_rules],
        indexing="ij",
    )
    weight_mesh = np.meshgrid(
        *[weights for _nodes, weights in one_dimensional_rules],
        indexing="ij",
    )
    nodes = np.column_stack([values.ravel() for values in node_mesh])
    weights = np.ones(node_count, dtype=float)
    for values in weight_mesh:
        weights *= values.ravel()
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
        and abs(float(np.sum(weights)) - 1.0) <= 1.0e-9
    ):
        raise FloatingPointError(
            "anisotropic active quadrature construction failed"
        )
    return nodes, weights


def _stable_rational_pullback(
    transform: RationalSineTransform,
    lawson_values: np.ndarray,
    exponent: np.ndarray,
) -> np.ndarray:
    values = np.asarray(lawson_values, dtype=float)
    q = np.asarray(exponent, dtype=float)
    if transform.scale == 0.0:
        return np.zeros(np.broadcast_shapes(values.shape, q.shape), dtype=float)
    values, q = np.broadcast_arrays(values, q)
    absolute = np.abs(values)
    log_values = np.full(values.shape, -np.inf, dtype=float)
    nonzero = absolute > 0.0
    log_values[nonzero] = np.log(absolute[nonzero])
    log_denominator = np.logaddexp(
        2.0 * np.log(transform.width),
        2.0 * log_values + 2.0 * q,
    )
    result = np.zeros(values.shape, dtype=float)
    log_magnitude = (
        np.log(abs(transform.scale))
        + log_values[nonzero]
        - log_denominator[nonzero]
    )
    result[nonzero] = (
        np.sign(transform.scale * values[nonzero])
        * np.exp(log_magnitude)
    )
    return result


def _stable_log_pullback(
    transform: LogQuadraticCosineTransform,
    lawson_values: np.ndarray,
    exponent: np.ndarray,
) -> np.ndarray:
    values = np.asarray(lawson_values, dtype=float)
    q = np.asarray(exponent, dtype=float)
    if transform.scale == 0.0:
        return np.zeros(np.broadcast_shapes(values.shape, q.shape), dtype=float)
    values, q = np.broadcast_arrays(values, q)
    absolute = np.abs(values)
    log_ratio = np.full(values.shape, -np.inf, dtype=float)
    nonzero = absolute > 0.0
    log_ratio[nonzero] = (
        np.log(absolute[nonzero]) - np.log(transform.width) + q[nonzero]
    )
    doubled = 2.0 * log_ratio
    result = np.zeros(values.shape, dtype=float)
    active = nonzero
    small = active & (doubled < -36.0)
    # In this branch log1p(exp(d))=exp(d) to relative roundoff.  Combining
    # exponents before exponentiation avoids an otherwise possible inf*0.
    small_log_magnitude = (
        np.log(abs(transform.scale)) + doubled[small] - q[small]
    )
    result[small] = np.sign(transform.scale) * np.exp(small_log_magnitude)
    regular = active & ~small
    logarithm = np.logaddexp(0.0, doubled[regular])
    regular_log_magnitude = (
        np.log(abs(transform.scale))
        + np.log(logarithm)
        - q[regular]
    )
    result[regular] = (
        np.sign(transform.scale) * np.exp(regular_log_magnitude)
    )
    return result


def _pulled_back_spectral_transforms(
    drift: AffineSpectralDrift,
    lawson_values: np.ndarray,
    exponent: np.ndarray,
    *,
    include_cosine: bool = True,
) -> np.ndarray:
    """Evaluate only the nonlinear pulled-back spectral terms."""

    if not isinstance(drift, AffineSpectralDrift):
        raise TypeError("residual closure requires AffineSpectralDrift")
    y, q = np.broadcast_arrays(
        np.asarray(lawson_values, dtype=float),
        np.asarray(exponent, dtype=float),
    )
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(q)):
        raise FloatingPointError("pullback inputs must be finite")
    result = np.zeros(y.shape, dtype=float)
    for transform in drift.transforms:
        if transform.scale == 0.0:
            continue
        if isinstance(transform, RationalSineTransform):
            result = result + _stable_rational_pullback(transform, y, q)
        elif isinstance(transform, LogQuadraticCosineTransform):
            result = result + _stable_log_pullback(transform, y, q)
        elif isinstance(transform, CosineTransform):
            if not include_cosine:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                physical = y * np.exp(q)
                result = result + (
                    transform.scale
                    * np.exp(-q)
                    * np.cos(transform.frequency * physical)
                )
        else:
            raise TypeError("unsupported spectral transform")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("pulled-back spectral transform is nonfinite")
    return result


def pulled_back_affine_spectral_drift(
    drift: AffineSpectralDrift,
    lawson_values: np.ndarray,
    exponent: np.ndarray,
) -> np.ndarray:
    """Evaluate ``exp(-q) b(y exp(q))`` without avoidable overflow."""

    y, q = np.broadcast_arrays(
        np.asarray(lawson_values, dtype=float),
        np.asarray(exponent, dtype=float),
    )
    nonlinear = _pulled_back_spectral_transforms(drift, y, q)
    result = drift.rate * y + nonlinear
    if drift.intercept != 0.0:
        with np.errstate(over="ignore", invalid="ignore"):
            result = result + drift.intercept * np.exp(-q)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("pulled-back drift is nonfinite")
    return result


def residual_coherent_cross_moment(
    basis: HermiteBasis,
    left_lawson_coefficients: np.ndarray,
    right_lawson_coefficients: np.ndarray,
    left_loading: np.ndarray,
    right_loading: np.ndarray,
    residual_covariance: float,
) -> float:
    """Return the exact two-time coherent moment of two active states.

    The inactive Gaussian coordinates enter only through their covariance.
    The active translations are evaluated exactly in the finite polynomial
    space; no chaos projection of the coherent outputs is performed.
    """

    if not isinstance(basis, HermiteBasis):
        raise TypeError("basis must be HermiteBasis")
    left = np.asarray(left_lawson_coefficients, dtype=float)
    right = np.asarray(right_lawson_coefficients, dtype=float)
    loading_left = np.asarray(left_loading, dtype=float)
    loading_right = np.asarray(right_loading, dtype=float)
    residual = float(residual_covariance)
    if left.shape != (basis.size,) or right.shape != (basis.size,):
        raise ValueError("Lawson coefficient vector has the wrong shape")
    if (
        loading_left.shape != (basis.dimension,)
        or loading_right.shape != (basis.dimension,)
    ):
        raise ValueError("active loading has the wrong shape")
    if not all(
        np.all(np.isfinite(values))
        for values in (left, right, loading_left, loading_right)
    ) or not np.isfinite(residual):
        raise ValueError("cross-moment inputs must be finite")
    left_shifted = shifted_polynomial_coefficients(
        basis,
        left,
        loading_right,
    )
    right_shifted = shifted_polynomial_coefficients(
        basis,
        right,
        loading_left,
    )
    exponent = float(np.dot(loading_left, loading_right) + residual)
    weight = float(np.exp(exponent))
    if not np.isfinite(weight):
        raise OverflowError("coherent cross-moment weight exceeds float range")
    moment = float(weight * np.dot(left_shifted, right_shifted))
    if not np.isfinite(moment):
        raise FloatingPointError("coherent cross moment is nonfinite")
    return moment


@dataclass(frozen=True, slots=True)
class ResidualClosureDiagnostic:
    """Per-stage projection audit.

    ``residual_node_count`` is the largest one-dimensional residual rule
    actually used by any transform at the stage; exact zero-variance limits
    count as one node.
    """

    active_node_count: int
    residual_node_count: int
    residual_variance: float
    projected_rhs_norm: float
    cosine_residual_method: str = "real_gauss_hermite"
    maximum_cosine_phase: float = 0.0


class ResidualCoherentLawsonProjection:
    """Pseudospectrally project the coherent Lawson field onto active chaos."""

    def __init__(
        self,
        basis: HermiteBasis,
        drift: AffineSpectralDrift,
        *,
        active_quadrature_order: int | tuple[int, ...],
        residual_quadrature_order: int,
        cosine_residual_rule: ContourShiftedLognormalRule | None = None,
        active_quadrature_rule: str | tuple[str, ...] = "gauss_hermite",
        active_truncation: float | tuple[float | None, ...] | None = None,
        max_active_nodes: int = 2_000_000,
        store_projection_matrix: bool = True,
        gram_chunk_size: int = 32_768,
    ) -> None:
        if not isinstance(basis, HermiteBasis):
            raise TypeError("basis must be HermiteBasis")
        if not isinstance(drift, AffineSpectralDrift):
            raise TypeError("drift must be AffineSpectralDrift")
        if cosine_residual_rule is not None and not isinstance(
            cosine_residual_rule,
            ContourShiftedLognormalRule,
        ):
            raise TypeError(
                "cosine_residual_rule must be ContourShiftedLognormalRule"
            )
        selected_active_order = _active_order_control(
            active_quadrature_order,
            basis.dimension,
        )
        selected_active_rules = _active_rule_control(
            active_quadrature_rule,
            basis.dimension,
        )
        selected_active_truncations = _active_truncation_control(
            active_truncation,
            selected_active_rules,
        )
        selected_residual_order = _integer_control(
            residual_quadrature_order,
            "residual_quadrature_order",
            minimum=1,
        )
        selected_max_active_nodes = _integer_control(
            max_active_nodes,
            "max_active_nodes",
            minimum=1,
        )
        selected_gram_chunk_size = _integer_control(
            gram_chunk_size,
            "gram_chunk_size",
            minimum=1,
        )
        if not isinstance(store_projection_matrix, (bool, np.bool_)):
            raise ValueError("store_projection_matrix must be boolean")
        selected_store_projection = bool(store_projection_matrix)
        active_nodes, active_weights = _active_normal_rule(
            basis.dimension,
            selected_active_order,
            rules=selected_active_rules,
            truncations=selected_active_truncations,
            max_nodes=selected_max_active_nodes,
        )
        residual_nodes, residual_weights = _normal_rule(selected_residual_order)
        basis_matrix = basis.evaluate(active_nodes)
        if selected_store_projection:
            projection: np.ndarray | None = basis_matrix.T * active_weights
            gram = projection @ basis_matrix
        else:
            projection = None
            gram = np.zeros((basis.size, basis.size), dtype=float)
            for start in range(0, active_nodes.shape[0], selected_gram_chunk_size):
                stop = min(
                    start + selected_gram_chunk_size,
                    active_nodes.shape[0],
                )
                selected_basis = basis_matrix[start:stop]
                selected_weights = active_weights[start:stop]
                gram += selected_basis.T @ (
                    selected_weights[:, None] * selected_basis
                )
        gram_error = float(
            np.linalg.norm(gram - np.eye(basis.size), ord=2)
        )
        if not np.isfinite(gram_error) or gram_error > 5e-11:
            raise FloatingPointError("active quadrature does not resolve the basis")
        self.basis = basis
        self.drift = drift
        self.active_quadrature_order = selected_active_order
        self.active_quadrature_rule = selected_active_rules
        self.active_truncation = selected_active_truncations
        self.residual_quadrature_order = selected_residual_order
        self.max_active_nodes = selected_max_active_nodes
        self.store_projection_matrix = selected_store_projection
        self.gram_chunk_size = selected_gram_chunk_size
        self.active_nodes = active_nodes
        self.active_weights = active_weights
        self.residual_nodes = residual_nodes
        self.residual_weights = residual_weights
        self.basis_matrix = basis_matrix
        self.projection = projection
        self.gram_error = gram_error
        self.cosine_residual_rule = cosine_residual_rule

    def project(
        self,
        coefficients: np.ndarray,
        active_loading: np.ndarray,
        residual_variance: float,
    ) -> tuple[np.ndarray, ResidualClosureDiagnostic]:
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
            raise ValueError("residual-closure inputs must be finite/nonnegative")
        lawson_values = self.basis_matrix @ state
        active_scalar = self.active_nodes @ loading
        active_variance = float(np.dot(loading, loading))
        contour_rule = self.cosine_residual_rule
        use_real_cosine = contour_rule is None
        evaluated_residual_nodes = 0
        has_real_axis_terms = any(
            transform.scale != 0.0
            and (use_real_cosine or not isinstance(transform, CosineTransform))
            for transform in self.drift.transforms
        )
        if has_real_axis_terms:
            total_variance = active_variance + variance
            if variance == 0.0:
                exponent = active_scalar + 0.5 * total_variance
                conditional = _pulled_back_spectral_transforms(
                    self.drift,
                    lawson_values,
                    exponent,
                    include_cosine=use_real_cosine,
                )
                evaluated_residual_nodes = 1
            else:
                residual_scalar = np.sqrt(variance) * self.residual_nodes
                exponent = (
                    active_scalar[:, None]
                    + residual_scalar[None, :]
                    + 0.5 * total_variance
                )
                pulled_back = _pulled_back_spectral_transforms(
                    self.drift,
                    lawson_values[:, None],
                    exponent,
                    include_cosine=use_real_cosine,
                )
                conditional = pulled_back @ self.residual_weights
                evaluated_residual_nodes = int(self.residual_nodes.size)
        else:
            conditional = np.zeros(self.active_nodes.shape[0], dtype=float)

        maximum_cosine_phase = 0.0
        cosine_method = "real_gauss_hermite"
        if contour_rule is not None:
            used_contour_cosine = False
            for transform in self.drift.transforms:
                if (
                    not isinstance(transform, CosineTransform)
                    or transform.scale == 0.0
                ):
                    continue
                used_contour_cosine = True
                cosine_conditional, cosine_diagnostic = (
                    contour_rule.pulled_back_cosine(
                        lawson_values,
                        active_scalar,
                        active_variance,
                        variance,
                        scale=transform.scale,
                        frequency=transform.frequency,
                    )
                )
                conditional = conditional + cosine_conditional
                maximum_cosine_phase = max(
                    maximum_cosine_phase,
                    cosine_diagnostic.maximum_phase,
                )
                evaluated_residual_nodes = max(
                    evaluated_residual_nodes,
                    1 if variance == 0.0 else int(contour_rule.order),
                )
            if used_contour_cosine:
                cosine_method = "esscher_contour_shifted_gauss_hermite"
        # The affine Lawson terms are available in closed form.  Evaluating
        # them analytically prevents a large-residual exponential moment from
        # being mistaken for a nonlinear quadrature error.
        if self.projection is None:
            nonlinear_projection = self.basis_matrix.T @ (
                self.active_weights * conditional
            )
        else:
            nonlinear_projection = self.projection @ conditional
        rhs = (
            self.drift.rate * state
            + self.drift.intercept
            * self.basis.exact_wick_exponential_coefficients(-loading)
            + nonlinear_projection
        )
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError("residual-closure projection is nonfinite")
        return rhs, ResidualClosureDiagnostic(
            active_node_count=int(self.active_nodes.shape[0]),
            residual_node_count=evaluated_residual_nodes,
            residual_variance=variance,
            projected_rhs_norm=float(np.linalg.norm(rhs, ord=np.inf)),
            cosine_residual_method=cosine_method,
            maximum_cosine_phase=maximum_cosine_phase,
        )


@dataclass(frozen=True, slots=True)
class ResidualCoherentLawsonResult:
    times: np.ndarray
    lawson_coefficients: np.ndarray
    coherent_means: np.ndarray
    coherent_second_moments: np.ndarray
    coherent_variances: np.ndarray
    retained_physical_second_moments: np.ndarray
    omitted_energy: np.ndarray
    residual_variances: np.ndarray
    coherent_cross_moments: np.ndarray
    coherent_covariance_matrix: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "times",
            "lawson_coefficients",
            "coherent_means",
            "coherent_second_moments",
            "coherent_variances",
            "retained_physical_second_moments",
            "omitted_energy",
            "residual_variances",
            "coherent_cross_moments",
            "coherent_covariance_matrix",
        ):
            values = np.asarray(getattr(self, name)).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class ResidualCoherentLawsonStage:
    """Immutable RK-stage snapshot for governed numerical audits."""

    time: float
    coefficient_state: np.ndarray
    projected_rhs: np.ndarray
    active_loading: np.ndarray
    residual_variance: float
    diagnostic: ResidualClosureDiagnostic

    def __post_init__(self) -> None:
        source_state = np.ascontiguousarray(self.coefficient_state, dtype=float)
        source_rhs = np.ascontiguousarray(self.projected_rhs, dtype=float)
        source_loading = np.ascontiguousarray(self.active_loading, dtype=float)
        state = np.frombuffer(
            source_state.tobytes(),
            dtype=source_state.dtype,
        ).reshape(source_state.shape)
        loading = np.frombuffer(
            source_loading.tobytes(),
            dtype=source_loading.dtype,
        ).reshape(source_loading.shape)
        projected_rhs = np.frombuffer(
            source_rhs.tobytes(),
            dtype=source_rhs.dtype,
        ).reshape(source_rhs.shape)
        object.__setattr__(self, "coefficient_state", state)
        object.__setattr__(self, "projected_rhs", projected_rhs)
        object.__setattr__(self, "active_loading", loading)


class ResidualCoherentLawsonSolver:
    """Deterministic active-chaos solver with residual marginalization."""

    implementation_label = "residual_covariance_coherent_lawson_v1_prototype"

    def __init__(
        self,
        factor: GaussianPathFactor,
        degree: int,
        beta: float,
        *,
        drift: AffineSpectralDrift,
        active_quadrature_order: int | tuple[int, ...],
        residual_quadrature_order: int,
        drift_substeps: int = 1,
        cosine_residual_rule: ContourShiftedLognormalRule | None = None,
        active_quadrature_rule: str | tuple[str, ...] = "gauss_hermite",
        active_truncation: float | tuple[float | None, ...] | None = None,
        max_active_nodes: int = 2_000_000,
        store_projection_matrix: bool = True,
        gram_chunk_size: int = 32_768,
    ) -> None:
        if not isinstance(factor, GaussianPathFactor):
            raise TypeError("factor must be GaussianPathFactor")
        raw_loadings = np.asarray(factor.loadings, dtype=float)
        if raw_loadings.ndim != 2 or raw_loadings.shape[1] < 1:
            raise ValueError("Gaussian path factor loadings must be two-dimensional")
        factor_modes = int(raw_loadings.shape[1])
        selected_degree = _integer_control(degree, "degree", minimum=0)
        selected_substeps = _integer_control(
            drift_substeps,
            "drift_substeps",
            minimum=1,
        )
        selected_active_order = _active_order_control(
            active_quadrature_order,
            factor_modes,
        )
        selected_active_rules = _active_rule_control(
            active_quadrature_rule,
            factor_modes,
        )
        selected_active_truncations = _active_truncation_control(
            active_truncation,
            selected_active_rules,
        )
        selected_residual_order = _integer_control(
            residual_quadrature_order,
            "residual_quadrature_order",
            minimum=1,
        )
        selected_beta = float(beta)
        if not np.isfinite(selected_beta):
            raise ValueError("beta must be finite")
        times = np.asarray(factor.times, dtype=float)
        loadings = raw_loadings
        covariance = np.asarray(factor.covariance, dtype=float)
        residual = np.asarray(factor.residual, dtype=float)
        node_count = times.size
        if (
            times.ndim != 1
            or node_count < 1
            or times[0] <= 0.0
            or np.any(np.diff(times) <= 0.0)
            or loadings.shape != (node_count, factor_modes)
            or covariance.shape != (node_count, node_count)
            or residual.shape != (node_count, node_count)
            or not all(
                np.all(np.isfinite(values))
                for values in (times, loadings, covariance, residual)
            )
        ):
            raise ValueError("Gaussian path factor arrays are invalid")
        symmetry_scale = max(float(np.linalg.norm(residual, ord=2)), 1.0)
        if float(np.linalg.norm(residual - residual.T, ord=2)) > (
            2.0e-12 * symmetry_scale
        ):
            raise ValueError("Gaussian path residual covariance is not symmetric")
        residual_eigenvalues = np.linalg.eigvalsh(0.5 * (residual + residual.T))
        if float(np.min(residual_eigenvalues)) < -2.0e-11 * symmetry_scale:
            raise ValueError("Gaussian path residual covariance is not PSD")
        if factor.covariance_reconstruction_error > 5.0e-11:
            raise ValueError("Gaussian path factor fails covariance reconstruction")
        self.factor = factor
        self.basis = HermiteBasis(factor_modes, selected_degree)
        self.beta = selected_beta
        self.drift = drift
        self.drift_substeps = selected_substeps
        self.projection = ResidualCoherentLawsonProjection(
            self.basis,
            drift,
            active_quadrature_order=selected_active_order,
            residual_quadrature_order=selected_residual_order,
            cosine_residual_rule=cosine_residual_rule,
            active_quadrature_rule=selected_active_rules,
            active_truncation=selected_active_truncations,
            max_active_nodes=max_active_nodes,
            store_projection_matrix=store_projection_matrix,
            gram_chunk_size=gram_chunk_size,
        )

    def _loading_and_residual(
        self,
        time: float,
        left_index: int,
        fraction: float,
    ) -> tuple[np.ndarray, float]:
        selected_time = float(time)
        theta = float(fraction)
        if not np.isfinite(selected_time) or not 0.0 <= theta <= 1.0:
            raise ValueError("stage time/fraction must be finite and admissible")
        left_time = 0.0 if left_index == 0 else float(
            self.factor.times[left_index - 1]
        )
        right_time = float(self.factor.times[left_index])
        interpolated_time = left_time + theta * (right_time - left_time)
        time_scale = max(abs(left_time), abs(right_time), 1.0)
        if abs(selected_time - interpolated_time) > (
            256.0 * np.finfo(float).eps * time_scale
        ):
            raise ValueError("stage time and interpolation fraction disagree")
        zero = np.zeros(self.factor.modes, dtype=float)
        left = zero if left_index == 0 else self.factor.loadings[left_index - 1]
        right = self.factor.loadings[left_index]
        active = left + theta * (right - left)

        residual_right = float(self.factor.residual[left_index, left_index])
        if left_index == 0:
            residual_left = 0.0
            residual_cross = 0.0
            covariance_left = 0.0
            covariance_cross = 0.0
        else:
            residual_left = float(
                self.factor.residual[left_index - 1, left_index - 1]
            )
            residual_cross = float(
                self.factor.residual[left_index - 1, left_index]
            )
            covariance_left = float(
                self.factor.covariance[left_index - 1, left_index - 1]
            )
            covariance_cross = float(
                self.factor.covariance[left_index - 1, left_index]
            )
        residual = (
            (1.0 - theta) ** 2 * residual_left
            + 2.0 * theta * (1.0 - theta) * residual_cross
            + theta**2 * residual_right
        )
        covariance_right = float(
            self.factor.covariance[left_index, left_index]
        )
        total_variance = (
            (1.0 - theta) ** 2 * covariance_left
            + 2.0 * theta * (1.0 - theta) * covariance_cross
            + theta**2 * covariance_right
        )
        reconstructed = float(np.dot(active, active) + residual)
        scale = max(total_variance, reconstructed, 1.0)
        tolerance = 512.0 * np.finfo(float).eps * scale
        if abs(reconstructed - total_variance) > tolerance:
            raise FloatingPointError(
                "interpolated active/residual covariance does not reconstruct "
                "the interpolated path variance"
            )
        if residual < -tolerance:
            raise FloatingPointError(
                "interpolated residual path variance is negative"
            )
        residual = max(residual, 0.0)
        return self.beta * active, self.beta * self.beta * residual

    def solve(
        self,
        initial_value: float,
        *,
        stage_observer: Callable[[ResidualCoherentLawsonStage], None] | None = None,
    ) -> ResidualCoherentLawsonResult:
        selected_initial_value = float(initial_value)
        if not np.isfinite(selected_initial_value):
            raise ValueError("initial_value must be finite")
        if stage_observer is not None and not callable(stage_observer):
            raise TypeError("stage_observer must be callable")
        node_times = self.factor.times
        times = np.concatenate(([0.0], node_times))
        coefficients = np.zeros((times.size, self.basis.size), dtype=float)
        coefficients[0, 0] = selected_initial_value
        state = coefficients[0].copy()

        for step in range(node_times.size):
            left_time = times[step]
            right_time = times[step + 1]
            cell = right_time - left_time
            substep = cell / self.drift_substeps
            current = left_time

            def rhs(evaluation_time: float, values: np.ndarray) -> np.ndarray:
                fraction = min(
                    1.0,
                    max(0.0, (evaluation_time - left_time) / cell),
                )
                loading, residual = self._loading_and_residual(
                    evaluation_time,
                    step,
                    fraction,
                )
                projected, diagnostic = self.projection.project(
                    values,
                    loading,
                    residual,
                )
                if stage_observer is not None:
                    stage_observer(
                        ResidualCoherentLawsonStage(
                            time=float(evaluation_time),
                            coefficient_state=values,
                            projected_rhs=projected,
                            active_loading=loading,
                            residual_variance=float(residual),
                            diagnostic=diagnostic,
                        )
                    )
                return projected

            for _ in range(self.drift_substeps):
                k1 = rhs(current, state)
                k2 = rhs(current + 0.5 * substep, state + 0.5 * substep * k1)
                k3 = rhs(current + 0.5 * substep, state + 0.5 * substep * k2)
                k4 = rhs(current + substep, state + substep * k3)
                state = state + (substep / 6.0) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4
                )
                current += substep
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("residual-closure coefficient state is nonfinite")
            coefficients[step + 1] = state

        means = np.empty(times.size, dtype=float)
        seconds = np.empty(times.size, dtype=float)
        variances = np.empty(times.size, dtype=float)
        retained_seconds = np.empty(times.size, dtype=float)
        omitted = np.empty(times.size, dtype=float)
        residual_variances = np.empty(times.size, dtype=float)
        zero = np.zeros(self.factor.modes, dtype=float)
        for index, time in enumerate(times):
            loading = zero if index == 0 else self.beta * self.factor.loadings[index - 1]
            if index == 0:
                residual = 0.0
            else:
                raw_residual = float(
                    self.factor.residual[index - 1, index - 1]
                )
                residual_scale = max(
                    float(self.factor.covariance[index - 1, index - 1]),
                    1.0,
                )
                if raw_residual < (
                    -512.0 * np.finfo(float).eps * residual_scale
                ):
                    raise FloatingPointError(
                        "nodal residual path variance is negative"
                    )
                residual = self.beta * self.beta * max(raw_residual, 0.0)
            # ``coherent_reconstruction`` accepts retained physical
            # coefficients, not Lawson coefficients.  Form P_A D_a Y_A
            # first; the inverse finite creation map then recovers Y_A and
            # the routine evaluates the complete coherent active output.
            retained_physical = self.basis.apply_wick_exponential(
                coefficients[index], loading
            )
            active = coherent_reconstruction(
                self.basis,
                retained_physical,
                loading,
            )
            residual_multiplier = float(np.exp(residual))
            residual_excess = float(np.expm1(residual))
            full_second = float(
                residual_multiplier * active.coherent_second_moment
            )
            _mean, _retained_variance, retained_second = gaussian_moments(
                retained_physical
            )
            means[index] = active.coherent_mean
            seconds[index] = full_second
            variances[index] = (
                residual_multiplier * active.coherent_variance
                + active.coherent_mean**2 * residual_excess
            )
            retained_seconds[index] = retained_second
            omitted[index] = (
                active.tau_squared
                + residual_excess * active.coherent_second_moment
            )
            residual_variances[index] = residual
        arrays = (
            coefficients,
            means,
            seconds,
            variances,
            retained_seconds,
            omitted,
            residual_variances,
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise FloatingPointError("residual-closure result is nonfinite")
        if np.any(variances < -5e-12) or np.any(omitted < -5e-12):
            raise FloatingPointError("residual-closure energy identity is negative")

        loadings = np.vstack(
            (
                np.zeros((1, self.factor.modes), dtype=float),
                self.beta * self.factor.loadings,
            )
        )
        residual_covariance = np.zeros((times.size, times.size), dtype=float)
        residual_covariance[1:, 1:] = (
            self.beta * self.beta * self.factor.residual
        )
        cross_moments = np.empty((times.size, times.size), dtype=float)
        for left in range(times.size):
            for right in range(left, times.size):
                value = residual_coherent_cross_moment(
                    self.basis,
                    coefficients[left],
                    coefficients[right],
                    loadings[left],
                    loadings[right],
                    residual_covariance[left, right],
                )
                cross_moments[left, right] = value
                cross_moments[right, left] = value
        covariance_matrix = cross_moments - np.outer(means, means)
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        diagonal_error = float(
            np.max(np.abs(np.diag(cross_moments) - seconds), initial=0.0)
        )
        diagonal_scale = max(float(np.max(np.abs(seconds), initial=0.0)), 1.0)
        if diagonal_error > 2.0e-11 * diagonal_scale:
            raise FloatingPointError(
                "coherent cross-moment diagonal does not match second moments"
            )
        np.fill_diagonal(cross_moments, seconds)
        np.fill_diagonal(covariance_matrix, np.maximum(variances, 0.0))
        minimum_covariance_eigenvalue = float(
            np.min(np.linalg.eigvalsh(covariance_matrix), initial=0.0)
        )
        covariance_scale = max(
            float(np.linalg.norm(covariance_matrix, ord=2)),
            1.0,
        )
        if minimum_covariance_eigenvalue < -2.0e-10 * covariance_scale:
            raise FloatingPointError(
                "coherent covariance matrix is not positive semidefinite"
            )
        return ResidualCoherentLawsonResult(
            times,
            coefficients,
            means,
            seconds,
            np.maximum(variances, 0.0),
            retained_seconds,
            np.maximum(omitted, 0.0),
            residual_variances,
            cross_moments,
            covariance_matrix,
        )

    def sample_coherent_paths(
        self,
        result: ResidualCoherentLawsonResult,
        sample_count: int,
        *,
        seed: int,
    ) -> np.ndarray:
        """Sample joint grid paths from an already propagated active state."""

        if not isinstance(result, ResidualCoherentLawsonResult):
            raise TypeError("result must be ResidualCoherentLawsonResult")
        count = _integer_control(sample_count, "sample_count", minimum=1)
        selected_seed = _integer_control(seed, "seed", minimum=0)
        expected_times = np.concatenate(([0.0], self.factor.times))
        if (
            result.times.shape != expected_times.shape
            or not np.array_equal(result.times, expected_times)
        ):
            raise ValueError("result time grid is incompatible with the solver")
        if result.lawson_coefficients.shape != (
            result.times.size,
            self.basis.size,
        ):
            raise ValueError("result coefficient trajectory is incompatible")

        residual = 0.5 * (self.factor.residual + self.factor.residual.T)
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
        if float(np.min(eigenvalues, initial=0.0)) < -2.0e-11 * scale:
            raise FloatingPointError("path residual covariance is not PSD")
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
        generator = np.random.default_rng(selected_seed)
        active_samples = generator.standard_normal(
            (count, self.factor.modes)
        )
        residual_samples = (
            generator.standard_normal((count, self.factor.times.size))
            @ root.T
        )
        paths = np.empty((count, result.times.size), dtype=float)
        paths[:, 0] = self.basis.evaluate(active_samples) @ (
            result.lawson_coefficients[0]
        )
        for index in range(1, result.times.size):
            loading = self.beta * self.factor.loadings[index - 1]
            residual_value = self.beta * residual_samples[:, index - 1]
            total_variance = self.beta * self.beta * float(
                self.factor.covariance[index - 1, index - 1]
            )
            lawson_values = self.basis.evaluate(
                active_samples - loading
            ) @ result.lawson_coefficients[index]
            exponent = (
                active_samples @ loading
                + residual_value
                - 0.5 * total_variance
            )
            paths[:, index] = lawson_values * np.exp(exponent)
        if not np.all(np.isfinite(paths)):
            raise FloatingPointError("sampled coherent path is nonfinite")
        return paths


__all__ = [
    "ResidualClosureDiagnostic",
    "ResidualCoherentLawsonProjection",
    "ResidualCoherentLawsonResult",
    "ResidualCoherentLawsonStage",
    "ResidualCoherentLawsonSolver",
    "pulled_back_affine_spectral_drift",
    "residual_coherent_cross_moment",
]
