"""Unfrozen, non-evidence streamed active-GL RHS projection draft.

This module is an implementation candidate for prospective Paper IX work.  It
is deliberately not imported by :mod:`wis_chaos` and has no executable CLI.
Nothing in this file authorizes a numerical run or changes any frozen V5,
V5R1, V5R2, or V5R2A artifact.

The production ``ResidualCoherentLawsonProjection`` materializes the complete
active tensor nodes, weights, and basis matrix.  The candidate below preserves
the same one-dimensional rules, C-order tensor traversal, pointwise drift
math, residual quadrature, contour rule, and affine control, but constructs
only bounded active-node chunks.  The tensor Gram matrix is evaluated from
the exact product of one-dimensional Gram matrices, so setup is independent
of the full tensor-node count.

Each nonlinear projection is accumulated twice:

* deterministic balanced pairwise float64 sums of chunk contributions; and
* an independent Kahan-compensated audit sum whose subchunk products and sums
  use ``numpy.longdouble`` where it is wider than float64 (and compensated
  float64 on platforms, including Windows, where it is not).

The two answers must satisfy the configured fail-closed discrepancy bound.
All BLAS-backed work is executed under an enforced one-thread limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterator

import numpy as np
from scipy.special import roots_legendre

try:
    from threadpoolctl import threadpool_limits
except ImportError as error:  # pragma: no cover - exercised by import failure
    raise ImportError(
        "the streamed projection draft requires threadpoolctl for an "
        "enforced single-thread BLAS scope"
    ) from error

from .basis import HermiteBasis
from .contour_shifted_lognormal import ContourShiftedLognormalRule
from .residual_coherent_lawson import (
    _active_order_control,
    _active_rule_control,
    _active_truncation_control,
    _integer_control,
    _normal_rule,
    _pulled_back_spectral_transforms,
)
from .spectral_projection import (
    AffineSpectralDrift,
    CosineTransform,
)


NON_EVIDENCE_DRAFT_UNFROZEN = True
IMPLEMENTATION_LABEL = (
    "non_evidence_draft_unfrozen_streamed_active_gl_rhs_v1"
)
SCIPY_RULE_IMPLEMENTATION = "scipy_special_roots_legendre"
NUMPY_SENTINEL_RULE_IMPLEMENTATION = "numpy_polynomial_legendre_leggauss"
RULE_IMPLEMENTATIONS = frozenset(
    {SCIPY_RULE_IMPLEMENTATION, NUMPY_SENTINEL_RULE_IMPLEMENTATION}
)


def _finite_nonnegative(value: object, name: str) -> float:
    selected = float(value)
    if not np.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return selected


def _one_dimensional_gl_rule(
    order: int,
    truncation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly the truncated-normal GL rule used by the baseline."""

    base_nodes, base_weights = roots_legendre(order)
    nodes = np.asarray(truncation * base_nodes, dtype=float)
    weights = np.asarray(
        truncation
        * base_weights
        * np.exp(-0.5 * nodes * nodes)
        / np.sqrt(2.0 * np.pi),
        dtype=float,
    )
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
    ):
        raise FloatingPointError("active GL rule construction failed")
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _one_dimensional_numpy_leggauss_rule(
    order: int,
    truncation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent-implementation sentinel for the same governed GL rule."""

    base_nodes, base_weights = np.polynomial.legendre.leggauss(order)
    nodes = np.asarray(truncation * base_nodes, dtype=float)
    weights = np.asarray(
        truncation
        * base_weights
        * np.exp(-0.5 * nodes * nodes)
        / np.sqrt(2.0 * np.pi),
        dtype=float,
    )
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
    ):
        raise FloatingPointError(
            "independent active GL sentinel rule construction failed"
        )
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _factorized_gram(
    basis: HermiteBasis,
    rules: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Build the tensor Gram matrix without enumerating tensor nodes."""

    univariate_basis = HermiteBasis(1, basis.degree)
    one_dimensional_grams: list[np.ndarray] = []
    with threadpool_limits(limits=1, user_api="blas"):
        for nodes, weights in rules:
            values = univariate_basis.evaluate(nodes)
            one_dimensional_grams.append(
                values.T @ (weights[:, None] * values)
            )
    indices = np.asarray(basis.indices, dtype=np.int64)
    gram = np.ones((basis.size, basis.size), dtype=float)
    for coordinate, coordinate_gram in enumerate(one_dimensional_grams):
        orders = indices[:, coordinate]
        gram *= coordinate_gram[orders[:, None], orders[None, :]]
    return gram


class _PairwiseFloat64Accumulator:
    """Online balanced pairwise sum with O(log(number of chunks)) storage."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape
        self._bins: list[np.ndarray | None] = []
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        item = np.asarray(values, dtype=np.float64)
        if item.shape != self._shape or not np.all(np.isfinite(item)):
            raise FloatingPointError("pairwise accumulation input is invalid")
        item = item.copy()
        level = 0
        while level < len(self._bins) and self._bins[level] is not None:
            item = self._bins[level] + item  # type: ignore[operator]
            self._bins[level] = None
            level += 1
        if level == len(self._bins):
            self._bins.append(item)
        else:
            self._bins[level] = item
        self.count += 1

    def total(self) -> np.ndarray:
        result: np.ndarray | None = None
        # Higher bins contain earlier, larger adjacent blocks.  Combining in
        # descending order preserves deterministic left-to-right block order.
        for item in reversed(self._bins):
            if item is None:
                continue
            result = item.copy() if result is None else result + item
        if result is None:
            return np.zeros(self._shape, dtype=np.float64)
        return result


class _CompensatedLongDoubleAccumulator:
    """Vector Kahan sum in longdouble (or compensated float64 on Windows)."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape
        self._total = np.zeros(shape, dtype=np.longdouble)
        self._correction = np.zeros(shape, dtype=np.longdouble)
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        item = np.asarray(values, dtype=np.longdouble)
        if item.shape != self._shape or not np.all(np.isfinite(item)):
            raise FloatingPointError("compensated accumulation input is invalid")
        adjusted = item - self._correction
        updated = self._total + adjusted
        self._correction = (updated - self._total) - adjusted
        self._total = updated
        self.count += 1

    def total(self) -> np.ndarray:
        return self._total.copy()


@dataclass(frozen=True, slots=True)
class StreamedResidualClosureDiagnostic:
    """Per-call diagnostics, including the independent accumulation check."""

    active_node_count: int
    residual_node_count: int
    residual_variance: float
    projected_rhs_norm: float
    cosine_residual_method: str
    maximum_cosine_phase: float
    active_chunk_count: int
    active_chunk_size: int
    primary_accumulation_method: str
    shadow_accumulation_method: str
    shadow_precision_bits: int
    accumulation_discrepancy_inf: float
    accumulation_tolerance: float
    accumulation_consistent: bool
    active_rule_implementation: str


class NonEvidenceDraftStreamedResidualGLProjection:
    """Bounded-memory active-GL version of the residual coherent projector.

    This class intentionally accepts only coordinate-wise truncated
    Gauss--Legendre active rules.  It cannot silently replace the existing
    projector in a frozen runner because it does not expose full
    ``active_nodes``, ``active_weights``, ``basis_matrix``, or ``projection``
    arrays.
    """

    implementation_label = IMPLEMENTATION_LABEL
    store_projection_matrix = False
    blas_thread_limit = 1

    def __init__(
        self,
        basis: HermiteBasis,
        drift: AffineSpectralDrift,
        *,
        active_quadrature_order: tuple[int, ...],
        residual_quadrature_order: int,
        active_truncation: tuple[float, ...],
        cosine_residual_rule: ContourShiftedLognormalRule | None = None,
        active_quadrature_rule: str | tuple[str, ...] = (
            "truncated_gauss_legendre"
        ),
        max_active_nodes: int = 10_000_000,
        active_chunk_size: int = 4_096,
        shadow_subchunk_size: int = 257,
        accumulation_rtol: float = 2.0e-11,
        accumulation_atol: float = 5.0e-14,
        active_rule_implementation: str = SCIPY_RULE_IMPLEMENTATION,
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
        selected_order = _active_order_control(
            active_quadrature_order,
            basis.dimension,
        )
        if isinstance(selected_order, int):
            raise ValueError(
                "streamed active-GL draft requires an explicit order per axis"
            )
        selected_rules = _active_rule_control(
            active_quadrature_rule,
            basis.dimension,
        )
        if any(rule != "truncated_gauss_legendre" for rule in selected_rules):
            raise ValueError(
                "streamed active-GL draft accepts only truncated GL axes"
            )
        selected_truncations = _active_truncation_control(
            active_truncation,
            selected_rules,
        )
        if any(value is None for value in selected_truncations):
            raise ValueError("every streamed active-GL axis needs a truncation")
        selected_residual_order = _integer_control(
            residual_quadrature_order,
            "residual_quadrature_order",
            minimum=1,
        )
        selected_max_nodes = _integer_control(
            max_active_nodes,
            "max_active_nodes",
            minimum=1,
        )
        selected_chunk_size = _integer_control(
            active_chunk_size,
            "active_chunk_size",
            minimum=1,
        )
        selected_shadow_size = _integer_control(
            shadow_subchunk_size,
            "shadow_subchunk_size",
            minimum=1,
        )
        selected_rtol = _finite_nonnegative(
            accumulation_rtol,
            "accumulation_rtol",
        )
        selected_atol = _finite_nonnegative(
            accumulation_atol,
            "accumulation_atol",
        )
        if (
            not isinstance(active_rule_implementation, str)
            or active_rule_implementation not in RULE_IMPLEMENTATIONS
        ):
            raise ValueError(
                "active_rule_implementation is not an exact governed choice"
            )
        orders = tuple(int(value) for value in selected_order)
        node_count = int(prod(orders))
        if node_count > selected_max_nodes:
            raise MemoryError(
                f"anisotropic tensor rule would have {node_count:,} nodes"
            )
        rule_factory = (
            _one_dimensional_gl_rule
            if active_rule_implementation == SCIPY_RULE_IMPLEMENTATION
            else _one_dimensional_numpy_leggauss_rule
        )
        one_dimensional_rules = tuple(
            rule_factory(order, float(truncation))
            for order, truncation in zip(
                orders,
                selected_truncations,
                strict=True,
            )
        )
        active_weight_mass = float(
            prod(float(np.sum(weights)) for _nodes, weights in one_dimensional_rules)
        )
        if (
            not np.isfinite(active_weight_mass)
            or abs(active_weight_mass - 1.0) > 1.0e-9
        ):
            raise FloatingPointError(
                "streamed active quadrature mass is outside tolerance"
            )
        gram = _factorized_gram(basis, one_dimensional_rules)
        gram_error = float(np.linalg.norm(gram - np.eye(basis.size), ord=2))
        if not np.isfinite(gram_error) or gram_error > 5.0e-11:
            raise FloatingPointError("active quadrature does not resolve the basis")
        residual_nodes, residual_weights = _normal_rule(selected_residual_order)

        self.basis = basis
        self.drift = drift
        self.active_quadrature_order = orders
        self.active_quadrature_rule = selected_rules
        self.active_truncation = tuple(
            float(value) for value in selected_truncations if value is not None
        )
        self.residual_quadrature_order = selected_residual_order
        self.max_active_nodes = selected_max_nodes
        self.active_chunk_size = selected_chunk_size
        self.shadow_subchunk_size = selected_shadow_size
        self.accumulation_rtol = selected_rtol
        self.accumulation_atol = selected_atol
        self.active_rule_implementation = active_rule_implementation
        self.active_node_count = node_count
        self.active_weight_mass = active_weight_mass
        self.gram_error = gram_error
        self.cosine_residual_rule = cosine_residual_rule
        self._one_dimensional_rules = one_dimensional_rules
        self._residual_nodes = residual_nodes
        self._residual_weights = residual_weights

    @property
    def persistent_quadrature_bytes(self) -> int:
        """Bytes retained for one-dimensional active and residual rules."""

        return int(
            sum(
                nodes.nbytes + weights.nbytes
                for nodes, weights in self._one_dimensional_rules
            )
            + self._residual_nodes.nbytes
            + self._residual_weights.nbytes
        )

    def iter_active_chunks(
        self,
    ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        """Yield C-order tensor slabs without retaining a full tensor array."""

        orders = self.active_quadrature_order
        for start in range(0, self.active_node_count, self.active_chunk_size):
            stop = min(start + self.active_chunk_size, self.active_node_count)
            linear = np.arange(start, stop, dtype=np.int64)
            coordinates = np.unravel_index(linear, orders, order="C")
            nodes = np.empty((stop - start, self.basis.dimension), dtype=float)
            weights = np.ones(stop - start, dtype=float)
            for coordinate, ((axis_nodes, axis_weights), indices) in enumerate(
                zip(self._one_dimensional_rules, coordinates, strict=True)
            ):
                nodes[:, coordinate] = axis_nodes[indices]
                weights *= axis_weights[indices]
            if not (
                np.all(np.isfinite(nodes))
                and np.all(np.isfinite(weights))
                and np.all(weights > 0.0)
            ):
                raise FloatingPointError("streamed tensor chunk is invalid")
            yield start, stop, nodes, weights

    def _conditional_nonlinearity(
        self,
        lawson_values: np.ndarray,
        active_scalar: np.ndarray,
        active_variance: float,
        residual_variance: float,
    ) -> tuple[np.ndarray, int, str, float]:
        contour_rule = self.cosine_residual_rule
        use_real_cosine = contour_rule is None
        evaluated_residual_nodes = 0
        has_real_axis_terms = any(
            transform.scale != 0.0
            and (use_real_cosine or not isinstance(transform, CosineTransform))
            for transform in self.drift.transforms
        )
        if has_real_axis_terms:
            total_variance = active_variance + residual_variance
            if residual_variance == 0.0:
                exponent = active_scalar + 0.5 * total_variance
                conditional = _pulled_back_spectral_transforms(
                    self.drift,
                    lawson_values,
                    exponent,
                    include_cosine=use_real_cosine,
                )
                evaluated_residual_nodes = 1
            else:
                residual_scalar = (
                    np.sqrt(residual_variance) * self._residual_nodes
                )
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
                conditional = pulled_back @ self._residual_weights
                evaluated_residual_nodes = int(self._residual_nodes.size)
        else:
            conditional = np.zeros(lawson_values.shape, dtype=float)

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
                        residual_variance,
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
                    1
                    if residual_variance == 0.0
                    else int(contour_rule.order),
                )
            if used_contour_cosine:
                cosine_method = "esscher_contour_shifted_gauss_hermite"
        if not np.all(np.isfinite(conditional)):
            raise FloatingPointError("streamed conditional drift is nonfinite")
        return (
            np.asarray(conditional, dtype=float),
            evaluated_residual_nodes,
            cosine_method,
            maximum_cosine_phase,
        )

    def project(
        self,
        coefficients: np.ndarray,
        active_loading: np.ndarray,
        residual_variance: float,
    ) -> tuple[np.ndarray, StreamedResidualClosureDiagnostic]:
        """Project one RHS with bounded memory and dual accumulation."""

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

        primary = _PairwiseFloat64Accumulator((self.basis.size,))
        shadow = _CompensatedLongDoubleAccumulator((self.basis.size,))
        active_variance = float(np.dot(loading, loading))
        evaluated_residual_nodes = 0
        cosine_method = "real_gauss_hermite"
        maximum_cosine_phase = 0.0

        with threadpool_limits(limits=1, user_api="blas"):
            for _start, _stop, nodes, weights in self.iter_active_chunks():
                basis_matrix = self.basis.evaluate(nodes)
                lawson_values = basis_matrix @ state
                active_scalar = nodes @ loading
                (
                    conditional,
                    residual_nodes,
                    chunk_cosine_method,
                    chunk_maximum_phase,
                ) = self._conditional_nonlinearity(
                    lawson_values,
                    active_scalar,
                    active_variance,
                    variance,
                )
                weighted = weights * conditional
                primary.add(basis_matrix.T @ weighted)

                # The audit path deliberately uses a different partition and
                # accumulation algorithm from the primary BLAS dot product.
                for shadow_start in range(
                    0,
                    weighted.size,
                    self.shadow_subchunk_size,
                ):
                    shadow_stop = min(
                        shadow_start + self.shadow_subchunk_size,
                        weighted.size,
                    )
                    selected_basis = np.asarray(
                        basis_matrix[shadow_start:shadow_stop],
                        dtype=np.longdouble,
                    )
                    selected_weighted = np.asarray(
                        weighted[shadow_start:shadow_stop],
                        dtype=np.longdouble,
                    )
                    products = selected_basis * selected_weighted[:, None]
                    shadow.add(
                        np.sum(products, axis=0, dtype=np.longdouble)
                    )
                evaluated_residual_nodes = max(
                    evaluated_residual_nodes,
                    residual_nodes,
                )
                maximum_cosine_phase = max(
                    maximum_cosine_phase,
                    chunk_maximum_phase,
                )
                if chunk_cosine_method != "real_gauss_hermite":
                    cosine_method = chunk_cosine_method

        nonlinear_projection = primary.total()
        shadow_projection_long = shadow.total()
        if not np.all(np.isfinite(shadow_projection_long)):
            raise FloatingPointError("shadow projection is nonfinite")
        shadow_projection = np.asarray(shadow_projection_long, dtype=float)
        affine = (
            self.drift.rate * state
            + self.drift.intercept
            * self.basis.exact_wick_exponential_coefficients(-loading)
        )
        rhs = affine + nonlinear_projection
        shadow_rhs = affine + shadow_projection
        if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(shadow_rhs)):
            raise FloatingPointError("streamed residual projection is nonfinite")
        discrepancy = float(np.linalg.norm(rhs - shadow_rhs, ord=np.inf))
        shadow_scale = float(np.linalg.norm(shadow_rhs, ord=np.inf))
        tolerance = float(
            self.accumulation_atol + self.accumulation_rtol * shadow_scale
        )
        consistent = bool(discrepancy <= tolerance)
        if not consistent:
            raise FloatingPointError(
                "primary/shadow streamed accumulation discrepancy exceeds "
                f"its bound ({discrepancy:.17g} > {tolerance:.17g})"
            )
        return rhs, StreamedResidualClosureDiagnostic(
            active_node_count=self.active_node_count,
            residual_node_count=evaluated_residual_nodes,
            residual_variance=variance,
            projected_rhs_norm=float(np.linalg.norm(rhs, ord=np.inf)),
            cosine_residual_method=cosine_method,
            maximum_cosine_phase=maximum_cosine_phase,
            active_chunk_count=primary.count,
            active_chunk_size=self.active_chunk_size,
            primary_accumulation_method=(
                "balanced_pairwise_float64_chunk_dot"
            ),
            shadow_accumulation_method=(
                "kahan_compensated_longdouble_subchunk_sum"
            ),
            shadow_precision_bits=int(np.finfo(np.longdouble).nmant + 1),
            accumulation_discrepancy_inf=discrepancy,
            accumulation_tolerance=tolerance,
            accumulation_consistent=consistent,
            active_rule_implementation=self.active_rule_implementation,
        )


def main() -> None:
    """Fail closed: this draft is a library candidate, never a runner."""

    raise SystemExit(
        "NON_EVIDENCE_DRAFT_UNFROZEN: no CLI or numerical execution is "
        "authorized"
    )


if __name__ == "__main__":  # pragma: no cover - subprocess test covers it
    main()


__all__ = [
    "IMPLEMENTATION_LABEL",
    "NUMPY_SENTINEL_RULE_IMPLEMENTATION",
    "NON_EVIDENCE_DRAFT_UNFROZEN",
    "RULE_IMPLEMENTATIONS",
    "SCIPY_RULE_IMPLEMENTATION",
    "NonEvidenceDraftStreamedResidualGLProjection",
    "StreamedResidualClosureDiagnostic",
]
