"""Transform-admissible analytic-tail Galerkin projection.

The retained Hermite state is conditioned on a low-dimensional Gaussian
core.  Conditional on that core, the remaining Gaussian tail is a noncentral
quadratic form.  Its characteristic function and the first two
Hermite-weighted characteristic moments are available in closed form.

For a real residual represented by a finite spectral quadrature rule

``r_Q(x) = c + Re sum_q w_q (exp(i*u_q*x) - a_q)``,

the complete retained Galerkin vector can therefore be evaluated without
sampling the Gaussian tail.  Only core Gauss--Hermite quadrature and the
one-dimensional transform rule used to construct ``(u_q, w_q, a_q)`` remain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_laguerre

from .basis import (
    HermiteBasis,
    StructuredQuadraticTailSobolProjection,
    TensorGaussHermite,
)


@dataclass(frozen=True, slots=True)
class SpectralQuadratureRule:
    r"""Finite real spectral rule.

    The represented function is

    .. math::

       c + \operatorname{Re}\sum_q
       w_q\{\exp(iu_qx)-a_q\}.

    Nonzero anchors are useful for removable zero-frequency singularities,
    such as the transform of ``log(1+x*x)``.
    """

    frequencies: tuple[float, ...]
    weights: tuple[complex, ...]
    anchors: tuple[complex, ...]
    constant: float = 0.0
    label: str = "spectral_rule"

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.frequencies)
        weights = tuple(complex(value) for value in self.weights)
        anchors = tuple(complex(value) for value in self.anchors)
        if not (len(frequencies) == len(weights) == len(anchors)):
            raise ValueError(
                "frequencies, weights, and anchors must have equal length"
            )
        if any(value < 0.0 or not np.isfinite(value) for value in frequencies):
            raise ValueError("spectral frequencies must be finite and nonnegative")
        if any(
            not (np.isfinite(value.real) and np.isfinite(value.imag))
            for value in weights + anchors
        ):
            raise ValueError("spectral weights and anchors must be finite")
        constant = float(self.constant)
        if not np.isfinite(constant):
            raise ValueError("spectral constant must be finite")
        label = str(self.label)
        if not label:
            raise ValueError("spectral rule label must be nonempty")
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "label", label)

    @property
    def size(self) -> int:
        return len(self.frequencies)

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        """Evaluate the finite spectral rule pointwise."""

        state = np.asarray(values, dtype=float)
        result = np.full(state.shape, self.constant, dtype=float)
        for frequency, weight, anchor in zip(
            self.frequencies,
            self.weights,
            self.anchors,
        ):
            result += np.real(
                weight
                * (np.exp(1.0j * frequency * state) - anchor)
            )
        return result


def combine_spectral_rules(
    rules: tuple[SpectralQuadratureRule, ...],
    *,
    label: str = "combined_spectral_rule",
) -> SpectralQuadratureRule:
    """Concatenate independent rules without changing their finite sum."""

    frequencies: list[float] = []
    weights: list[complex] = []
    anchors: list[complex] = []
    constant = 0.0
    for rule in rules:
        if not isinstance(rule, SpectralQuadratureRule):
            raise TypeError("rules must contain SpectralQuadratureRule objects")
        frequencies.extend(rule.frequencies)
        weights.extend(rule.weights)
        anchors.extend(rule.anchors)
        constant += rule.constant
    return SpectralQuadratureRule(
        tuple(frequencies),
        tuple(weights),
        tuple(anchors),
        constant=constant,
        label=label,
    )


@lru_cache(maxsize=None)
def _laguerre_nodes_weights(
    order: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if int(order) < 1:
        raise ValueError("laguerre_order must be positive")
    nodes, weights = roots_laguerre(int(order))
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and float(np.sum(weights)) > 0.0
    ):
        raise FloatingPointError(
            "Gauss--Laguerre construction returned nonfinite data"
        )
    return tuple(float(value) for value in nodes), tuple(
        float(value) for value in weights
    )


@dataclass(frozen=True, slots=True)
class RationalSineTransform:
    r"""Term ``scale*x/(width**2+x**2)``."""

    scale: float
    width: float = 1.0

    def __post_init__(self) -> None:
        scale = float(self.scale)
        width = float(self.width)
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("width must be finite and positive")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "width", width)

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        magnitude = np.abs(state)
        result = np.empty_like(state, dtype=float)
        small = magnitude <= self.width
        ratio = state[small] / self.width
        result[small] = (
            (self.scale / self.width)
            * ratio
            / (1.0 + ratio * ratio)
        )
        inverse_ratio = self.width / state[~small]
        result[~small] = (
            (self.scale / state[~small])
            / (1.0 + inverse_ratio * inverse_ratio)
        )
        return result

    def spectral_rule(self, laguerre_order: int) -> SpectralQuadratureRule:
        nodes, laguerre_weights = _laguerre_nodes_weights(laguerre_order)
        multiplier = self.scale / self.width
        return SpectralQuadratureRule(
            tuple(node / self.width for node in nodes),
            tuple(-1.0j * multiplier * weight for weight in laguerre_weights),
            tuple(0.0j for _ in nodes),
            label="rational_sine_laguerre",
        )


@dataclass(frozen=True, slots=True)
class CosineTransform:
    r"""Term ``scale*cos(frequency*x)`` represented by one atom."""

    scale: float
    frequency: float = 1.0

    def __post_init__(self) -> None:
        scale = float(self.scale)
        frequency = float(self.frequency)
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency must be finite and nonnegative")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "frequency", frequency)

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        return self.scale * np.cos(
            self.frequency * np.asarray(values, dtype=float)
        )

    def spectral_rule(self, _laguerre_order: int) -> SpectralQuadratureRule:
        if self.frequency == 0.0:
            return SpectralQuadratureRule(
                (),
                (),
                (),
                constant=self.scale,
                label="constant_cosine_atom",
            )
        return SpectralQuadratureRule(
            (self.frequency,),
            (complex(self.scale),),
            (0.0j,),
            label="cosine_atom",
        )


@dataclass(frozen=True, slots=True)
class LogQuadraticCosineTransform:
    r"""Term ``scale*log(1+(x/width)**2)``."""

    scale: float
    width: float = 1.0

    def __post_init__(self) -> None:
        scale = float(self.scale)
        width = float(self.width)
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("width must be finite and positive")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "width", width)

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        magnitude = np.abs(state)
        result = np.empty_like(state, dtype=float)
        small = magnitude <= self.width
        ratio = state[small] / self.width
        result[small] = self.scale * np.log1p(ratio * ratio)
        inverse_ratio = self.width / state[~small]
        logarithm = (
            2.0
            * (np.log(magnitude[~small]) - np.log(self.width))
            + np.log1p(inverse_ratio * inverse_ratio)
        )
        result[~small] = self.scale * logarithm
        return result

    def spectral_rule(self, laguerre_order: int) -> SpectralQuadratureRule:
        nodes, laguerre_weights = _laguerre_nodes_weights(laguerre_order)
        # 2*log identity with an anchor at u=0:
        # log(1+x^2/a^2) = 2 int exp(-v)(1-cos(v*x/a))/v dv.
        return SpectralQuadratureRule(
            tuple(node / self.width for node in nodes),
            tuple(
                -2.0 * self.scale * weight / node
                for node, weight in zip(nodes, laguerre_weights)
            ),
            tuple(1.0 + 0.0j for _ in nodes),
            label="log_quadratic_cosine_laguerre",
        )


SpectralTransform = (
    RationalSineTransform
    | CosineTransform
    | LogQuadraticCosineTransform
)


@dataclass(frozen=True, slots=True)
class AffineSpectralDrift:
    """Fail-closed affine plus transform-admissible drift specification."""

    intercept: float
    rate: float
    transforms: tuple[SpectralTransform, ...]

    def __post_init__(self) -> None:
        intercept = float(self.intercept)
        rate = float(self.rate)
        transforms = tuple(self.transforms)
        if not np.isfinite(intercept) or not np.isfinite(rate):
            raise ValueError("intercept and rate must be finite")
        if not transforms:
            raise ValueError("at least one spectral transform is required")
        allowed = (
            RationalSineTransform,
            CosineTransform,
            LogQuadraticCosineTransform,
        )
        if any(not isinstance(term, allowed) for term in transforms):
            raise TypeError("unsupported spectral transform specification")
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "transforms", transforms)

    def __call__(self, _time: float, values: np.ndarray) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        result = self.intercept + self.rate * state
        for transform in self.transforms:
            result = result + transform.evaluate(state)
        return result

    def residual_rule(self, laguerre_order: int) -> SpectralQuadratureRule:
        return combine_spectral_rules(
            tuple(
                transform.spectral_rule(int(laguerre_order))
                for transform in self.transforms
            ),
            label="affine_spectral_drift_residual",
        )


@dataclass(frozen=True)
class CoreModulatedSpectralProjection:
    """Analytically eliminate a quadratic Gaussian tail for a spectral drift."""

    basis: HermiteBasis
    core_dimensions: int
    drift: AffineSpectralDrift
    modulation_core_coordinate: int = 0
    core_order: int = 9
    laguerre_order: int = 24
    max_core_nodes: int = 2_000_000

    def __post_init__(self) -> None:
        core = int(self.core_dimensions)
        coordinate = int(self.modulation_core_coordinate)
        if not isinstance(self.drift, AffineSpectralDrift):
            raise TypeError("drift must be an AffineSpectralDrift")
        if self.core_order < 1:
            raise ValueError("core_order must be positive")
        if self.laguerre_order < 1:
            raise ValueError("laguerre_order must be positive")
        if not 0 <= coordinate < core:
            raise ValueError(
                "modulation_core_coordinate must identify a core coordinate"
            )

        structure = StructuredQuadraticTailSobolProjection(
            self.basis,
            core,
            node_count=1,
            seed=0,
            block_size=1,
        )
        maximum_core_degree = max(
            max((int(value) for value in alpha), default=0)
            for core_basis in (
                structure._core_basis,
                structure._linear_core_basis,
            )
            for alpha in core_basis.index_set
        )
        if 2 * int(self.core_order) - 1 < maximum_core_degree:
            raise ValueError(
                "core_order is algebraically underresolved: tensor "
                "Gauss--Hermite exactness requires "
                "2*core_order-1 >= the maximum retained per-coordinate "
                f"core degree ({maximum_core_degree})"
            )
        expected_core_alpha = [0] * core
        expected_core_alpha[coordinate] = 1
        expected_core_alpha_tuple = tuple(expected_core_alpha)
        core_patterns = tuple(
            tuple(int(value) for value in alpha)
            for alpha in structure._core_basis.index_set
        )
        for column in structure._modulated_core_columns:
            if core_patterns[int(column)] != expected_core_alpha_tuple:
                raise ValueError(
                    "every modulated tail pair must use exactly the selected "
                    "degree-one core Hermite coordinate"
                )

        core_rule = TensorGaussHermite(
            core,
            int(self.core_order),
            max_nodes=int(self.max_core_nodes),
        )
        core_nodes, core_weights = core_rule.nodes_and_weights
        core_matrix = structure._core_basis.evaluate(core_nodes)
        linear_core_matrix = structure._linear_core_basis.evaluate(core_nodes)

        modulation_values = core_nodes[:, coordinate]
        unique_values, inverse = np.unique(
            modulation_values,
            return_inverse=True,
        )
        groups = tuple(
            np.flatnonzero(inverse == group)
            for group in range(unique_values.size)
        )
        if unique_values.size != int(self.core_order):
            raise RuntimeError(
                "tensor core rule did not expose the expected number of "
                "modulation-coordinate nodes"
            )

        spectral_rule = self.drift.residual_rule(int(self.laguerre_order))
        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "_core_nodes", core_nodes)
        object.__setattr__(self, "_core_weights", core_weights)
        object.__setattr__(self, "_core_matrix", core_matrix)
        object.__setattr__(self, "_linear_core_matrix", linear_core_matrix)
        object.__setattr__(self, "_modulation_values", unique_values)
        object.__setattr__(self, "_core_groups", groups)
        object.__setattr__(self, "_spectral_rule", spectral_rule)

    @property
    def core_node_count(self) -> int:
        return int(self._core_nodes.shape[0])

    @property
    def spectral_node_count(self) -> int:
        return self._spectral_rule.size

    @property
    def diagonalization_count(self) -> int:
        return len(self._core_groups)

    def _quadratic_matrices(
        self,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        structure = self._structure
        core_values, linear_values, _, symmetric_zero, modulated_values = (
            structure._structured_coefficients(state)
        )
        symmetric_one = np.zeros_like(symmetric_zero)
        for value, left, right, is_diagonal in zip(
            modulated_values,
            structure._modulated_left,
            structure._modulated_right,
            structure._modulated_diagonal,
        ):
            i = int(left)
            j = int(right)
            coefficient = float(value)
            if bool(is_diagonal):
                symmetric_one[i, i] += coefficient / np.sqrt(2.0)
            else:
                symmetric_one[i, j] += 0.5 * coefficient
                symmetric_one[j, i] += 0.5 * coefficient
        return core_values, linear_values, symmetric_zero, symmetric_one

    def project_residual(self, coefficients: np.ndarray) -> np.ndarray:
        """Project only the transform-admissible residual."""

        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")

        structure = self._structure
        (
            core_values,
            linear_values,
            symmetric_zero,
            symmetric_one,
        ) = self._quadratic_matrices(state)
        core_polynomial = self._core_matrix @ core_values
        tail_dimension = symmetric_zero.shape[0]
        linear_core_size = structure._linear_core_basis.size

        projected_core = np.zeros(structure._core_basis.size, dtype=float)
        projected_linear = np.zeros(
            (tail_dimension, linear_core_size),
            dtype=float,
        )
        projected_second = np.zeros(
            (tail_dimension, tail_dimension),
            dtype=float,
        )
        projected_modulated_second = np.zeros_like(projected_second)

        for modulation_value, group in zip(
            self._modulation_values,
            self._core_groups,
        ):
            eta_c = float(modulation_value)
            symmetric = symmetric_zero + eta_c * symmetric_one
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            rotated_linear_map = linear_values.T @ eigenvectors
            trace = float(np.trace(symmetric))

            core_matrix = self._core_matrix[group]
            linear_core_matrix = self._linear_core_matrix[group]
            core_weights = self._core_weights[group]
            core_values_at_nodes = core_polynomial[group]
            anchor_core_moment = core_matrix.T @ core_weights

            projected_linear_eigen = np.zeros(
                (tail_dimension, linear_core_size),
                dtype=float,
            )
            projected_second_eigen = np.zeros(
                (tail_dimension, tail_dimension),
                dtype=float,
            )

            for frequency, spectral_weight, anchor in zip(
                self._spectral_rule.frequencies,
                self._spectral_rule.weights,
                self._spectral_rule.anchors,
            ):
                denominator = 1.0 - 2.0j * frequency * eigenvalues
                inverse_denominator = 1.0 / denominator
                log_determinant = np.sum(np.log(denominator))
                quadratic_kernel = (
                    rotated_linear_map * inverse_denominator
                ) @ rotated_linear_map.T
                quadratic = np.sum(
                    (linear_core_matrix @ quadratic_kernel)
                    * linear_core_matrix,
                    axis=1,
                )
                characteristic = np.exp(
                    1.0j
                    * frequency
                    * (core_values_at_nodes - trace)
                    - 0.5 * log_determinant
                    - 0.5 * frequency * frequency * quadratic
                )
                tilted_mean_map = (
                    1.0j
                    * frequency
                    * rotated_linear_map
                    * inverse_denominator
                )
                characteristic_weights = core_weights * characteristic
                characteristic_gram = (
                    linear_core_matrix.T
                    @ (
                        characteristic_weights[:, None]
                        * linear_core_matrix
                    )
                )

                core_characteristic_moment = (
                    core_matrix.T @ characteristic_weights
                )
                projected_core += np.real(
                    spectral_weight
                    * (
                        core_characteristic_moment
                        - anchor * anchor_core_moment
                    )
                )
                projected_linear_eigen += np.real(
                    spectral_weight
                    * (tilted_mean_map.T @ characteristic_gram)
                )
                second_characteristic_eigen = (
                    np.diag(
                        np.sum(characteristic_weights)
                        * (inverse_denominator - 1.0)
                    )
                    + tilted_mean_map.T
                    @ characteristic_gram
                    @ tilted_mean_map
                )
                projected_second_eigen += np.real(
                    spectral_weight * second_characteristic_eigen
                )

            projected_linear += eigenvectors @ projected_linear_eigen
            group_second = (
                eigenvectors
                @ projected_second_eigen
                @ eigenvectors.T
            )
            projected_second += group_second
            projected_modulated_second += eta_c * group_second

        result = np.zeros(self.basis.size, dtype=float)
        result[structure._core_positions] = projected_core
        result[structure._linear_positions] = projected_linear
        result[structure._diagonal_positions] = (
            np.diag(projected_second) / np.sqrt(2.0)
        )
        result[structure._pair_positions] = projected_second[
            structure._pair_left,
            structure._pair_right,
        ]
        if structure._modulated_positions.size:
            modulated_values = projected_modulated_second[
                structure._modulated_left,
                structure._modulated_right,
            ].copy()
            modulated_values[structure._modulated_diagonal] /= np.sqrt(2.0)
            result[structure._modulated_positions] = modulated_values
        result[0] += self._spectral_rule.constant
        return result

    def project_with_affine_control(
        self,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Apply the affine part exactly and the spectral residual analytically."""

        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        projected = self.drift.rate * state + self.project_residual(state)
        projected[0] += self.drift.intercept
        return projected, float("nan")


__all__ = [
    "AffineSpectralDrift",
    "CoreModulatedSpectralProjection",
    "CosineTransform",
    "LogQuadraticCosineTransform",
    "RationalSineTransform",
    "SpectralQuadratureRule",
    "combine_spectral_rules",
]
