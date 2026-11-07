"""Analytic-tail projector for one core-modulated quadratic layer.

This module is intentionally separate from :mod:`wis_chaos.basis`.  It
extends the conditional-characteristic rational projector to the
operator-geometry support enrichment

``e_c + e_i + e_j,  i <= j``.

For the supported state, conditioning on the core variable ``eta`` gives

``X = a(eta) + b(eta).T z + z.T S(eta) z - trace(S(eta))``

with ``S(eta) = S0 + eta[c] * S1``.  Thus the Gaussian tail remains analytic.
Only the low-dimensional core Gauss--Hermite rule and the one-dimensional
Gauss--Laguerre transform integral are numerical.

The class is used by the fail-closed
``structured_modulated_rational_laguerre`` solver backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_laguerre

from .basis import (
    HermiteBasis,
    StructuredQuadraticTailSobolProjection,
    TensorGaussHermite,
)


@dataclass(frozen=True)
class CoreModulatedRationalLaguerreProjection:
    r"""Project ``scale*x/(1+x*x)`` with analytic Gaussian-tail elimination.

    Every core-modulated quadratic-tail multi-index must use the same degree
    one core Hermite coordinate, selected by ``modulation_core_coordinate``.
    With normalized probabilists' Hermites this is exactly ``eta_c``.  The
    tail pair set may be any sparse subset, including the frozen ``R=2N``
    operator-geometry prefix.  Mixed modulation coordinates and higher-degree
    core modulation are rejected rather than approximated silently.

    The implementation groups tensor core nodes by ``eta_c``.  Consequently
    only ``core_order`` matrices ``S0 + eta_c*S1`` are diagonalized, rather
    than one matrix at each of the ``core_order**core_dimensions`` nodes.
    """

    basis: HermiteBasis
    core_dimensions: int
    modulation_core_coordinate: int = 0
    core_order: int = 9
    laguerre_order: int = 24
    max_core_nodes: int = 2_000_000

    def __post_init__(self) -> None:
        core = int(self.core_dimensions)
        coordinate = int(self.modulation_core_coordinate)
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
        expected_core_alpha = [0] * core
        expected_core_alpha[coordinate] = 1
        expected_core_alpha = tuple(expected_core_alpha)
        core_patterns = tuple(
            tuple(int(value) for value in alpha)
            for alpha in structure._core_basis.index_set
        )
        for column in structure._modulated_core_columns:
            if core_patterns[int(column)] != expected_core_alpha:
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

        laguerre_nodes, laguerre_weights = roots_laguerre(
            int(self.laguerre_order)
        )
        if not (
            np.all(np.isfinite(laguerre_nodes))
            and np.all(np.isfinite(laguerre_weights))
            and float(np.sum(laguerre_weights)) > 0.0
        ):
            raise FloatingPointError(
                "Gauss--Laguerre construction returned nonfinite data"
            )

        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "_core_nodes", core_nodes)
        object.__setattr__(self, "_core_weights", core_weights)
        object.__setattr__(self, "_core_matrix", core_matrix)
        object.__setattr__(self, "_linear_core_matrix", linear_core_matrix)
        object.__setattr__(self, "_modulation_values", unique_values)
        object.__setattr__(self, "_core_groups", groups)
        object.__setattr__(self, "_laguerre_nodes", laguerre_nodes)
        object.__setattr__(self, "_laguerre_weights", laguerre_weights)

    @property
    def core_node_count(self) -> int:
        """Number of tensor Gauss--Hermite nodes in all core coordinates."""

        return int(self._core_nodes.shape[0])

    @property
    def diagonalization_count(self) -> int:
        """Number of conditional quadratic matrices per projection call."""

        return len(self._core_groups)

    def _quadratic_matrices(
        self,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract ``a``, the tail-linear map, ``S0``, and ``S1``."""

        structure = self._structure
        core_values, linear_values, _, symmetric_zero, modulated_values = (
            structure._structured_coefficients(state)
        )
        tail_dimension = symmetric_zero.shape[0]
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
                # c*eta*(z_i**2-1)/sqrt(2)
                symmetric_one[i, i] += coefficient / np.sqrt(2.0)
            else:
                # z.T*S*z contains 2*S_ij*z_i*z_j.
                symmetric_one[i, j] += 0.5 * coefficient
                symmetric_one[j, i] += 0.5 * coefficient
        if symmetric_one.shape != (tail_dimension, tail_dimension):
            raise RuntimeError("internal quadratic-tail shape mismatch")
        return core_values, linear_values, symmetric_zero, symmetric_one

    def project_rational_residual(
        self,
        coefficients: np.ndarray,
        scale: float = 4.0,
    ) -> np.ndarray:
        """Project ``scale*X/(1+X**2)`` onto every retained coefficient."""

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
        multiplier = float(scale)

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

            projected_linear_eigen = np.zeros(
                (tail_dimension, linear_core_size),
                dtype=float,
            )
            projected_second_eigen = np.zeros(
                (tail_dimension, tail_dimension),
                dtype=float,
            )

            for frequency, laguerre_weight in zip(
                self._laguerre_nodes,
                self._laguerre_weights,
            ):
                denominator = 1.0 - 2.0j * frequency * eigenvalues
                inverse_denominator = 1.0 / denominator
                # Principal logs are branch-safe because Re(denominator)=1.
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

                weighted_scale = multiplier * float(laguerre_weight)
                characteristic_weights = core_weights * characteristic
                characteristic_gram = (
                    linear_core_matrix.T
                    @ (
                        characteristic_weights[:, None]
                        * linear_core_matrix
                    )
                )
                projected_core += weighted_scale * np.imag(
                    core_matrix.T @ characteristic_weights
                )
                projected_linear_eigen += weighted_scale * np.imag(
                    tilted_mean_map.T @ characteristic_gram
                )
                # Ordinary transpose is required; no conjugation appears in
                # analytic characteristic derivatives.
                second_characteristic_eigen = (
                    np.diag(
                        np.sum(characteristic_weights)
                        * (inverse_denominator - 1.0)
                    )
                    + tilted_mean_map.T
                    @ characteristic_gram
                    @ tilted_mean_map
                )
                projected_second_eigen += weighted_scale * np.imag(
                    second_characteristic_eigen
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
        return result

    def project_with_linear_control(
        self,
        coefficients: np.ndarray,
        linear_rate: float,
        rational_scale: float = 4.0,
        affine_intercept: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Apply an affine drift exactly and project only its rational part."""

        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        projected = (
            float(linear_rate) * state
            + self.project_rational_residual(state, rational_scale)
        )
        projected[0] += float(affine_intercept)
        return projected, float("nan")


__all__ = ["CoreModulatedRationalLaguerreProjection"]
