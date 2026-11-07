"""Batched Hermite-weighted characteristic moments for research backends.

This module is version-isolated from the centered-Filon implementation used by
the sealed N32 experiments.  It prepares the same conditional quadratic-tail
algebra but evaluates a vector of frequencies with array contractions, making
it suitable for a later level-synchronous Filon integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centered_filon_projection import CoreModulatedCenteredFilonProjection


@dataclass(frozen=True, slots=True)
class _PreparedGroup:
    eta_c: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rotated_linear_map: np.ndarray
    trace: float
    core_matrix: np.ndarray
    linear_core_matrix: np.ndarray
    core_weights: np.ndarray
    core_values_at_nodes: np.ndarray


class BatchedHermiteCharacteristicEvaluator:
    """Evaluate retained ``E[Psi_alpha exp(i*u*X)]`` in frequency batches."""

    def __init__(
        self,
        projection: CoreModulatedCenteredFilonProjection,
        state: np.ndarray,
        *,
        chunk_size: int = 32,
    ) -> None:
        if not isinstance(projection, CoreModulatedCenteredFilonProjection):
            raise TypeError(
                "projection must be CoreModulatedCenteredFilonProjection"
            )
        if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
            chunk_size, (int, np.integer)
        ):
            raise TypeError("chunk_size must be an integer")
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError("chunk_size must be positive")
        raw_coefficients = np.asarray(state)
        if np.iscomplexobj(raw_coefficients):
            raise TypeError("state must be real-valued")
        coefficients = np.asarray(raw_coefficients, dtype=float)
        if coefficients.shape != (projection.basis.size,):
            raise ValueError(
                "state shape must equal the retained Hermite basis size"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("state must be finite")

        base = projection._base
        structure = base._structure
        (
            core_values,
            linear_values,
            symmetric_zero,
            symmetric_one,
        ) = base._quadratic_matrices(coefficients)
        core_polynomial = base._core_matrix @ core_values
        groups = []
        for modulation_value, group in zip(
            base._modulation_values, base._core_groups
        ):
            eta_c = float(modulation_value)
            symmetric = symmetric_zero + eta_c * symmetric_one
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            groups.append(
                _PreparedGroup(
                    eta_c=eta_c,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    rotated_linear_map=linear_values.T @ eigenvectors,
                    trace=float(np.trace(symmetric)),
                    core_matrix=base._core_matrix[group],
                    linear_core_matrix=base._linear_core_matrix[group],
                    core_weights=base._core_weights[group],
                    core_values_at_nodes=core_polynomial[group],
                )
            )

        self._projection = projection
        self._structure = structure
        self._groups = tuple(groups)
        self._tail_dimension = symmetric_zero.shape[0]
        self._linear_core_size = structure._linear_core_basis.size
        self._chunk_size = chunk
        unit = np.zeros(projection.basis.size, dtype=complex)
        unit[0] = 1.0
        self._unit = unit

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def diagonalization_count(self) -> int:
        return len(self._groups)

    def _positive_chunk(self, frequencies: np.ndarray) -> np.ndarray:
        structure = self._structure
        count = frequencies.size
        tail = self._tail_dimension
        linear_core = self._linear_core_size

        projected_core = np.zeros(
            (count, structure._core_basis.size), dtype=complex
        )
        projected_linear = np.zeros(
            (count, tail, linear_core), dtype=complex
        )
        projected_second = np.zeros(
            (count, tail, tail), dtype=complex
        )
        projected_modulated_second = np.zeros_like(projected_second)

        for group in self._groups:
            eigenvalues = group.eigenvalues
            eigenvectors = group.eigenvectors
            rotated = group.rotated_linear_map
            core_matrix = group.core_matrix
            linear_matrix = group.linear_core_matrix

            denominator = 1.0 - 2.0j * frequencies[:, None] * eigenvalues
            inverse = 1.0 / denominator
            log_determinant = np.sum(np.log(denominator), axis=1)

            quadratic_kernel = np.einsum(
                "ad,fd,bd->fab",
                rotated,
                inverse,
                rotated,
                optimize=True,
            )
            quadratic = np.einsum(
                "na,fab,nb->fn",
                linear_matrix,
                quadratic_kernel,
                linear_matrix,
                optimize=True,
            )
            characteristic = np.exp(
                1.0j
                * frequencies[:, None]
                * (group.core_values_at_nodes[None, :] - group.trace)
                - 0.5 * log_determinant[:, None]
                - 0.5 * frequencies[:, None] ** 2 * quadratic
            )
            characteristic_weights = (
                characteristic * group.core_weights[None, :]
            )
            characteristic_gram = np.einsum(
                "na,fn,nb->fab",
                linear_matrix,
                characteristic_weights,
                linear_matrix,
                optimize=True,
            )
            projected_core += characteristic_weights @ core_matrix

            tilted = (
                1.0j
                * frequencies[:, None, None]
                * rotated[None, :, :]
                * inverse[:, None, :]
            )
            projected_linear_eigen = np.einsum(
                "fda,fab->fdb",
                np.swapaxes(tilted, 1, 2),
                characteristic_gram,
                optimize=True,
            )
            projected_linear += np.einsum(
                "ij,fjk->fik",
                eigenvectors,
                projected_linear_eigen,
                optimize=True,
            )

            tilted_quadratic = np.einsum(
                "fad,fab,fbe->fde",
                tilted,
                characteristic_gram,
                tilted,
                optimize=True,
            )
            diagonal = (
                np.sum(characteristic_weights, axis=1)[:, None]
                * (inverse - 1.0)
            )
            second_eigen = tilted_quadratic.copy()
            diagonal_indices = np.arange(tail)
            second_eigen[:, diagonal_indices, diagonal_indices] += diagonal
            group_second = np.einsum(
                "ij,fjk,lk->fil",
                eigenvectors,
                second_eigen,
                eigenvectors,
                optimize=True,
            )
            projected_second += group_second
            projected_modulated_second += group.eta_c * group_second

        result = np.zeros(
            (count, self._projection.basis.size), dtype=complex
        )
        result[:, structure._core_positions] = projected_core
        result[:, structure._linear_positions] = projected_linear
        result[:, structure._diagonal_positions] = (
            np.diagonal(projected_second, axis1=1, axis2=2) / np.sqrt(2.0)
        )
        result[:, structure._pair_positions] = projected_second[
            :, structure._pair_left, structure._pair_right
        ]
        if structure._modulated_positions.size:
            modulated = projected_modulated_second[
                :, structure._modulated_left, structure._modulated_right
            ].copy()
            modulated[:, structure._modulated_diagonal] /= np.sqrt(2.0)
            result[:, structure._modulated_positions] = modulated
        return result

    def evaluate_many(self, frequencies: np.ndarray) -> np.ndarray:
        """Return a ``(frequency_count, basis_size)`` complex array."""

        raw_values = np.asarray(frequencies)
        if np.iscomplexobj(raw_values):
            raise TypeError("frequencies must be real-valued")
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("frequencies must be one-dimensional")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("frequencies must be finite and nonnegative")
        result = np.empty(
            (values.size, self._projection.basis.size), dtype=complex
        )
        zero = values == 0.0
        result[zero] = self._unit
        positive_positions = np.flatnonzero(~zero)
        for start in range(0, positive_positions.size, self._chunk_size):
            positions = positive_positions[start : start + self._chunk_size]
            result[positions] = self._positive_chunk(values[positions])
        return result

    def __call__(self, frequency: float) -> np.ndarray:
        return self.evaluate_many(np.asarray([frequency], dtype=float))[0]


__all__ = ["BatchedHermiteCharacteristicEvaluator"]
