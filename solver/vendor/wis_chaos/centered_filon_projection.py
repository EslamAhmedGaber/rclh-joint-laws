"""Centered oscillatory analytic-tail Galerkin projection.

Large state means make a fixed Gauss--Laguerre rule alias the Fourier phase.
This projector factors the exact mean phase from every Hermite-weighted
characteristic moment and subtracts the deterministic mean response.  The
remaining smooth vector amplitude is integrated by adaptive Filon quadrature,
while all Gaussian tail coordinates are still eliminated analytically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import exp1

from .basis import HermiteBasis
from .filon import AdaptiveExponentialFilon
from .spectral_projection import (
    AffineSpectralDrift,
    CoreModulatedSpectralProjection,
    CosineTransform,
    LogQuadraticCosineTransform,
    RationalSineTransform,
)


@dataclass(frozen=True, slots=True)
class CenteredProjectionDiagnostic:
    estimated_filon_error: float
    transform_tail_bound: float
    evaluation_count: int
    panel_count: int
    converged: bool

    @property
    def total_transform_error_estimate(self) -> float:
        if not self.converged:
            return float("inf")
        return float(self.estimated_filon_error + self.transform_tail_bound)


@dataclass(frozen=True)
class CoreModulatedCenteredFilonProjection:
    """Project an affine-spectral drift with centered adaptive Filon rules."""

    basis: HermiteBasis
    core_dimensions: int
    drift: AffineSpectralDrift
    modulation_core_coordinate: int = 0
    core_order: int = 9
    cutoff: float = 28.0
    absolute_tolerance: float = 1.0e-8
    relative_tolerance: float = 1.0e-8
    max_panels: int = 2048
    max_core_nodes: int = 2_000_000

    def __post_init__(self) -> None:
        # Reuse the independently tested support validator, structured
        # coefficient extractor, and core quadrature construction.  Its
        # one-node Laguerre rule is never used by this projector.
        base = CoreModulatedSpectralProjection(
            self.basis,
            int(self.core_dimensions),
            self.drift,
            modulation_core_coordinate=int(
                self.modulation_core_coordinate
            ),
            core_order=int(self.core_order),
            laguerre_order=1,
            max_core_nodes=int(self.max_core_nodes),
        )
        integrator = AdaptiveExponentialFilon(
            cutoff=float(self.cutoff),
            absolute_tolerance=float(self.absolute_tolerance),
            relative_tolerance=float(self.relative_tolerance),
            max_panels=int(self.max_panels),
        )
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_integrator", integrator)

    @property
    def core_node_count(self) -> int:
        return self._base.core_node_count

    @property
    def diagonalization_count(self) -> int:
        return self._base.diagonalization_count

    def _characteristic_evaluator(
        self,
        state: np.ndarray,
    ):
        base = self._base
        structure = base._structure
        (
            core_values,
            linear_values,
            symmetric_zero,
            symmetric_one,
        ) = base._quadratic_matrices(state)
        core_polynomial = base._core_matrix @ core_values
        tail_dimension = symmetric_zero.shape[0]
        linear_core_size = structure._linear_core_basis.size

        prepared_groups = []
        for modulation_value, group in zip(
            base._modulation_values,
            base._core_groups,
        ):
            eta_c = float(modulation_value)
            symmetric = symmetric_zero + eta_c * symmetric_one
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            prepared_groups.append(
                (
                    eta_c,
                    eigenvalues,
                    eigenvectors,
                    linear_values.T @ eigenvectors,
                    float(np.trace(symmetric)),
                    base._core_matrix[group],
                    base._linear_core_matrix[group],
                    base._core_weights[group],
                    core_polynomial[group],
                )
            )

        unit = np.zeros(self.basis.size, dtype=complex)
        unit[0] = 1.0
        cache: dict[float, np.ndarray] = {0.0: unit}

        def evaluate(frequency: float) -> np.ndarray:
            key = float(frequency)
            if key in cache:
                return cache[key]
            if not np.isfinite(key) or key < 0.0:
                raise ValueError(
                    "characteristic frequency must be finite and nonnegative"
                )

            projected_core = np.zeros(
                structure._core_basis.size,
                dtype=complex,
            )
            projected_linear = np.zeros(
                (tail_dimension, linear_core_size),
                dtype=complex,
            )
            projected_second = np.zeros(
                (tail_dimension, tail_dimension),
                dtype=complex,
            )
            projected_modulated_second = np.zeros_like(projected_second)

            for (
                eta_c,
                eigenvalues,
                eigenvectors,
                rotated_linear_map,
                trace,
                core_matrix,
                linear_core_matrix,
                core_weights,
                core_values_at_nodes,
            ) in prepared_groups:
                denominator = 1.0 - 2.0j * key * eigenvalues
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
                    * key
                    * (core_values_at_nodes - trace)
                    - 0.5 * log_determinant
                    - 0.5 * key * key * quadratic
                )
                tilted_mean_map = (
                    1.0j
                    * key
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
                projected_core += core_matrix.T @ characteristic_weights
                projected_linear_eigen = (
                    tilted_mean_map.T @ characteristic_gram
                )
                projected_linear += (
                    eigenvectors @ projected_linear_eigen
                )
                second_eigen = (
                    np.diag(
                        np.sum(characteristic_weights)
                        * (inverse_denominator - 1.0)
                    )
                    + tilted_mean_map.T
                    @ characteristic_gram
                    @ tilted_mean_map
                )
                group_second = (
                    eigenvectors @ second_eigen @ eigenvectors.T
                )
                projected_second += group_second
                projected_modulated_second += eta_c * group_second

            result = np.zeros(self.basis.size, dtype=complex)
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
                modulated_values[structure._modulated_diagonal] /= np.sqrt(
                    2.0
                )
                result[structure._modulated_positions] = modulated_values
            cache[key] = result
            return result

        return evaluate

    def project_residual_with_diagnostic(
        self,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, CenteredProjectionDiagnostic]:
        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("coefficient vector must be finite")

        characteristic = self._characteristic_evaluator(state)
        mean = float(state[0])
        centered_state = state.copy()
        centered_state[0] = 0.0
        standard_deviation = float(np.linalg.norm(centered_state))
        unit = np.zeros(self.basis.size, dtype=complex)
        unit[0] = 1.0
        result = np.zeros(self.basis.size, dtype=float)
        estimated_error = 0.0
        tail_bound = 0.0
        evaluation_count = 0
        panel_count = 0
        converged = True
        coefficient_bound = float(np.sqrt(self.basis.size + 3.0))

        for transform in self.drift.transforms:
            if isinstance(transform, CosineTransform):
                if transform.frequency == 0.0:
                    result[0] += transform.scale
                elif transform.scale != 0.0:
                    result += transform.scale * np.real(
                        characteristic(transform.frequency)
                    )
                continue

            width = float(transform.width)
            scaled_mean = mean / width
            exponent = -1.0 + 1.0j * scaled_mean
            gamma = standard_deviation / width
            feature_scale = min(
                1.0,
                width / max(standard_deviation, np.finfo(float).tiny),
            )
            centered_derivative = 1.0j * centered_state / width
            reference_linear = centered_derivative + gamma * unit
            reference_denominator = gamma - exponent

            def centered_moment(value: float) -> np.ndarray:
                frequency = value / width
                return (
                    np.exp(-1.0j * frequency * mean)
                    * characteristic(frequency)
                )

            if isinstance(transform, RationalSineTransform):
                if transform.scale == 0.0:
                    continue

                def amplitude(value: float) -> np.ndarray:
                    reference = np.exp(-gamma * value) * (
                        unit + reference_linear * value
                    )
                    return centered_moment(value) - reference

                integration = self._integrator.integrate(
                    amplitude,
                    exponent,
                    feature_scale=feature_scale,
                )
                multiplier = transform.scale / width
                analytic_reference = (
                    unit / reference_denominator
                    + reference_linear / reference_denominator**2
                )
                result += multiplier * np.imag(
                    integration.value + analytic_reference
                )
                estimated_error += abs(multiplier) * (
                    integration.estimated_error
                )
                cutoff = self._integrator.cutoff
                decay = 1.0 + gamma
                reference_tail = np.exp(-decay * cutoff) * (
                    1.0 / decay
                    + float(np.max(np.abs(reference_linear)))
                    * (cutoff / decay + 1.0 / decay**2)
                )
                tail_bound += abs(multiplier) * (
                    coefficient_bound * np.exp(-cutoff)
                    + reference_tail
                )
            elif isinstance(transform, LogQuadraticCosineTransform):
                if transform.scale == 0.0:
                    continue

                def amplitude(value: float) -> np.ndarray:
                    if value == 0.0:
                        return np.zeros(self.basis.size, dtype=complex)
                    exponential_reference = np.exp(-gamma * value)
                    return (
                        (
                            centered_moment(value)
                            - unit * exponential_reference
                        )
                        / value
                        - reference_linear * exponential_reference
                    )

                integration = self._integrator.integrate(
                    amplitude,
                    exponent,
                    feature_scale=feature_scale,
                )
                multiplier = -2.0 * transform.scale
                analytic_reference = (
                    reference_linear / reference_denominator
                    - np.log(reference_denominator) * unit
                )
                result += np.real(
                    multiplier
                    * (integration.value + analytic_reference)
                )
                estimated_error += abs(multiplier) * (
                    integration.estimated_error
                )
                cutoff = self._integrator.cutoff
                decay = 1.0 + gamma
                reference_tail = float(exp1(decay * cutoff)) + (
                    float(np.max(np.abs(reference_linear)))
                    * np.exp(-decay * cutoff)
                    / decay
                )
                tail_bound += abs(multiplier) * (
                    coefficient_bound * float(exp1(cutoff))
                    + reference_tail
                )
            else:
                raise TypeError("unsupported spectral transform")

            evaluation_count += integration.evaluation_count
            panel_count += integration.panel_count
            converged = converged and integration.converged

        diagnostic = CenteredProjectionDiagnostic(
            estimated_filon_error=float(estimated_error),
            transform_tail_bound=float(tail_bound),
            evaluation_count=int(evaluation_count),
            panel_count=int(panel_count),
            converged=bool(converged),
        )
        return result, diagnostic

    def project_residual(self, coefficients: np.ndarray) -> np.ndarray:
        result, diagnostic = self.project_residual_with_diagnostic(coefficients)
        if not diagnostic.converged:
            raise RuntimeError(
                "centered Filon transform quadrature did not converge"
            )
        return result

    def project_with_affine_control(
        self,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        state = np.asarray(coefficients, dtype=float)
        residual, diagnostic = self.project_residual_with_diagnostic(state)
        projected = self.drift.rate * state + residual
        projected[0] += self.drift.intercept
        return projected, diagnostic.total_transform_error_estimate


__all__ = [
    "CenteredProjectionDiagnostic",
    "CoreModulatedCenteredFilonProjection",
]
