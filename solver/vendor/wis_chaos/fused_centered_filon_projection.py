"""Centered analytic-tail projection with panel-level eigenbasis fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import exp1

from .basis import HermiteBasis
from .centered_filon_projection import (
    CenteredProjectionDiagnostic,
    CoreModulatedCenteredFilonProjection,
)
from .eigenbasis_centered_amplitude import StableCenteredEigenAmplitude
from .fused_eigen_filon import FusedEigenAdaptiveFilon
from .spectral_projection import (
    AffineSpectralDrift,
    CosineTransform,
    LogQuadraticCosineTransform,
    RationalSineTransform,
)


@dataclass(frozen=True, slots=True)
class FusedProjectionInstrumentation:
    """Counts that distinguish node evaluation from panel rotations."""

    compact_node_count: int
    active_panel_count: int
    panel_creation_count: int
    group_rotation_count: int
    state_diagonalization_count: int
    noncosine_transform_count: int


@dataclass(frozen=True)
class FusedCoreModulatedCenteredFilonProjection:
    """Analytic-tail projector with high/low contraction in eigenbases."""

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
    refinement_batch_size: int = 1
    frequency_chunk_size: int = 32

    def __post_init__(self) -> None:
        scalar_projection = CoreModulatedCenteredFilonProjection(
            self.basis,
            int(self.core_dimensions),
            self.drift,
            modulation_core_coordinate=int(
                self.modulation_core_coordinate
            ),
            core_order=int(self.core_order),
            cutoff=float(self.cutoff),
            absolute_tolerance=float(self.absolute_tolerance),
            relative_tolerance=float(self.relative_tolerance),
            max_panels=int(self.max_panels),
            max_core_nodes=int(self.max_core_nodes),
        )
        integrator = FusedEigenAdaptiveFilon(
            cutoff=float(self.cutoff),
            absolute_tolerance=float(self.absolute_tolerance),
            relative_tolerance=float(self.relative_tolerance),
            max_panels=int(self.max_panels),
            refinement_batch_size=int(self.refinement_batch_size),
            frequency_chunk_size=int(self.frequency_chunk_size),
        )
        object.__setattr__(self, "_scalar_projection", scalar_projection)
        object.__setattr__(self, "_base", scalar_projection._base)
        object.__setattr__(self, "_integrator", integrator)

    @property
    def core_node_count(self) -> int:
        return self._base.core_node_count

    @property
    def diagonalization_count(self) -> int:
        return self._base.diagonalization_count

    def project_residual_with_instrumentation(
        self,
        coefficients: np.ndarray,
    ) -> tuple[
        np.ndarray,
        CenteredProjectionDiagnostic,
        FusedProjectionInstrumentation,
    ]:
        raw_state = np.asarray(coefficients)
        if np.iscomplexobj(raw_state):
            raise TypeError("coefficient vector must be real-valued")
        state = np.asarray(raw_state, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("coefficient vector must be finite")

        noncosine = tuple(
            transform
            for transform in self.drift.transforms
            if isinstance(
                transform,
                (RationalSineTransform, LogQuadraticCosineTransform),
            )
            and transform.scale != 0.0
        )
        first_width = (
            float(noncosine[0].width) if noncosine else 1.0
        )
        prepared = StableCenteredEigenAmplitude(
            self._scalar_projection,
            state,
            first_width,
        )
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
        panel_creation_count = 0
        rotation_count = 0
        converged = True
        coefficient_bound = float(np.sqrt(self.basis.size + 3.0))

        for transform in self.drift.transforms:
            if isinstance(transform, CosineTransform):
                if transform.frequency == 0.0:
                    result[0] += transform.scale
                elif transform.scale != 0.0:
                    characteristic = prepared.characteristic_many(
                        np.asarray([transform.frequency], dtype=float)
                    )[0]
                    result += transform.scale * np.real(characteristic)
                continue

            width = float(transform.width)
            evaluator = (
                prepared
                if width == prepared.width
                else prepared.for_width(width)
            )
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

            if isinstance(transform, RationalSineTransform):
                if transform.scale == 0.0:
                    continue
                integration = self._integrator.integrate(
                    evaluator,
                    exponent,
                    rational=True,
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
                decay_rate = 1.0 + gamma
                reference_tail = np.exp(-decay_rate * cutoff) * (
                    1.0 / decay_rate
                    + float(np.max(np.abs(reference_linear)))
                    * (cutoff / decay_rate + 1.0 / decay_rate**2)
                )
                tail_bound += abs(multiplier) * (
                    coefficient_bound * np.exp(-cutoff)
                    + reference_tail
                )
            elif isinstance(transform, LogQuadraticCosineTransform):
                if transform.scale == 0.0:
                    continue
                integration = self._integrator.integrate(
                    evaluator,
                    exponent,
                    rational=False,
                    feature_scale=feature_scale,
                )
                multiplier = -2.0 * transform.scale
                analytic_reference = (
                    reference_linear / reference_denominator
                    - np.log(reference_denominator) * unit
                )
                result += np.real(
                    multiplier * (integration.value + analytic_reference)
                )
                estimated_error += abs(multiplier) * (
                    integration.estimated_error
                )
                cutoff = self._integrator.cutoff
                decay_rate = 1.0 + gamma
                reference_tail = float(exp1(decay_rate * cutoff)) + (
                    float(np.max(np.abs(reference_linear)))
                    * np.exp(-decay_rate * cutoff)
                    / decay_rate
                )
                tail_bound += abs(multiplier) * (
                    coefficient_bound * float(exp1(cutoff))
                    + reference_tail
                )
            else:
                raise TypeError("unsupported spectral transform")

            evaluation_count += integration.evaluation_count
            panel_count += integration.panel_count
            panel_creation_count += integration.panel_creation_count
            rotation_count += integration.rotation_count
            converged = converged and integration.converged

        diagnostic = CenteredProjectionDiagnostic(
            estimated_filon_error=float(estimated_error),
            transform_tail_bound=float(tail_bound),
            evaluation_count=int(evaluation_count),
            panel_count=int(panel_count),
            converged=bool(converged),
        )
        instrumentation = FusedProjectionInstrumentation(
            compact_node_count=int(evaluation_count),
            active_panel_count=int(panel_count),
            panel_creation_count=int(panel_creation_count),
            group_rotation_count=int(rotation_count),
            state_diagonalization_count=int(prepared.group_count),
            noncosine_transform_count=len(noncosine),
        )
        return result, diagnostic, instrumentation

    def project_residual_with_diagnostic(
        self,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, CenteredProjectionDiagnostic]:
        result, diagnostic, _instrumentation = (
            self.project_residual_with_instrumentation(coefficients)
        )
        return result, diagnostic

    def project_residual(self, coefficients: np.ndarray) -> np.ndarray:
        result, diagnostic = self.project_residual_with_diagnostic(coefficients)
        if not diagnostic.converged:
            raise RuntimeError(
                "fused centered Filon transform quadrature did not converge"
            )
        return result

    def project_with_affine_control(
        self,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        raw_state = np.asarray(coefficients)
        if np.iscomplexobj(raw_state):
            raise TypeError("coefficient vector must be real-valued")
        state = np.asarray(raw_state, dtype=float)
        residual, diagnostic = self.project_residual_with_diagnostic(state)
        projected = self.drift.rate * state + residual
        projected[0] += self.drift.intercept
        return projected, diagnostic.total_transform_error_estimate


__all__ = [
    "FusedCoreModulatedCenteredFilonProjection",
    "FusedProjectionInstrumentation",
]
