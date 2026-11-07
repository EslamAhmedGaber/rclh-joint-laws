"""Version-isolated deterministic WIS solver with fused eigen-Filon RHS."""

from __future__ import annotations

from .fbm import GaussianPathFactor
from .fused_centered_filon_projection import (
    FusedCoreModulatedCenteredFilonProjection,
)
from .spectral_projection import AffineSpectralDrift
from .spectral_solver import SpectralDeterministicWISChaosSolver


class FusedSpectralDeterministicWISChaosSolver(
    SpectralDeterministicWISChaosSolver
):
    """Full-time solver using panel-level conditional-eigenbasis fusion."""

    implementation_label = "spectral_analytic_tail_fused_eigen_filon_v1"

    def __init__(
        self,
        factor: GaussianPathFactor,
        degree: int,
        beta: float,
        *,
        drift: AffineSpectralDrift,
        structured_core_dimensions: int,
        core_quadrature_order: int,
        transform_cutoff: float = 28.0,
        transform_absolute_tolerance: float = 1.0e-8,
        transform_relative_tolerance: float = 1.0e-8,
        transform_max_panels: int = 2048,
        modulation_core_coordinate: int = 0,
        drift_substeps: int = 1,
        nonlinear_propagator: str = "integrating_factor",
        index_set: tuple[tuple[int, ...], ...] | None = None,
        max_core_nodes: int = 2_000_000,
        refinement_batch_size: int = 1,
        frequency_chunk_size: int = 32,
    ) -> None:
        super().__init__(
            factor,
            degree,
            beta,
            drift=drift,
            structured_core_dimensions=int(structured_core_dimensions),
            core_quadrature_order=int(core_quadrature_order),
            transform_quadrature="centered_filon",
            transform_cutoff=float(transform_cutoff),
            transform_absolute_tolerance=float(
                transform_absolute_tolerance
            ),
            transform_relative_tolerance=float(
                transform_relative_tolerance
            ),
            transform_max_panels=int(transform_max_panels),
            modulation_core_coordinate=int(modulation_core_coordinate),
            drift_substeps=int(drift_substeps),
            nonlinear_propagator=nonlinear_propagator,
            index_set=index_set,
            max_core_nodes=int(max_core_nodes),
        )
        self._spectral_projection = (
            FusedCoreModulatedCenteredFilonProjection(
                self.basis,
                int(structured_core_dimensions),
                drift,
                modulation_core_coordinate=int(
                    modulation_core_coordinate
                ),
                core_order=int(core_quadrature_order),
                cutoff=float(transform_cutoff),
                absolute_tolerance=float(
                    transform_absolute_tolerance
                ),
                relative_tolerance=float(
                    transform_relative_tolerance
                ),
                max_panels=int(transform_max_panels),
                max_core_nodes=int(max_core_nodes),
                refinement_batch_size=int(refinement_batch_size),
                frequency_chunk_size=int(frequency_chunk_size),
            )
        )


__all__ = ["FusedSpectralDeterministicWISChaosSolver"]
