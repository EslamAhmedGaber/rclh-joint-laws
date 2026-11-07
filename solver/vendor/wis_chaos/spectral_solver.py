"""Version-isolated deterministic WIS solver for spectral nonlinearities.

This module deliberately extends, rather than edits, the frozen legacy
solver implementation.  Earlier experiment artifacts commit the byte-level
fingerprint of :mod:`wis_chaos.solver`; keeping this backend separate
preserves the meaning of those completed gates.
"""

from __future__ import annotations

import numpy as np

from .centered_filon_projection import (
    CoreModulatedCenteredFilonProjection,
)
from .fbm import GaussianPathFactor
from .solver import DeterministicWISChaosSolver
from .spectral_projection import (
    AffineSpectralDrift,
    CoreModulatedSpectralProjection,
)


class SpectralDeterministicWISChaosSolver(DeterministicWISChaosSolver):
    """Full-time Wick--Lawson chaos solver with analytic Gaussian-tail drift.

    The inherited time propagators and coherent finite-chaos state semantics
    are unchanged.  Only the nonlinear Galerkin RHS is replaced by the
    transform-admissible conditional analytic-tail projector.
    """

    implementation_label = "spectral_analytic_tail_v1"

    def __init__(
        self,
        factor: GaussianPathFactor,
        degree: int,
        beta: float,
        *,
        drift: AffineSpectralDrift,
        structured_core_dimensions: int,
        core_quadrature_order: int,
        laguerre_order: int = 24,
        transform_quadrature: str = "centered_filon",
        transform_cutoff: float = 28.0,
        transform_absolute_tolerance: float = 1.0e-8,
        transform_relative_tolerance: float = 1.0e-8,
        transform_max_panels: int = 2048,
        modulation_core_coordinate: int = 0,
        drift_substeps: int = 1,
        nonlinear_propagator: str = "integrating_factor",
        index_set: tuple[tuple[int, ...], ...] | None = None,
        max_core_nodes: int = 2_000_000,
    ) -> None:
        if not isinstance(drift, AffineSpectralDrift):
            raise ValueError(
                "spectral solver requires drift=AffineSpectralDrift(...)"
            )
        core_dimensions = int(structured_core_dimensions)
        core_order = int(core_quadrature_order)
        laguerre = int(laguerre_order)
        coordinate = int(modulation_core_coordinate)
        if core_dimensions < 1:
            raise ValueError("structured_core_dimensions must be positive")
        if core_order < 1:
            raise ValueError("core_quadrature_order must be positive")
        if laguerre < 1:
            raise ValueError("laguerre_order must be positive")
        if transform_quadrature not in {"centered_filon", "laguerre"}:
            raise ValueError(
                "transform_quadrature must be 'centered_filon' or 'laguerre'"
            )

        # Construct the unchanged finite-chaos machinery without allocating a
        # legacy nonlinear projector, then activate the isolated spectral RHS.
        super().__init__(
            factor,
            degree,
            beta,
            linear_drift_rate=0.0,
            drift_substeps=drift_substeps,
            nonlinear_propagator=nonlinear_propagator,
            index_set=index_set,
        )
        self.linear_drift_rate = None
        self.drift = drift
        self.projection_backend = "structured_modulated_spectral"
        self.modulation_core_coordinate = coordinate
        try:
            if transform_quadrature == "centered_filon":
                self._spectral_projection = (
                    CoreModulatedCenteredFilonProjection(
                        self.basis,
                        core_dimensions,
                        drift,
                        modulation_core_coordinate=coordinate,
                        core_order=core_order,
                        cutoff=float(transform_cutoff),
                        absolute_tolerance=float(
                            transform_absolute_tolerance
                        ),
                        relative_tolerance=float(
                            transform_relative_tolerance
                        ),
                        max_panels=int(transform_max_panels),
                        max_core_nodes=int(max_core_nodes),
                    )
                )
            else:
                self._spectral_projection = CoreModulatedSpectralProjection(
                    self.basis,
                    core_dimensions,
                    drift,
                    modulation_core_coordinate=coordinate,
                    core_order=core_order,
                    laguerre_order=laguerre,
                    max_core_nodes=int(max_core_nodes),
                )
        except ValueError as error:
            raise ValueError(
                "structured modulated spectral support validation failed: "
                "the basis must contain complete core, tail-linear, and "
                "core-free quadratic-tail layers, while every retained "
                "core-modulated tail pair must use the selected degree-one "
                f"modulation coordinate; details: {error}"
            ) from error
        self.projection_gram_error = 0.0
        self.projection_condition_number = 1.0

    def _projected_drift_rhs(
        self,
        _time: float,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        return self._spectral_projection.project_with_affine_control(
            coefficients
        )


__all__ = ["SpectralDeterministicWISChaosSolver"]
