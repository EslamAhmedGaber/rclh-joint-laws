"""Contour-shifted lognormal cosine expectation.

This module supplies the optional oscillatory-cosine backend used by the
residual-covariance coherent Lawson projector.  It remains a research
candidate and does not alter the separately fingerprinted official fused
solver.

At one active quadrature node, put

``q = A + sqrt(Vr) Z + (Va + Vr)/2``,  ``Z ~ N(0, 1)``.

Gaussian exponential tilting gives the exact Esscher identity

``E[exp(-q) cos(omega*y*exp(q))]``
``= exp(-A-Va/2) E[cos(lambda*exp(sigma*W))]``,

where ``sigma=sqrt(Vr)`` and

``lambda=abs(omega*y)*exp(A+Va/2-Vr/2)``.

The residual random amplitude has disappeared, so the real-axis integrand
is bounded.  Plain Gauss--Hermite can nevertheless remain nonmonotone when
the lognormal phase is oscillatory.  For ``lambda >= 0`` define

``chi(lambda,sigma)=E[exp(i*lambda*exp(sigma*W))]``.

The integrand is entire.  Shifting the contour from ``z=x`` to
``z=x+i*delta`` yields

``chi = exp(delta**2/2) E[exp(-i*delta*W)``
``       * exp(i*lambda*exp(sigma*W+i*sigma*delta))]``.

For ``0 < sigma*delta < pi`` the vertical sides of the closing rectangle
vanish: on the right side the lognormal factor has modulus
``exp(-lambda*exp(sigma*x)*sin(sigma*y)) <= 1``, and on the left side it
tends to one while the Gaussian factor decays.  Along the shifted line the
non-Gaussian factor is bounded by ``exp(delta**2/2)``.  The finite rule below
uses a deliberately narrower, empirically audited domain.  Calls outside
that domain fail closed rather than extrapolating a numerical certificate.

Contour deformation for lognormal characteristic functions is classical;
this helper is a case-specific numerical stabilization, not a novelty claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_hermitenorm


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not np.isfinite(scalar) or scalar < 1.0 or scalar != np.floor(scalar):
        raise ValueError(f"{name} must be a positive integer")
    return int(scalar)


@lru_cache(maxsize=None)
def _normal_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    selected = _positive_integer(order, "order")
    nodes, weights = roots_hermitenorm(selected)
    weights = weights / np.sqrt(2.0 * np.pi)
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
        and abs(float(np.sum(weights)) - 1.0) <= 2.0e-14
    ):
        raise FloatingPointError("normal quadrature construction failed")
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@dataclass(frozen=True, slots=True)
class ContourShiftedLognormalDiagnostic:
    """Audit data for one vectorized cosine expectation."""

    order: int
    contour_shift: float
    sigma: float
    maximum_phase: float
    contour_amplification: float


@dataclass(frozen=True, slots=True)
class ContourShiftedLognormalRule:
    """Fail-closed shifted Gauss--Hermite rule for a certified domain.

    The defaults are a conservative candidate for the C problem with
    ``H >= 0.10``, grids through N=64, active dimensions 1--3 and
    ``beta=0.5``.  They are discovery controls, not official evidence.
    A caller must create a separately governed rule before extending either
    the residual standard deviation or the lognormal phase range.
    """

    order: int = 33
    contour_shift: float = 2.0
    max_sigma: float = 0.4
    max_phase: float = 1000.0
    phase_chunk_size: int = 4096

    def __post_init__(self) -> None:
        order = _positive_integer(self.order, "order")
        shift = float(self.contour_shift)
        maximum_sigma = float(self.max_sigma)
        maximum_phase = float(self.max_phase)
        chunk = _positive_integer(self.phase_chunk_size, "phase_chunk_size")
        if not np.isfinite(shift) or shift <= 0.0:
            raise ValueError("contour_shift must be finite and positive")
        if not np.isfinite(maximum_sigma) or maximum_sigma < 0.0:
            raise ValueError("max_sigma must be finite and nonnegative")
        if not np.isfinite(maximum_phase) or maximum_phase <= 0.0:
            raise ValueError("max_phase must be finite and positive")
        if maximum_sigma * shift >= np.pi:
            raise ValueError(
                "certified domain violates sigma*contour_shift < pi"
            )
        amplification = float(np.exp(0.5 * shift * shift))
        if not np.isfinite(amplification):
            raise ValueError("contour amplification exceeds float range")
        _normal_rule(order)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "contour_shift", shift)
        object.__setattr__(self, "max_sigma", maximum_sigma)
        object.__setattr__(self, "max_phase", maximum_phase)
        object.__setattr__(self, "phase_chunk_size", chunk)

    @property
    def contour_amplification(self) -> float:
        return float(np.exp(0.5 * self.contour_shift**2))

    def characteristic_cosine(
        self,
        phase: np.ndarray,
        sigma: float,
    ) -> tuple[np.ndarray, ContourShiftedLognormalDiagnostic]:
        """Return ``E[cos(abs(phase)*exp(sigma*Z))]``.

        ``phase`` may have any shape.  The result has the same shape.  The
        exact limits ``sigma=0`` and ``phase=0`` are applied explicitly to
        avoid unnecessary complex cancellation.
        """

        values = np.abs(np.asarray(phase, dtype=float))
        selected_sigma = float(sigma)
        if not np.all(np.isfinite(values)) or not np.isfinite(selected_sigma):
            raise ValueError("phase and sigma must be finite")
        if selected_sigma < 0.0 or selected_sigma > self.max_sigma:
            raise ValueError("sigma lies outside the certified rule domain")
        maximum = float(np.max(values)) if values.size else 0.0
        tolerance = 64.0 * np.finfo(float).eps * self.max_phase
        if maximum > self.max_phase + tolerance:
            raise ValueError(
                "phase lies outside the certified rule domain "
                f"(maximum={maximum:.17g}, limit={self.max_phase:.17g})"
            )
        if selected_sigma * self.contour_shift >= np.pi:
            raise ValueError("sigma*contour_shift must be smaller than pi")

        diagnostic = ContourShiftedLognormalDiagnostic(
            order=self.order,
            contour_shift=self.contour_shift,
            sigma=selected_sigma,
            maximum_phase=maximum,
            contour_amplification=self.contour_amplification,
        )
        if values.size == 0:
            return values.copy(), diagnostic
        if selected_sigma == 0.0:
            return np.cos(values), diagnostic

        nodes, weights = _normal_rule(self.order)
        shift = self.contour_shift
        angle = selected_sigma * shift
        shifted_exponential = np.exp(
            selected_sigma * nodes + 1.0j * angle
        )
        shifted_weights = weights * np.exp(-1.0j * shift * nodes)
        flattened = values.reshape(-1)
        result = np.empty(flattened.size, dtype=float)
        for start in range(0, flattened.size, self.phase_chunk_size):
            stop = min(start + self.phase_chunk_size, flattened.size)
            selected = flattened[start:stop]
            kernel = np.exp(
                1.0j * selected[:, None] * shifted_exponential[None, :]
            )
            transformed = self.contour_amplification * (
                kernel @ shifted_weights
            )
            result[start:stop] = np.real(transformed)
        result[flattened == 0.0] = 1.0
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "contour-shifted lognormal expectation is nonfinite"
            )
        return result.reshape(values.shape), diagnostic

    def pulled_back_cosine(
        self,
        lawson_values: np.ndarray,
        active_scalar: np.ndarray,
        active_variance: float,
        residual_variance: float,
        *,
        scale: float,
        frequency: float,
    ) -> tuple[np.ndarray, ContourShiftedLognormalDiagnostic]:
        """Evaluate the residual conditional cosine by the Esscher identity.

        ``active_scalar`` is ``A=x dot a`` at each active node and
        ``active_variance`` is ``Va=||a||**2``.  The returned array contains
        conditional values at those nodes; active Hermite projection remains
        a separate numerical axis.
        """

        state, active = np.broadcast_arrays(
            np.asarray(lawson_values, dtype=float),
            np.asarray(active_scalar, dtype=float),
        )
        variance_active = float(active_variance)
        variance_residual = float(residual_variance)
        selected_scale = float(scale)
        selected_frequency = float(frequency)
        if not (
            np.all(np.isfinite(state))
            and np.all(np.isfinite(active))
            and np.isfinite(variance_active)
            and np.isfinite(variance_residual)
            and np.isfinite(selected_scale)
            and np.isfinite(selected_frequency)
        ):
            raise ValueError("pulled-back cosine inputs must be finite")
        if variance_active < 0.0 or variance_residual < 0.0:
            raise ValueError("active and residual variances must be nonnegative")
        if selected_frequency < 0.0:
            raise ValueError("frequency must be nonnegative")

        sigma = float(np.sqrt(variance_residual))
        if sigma > self.max_sigma:
            raise ValueError("residual standard deviation is outside the domain")
        if selected_scale == 0.0:
            zeros = np.zeros(state.shape, dtype=float)
            diagnostic = ContourShiftedLognormalDiagnostic(
                order=self.order,
                contour_shift=self.contour_shift,
                sigma=sigma,
                maximum_phase=0.0,
                contour_amplification=self.contour_amplification,
            )
            return zeros, diagnostic

        absolute_argument = np.abs(selected_frequency * state)
        phase = np.zeros(state.shape, dtype=float)
        nonzero = absolute_argument > 0.0
        log_phase = np.empty(np.count_nonzero(nonzero), dtype=float)
        if log_phase.size:
            log_phase[:] = (
                np.log(absolute_argument[nonzero])
                + active[nonzero]
                + 0.5 * variance_active
                - 0.5 * variance_residual
            )
            maximum_log_phase = float(np.max(log_phase))
            if maximum_log_phase > (
                np.log(self.max_phase)
                + 64.0 * np.finfo(float).eps
            ):
                raise ValueError(
                    "phase lies outside the certified rule domain "
                    f"(maximum_log={maximum_log_phase:.17g}, "
                    f"limit_log={np.log(self.max_phase):.17g})"
                )
            phase[nonzero] = np.exp(log_phase)

        characteristic, diagnostic = self.characteristic_cosine(phase, sigma)
        with np.errstate(over="ignore", invalid="ignore"):
            prefactor = selected_scale * np.exp(
                -active - 0.5 * variance_active
            )
            result = prefactor * characteristic
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Esscher cosine pullback is nonfinite")
        return result, diagnostic


__all__ = [
    "ContourShiftedLognormalDiagnostic",
    "ContourShiftedLognormalRule",
]
