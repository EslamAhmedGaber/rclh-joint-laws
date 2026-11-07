"""Adaptive vector-valued Filon quadrature for a linear oscillator.

The routine integrates

``integral_0^U exp(z*x) * amplitude(x) dx``

with ``Re(z) <= 0``.  On every panel the amplitude, rather than the full
oscillatory integrand, is interpolated at nested Chebyshev--Lobatto nodes.
The polynomial times the exponential is then integrated analytically.  This
keeps the number of amplitude evaluations governed mainly by amplitude
variation instead of ``abs(Im(z))``.

The embedded high/low difference is an a posteriori numerical estimator, not
a theorem-level error certificate.  Callers must add a separate truncation
tail bound and verify refinement against an independent reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
from typing import Callable

import numpy as np


VectorAmplitude = Callable[[float], np.ndarray]


@dataclass(frozen=True, slots=True)
class FilonResult:
    value: np.ndarray
    estimated_error: float
    evaluation_count: int
    panel_count: int
    converged: bool


@dataclass(frozen=True, slots=True)
class _Panel:
    left: float
    right: float
    value: np.ndarray
    error: float


@lru_cache(maxsize=None)
def _interpolation_data(
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = int(degree)
    if order < 1:
        raise ValueError("interpolation degree must be positive")
    index = np.arange(order + 1, dtype=float)
    nodes = np.sort(0.5 * (1.0 - np.cos(np.pi * index / order)))
    vandermonde = np.polynomial.polynomial.polyvander(nodes, order)
    inverse = np.linalg.inv(vandermonde)
    return nodes, inverse


def _exponential_moments(exponent: complex, degree: int) -> np.ndarray:
    r"""Return ``int_0^1 s**k exp(exponent*s) ds`` for ``k=0..degree``."""

    argument = complex(exponent)
    order = int(degree)
    result = np.empty(order + 1, dtype=complex)
    if abs(argument) < 1.0:
        # Entire power series; this avoids cancellation in the recurrence at
        # a small complex argument.
        for power in range(order + 1):
            total = 0.0j
            term = 1.0 + 0.0j
            for series_order in range(256):
                addition = term / (series_order + power + 1)
                updated = total + addition
                if (
                    series_order >= 6
                    and abs(addition)
                    <= 4.0
                    * np.finfo(float).eps
                    * max(1.0, abs(updated))
                ):
                    total = updated
                    break
                total = updated
                term *= argument / (series_order + 1)
            else:
                raise FloatingPointError(
                    "exponential-moment series did not converge"
                )
            result[power] = total
        return result

    exponential = np.exp(argument)
    result[0] = np.expm1(argument) / argument
    for power in range(1, order + 1):
        result[power] = (
            exponential / argument
            - power * result[power - 1] / argument
        )
    return result


@dataclass(frozen=True, slots=True)
class AdaptiveExponentialFilon:
    """Nested-panel Filon integrator with a global refinement queue."""

    cutoff: float = 28.0
    absolute_tolerance: float = 1.0e-9
    relative_tolerance: float = 1.0e-9
    high_degree: int = 8
    low_degree: int = 4
    max_panels: int = 4096

    def __post_init__(self) -> None:
        cutoff = float(self.cutoff)
        absolute = float(self.absolute_tolerance)
        relative = float(self.relative_tolerance)
        high = int(self.high_degree)
        low = int(self.low_degree)
        panels = int(self.max_panels)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be finite and positive")
        if not np.isfinite(absolute) or absolute <= 0.0:
            raise ValueError("absolute_tolerance must be finite and positive")
        if not np.isfinite(relative) or relative < 0.0:
            raise ValueError("relative_tolerance must be finite and nonnegative")
        if high < 2 or low < 1 or high <= low:
            raise ValueError("require high_degree > low_degree >= 1")
        if high % low != 0:
            raise ValueError(
                "low-degree Chebyshev nodes must nest in the high rule"
            )
        if panels < 1:
            raise ValueError("max_panels must be positive")
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "absolute_tolerance", absolute)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "high_degree", high)
        object.__setattr__(self, "low_degree", low)
        object.__setattr__(self, "max_panels", panels)

    def _initial_breakpoints(self, feature_scale: float) -> np.ndarray:
        scale = float(feature_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("feature_scale must be finite and positive")
        scale = min(scale, self.cutoff)
        points = {0.0, self.cutoff}
        for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
            point = multiplier * scale
            if 0.0 < point < self.cutoff:
                points.add(point)
        point = max(4.0 * scale, scale)
        while point < self.cutoff:
            points.add(point)
            point *= 2.0
        return np.asarray(sorted(points), dtype=float)

    def integrate(
        self,
        amplitude: VectorAmplitude,
        exponent: complex,
        *,
        feature_scale: float = 1.0,
    ) -> FilonResult:
        """Integrate a complex vector amplitude on ``[0, cutoff]``."""

        oscillator = complex(exponent)
        if not (
            np.isfinite(oscillator.real)
            and np.isfinite(oscillator.imag)
        ):
            raise ValueError("exponent must be finite")
        if oscillator.real > 16.0 * np.finfo(float).eps:
            raise ValueError("Filon exponent must have nonpositive real part")

        high_nodes, high_inverse = _interpolation_data(self.high_degree)
        low_nodes, low_inverse = _interpolation_data(self.low_degree)
        cache: dict[float, np.ndarray] = {}
        output_shape: tuple[int, ...] | None = None

        def evaluate(location: float) -> np.ndarray:
            nonlocal output_shape
            key = float(location)
            if key not in cache:
                value = np.asarray(amplitude(key), dtype=complex)
                if value.ndim != 1:
                    raise ValueError("amplitude must return a one-dimensional vector")
                if not (
                    np.all(np.isfinite(value.real))
                    and np.all(np.isfinite(value.imag))
                ):
                    raise FloatingPointError("amplitude returned nonfinite data")
                if output_shape is None:
                    output_shape = value.shape
                elif value.shape != output_shape:
                    raise ValueError("amplitude output shape changed")
                cache[key] = value
            return cache[key]

        def panel_rule(
            left: float,
            right: float,
            nodes: np.ndarray,
            inverse: np.ndarray,
        ) -> np.ndarray:
            width = right - left
            values = np.stack(
                [evaluate(left + width * node) for node in nodes],
                axis=0,
            )
            moments = _exponential_moments(
                oscillator * width,
                nodes.size - 1,
            )
            weights = moments @ inverse
            return (
                np.exp(oscillator * left)
                * width
                * (weights @ values)
            )

        def make_panel(left: float, right: float) -> _Panel:
            high_value = panel_rule(
                left,
                right,
                high_nodes,
                high_inverse,
            )
            low_value = panel_rule(
                left,
                right,
                low_nodes,
                low_inverse,
            )
            error = float(np.max(np.abs(high_value - low_value)))
            return _Panel(left, right, high_value, error)

        breakpoints = self._initial_breakpoints(feature_scale)
        panels: dict[int, _Panel] = {}
        heap: list[tuple[float, int]] = []
        next_identifier = 0
        total_value: np.ndarray | None = None
        total_error = 0.0

        def add_panel(panel: _Panel) -> None:
            nonlocal next_identifier, total_value, total_error
            identifier = next_identifier
            next_identifier += 1
            panels[identifier] = panel
            heapq.heappush(heap, (-panel.error, identifier))
            total_value = (
                panel.value.copy()
                if total_value is None
                else total_value + panel.value
            )
            total_error += panel.error

        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            add_panel(make_panel(float(left), float(right)))

        assert total_value is not None

        def target() -> float:
            return self.absolute_tolerance + self.relative_tolerance * float(
                np.max(np.abs(total_value))
            )

        while total_error > target() and len(panels) < self.max_panels:
            while heap:
                _negative_error, identifier = heapq.heappop(heap)
                if identifier in panels:
                    break
            else:
                break
            old = panels.pop(identifier)
            total_value = total_value - old.value
            total_error -= old.error
            midpoint = 0.5 * (old.left + old.right)
            add_panel(make_panel(old.left, midpoint))
            add_panel(make_panel(midpoint, old.right))

        return FilonResult(
            value=total_value,
            estimated_error=float(max(total_error, 0.0)),
            evaluation_count=len(cache),
            panel_count=len(panels),
            converged=bool(total_error <= target()),
        )


__all__ = ["AdaptiveExponentialFilon", "FilonResult"]
