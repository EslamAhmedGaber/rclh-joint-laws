"""Wave-batched adaptive vector Filon quadrature.

The scalar adaptive Filon rule evaluates one frequency at a time.  This
version keeps the same embedded high/low panel rules, but requests all missing
amplitude values for a small wave of panels at once.  It is version-isolated
so the sealed centered-Filon experiments remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable

import numpy as np

from .filon import FilonResult, _exponential_moments, _interpolation_data


BatchVectorAmplitude = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class _BatchedPanel:
    left: float
    right: float
    value: np.ndarray
    error: float


@dataclass(frozen=True, slots=True)
class BatchedAdaptiveExponentialFilon:
    """Adaptive Filon rule with deterministic panel-refinement waves."""

    cutoff: float = 28.0
    absolute_tolerance: float = 1.0e-9
    relative_tolerance: float = 1.0e-9
    high_degree: int = 8
    low_degree: int = 4
    max_panels: int = 4096
    refinement_batch_size: int = 4
    frequency_chunk_size: int = 32

    def __post_init__(self) -> None:
        cutoff = float(self.cutoff)
        absolute = float(self.absolute_tolerance)
        relative = float(self.relative_tolerance)
        high = int(self.high_degree)
        low = int(self.low_degree)
        panels = int(self.max_panels)
        wave = int(self.refinement_batch_size)
        chunk = int(self.frequency_chunk_size)
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
        if wave < 1:
            raise ValueError("refinement_batch_size must be positive")
        if chunk < 1:
            raise ValueError("frequency_chunk_size must be positive")
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "absolute_tolerance", absolute)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "high_degree", high)
        object.__setattr__(self, "low_degree", low)
        object.__setattr__(self, "max_panels", panels)
        object.__setattr__(self, "refinement_batch_size", wave)
        object.__setattr__(self, "frequency_chunk_size", chunk)

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
        amplitude_many: BatchVectorAmplitude,
        exponent: complex,
        *,
        feature_scale: float = 1.0,
    ) -> FilonResult:
        """Integrate a batched complex-vector amplitude on ``[0, cutoff]``."""

        oscillator = complex(exponent)
        if not (
            np.isfinite(oscillator.real) and np.isfinite(oscillator.imag)
        ):
            raise ValueError("exponent must be finite")
        if oscillator.real > 16.0 * np.finfo(float).eps:
            raise ValueError("Filon exponent must have nonpositive real part")

        high_nodes, high_inverse = _interpolation_data(self.high_degree)
        low_nodes, low_inverse = _interpolation_data(self.low_degree)
        cache: dict[float, np.ndarray] = {}
        output_size: int | None = None

        def evaluate_locations(locations: np.ndarray) -> np.ndarray:
            nonlocal output_size
            keys = [float(value) for value in np.asarray(locations).ravel()]
            missing: list[float] = []
            pending: set[float] = set()
            for key in keys:
                if key not in cache and key not in pending:
                    missing.append(key)
                    pending.add(key)
            for start in range(0, len(missing), self.frequency_chunk_size):
                chunk = missing[start : start + self.frequency_chunk_size]
                values = np.asarray(
                    amplitude_many(np.asarray(chunk, dtype=float)),
                    dtype=complex,
                )
                if values.ndim != 2 or values.shape[0] != len(chunk):
                    raise ValueError(
                        "amplitude_many must return a two-dimensional array "
                        "with one row per location"
                    )
                if not (
                    np.all(np.isfinite(values.real))
                    and np.all(np.isfinite(values.imag))
                ):
                    raise FloatingPointError(
                        "amplitude_many returned nonfinite data"
                    )
                if output_size is None:
                    output_size = int(values.shape[1])
                elif values.shape[1] != output_size:
                    raise ValueError("amplitude output size changed")
                for key, value in zip(chunk, values):
                    cache[key] = value.copy()
            if not keys:
                if output_size is None:
                    raise ValueError("cannot infer amplitude size from no values")
                return np.empty((0, output_size), dtype=complex)
            return np.stack([cache[key] for key in keys], axis=0)

        def panel_locations(
            left: float,
            right: float,
            nodes: np.ndarray,
        ) -> np.ndarray:
            return left + (right - left) * nodes

        def panel_rule(
            left: float,
            right: float,
            nodes: np.ndarray,
            inverse: np.ndarray,
        ) -> np.ndarray:
            width = right - left
            values = evaluate_locations(panel_locations(left, right, nodes))
            moments = _exponential_moments(
                oscillator * width,
                nodes.size - 1,
            )
            weights = moments @ inverse
            return np.exp(oscillator * left) * width * (weights @ values)

        def make_panels(
            intervals: list[tuple[float, float]],
        ) -> list[_BatchedPanel]:
            all_locations = []
            for left, right in intervals:
                all_locations.append(panel_locations(left, right, high_nodes))
                all_locations.append(panel_locations(left, right, low_nodes))
            if all_locations:
                evaluate_locations(np.concatenate(all_locations))
            made = []
            for left, right in intervals:
                high_value = panel_rule(
                    left, right, high_nodes, high_inverse
                )
                low_value = panel_rule(left, right, low_nodes, low_inverse)
                error = float(np.max(np.abs(high_value - low_value)))
                made.append(_BatchedPanel(left, right, high_value, error))
            return made

        breakpoints = self._initial_breakpoints(feature_scale)
        initial_intervals = [
            (float(left), float(right))
            for left, right in zip(breakpoints[:-1], breakpoints[1:])
        ]
        panels: dict[int, _BatchedPanel] = {}
        heap: list[tuple[float, int]] = []
        next_identifier = 0
        total_value: np.ndarray | None = None
        total_error = 0.0

        def add_panel(panel: _BatchedPanel) -> None:
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

        for panel in make_panels(initial_intervals):
            add_panel(panel)
        assert total_value is not None

        def target() -> float:
            assert total_value is not None
            return self.absolute_tolerance + self.relative_tolerance * float(
                np.max(np.abs(total_value))
            )

        while total_error > target() and len(panels) < self.max_panels:
            available_splits = self.max_panels - len(panels)
            take = min(self.refinement_batch_size, available_splits)
            selected: list[tuple[int, _BatchedPanel]] = []
            while heap and len(selected) < take:
                _negative_error, identifier = heapq.heappop(heap)
                if identifier in panels:
                    selected.append((identifier, panels.pop(identifier)))
            if not selected:
                break
            child_intervals: list[tuple[float, float]] = []
            for _identifier, old in selected:
                total_value = total_value - old.value
                total_error -= old.error
                midpoint = 0.5 * (old.left + old.right)
                child_intervals.extend(
                    [(old.left, midpoint), (midpoint, old.right)]
                )
            for panel in make_panels(child_intervals):
                add_panel(panel)

        return FilonResult(
            value=total_value,
            estimated_error=float(max(total_error, 0.0)),
            evaluation_count=len(cache),
            panel_count=len(panels),
            converged=bool(total_error <= target()),
        )


__all__ = ["BatchedAdaptiveExponentialFilon"]
