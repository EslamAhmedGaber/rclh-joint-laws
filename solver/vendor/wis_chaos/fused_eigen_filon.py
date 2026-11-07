"""Adaptive Filon integration fused with conditional eigenbasis assembly.

For every panel, the high rule and the high-minus-low embedded rule are
contracted while linear and quadratic tail moments remain in their
conditional eigenbases.  The two rule outputs are then rotated and scattered
to physical retained Hermite coefficients together.  Consequently the
adaptive error remains the physical coefficient infinity norm used by the
validated centered-Filon algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .eigenbasis_centered_amplitude import (
    CompactCenteredGroupBatch,
    NativeGroupBatch,
    StableCenteredEigenAmplitude,
)
from .filon import _exponential_moments, _interpolation_data


@dataclass(frozen=True, slots=True)
class FusedPanelOracle:
    """Physical high/low results for one frozen Filon panel."""

    high_value: np.ndarray
    low_value: np.ndarray
    difference: np.ndarray
    error: float
    evaluation_count: int
    rotation_count: int


@dataclass(frozen=True, slots=True)
class FusedEigenFilonResult:
    """Adaptive fused-Filon result and implementation instrumentation."""

    value: np.ndarray
    estimated_error: float
    evaluation_count: int
    panel_count: int
    converged: bool
    panel_creation_count: int
    rotation_count: int


@dataclass(frozen=True, slots=True)
class _FusedPanel:
    left: float
    right: float
    value: np.ndarray
    error: float


_CachedGroupNode = tuple[np.ndarray, np.ndarray, complex]
_CachedNode = tuple[_CachedGroupNode, ...]


@dataclass(frozen=True, slots=True)
class FusedEigenAdaptiveFilon:
    """Adaptive vector Filon rule with panel-level eigenbasis fusion."""

    cutoff: float = 28.0
    absolute_tolerance: float = 1.0e-9
    relative_tolerance: float = 1.0e-9
    high_degree: int = 8
    low_degree: int = 4
    max_panels: int = 4096
    refinement_batch_size: int = 1
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

    @staticmethod
    def _validate_exponent(exponent: complex) -> complex:
        oscillator = complex(exponent)
        if not (
            np.isfinite(oscillator.real)
            and np.isfinite(oscillator.imag)
        ):
            raise ValueError("exponent must be finite")
        if oscillator.real > 16.0 * np.finfo(float).eps:
            raise ValueError("Filon exponent must have nonpositive real part")
        return oscillator

    def _embedded_panel_data(
        self,
        left: float,
        right: float,
        oscillator: complex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return high locations and physical high/high-minus-low weights."""

        a = float(left)
        b = float(right)
        if not np.isfinite(a) or not np.isfinite(b) or not 0.0 <= a < b:
            raise ValueError("panel endpoints must satisfy 0 <= left < right")
        high_nodes, high_inverse = _interpolation_data(self.high_degree)
        _low_nodes, low_inverse = _interpolation_data(self.low_degree)
        width = b - a
        locations = a + width * high_nodes
        high_moments = _exponential_moments(
            oscillator * width,
            self.high_degree,
        )
        low_moments = _exponential_moments(
            oscillator * width,
            self.low_degree,
        )
        high_weights = high_moments @ high_inverse
        low_small = low_moments @ low_inverse
        low_weights = np.zeros(high_nodes.size, dtype=complex)
        ratio = self.high_degree // self.low_degree
        low_indices = ratio * np.arange(self.low_degree + 1)
        low_weights[low_indices] = low_small
        prefactor = np.exp(oscillator * a) * width
        high_weights = prefactor * high_weights
        low_weights = prefactor * low_weights
        return locations, np.stack(
            [high_weights, high_weights - low_weights],
            axis=0,
        )

    @staticmethod
    def _stack_cached_group(
        cache: dict[float, _CachedNode],
        locations: np.ndarray,
        group_index: int,
    ) -> CompactCenteredGroupBatch:
        selected = [
            cache[float(location)][group_index] for location in locations
        ]
        return CompactCenteredGroupBatch(
            core_d=np.stack([value[0] for value in selected], axis=0),
            delta_gram=np.stack([value[1] for value in selected], axis=0),
            delta_mass=np.asarray(
                [value[2] for value in selected], dtype=complex
            ),
        )

    @staticmethod
    def _contract_group(
        evaluator: StableCenteredEigenAmplitude,
        group_index: int,
        locations: np.ndarray,
        rule_weights: np.ndarray,
        compact: CompactCenteredGroupBatch,
        *,
        rational: bool,
    ) -> NativeGroupBatch:
        """Contract high and difference rules without nodewise rotations."""

        group = evaluator.groups[group_index]
        values = np.asarray(locations, dtype=float)
        kappa = values if rational else np.ones_like(values)
        effective_weights = rule_weights * kappa[None, :]
        frequency = values / evaluator.width
        inverse = 1.0 / (
            1.0
            - 2.0j
            * frequency[:, None]
            * group.eigenvalues[None, :]
        )
        delta_inverse = (
            2.0j
            * frequency[:, None]
            * group.eigenvalues[None, :]
            * inverse
            - np.expm1(-evaluator.gamma * values)[:, None]
        )

        core = np.einsum(
            "rf,fc->rc",
            effective_weights,
            compact.core_d,
            optimize=True,
        )
        b_transpose = group.rotated_linear_map.T
        b_delta_gram = np.einsum(
            "ta,fab->ftb",
            b_transpose,
            compact.delta_gram,
            optimize=True,
        )
        b_base_gram = b_transpose @ group.weighted_linear_gram
        linear_nodes = (
            1.0j
            / evaluator.width
            * (
                inverse[:, :, None] * b_delta_gram
                + delta_inverse[:, :, None]
                * b_base_gram[None, :, :]
            )
        )
        linear = np.einsum(
            "rf,ftb->rtb",
            effective_weights,
            linear_nodes,
            optimize=True,
        )

        diagonal_nodes = (
            2.0j
            / evaluator.width
            * group.eigenvalues[None, :]
            * (
                inverse * compact.delta_mass[:, None]
                + delta_inverse * group.group_mass
            )
        )
        diagonal = np.einsum(
            "rf,ft->rt",
            effective_weights,
            diagonal_nodes,
            optimize=True,
        )
        t_bar = (
            1.0j
            / evaluator.width
            * group.rotated_linear_map[None, :, :]
            * inverse[:, None, :]
        )
        full_gram = (
            group.weighted_linear_gram[None, :, :]
            + compact.delta_gram
        )
        quartic_weights = effective_weights * values[None, :]
        second = np.einsum(
            "rf,fat,fab,fbe->rte",
            quartic_weights,
            t_bar,
            full_gram,
            t_bar,
            optimize=True,
        )
        diagonal_indices = np.arange(evaluator.tail_dimension)
        second[:, diagonal_indices, diagonal_indices] += diagonal
        return NativeGroupBatch(
            core=core,
            linear_eigen=linear,
            second_eigen=second,
        )

    def _make_cache_functions(
        self,
        evaluator: StableCenteredEigenAmplitude,
    ):
        cache: dict[float, _CachedNode] = {}

        def evaluate_locations(locations: np.ndarray) -> None:
            keys = [float(value) for value in np.asarray(locations).ravel()]
            missing: list[float] = []
            pending: set[float] = set()
            for key in keys:
                if key not in cache and key not in pending:
                    missing.append(key)
                    pending.add(key)
            for start in range(0, len(missing), self.frequency_chunk_size):
                chunk = np.asarray(
                    missing[start : start + self.frequency_chunk_size],
                    dtype=float,
                )
                batches = evaluator.compact_many(chunk)
                for position, key in enumerate(chunk):
                    node_groups: list[_CachedGroupNode] = []
                    for batch in batches:
                        core = batch.core_d[position].copy()
                        gram = batch.delta_gram[position].copy()
                        mass = complex(batch.delta_mass[position])
                        if not (
                            np.all(np.isfinite(core.real))
                            and np.all(np.isfinite(core.imag))
                            and np.all(np.isfinite(gram.real))
                            and np.all(np.isfinite(gram.imag))
                            and np.isfinite(mass.real)
                            and np.isfinite(mass.imag)
                        ):
                            raise FloatingPointError(
                                "compact centered amplitude is nonfinite"
                            )
                        node_groups.append((core, gram, mass))
                    cache[float(key)] = tuple(node_groups)

        return cache, evaluate_locations

    def _panel_from_cache(
        self,
        evaluator: StableCenteredEigenAmplitude,
        left: float,
        right: float,
        oscillator: complex,
        cache: dict[float, _CachedNode],
        evaluate_locations,
        *,
        rational: bool,
    ) -> FusedPanelOracle:
        locations, rule_weights = self._embedded_panel_data(
            left, right, oscillator
        )
        evaluate_locations(locations)
        native = []
        for group_index in range(evaluator.group_count):
            compact = self._stack_cached_group(
                cache, locations, group_index
            )
            native.append(
                self._contract_group(
                    evaluator,
                    group_index,
                    locations,
                    rule_weights,
                    compact,
                    rational=rational,
                )
            )
        physical = evaluator.assemble_native(tuple(native))
        high = physical[0]
        difference = physical[1]
        low = high - difference
        return FusedPanelOracle(
            high_value=high,
            low_value=low,
            difference=difference,
            error=float(np.max(np.abs(difference))),
            evaluation_count=len(cache),
            rotation_count=evaluator.group_count,
        )

    def panel_oracle(
        self,
        evaluator: StableCenteredEigenAmplitude,
        exponent: complex,
        left: float,
        right: float,
        *,
        rational: bool = False,
    ) -> FusedPanelOracle:
        """Evaluate one panel, exposing high and low physical vectors."""

        if not isinstance(evaluator, StableCenteredEigenAmplitude):
            raise TypeError("evaluator must be StableCenteredEigenAmplitude")
        oscillator = self._validate_exponent(exponent)
        cache, evaluate_locations = self._make_cache_functions(evaluator)
        return self._panel_from_cache(
            evaluator,
            float(left),
            float(right),
            oscillator,
            cache,
            evaluate_locations,
            rational=bool(rational),
        )

    def integrate(
        self,
        evaluator: StableCenteredEigenAmplitude,
        exponent: complex,
        *,
        rational: bool = False,
        feature_scale: float = 1.0,
    ) -> FusedEigenFilonResult:
        """Integrate a stable centered group amplitude on ``[0, cutoff]``."""

        if not isinstance(evaluator, StableCenteredEigenAmplitude):
            raise TypeError("evaluator must be StableCenteredEigenAmplitude")
        oscillator = self._validate_exponent(exponent)
        cache, evaluate_locations = self._make_cache_functions(evaluator)
        rotation_count = 0
        panel_creation_count = 0

        def make_panels(
            intervals: list[tuple[float, float]],
        ) -> list[_FusedPanel]:
            nonlocal rotation_count, panel_creation_count
            all_locations = [
                self._embedded_panel_data(left, right, oscillator)[0]
                for left, right in intervals
            ]
            if all_locations:
                evaluate_locations(np.concatenate(all_locations))
            made: list[_FusedPanel] = []
            for left, right in intervals:
                oracle = self._panel_from_cache(
                    evaluator,
                    left,
                    right,
                    oscillator,
                    cache,
                    evaluate_locations,
                    rational=bool(rational),
                )
                rotation_count += oracle.rotation_count
                panel_creation_count += 1
                made.append(
                    _FusedPanel(
                        left=left,
                        right=right,
                        value=oracle.high_value,
                        error=oracle.error,
                    )
                )
            return made

        breakpoints = self._initial_breakpoints(feature_scale)
        initial_intervals = [
            (float(left), float(right))
            for left, right in zip(breakpoints[:-1], breakpoints[1:])
        ]
        panels: dict[int, _FusedPanel] = {}
        heap: list[tuple[float, int]] = []
        next_identifier = 0
        total_value: np.ndarray | None = None
        total_error = 0.0

        def add_panel(panel: _FusedPanel) -> None:
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
            selected: list[tuple[int, _FusedPanel]] = []
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

        return FusedEigenFilonResult(
            value=total_value,
            estimated_error=float(max(total_error, 0.0)),
            evaluation_count=len(cache),
            panel_count=len(panels),
            converged=bool(total_error <= target()),
            panel_creation_count=panel_creation_count,
            rotation_count=rotation_count,
        )


__all__ = [
    "FusedEigenAdaptiveFilon",
    "FusedEigenFilonResult",
    "FusedPanelOracle",
]
