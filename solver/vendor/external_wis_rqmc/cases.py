"""Frozen scalar WIS benchmark cases.

This module is intentionally independent of ``wis_chaos``.  The drift
functions operate only on ordinary floating-point arrays and do not expose
Hermite bases, chaos coefficients, or intrusive projection machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    """One frozen quasilinear scalar WIS problem."""

    code: str
    alpha: float
    beta: float
    initial_value: float
    drift_name: str

    def drift(self, time: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the non-linear drift ``a(t, x)`` stably."""

        del time  # All four frozen cases are autonomous.
        values = np.asarray(state, dtype=float)
        if self.code == "G":
            return np.zeros_like(values)
        if self.code == "R":
            magnitude = np.abs(values)
            output = np.empty_like(values)
            small = magnitude <= 1.0
            output[small] = (
                4.0
                * values[small]
                / (1.0 + values[small] * values[small])
            )
            large = ~small
            output[large] = 4.0 / (
                values[large] + 1.0 / values[large]
            )
            return output
        if self.code == "C":
            return np.cos(values)
        if self.code == "L":
            magnitude = np.abs(values)
            output = np.empty_like(values)
            small = magnitude <= 1.0
            output[small] = 25.0 * np.log1p(
                values[small] * values[small]
            )
            large = ~small
            output[large] = 25.0 * (
                2.0 * np.log(magnitude[large])
                + np.log1p(1.0 / (magnitude[large] * magnitude[large]))
            )
            return output
        raise RuntimeError(f"unsupported frozen case {self.code!r}")


_CASES = {
    "G": BenchmarkCase("G", 1.0, 1.0, 1.0, "zero"),
    "R": BenchmarkCase("R", 1.0, 1.0, 1.0, "4x/(1+x^2)"),
    "C": BenchmarkCase("C", -1.0, 0.5, 10.0, "cos(x)"),
    "L": BenchmarkCase("L", 0.0, 5.0, 25.0, "25 log(1+x^2)"),
}

FROZEN_CASES = MappingProxyType(_CASES)
FROZEN_HURST_VALUES = (0.10, 0.25, 0.50, 0.75)


def benchmark_case(code: str) -> BenchmarkCase:
    """Return a frozen case by its single-letter code."""

    key = str(code).upper()
    try:
        return FROZEN_CASES[key]
    except KeyError as error:
        raise ValueError(
            f"case must be one of {tuple(FROZEN_CASES)}"
        ) from error


__all__ = [
    "BenchmarkCase",
    "FROZEN_CASES",
    "FROZEN_HURST_VALUES",
    "benchmark_case",
]
