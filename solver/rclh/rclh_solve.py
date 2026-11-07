"""One documented entry point to the residual-covariance Lawson-Hermite (RCLH) solver.

Problem class
-------------
    dX_t = (a X_t + b(X_t)) dt + beta X_t <> dB^H_t,   X_0 = x0,   0 <= t <= T,

where <> is the Wick product and B^H is fractional Brownian motion with Hurst index H in (0, 1).
The solver targets the finite problem in which B^H is represented by its exact Gaussian law on the
grid t_1 < ... < t_N (default t_i = iT/N), interpolated linearly between grid points, and it returns
the moments of the coherent reconstruction X_coh on that grid.

Supported drifts b
------------------
* Polynomial: b(x) = sum_k c_k x^k with k = 0 or k >= 2 (a degree-one term is merged into ``a``).
  The residual closure is evaluated in closed form (``PolynomialResidualLawsonSolver``).
* The built-in non-polynomial families of ``wis_chaos.spectral_projection``:
  ``rational(scale, width)``      -> scale * x / (width^2 + x^2),
  ``cosine(scale, frequency)``    -> scale * cos(frequency * x),
  ``log_quadratic(scale, width)`` -> scale * log(1 + (x / width)^2),
  plus a constant; the residual closure then uses scalar residual quadrature
  (``ResidualCoherentLawsonSolver``).  Only the rational family has been tested against an
  independent reference in an earlier inconclusive test. Further validation
  studies must be reported separately; this wrapper does not validate accuracy.

Scope: scalar equations with multiplicative noise beta*X only.  This module adds no numerical method;
it validates inputs, builds the factor and the solver, collects diagnostics and packages the outputs.
The engine modules are imported only from ``../vendor`` and checked against
``vendor/MANIFEST_SHA256.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = Path(os.environ.get("RCLH_VENDOR_DIR", _PACKAGE_DIR.parent / "vendor")).resolve()
_MANIFEST_PATH = VENDOR_DIR / "MANIFEST_SHA256.json"
_ENGINE_FILES = (
    "polynomial_residual_lawson_v1.py",
    "wis_chaos/residual_coherent_lawson.py",
    "wis_chaos/basis.py",
    "wis_chaos/coherent.py",
    "wis_chaos/fbm.py",
    "wis_chaos/hierarchical_fbm.py",
    "wis_chaos/spectral_projection.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_engine() -> dict[str, str]:
    """Put the vendored engine first on sys.path and verify its pinned hashes."""

    if not _MANIFEST_PATH.is_file():
        raise ImportError(f"vendor manifest not found: {_MANIFEST_PATH}")
    pins = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))["sha256"]
    digests: dict[str, str] = {}
    for relative in pins:
        path = VENDOR_DIR / relative
        if VENDOR_DIR not in path.resolve().parents:
            raise ImportError("vendor manifest contains a path outside its root")
        digest = _sha256(path)
        if pins.get(relative) != digest:
            raise ImportError(f"engine file differs from the pinned hash: {relative}")
        digests[relative] = digest
    if not all(relative in pins for relative in _ENGINE_FILES):
        raise ImportError("vendor manifest omits required engine files")
    vendor = str(VENDOR_DIR)
    sys.path[:] = [vendor] + [entry for entry in sys.path if entry != vendor]
    for name in tuple(sys.modules):
        if not (name == "polynomial_residual_lawson_v1" or name == "wis_chaos" or name.startswith("wis_chaos.")):
            continue
        module = sys.modules.get(name)
        if module is not None:
            origin = Path(getattr(module, "__file__", "") or "").resolve()
            if VENDOR_DIR not in origin.parents:
                raise ImportError(
                    f"module {name!r} was already imported from {origin}; start a fresh process"
                )
    return digests


ENGINE_SHA256 = _bootstrap_engine()

from polynomial_residual_lawson_v1 import (  # noqa: E402
    PolynomialDrift,
    PolynomialResidualLawsonSolver,
)
from wis_chaos.fbm import (  # noqa: E402
    GaussianPathFactor,
    endpoint_conditioned_factor,
    path_kl_factor,
)
from wis_chaos.hierarchical_fbm import endpoint_hierarchical_dyadic_factor  # noqa: E402
from wis_chaos.residual_coherent_lawson import ResidualCoherentLawsonSolver  # noqa: E402
from wis_chaos.spectral_projection import (  # noqa: E402
    AffineSpectralDrift,
    CosineTransform,
    LogQuadraticCosineTransform,
    RationalSineTransform,
)

for _name in tuple(sys.modules):
    if not (_name == "polynomial_residual_lawson_v1" or _name == "wis_chaos" or _name.startswith("wis_chaos.")):
        continue
    _origin = Path(sys.modules[_name].__file__).resolve()
    if VENDOR_DIR not in _origin.parents:
        raise ImportError(f"{_name} resolved outside the vendor directory: {_origin}")


# ----------------------------------------------------------------------------------------------
# Drift specifications
# ----------------------------------------------------------------------------------------------


def rational(scale: float, width: float = 1.0) -> RationalSineTransform:
    """Term scale * x / (width^2 + x^2)."""

    return RationalSineTransform(float(scale), width=float(width))


def cosine(scale: float, frequency: float = 1.0) -> CosineTransform:
    """Term scale * cos(frequency * x)."""

    return CosineTransform(float(scale), frequency=float(frequency))


def log_quadratic(scale: float, width: float = 1.0) -> LogQuadraticCosineTransform:
    """Term scale * log(1 + (x / width)^2)."""

    return LogQuadraticCosineTransform(float(scale), width=float(width))


def trapezoid_weights(times: Sequence[float], scale: float = 1.0) -> np.ndarray:
    """Trapezoidal weights on the output grid (including t = 0), multiplied by ``scale``."""

    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("times must be a finite, strictly increasing one-dimensional grid")
    if not np.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    dt = np.diff(grid)
    weights = np.zeros(grid.size)
    weights[:-1] += 0.5 * dt
    weights[1:] += 0.5 * dt
    return float(scale) * weights


# ----------------------------------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------------------------------


@dataclass
class RCLHResult:
    """Moments of the coherent reconstruction on the grid ``times`` (index 0 is t = 0)."""

    times: np.ndarray
    mean: np.ndarray
    second_moment: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray
    cross_moments: np.ndarray
    projection_gap: np.ndarray
    residual_variance: np.ndarray
    lawson_coefficients: np.ndarray
    basis_indices: np.ndarray
    config: dict[str, Any]
    diagnostics: dict[str, Any]
    timings: dict[str, float]
    engine: dict[str, Any]
    raw: Any = field(default=None, repr=False)

    def index(self, t: float) -> int:
        """Grid index of the output time ``t`` (raises if ``t`` is not a grid time)."""

        if not np.isfinite(float(t)):
            raise ValueError("output time must be finite")
        position = int(np.argmin(np.abs(self.times - float(t))))
        if abs(self.times[position] - float(t)) > 1e-12 * max(1.0, abs(float(t))):
            raise ValueError(f"t = {t} is not an output time")
        return position

    def integral(self, weights: Sequence[float] | None = None, scale: float = 1.0) -> dict[str, float]:
        """Mean and variance of sum_i w_i X_coh(t_i); trapezoidal weights by default."""

        if not np.isfinite(float(scale)):
            raise ValueError("scale must be finite")
        w = trapezoid_weights(self.times, scale) if weights is None else float(scale) * np.asarray(weights, float)
        if w.shape != self.times.shape:
            raise ValueError("weights must have one entry per output time, including t = 0")
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite")
        return {"mean": float(w @ self.mean), "variance": float(w @ self.covariance @ w)}

    def es_projection_gap(self, gamma: float = 0.95) -> np.ndarray:
        """ES difference bound between coherent and retained physical laws.

        This is an equal-mean projection comparison. It is NOT a bound on the
        solver's error against the full WIS solution or an ES estimate itself.
        """
        gamma = float(gamma)
        if not np.isfinite(gamma) or not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        return math.sqrt(gamma / (1.0 - gamma)) * self.projection_gap

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value

        names = (
            "times", "mean", "second_moment", "variance", "covariance", "cross_moments",
            "projection_gap", "residual_variance", "lawson_coefficients", "basis_indices",
            "config", "diagnostics", "timings", "engine",
        )
        return {name: plain(getattr(self, name)) for name in names}

    def summary(self) -> str:
        d = self.diagnostics
        nodal = "n/a" if d["min_nodal_value"] is None else f"{d['min_nodal_value']:.4g}"
        return (
            f"RCLH {self.config['drift_label']}: H={self.config['hurst']}, N={self.config['N']}, "
            f"M={self.config['modes']}, P={self.config['degree']}, q={self.config['quadrature_order']}, "
            f"closure={self.config['closure']} | E X_T={self.mean[-1]:.6g}, Var X_T={self.variance[-1]:.6g} | "
            f"nodes={d['active_nodes']}, basis={d['basis_size']}, Gram defect={d['gram_error']:.1e}, "
            f"min nodal Lawson value={nodal}, wall={self.timings['total_seconds']:.3f} s"
        )


# ----------------------------------------------------------------------------------------------
# Factor construction
# ----------------------------------------------------------------------------------------------


def _scaled_hierarchical_factor(N: int, hurst: float, T: float, modes: int) -> GaussianPathFactor:
    """Endpoint-first dyadic factor on [0, T] by fBm self-similarity B_{cT} = c^H B_T."""

    base = endpoint_hierarchical_dyadic_factor(N, hurst, modes=modes)
    if T == 1.0:
        return base
    scale = T ** hurst
    return GaussianPathFactor(
        times=base.times * T,
        hurst=base.hurst,
        loadings=base.loadings * scale,
        covariance=base.covariance * scale * scale,
        residual=base.residual * scale * scale,
        method=base.method + f"_selfsimilar_T{T:g}",
    )


def build_factor(
    factor: str | GaussianPathFactor,
    *,
    hurst: float,
    T: float,
    N: int,
    modes: int,
    times: Sequence[float] | None = None,
) -> GaussianPathFactor:
    """Return the Gaussian path factor with ``modes`` retained coordinates."""

    N = _integer(N, "N", 1)
    modes = _integer(modes, "modes", 1)
    if not np.isfinite(T) or T <= 0 or not 0 < hurst < 1:
        raise ValueError("T must be finite and positive and hurst must lie in (0, 1)")
    if isinstance(factor, GaussianPathFactor):
        if factor.modes != modes:
            raise ValueError("a supplied factor must have exactly `modes` columns")
        if not np.isclose(factor.hurst, hurst, rtol=0, atol=1e-14):
            raise ValueError("supplied factor Hurst index disagrees with hurst")
        if times is not None and not np.array_equal(np.asarray(times, float), factor.times):
            raise ValueError("supplied factor and explicit times disagree")
        return factor
    grid = np.arange(1, N + 1, dtype=float) * (T / N) if times is None else np.asarray(times, float)
    if grid.ndim != 1 or grid.size < 1 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0) or grid[0] <= 0:
        raise ValueError("output times must be positive and strictly increasing")
    if factor == "endpoint":
        return endpoint_conditioned_factor(grid, hurst, modes=modes)
    if factor == "spectral":
        return path_kl_factor(grid, hurst, modes=modes)
    if factor == "hierarchy":
        if times is not None:
            raise ValueError("the dyadic hierarchy is defined on the uniform grid only")
        return _scaled_hierarchical_factor(N, hurst, T, modes)
    raise ValueError("factor must be 'endpoint', 'spectral', 'hierarchy' or a GaussianPathFactor")


# ----------------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------------


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        integer = int(value)
        scalar = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if not np.isfinite(scalar) or scalar != integer or integer < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return integer


def rclh_solve(
    *,
    x0: float,
    beta: float,
    hurst: float,
    linear_rate: float = 0.0,
    coefficients: Mapping[int, float] | None = None,
    transforms: Sequence[Any] | None = None,
    T: float = 1.0,
    N: int = 16,
    times: Sequence[float] | None = None,
    modes: int = 3,
    degree: int = 3,
    quadrature_order: int | None = None,
    substeps: int = 4,
    factor: str | GaussianPathFactor = "endpoint",
    closure: str = "residual",
    residual_quadrature_order: int = 25,
    stage_observer: Callable[[Any], None] | None = None,
    max_active_nodes: int = 2_000_000,
    label: str | None = None,
) -> RCLHResult:
    """Solve dX = (a X + b(X)) dt + beta X <> dB^H on the finite N-point Gaussian driver.

    Parameters
    ----------
    x0, beta, hurst : initial value, noise intensity and Hurst index.
    linear_rate : the coefficient a of X (projected without stochastic quadrature
        error; time integration still uses RK4).
    coefficients : polynomial part {k: c_k} of b (k = 0 or k >= 2; k = 1 is merged into a).
    transforms : non-polynomial terms, a sequence built with ``rational``, ``cosine`` or
        ``log_quadratic``; a constant may be given as coefficients={0: c}.  Exactly one of
        ``coefficients`` (with a nonconstant term) or ``transforms`` selects the solver branch.
    T, N, times : horizon and grid (t_i = iT/N unless ``times`` is given).
        Explicit times or a supplied factor determine the actual N and T
        reported in the result. A supplied factor must match hurst and modes.
    modes, degree : retained Gaussian coordinates M and total Hermite degree P.
    quadrature_order : Gauss-Hermite points q per coordinate (q^M active nodes); default P + 3.
        The engine refuses q < P + 1, the exact Gram floor.
    substeps : RK4 steps per grid interval.
    factor : 'endpoint' (endpoint-conditioned bridge KL, the paper's default), 'spectral'
        (trace-optimal path KL), 'hierarchy' (endpoint-first dyadic, N a power of two) or a
        ``GaussianPathFactor``.
    closure : 'residual' (RCLH) or 'active_only' (ablation: the drift closure ignores the residual
        variance; the reconstruction is unchanged).
    residual_quadrature_order : scalar residual rule for non-polynomial drifts.
    stage_observer : optional callable receiving every RK4 stage (engine stage object).

    Returns
    -------
    RCLHResult with times, mean, second_moment, variance, covariance, cross_moments,
    projection_gap tau_Lambda, residual_variance rho, Lawson coefficients and diagnostics.
    """

    started = time.perf_counter()
    x0 = float(x0)
    beta = float(beta)
    hurst = float(hurst)
    T = float(T)
    if not (0.0 < hurst < 1.0):
        raise ValueError("hurst must lie in (0, 1)")
    if not (np.isfinite(x0) and np.isfinite(beta) and np.isfinite(T) and T > 0):
        raise ValueError("x0 and beta must be finite and T positive")
    if closure not in ("residual", "active_only"):
        raise ValueError("closure must be 'residual' or 'active_only'")
    N = _integer(N, "N", 1)
    modes = _integer(modes, "modes", 1)
    degree = _integer(degree, "degree", 0)
    substeps = _integer(substeps, "substeps", 1)
    max_active_nodes = _integer(max_active_nodes, "max_active_nodes", 1)
    residual_quadrature_order = _integer(residual_quadrature_order, "residual_quadrature_order", 1)
    if stage_observer is not None and not callable(stage_observer):
        raise ValueError("stage_observer must be callable")
    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 1 or not np.isfinite(times).all() or times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, positive and strictly increasing")
        N = len(times)
    poly = {}
    for k, v in (coefficients or {}).items():
        exponent = _integer(k, "polynomial exponent", 0)
        if exponent in poly:
            raise ValueError("duplicate polynomial exponent after normalization")
        poly[exponent] = float(v)
    rate = float(linear_rate) + poly.pop(1, 0.0)
    constant = poly.get(0, 0.0)
    q = degree + 3 if quadrature_order is None else _integer(quadrature_order, "quadrature_order", 1)

    if transforms:
        if any(k not in (0,) for k in poly):
            raise ValueError("polynomial terms of degree >= 2 cannot be combined with transforms")
        drift = AffineSpectralDrift(intercept=constant, rate=rate, transforms=tuple(transforms))
        branch = "spectral_residual_quadrature"
        drift_label = label or "a*x + " + " + ".join(type(t).__name__ for t in transforms)
    else:
        if not poly:
            poly = {2: 0.0}
        drift = PolynomialDrift.from_mapping(rate, poly, label=label or "polynomial_drift")
        branch = "polynomial_closed_form"
        drift_label = label or (
            f"{rate:g}*x" + "".join(f" {'+' if c >= 0 else '-'} {abs(c):g}*x^{k}" for k, c in sorted(poly.items()))
        )

    tic = time.perf_counter()
    path_factor = build_factor(factor, hurst=hurst, T=T, N=N, modes=int(modes), times=times)
    N = int(len(path_factor.times))
    T = float(path_factor.times[-1])
    factor_seconds = time.perf_counter() - tic

    tic = time.perf_counter()
    if branch == "polynomial_closed_form":
        solver = PolynomialResidualLawsonSolver(
            path_factor,
            degree=int(degree),
            beta=beta,
            drift=drift,
            active_quadrature_order=q,
            drift_substeps=int(substeps),
            max_active_nodes=int(max_active_nodes),
        )
    else:
        solver = ResidualCoherentLawsonSolver(
            path_factor,
            degree=int(degree),
            beta=beta,
            drift=drift,
            active_quadrature_order=q,
            residual_quadrature_order=int(residual_quadrature_order),
            drift_substeps=int(substeps),
            max_active_nodes=int(max_active_nodes),
        )
    if closure == "active_only":
        original_project = solver.projection.project
        solver.projection.project = lambda c, a, rho: original_project(c, a, 0.0)
    setup_seconds = time.perf_counter() - tic

    basis_matrix = getattr(solver.projection, "basis_matrix", None)
    tracker = {"stages": 0, "min_nodal": math.inf, "max_energy_excess": -math.inf}

    def observe(stage: Any) -> None:
        tracker["stages"] += 1
        state = np.asarray(stage.coefficient_state, dtype=float)
        if basis_matrix is not None:
            tracker["min_nodal"] = min(tracker["min_nodal"], float(np.min(basis_matrix @ state)))
        excess = float(np.dot(state, stage.projected_rhs)) - rate * float(np.dot(state, state))
        tracker["max_energy_excess"] = max(tracker["max_energy_excess"], excess)
        if stage_observer is not None:
            stage_observer(stage)

    tic = time.perf_counter()
    result = solver.solve(x0, stage_observer=observe)
    solve_seconds = time.perf_counter() - tic

    tic = time.perf_counter()
    covariance = np.asarray(result.coherent_covariance_matrix, dtype=float)
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))))
    output_seconds = time.perf_counter() - tic

    nodes = int(solver.projection.active_nodes.shape[0])
    config = {
        "x0": x0, "beta": beta, "hurst": hurst, "linear_rate": rate,
        "coefficients": {str(k): v for k, v in sorted(poly.items())} if branch == "polynomial_closed_form" else {"0": constant},
        "transforms": [repr(t) for t in (transforms or ())],
        "T": T, "N": int(N), "modes": int(modes), "degree": int(degree), "quadrature_order": q,
        "substeps": int(substeps), "factor": path_factor.method, "closure": closure,
        "branch": branch, "residual_quadrature_order": int(residual_quadrature_order) if transforms else None,
        "drift_label": drift_label,
    }
    diagnostics = {
        "active_nodes": nodes,
        "basis_size": int(solver.basis.size),
        "gram_error": float(solver.projection.gram_error),
        "residual_trace_fraction": float(path_factor.residual_trace_fraction),
        "factor_reconstruction_error": float(path_factor.covariance_reconstruction_error),
        "rk4_stage_count": int(tracker["stages"]),
        "min_nodal_value": float(tracker["min_nodal"]) if basis_matrix is not None else None,
        "max_energy_rhs_excess": float(tracker["max_energy_excess"]),
        "min_covariance_eigenvalue": min_eig,
        "min_mean": float(np.min(result.coherent_means)),
        "min_variance": float(np.min(result.coherent_variances)),
    }
    timings = {
        "factor_seconds": factor_seconds,
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "output_seconds": output_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    engine = {
        "solver_class": type(solver).__name__,
        "implementation_label": getattr(solver, "implementation_label", ""),
        "vendor_dir": str(VENDOR_DIR),
        "sha256": dict(ENGINE_SHA256),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    return RCLHResult(
        times=np.asarray(result.times, dtype=float),
        mean=np.asarray(result.coherent_means, dtype=float),
        second_moment=np.asarray(result.coherent_second_moments, dtype=float),
        variance=np.asarray(result.coherent_variances, dtype=float),
        covariance=covariance,
        cross_moments=np.asarray(result.coherent_cross_moments, dtype=float),
        projection_gap=np.sqrt(np.maximum(np.asarray(result.omitted_energy, dtype=float), 0.0)),
        residual_variance=np.asarray(result.residual_variances, dtype=float),
        lawson_coefficients=np.asarray(result.lawson_coefficients, dtype=float),
        basis_indices=np.asarray(solver.basis.indices, dtype=int),
        config=config,
        diagnostics=diagnostics,
        timings=timings,
        engine=engine,
        raw=result,
    )


__all__ = [
    "RCLHResult",
    "rclh_solve",
    "build_factor",
    "trapezoid_weights",
    "rational",
    "cosine",
    "log_quadratic",
    "ENGINE_SHA256",
    "VENDOR_DIR",
]
