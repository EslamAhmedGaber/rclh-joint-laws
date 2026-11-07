"""Independent pointwise references for finite-feature WIS equations."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .fbm import GaussianPathFactor


ArrayFunction = Callable[[float, np.ndarray], np.ndarray]


def _validated_nodes(
    factor: GaussianPathFactor,
    nodes: np.ndarray,
) -> np.ndarray:
    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != factor.modes:
        raise ValueError(
            "nodes must have shape (n_nodes, factor.modes)"
        )
    return points


def _translated_grid_target_rk4_samples_from_paths(
    factor: GaussianPathFactor,
    paths: np.ndarray,
    initial_value: float,
    *,
    target_index: int,
    linear_drift_rate: float,
    beta: float,
    nonlinear_drift: ArrayFunction,
) -> np.ndarray:
    """Solve one grid-target Gjessing ODE from precomputed path values."""

    node_times = np.asarray(factor.times, dtype=float)
    loadings = np.asarray(factor.loadings, dtype=float)
    selected_target = int(target_index)
    if not 0 <= selected_target < node_times.size:
        raise ValueError(
            "target_index must identify a positive factor-grid time"
        )
    if paths.ndim != 2 or paths.shape[1] < selected_target + 1:
        raise ValueError("paths do not reach the selected factor-grid target")

    target_loading = loadings[selected_target]
    transformed = np.full(
        paths.shape[0],
        float(initial_value),
        dtype=float,
    )
    alpha = float(linear_drift_rate)
    diffusion = float(beta)
    diffusion_squared = diffusion * diffusion

    zero_loading = np.zeros(factor.modes, dtype=float)
    zero_path = np.zeros(paths.shape[0], dtype=float)
    left_time = 0.0
    left_loading = zero_loading
    left_path = zero_path

    def translation(
        time_value: float,
        loading: np.ndarray,
        path_value: np.ndarray,
    ) -> np.ndarray:
        deterministic_shift = (
            -alpha * time_value
            - 0.5
            * diffusion_squared
            * float(np.dot(loading, loading))
            + diffusion_squared
            * float(np.dot(loading, target_loading))
        )
        return np.exp(
            deterministic_shift - diffusion * path_value
        )

    def right_hand_side(
        time_value: float,
        state: np.ndarray,
        multiplier: np.ndarray,
    ) -> np.ndarray:
        physical = state / multiplier
        return multiplier * nonlinear_drift(
            time_value,
            physical,
        )

    for step in range(selected_target + 1):
        right_time = float(node_times[step])
        right_loading = loadings[step]
        right_path = paths[:, step]
        midpoint_time = 0.5 * (left_time + right_time)
        midpoint_loading = 0.5 * (
            left_loading + right_loading
        )
        midpoint_path = 0.5 * (left_path + right_path)
        step_size = right_time - left_time

        left_factor = translation(
            left_time,
            left_loading,
            left_path,
        )
        midpoint_factor = translation(
            midpoint_time,
            midpoint_loading,
            midpoint_path,
        )
        right_factor = translation(
            right_time,
            right_loading,
            right_path,
        )

        k1 = right_hand_side(
            left_time,
            transformed,
            left_factor,
        )
        k2 = right_hand_side(
            midpoint_time,
            transformed + 0.5 * step_size * k1,
            midpoint_factor,
        )
        k3 = right_hand_side(
            midpoint_time,
            transformed + 0.5 * step_size * k2,
            midpoint_factor,
        )
        k4 = right_hand_side(
            right_time,
            transformed + step_size * k3,
            right_factor,
        )
        transformed = transformed + (step_size / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        left_time = right_time
        left_loading = right_loading
        left_path = right_path

    target_factor = translation(
        float(node_times[selected_target]),
        target_loading,
        paths[:, selected_target],
    )
    return transformed / target_factor


def translated_grid_target_rk4_samples(
    factor: GaussianPathFactor,
    nodes: np.ndarray,
    initial_value: float,
    *,
    target_index: int,
    linear_drift_rate: float,
    beta: float,
    nonlinear_drift: ArrayFunction,
) -> np.ndarray:
    """Evaluate the WIS solution at one factor-grid target time.

    ``target_index`` selects the Gjessing target ``h_{t_j}``.  The RK4 solve
    runs only from zero through ``t_j`` using the same piecewise-linear
    finite Gaussian feature as :func:`translated_terminal_rk4_samples`.
    This target-specific construction is required for fractional Brownian
    motion: intermediate states of a solve targeted at the final time are
    generally not samples of ``X(t_j)`` when ``H != 1/2``.
    """

    points = _validated_nodes(factor, nodes)
    selected_target = int(target_index)
    if not 0 <= selected_target < factor.times.size:
        raise ValueError(
            "target_index must identify a positive factor-grid time"
        )
    paths = (
        points
        @ np.asarray(
            factor.loadings[: selected_target + 1],
            dtype=float,
        ).T
    )
    return _translated_grid_target_rk4_samples_from_paths(
        factor,
        paths,
        initial_value,
        target_index=selected_target,
        linear_drift_rate=linear_drift_rate,
        beta=beta,
        nonlinear_drift=nonlinear_drift,
    )


def translated_grid_rk4_sample_paths(
    factor: GaussianPathFactor,
    nodes: np.ndarray,
    initial_value: float,
    *,
    linear_drift_rate: float,
    beta: float,
    nonlinear_drift: ArrayFunction,
) -> np.ndarray:
    """Return target-correct samples at ``0,t_1,...,t_N``.

    The returned array has shape ``(n_nodes, N + 1)``.  Column zero is the
    deterministic initial value.  Column ``j + 1`` is computed with
    ``h_{t_j}`` as its own Gjessing target, so the columns are genuine
    same-factor path samples.  Reusing intermediate states from the single
    final-time-target ODE would be cheaper but is not valid for general fBM.

    The direct all-target implementation costs ``O(n_nodes * N**2)`` work
    and ``O(n_nodes * N)`` output memory.  Large reference experiments should
    therefore batch Gaussian nodes or stage individual target times.
    """

    points = _validated_nodes(factor, nodes)
    paths = points @ np.asarray(factor.loadings, dtype=float).T
    samples = np.empty(
        (points.shape[0], factor.times.size + 1),
        dtype=float,
    )
    samples[:, 0] = float(initial_value)
    for target_index in range(factor.times.size):
        samples[:, target_index + 1] = (
            _translated_grid_target_rk4_samples_from_paths(
                factor,
                paths,
                initial_value,
                target_index=target_index,
                linear_drift_rate=linear_drift_rate,
                beta=beta,
                nonlinear_drift=nonlinear_drift,
            )
        )
    return samples


def translated_terminal_rk4_samples(
    factor: GaussianPathFactor,
    nodes: np.ndarray,
    initial_value: float,
    *,
    linear_drift_rate: float,
    beta: float,
    nonlinear_drift: ArrayFunction,
) -> np.ndarray:
    """Evaluate the terminal-target Gjessing transform with RK4.

    The finite Gaussian feature is linearly interpolated between the factor
    grid points.  This routine has no chaos truncation or intrusive
    projection, so on a fixed factor it is an independent reference for a
    coefficient-space Lawson solver.
    """

    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != factor.modes:
        raise ValueError(
            "nodes must have shape (n_nodes, factor.modes)"
        )

    node_times = np.asarray(factor.times, dtype=float)
    loadings = np.asarray(factor.loadings, dtype=float)
    terminal_loading = loadings[-1]
    paths = points @ loadings.T
    transformed = np.full(
        points.shape[0],
        float(initial_value),
        dtype=float,
    )
    alpha = float(linear_drift_rate)
    diffusion = float(beta)
    diffusion_squared = diffusion * diffusion

    zero_loading = np.zeros(factor.modes, dtype=float)
    zero_path = np.zeros(points.shape[0], dtype=float)
    left_time = 0.0
    left_loading = zero_loading
    left_path = zero_path

    def translation(
        time_value: float,
        loading: np.ndarray,
        path_value: np.ndarray,
    ) -> np.ndarray:
        deterministic_shift = (
            -alpha * time_value
            - 0.5
            * diffusion_squared
            * float(np.dot(loading, loading))
            + diffusion_squared
            * float(np.dot(loading, terminal_loading))
        )
        return np.exp(
            deterministic_shift - diffusion * path_value
        )

    def right_hand_side(
        time_value: float,
        state: np.ndarray,
        multiplier: np.ndarray,
    ) -> np.ndarray:
        physical = state / multiplier
        return multiplier * nonlinear_drift(
            time_value,
            physical,
        )

    for step, right_time_raw in enumerate(node_times):
        right_time = float(right_time_raw)
        right_loading = loadings[step]
        right_path = paths[:, step]
        midpoint_time = 0.5 * (left_time + right_time)
        midpoint_loading = 0.5 * (
            left_loading + right_loading
        )
        midpoint_path = 0.5 * (left_path + right_path)
        step_size = right_time - left_time

        left_factor = translation(
            left_time,
            left_loading,
            left_path,
        )
        midpoint_factor = translation(
            midpoint_time,
            midpoint_loading,
            midpoint_path,
        )
        right_factor = translation(
            right_time,
            right_loading,
            right_path,
        )

        k1 = right_hand_side(
            left_time,
            transformed,
            left_factor,
        )
        k2 = right_hand_side(
            midpoint_time,
            transformed + 0.5 * step_size * k1,
            midpoint_factor,
        )
        k3 = right_hand_side(
            midpoint_time,
            transformed + 0.5 * step_size * k2,
            midpoint_factor,
        )
        k4 = right_hand_side(
            right_time,
            transformed + step_size * k3,
            right_factor,
        )
        transformed = transformed + (step_size / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        left_time = right_time
        left_loading = right_loading
        left_path = right_path

    terminal_factor = translation(
        float(node_times[-1]),
        terminal_loading,
        paths[:, -1],
    )
    return transformed / terminal_factor
