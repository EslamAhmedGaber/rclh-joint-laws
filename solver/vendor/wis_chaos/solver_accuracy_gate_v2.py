"""Reference-blind convergence accounting for Solver Accuracy V2.

The indicators in this module are conditional numerical saturation checks,
not a priori theorems.  They are evaluated before an external reference is
opened and use coherent full-output moment curves only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from typing import Mapping

import numpy as np

from .solver_accuracy_v2 import (
    ProjectorControlsV2,
    SolverAccuracyRunV2,
)


QUANTITIES = ("mean", "second_moment", "variance")
REFINEMENT_AXES = ("factor_grid", "time", "support_chaos", "projector")
REPORTING_NODES = np.linspace(0.0, 1.0, 33)
SAFETY_FACTOR = 1.25
REFINEMENT_ROUNDOFF_MULTIPLIER = 512
METHOD_ROUNDOFF_MULTIPLIER = 1024
CONTRACTION_LIMITS = {
    "factor_grid": 0.995,
    "time": 0.5,
    "support_chaos": 0.8,
    "projector": 0.5,
}


@dataclass(frozen=True, slots=True)
class AccuracyConfigurationV2:
    identifier: str
    grid_steps: int
    support_profile: str
    projector_profile: str
    drift_substeps: int
    triad_multiplier: int

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuantityRefinementV2:
    first_shift: float
    second_shift: float
    observed_contraction: float | None
    effective_contraction: float | None
    observed_order: float | None
    fine_tail_indicator: float | None
    resolved: bool
    classification: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AxisRefinementV2:
    axis: str
    contraction_floor: float
    contraction_limit: float
    safety_factor: float
    quantities: Mapping[str, QuantityRefinementV2]
    last_observed_shift: float
    fine_tail_indicator: float | None
    resolved: bool
    classification: str

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["quantities"] = {
            key: value.to_jsonable() for key, value in self.quantities.items()
        }
        return payload


def projector_profiles_v2() -> dict[str, ProjectorControlsV2]:
    """Three prospectively ordered projector resolutions."""

    return {
        "production": ProjectorControlsV2(
            core_order=7,
            cutoff=22.0,
            absolute_tolerance=2.0e-5,
            relative_tolerance=2.0e-5,
            maximum_panels=256,
        ),
        "audit_1": ProjectorControlsV2(
            core_order=9,
            cutoff=28.0,
            absolute_tolerance=2.0e-6,
            relative_tolerance=2.0e-6,
            maximum_panels=1024,
        ),
        "audit_2": ProjectorControlsV2(
            core_order=11,
            cutoff=28.0,
            absolute_tolerance=2.0e-7,
            relative_tolerance=2.0e-7,
            maximum_panels=2048,
        ),
    }


def producer_controls_v2(
    final_grid_steps: int,
    *,
    triad_multiplier: int,
    method_allowance_limit: float,
) -> dict[str, object]:
    """Return the exact JSON controls expected in a prospective manifest."""

    final = int(final_grid_steps)
    multiplier = int(triad_multiplier)
    limit = float(method_allowance_limit)
    if multiplier < 2 or multiplier % 2:
        raise ValueError(
            "official triad_multiplier must be a positive even integer at least two"
        )
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("method_allowance_limit must be finite and positive")
    projectors = projector_profiles_v2()
    configurations = accuracy_configurations_v2(
        final,
        triad_multiplier=multiplier,
    )
    return {
        "design": "nine_run_common_finest_four_axis_v1",
        "degree": 6,
        "core_modes": 3,
        "final_grid_steps": final,
        "factor_grid_steps": [final // 4, final // 2, final],
        "factor_method": "endpoint_conditioned_bridge_kl_full_rank_each_grid_v1",
        "coarse_curve_alignment": "linear_to_33_fixed_nodes_v1",
        "support_ladder": {
            "profile": "operator_triad_core6",
            "triad_multipliers": [0, multiplier // 2, multiplier],
            "prefix_semantics": "same_ranked_operator_pair_prefix_v1",
        },
        "triad_multiplier": multiplier,
        "time_substeps": [1, 2, 4],
        "projector_profiles": {
            name: controls.to_jsonable() for name, controls in projectors.items()
        },
        "axis_contraction_floors": {
            "factor_grid": "2^(-H)",
            "time": 0.0625,
            "support_chaos": 0.25,
            "projector": 0.1,
        },
        "axis_contraction_limits": dict(CONTRACTION_LIMITS),
        "tail_safety_factor": SAFETY_FACTOR,
        "refinement_roundoff_multiplier": REFINEMENT_ROUNDOFF_MULTIPLIER,
        "method_roundoff_multiplier": METHOD_ROUNDOFF_MULTIPLIER,
        "target_refinement_limit": 0.001,
        "method_allowance_limit": limit,
        "configuration_order": [item.to_jsonable() for item in configurations],
        "official_equivalence_prerequisite": {
            "classification": "PASS_FUSED_EIGENBASIS_EQUIVALENCE_V1",
            "verdict_path": (
                "06_results/fused_eigenbasis_equivalence_v1_verdict.json"
            ),
            "verdict_sha256": (
                "cfc79af9bdf947e5e686a2c495ed561cbe725bea9ca1f39c489c25547a8ddc15"
            ),
            "implementation_fingerprint": (
                "656e9a19bec2b04a02681c233a2639db3e2e1aa1039624ee5acb2493158130ab"
            ),
        },
        "checkpoint_resume": "same_attempt_exact_controls_committed_stages_v1",
        "official_rerun_policy": "no_new_attempt_after_terminal_package_v1",
    }


def accuracy_configurations_v2(
    final_grid_steps: int,
    *,
    triad_multiplier: int = 2,
) -> tuple[AccuracyConfigurationV2, ...]:
    """Return the nine-run common-finest four-axis design.

    The factor/grid axis is intentionally named as a coupled axis: changing
    the discrete fBM grid also changes its full endpoint-conditioned bridge
    factor.  Calling this a pure modal refinement would overstate what the
    computation isolates.
    """

    fine = int(final_grid_steps)
    if fine < 32 or fine % 4:
        raise ValueError("final_grid_steps must be a multiple of four and at least 32")
    multiplier = int(triad_multiplier)
    if multiplier < 1:
        raise ValueError("triad_multiplier must be positive")
    central = AccuracyConfigurationV2(
        "central",
        fine,
        "operator_triad_core6",
        "audit_2",
        4,
        multiplier,
    )
    middle_triad_multiplier = max(0, multiplier // 2)
    return (
        AccuracyConfigurationV2(
            "factor_grid_coarse",
            fine // 4,
            "operator_triad_core6",
            "audit_2",
            4,
            multiplier,
        ),
        AccuracyConfigurationV2(
            "factor_grid_middle",
            fine // 2,
            "operator_triad_core6",
            "audit_2",
            4,
            multiplier,
        ),
        AccuracyConfigurationV2(
            "support_chaos_coarse",
            fine,
            "operator_triad_core6",
            "audit_2",
            4,
            0,
        ),
        AccuracyConfigurationV2(
            "support_chaos_middle",
            fine,
            "operator_triad_core6",
            "audit_2",
            4,
            middle_triad_multiplier,
        ),
        AccuracyConfigurationV2(
            "time_coarse",
            fine,
            "operator_triad_core6",
            "audit_2",
            1,
            multiplier,
        ),
        AccuracyConfigurationV2(
            "time_middle",
            fine,
            "operator_triad_core6",
            "audit_2",
            2,
            multiplier,
        ),
        AccuracyConfigurationV2(
            "projector_coarse",
            fine,
            "operator_triad_core6",
            "production",
            4,
            multiplier,
        ),
        AccuracyConfigurationV2(
            "projector_middle",
            fine,
            "operator_triad_core6",
            "audit_1",
            4,
            multiplier,
        ),
        central,
    )


def axis_configuration_ids_v2() -> dict[str, tuple[str, str, str]]:
    return {
        "factor_grid": (
            "factor_grid_coarse",
            "factor_grid_middle",
            "central",
        ),
        "time": ("time_coarse", "time_middle", "central"),
        "support_chaos": (
            "support_chaos_coarse",
            "support_chaos_middle",
            "central",
        ),
        "projector": (
            "projector_coarse",
            "projector_middle",
            "central",
        ),
    }


def _normalized_curve_shift(
    times: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    scale_curve: np.ndarray,
) -> float:
    nodes = np.asarray(times, dtype=float)
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale_values = np.asarray(scale_curve, dtype=float)
    if (
        nodes.ndim != 1
        or first.shape != nodes.shape
        or second.shape != nodes.shape
        or scale_values.shape != nodes.shape
        or nodes.size < 2
        or np.any(np.diff(nodes) <= 0.0)
        or not all(np.all(np.isfinite(item)) for item in (first, second, scale_values))
    ):
        raise ValueError("refinement curves must be finite and aligned")
    scale = max(1.0, float(np.max(np.abs(scale_values))))
    difference = first - second
    infinity = float(np.max(np.abs(difference)) / scale)
    duration = float(nodes[-1] - nodes[0])
    time_l2 = float(
        np.sqrt(max(float(np.trapezoid(difference * difference, nodes)) / duration, 0.0))
        / scale
    )
    return max(infinity, time_l2)


def _quantity_refinement(
    first_shift: float,
    second_shift: float,
    *,
    contraction_floor: float,
    contraction_limit: float,
    safety_factor: float,
    roundoff_floor: float,
) -> QuantityRefinementV2:
    first = float(first_shift)
    second = float(second_shift)
    floor = float(contraction_floor)
    limit = float(contraction_limit)
    safety = float(safety_factor)
    if (
        not np.isfinite(first)
        or not np.isfinite(second)
        or first < 0.0
        or second < 0.0
    ):
        return QuantityRefinementV2(
            first,
            second,
            None,
            None,
            None,
            None,
            False,
            "STOP_NONFINITE_REFINEMENT",
        )
    if first <= roundoff_floor and second <= roundoff_floor:
        effective = floor
        tail = safety * roundoff_floor / (1.0 - effective)
        return QuantityRefinementV2(
            first,
            second,
            0.0,
            effective,
            None,
            float(tail),
            True,
            "RESOLVED_AT_ROUNDOFF",
        )
    if first <= roundoff_floor:
        return QuantityRefinementV2(
            first,
            second,
            None,
            None,
            None,
            None,
            False,
            "INCONCLUSIVE_ZERO_FIRST_SHIFT",
        )
    contraction = second / first
    effective = max(contraction, floor)
    order = None if second <= roundoff_floor else float(log2(first / second))
    if not np.isfinite(effective) or effective >= limit:
        return QuantityRefinementV2(
            first,
            second,
            float(contraction),
            float(effective),
            order,
            None,
            False,
            "INCONCLUSIVE_NO_SATURATION",
        )
    tail = safety * effective * max(second, roundoff_floor) / (1.0 - effective)
    return QuantityRefinementV2(
        first,
        second,
        float(contraction),
        float(effective),
        order,
        float(tail),
        True,
        "RESOLVED_CONDITIONAL_GEOMETRIC_TAIL",
    )


def factor_grid_contraction_floor(hurst: float) -> float:
    """Conservative fBM strong-roughness floor used in discovery.

    This floor is intentionally ``2**(-H)`` rather than a fitted rate.  It is
    a conservative design guard, not a theorem about the nonlinear solver's
    weak moment error.  The official value must be frozen before execution.
    """

    exponent = float(hurst)
    if not 0.0 < exponent < 1.0:
        raise ValueError("standard fBM requires 0 < H < 1")
    return float(2.0 ** (-exponent))


def analyze_axis_v2(
    axis: str,
    coarse: SolverAccuracyRunV2,
    middle: SolverAccuracyRunV2,
    fine: SolverAccuracyRunV2,
    *,
    reporting_nodes: np.ndarray = REPORTING_NODES,
    safety_factor: float = SAFETY_FACTOR,
) -> AxisRefinementV2:
    selected = str(axis)
    if selected not in REFINEMENT_AXES:
        raise ValueError("unknown Solver Accuracy V2 axis")
    if not (
        coarse.problem_code == middle.problem_code == fine.problem_code
        and coarse.hurst == middle.hurst == fine.hurst
    ):
        raise ValueError("refinement runs belong to different cells")
    nodes = np.asarray(reporting_nodes, dtype=float)
    curves = tuple(run.moment_curves_at(nodes) for run in (coarse, middle, fine))
    floor = (
        factor_grid_contraction_floor(fine.hurst)
        if selected == "factor_grid"
        else {"time": 0.0625, "support_chaos": 0.25, "projector": 0.1}[selected]
    )
    limit = CONTRACTION_LIMITS[selected]
    roundoff = REFINEMENT_ROUNDOFF_MULTIPLIER * np.finfo(float).eps
    quantities: dict[str, QuantityRefinementV2] = {}
    for quantity in QUANTITIES:
        first = _normalized_curve_shift(
            nodes,
            curves[0][quantity],
            curves[1][quantity],
            curves[2][quantity],
        )
        second = _normalized_curve_shift(
            nodes,
            curves[1][quantity],
            curves[2][quantity],
            curves[2][quantity],
        )
        quantities[quantity] = _quantity_refinement(
            first,
            second,
            contraction_floor=floor,
            contraction_limit=limit,
            safety_factor=safety_factor,
            roundoff_floor=roundoff,
        )
    unresolved = [name for name, item in quantities.items() if not item.resolved]
    last_shift = max(item.second_shift for item in quantities.values())
    if unresolved:
        tail = None
        classification = "INCONCLUSIVE_AXIS_REFINEMENT"
    else:
        tail = max(
            float(item.fine_tail_indicator)
            for item in quantities.values()
            if item.fine_tail_indicator is not None
        )
        classification = "PASS_CONDITIONAL_AXIS_SATURATION"
    return AxisRefinementV2(
        selected,
        float(floor),
        float(limit),
        float(safety_factor),
        quantities,
        float(last_shift),
        None if tail is None else float(tail),
        not unresolved,
        classification,
    )


def analyze_refinements_v2(
    runs: Mapping[str, SolverAccuracyRunV2],
) -> dict[str, AxisRefinementV2]:
    required = {item for ids in axis_configuration_ids_v2().values() for item in ids}
    if set(runs) != required:
        raise ValueError("accuracy run mapping does not match the exact nine-run design")
    output: dict[str, AxisRefinementV2] = {}
    for axis, identifiers in axis_configuration_ids_v2().items():
        output[axis] = analyze_axis_v2(
            axis,
            runs[identifiers[0]],
            runs[identifiers[1]],
            runs[identifiers[2]],
        )
    return output


def combined_method_allowance_v2(
    axes: Mapping[str, AxisRefinementV2],
) -> float | None:
    if set(axes) != set(REFINEMENT_AXES):
        raise ValueError("refinement-axis schema is not exact")
    if any(not item.resolved or item.fine_tail_indicator is None for item in axes.values()):
        return None
    roundoff = METHOD_ROUNDOFF_MULTIPLIER * np.finfo(float).eps
    return float(sum(float(item.fine_tail_indicator) for item in axes.values()) + roundoff)


__all__ = [
    "AccuracyConfigurationV2",
    "AxisRefinementV2",
    "CONTRACTION_LIMITS",
    "QUANTITIES",
    "REFINEMENT_AXES",
    "REFINEMENT_ROUNDOFF_MULTIPLIER",
    "REPORTING_NODES",
    "SAFETY_FACTOR",
    "METHOD_ROUNDOFF_MULTIPLIER",
    "QuantityRefinementV2",
    "accuracy_configurations_v2",
    "analyze_axis_v2",
    "analyze_refinements_v2",
    "axis_configuration_ids_v2",
    "combined_method_allowance_v2",
    "factor_grid_contraction_floor",
    "producer_controls_v2",
    "projector_profiles_v2",
]
