"""Deterministic normalized-Hermite prototype for finite-mode WIS equations."""

from .adaptivity import (
    HistoryRotation,
    NestedOperatorIndicatorScores,
    TransformedIndicatorScores,
    first_chaos_history,
    greedy_downward_closed_index_set,
    history_sensitivity_rotation,
    nested_lawson_operator_indicators,
    transformed_candidate_indicators,
)
from .basis import (
    HermiteBasis,
    HybridCoreGaussHermiteSobol,
    MatrixFreeSobolHermiteProjection,
    SobolHermiteLeastSquares,
    StructuredQuadraticTailSobolProjection,
    StructuredRationalLaguerreProjection,
    TensorGaussHermite,
    core_enriched_index_set,
    gaussian_moments,
    quadratic_tail_core_enriched_index_set,
    weighted_total_degree_index_set,
)
from .coherent import (
    CoherentReconstruction,
    coherent_reconstruction,
    recover_lawson_coefficients,
    shifted_polynomial_coefficients,
)
from .contour_shifted_lognormal import (
    ContourShiftedLognormalDiagnostic,
    ContourShiftedLognormalRule,
)
from .centered_filon_projection import (
    CenteredProjectionDiagnostic,
    CoreModulatedCenteredFilonProjection,
)
from .defect import (
    DeterministicDefectEnvelope,
    JointDefectIndicator,
    JointPhysicalLawsonDefect,
    deterministic_piecewise_affine_defect_envelope,
    joint_defect_indicator,
    physical_lawson_defect_samples,
    recover_lawson_trajectory,
    stable_affine_rational_lawson_field,
)
from .fbm import (
    GaussianPathFactor,
    canonical_feature_interpolation_error,
    endpoint_conditioned_factor,
    fbm_covariance,
    multi_anchor_conditioned_factor,
    path_kl_factor,
    pivoted_cholesky_factor,
    rotate_factor,
)
from .operator_support import (
    OperatorGeometryTailPair,
    operator_geometry_pair_fingerprint,
    operator_geometry_tail_pair_ranking,
    operator_geometry_triad_index_set,
)
from .modulated_rational import CoreModulatedRationalLaguerreProjection
from .spectral_projection import (
    AffineSpectralDrift,
    CoreModulatedSpectralProjection,
    CosineTransform,
    LogQuadraticCosineTransform,
    RationalSineTransform,
    SpectralQuadratureRule,
    combine_spectral_rules,
)
from .spectral_solver import SpectralDeterministicWISChaosSolver
from .solver import (
    AffineRationalDrift,
    DeterministicWISChaosSolver,
    SolverResult,
)
from .translated import (
    DeterministicTranslatedEulerChaosSolver,
    TranslatedEulerResult,
)
from .target_aligned_reference import (
    target_aligned_hybrid_moment_curves,
    target_loading_rotation,
)
from .reference import translated_terminal_rk4_samples
from .residual_coherent_lawson import (
    ResidualClosureDiagnostic,
    ResidualCoherentLawsonProjection,
    ResidualCoherentLawsonResult,
    ResidualCoherentLawsonSolver,
    ResidualCoherentLawsonStage,
    pulled_back_affine_spectral_drift,
    residual_coherent_cross_moment,
)

__all__ = [
    "AffineRationalDrift",
    "AffineSpectralDrift",
    "CoherentReconstruction",
    "ContourShiftedLognormalDiagnostic",
    "ContourShiftedLognormalRule",
    "CenteredProjectionDiagnostic",
    "DeterministicWISChaosSolver",
    "DeterministicDefectEnvelope",
    "DeterministicTranslatedEulerChaosSolver",
    "CoreModulatedRationalLaguerreProjection",
    "CoreModulatedCenteredFilonProjection",
    "CoreModulatedSpectralProjection",
    "CosineTransform",
    "GaussianPathFactor",
    "HermiteBasis",
    "HybridCoreGaussHermiteSobol",
    "MatrixFreeSobolHermiteProjection",
    "LogQuadraticCosineTransform",
    "RationalSineTransform",
    "ResidualClosureDiagnostic",
    "ResidualCoherentLawsonProjection",
    "ResidualCoherentLawsonResult",
    "ResidualCoherentLawsonSolver",
    "ResidualCoherentLawsonStage",
    "SpectralQuadratureRule",
    "SpectralDeterministicWISChaosSolver",
    "StructuredQuadraticTailSobolProjection",
    "StructuredRationalLaguerreProjection",
    "HistoryRotation",
    "JointDefectIndicator",
    "JointPhysicalLawsonDefect",
    "NestedOperatorIndicatorScores",
    "OperatorGeometryTailPair",
    "SolverResult",
    "SobolHermiteLeastSquares",
    "TensorGaussHermite",
    "TransformedIndicatorScores",
    "TranslatedEulerResult",
    "translated_terminal_rk4_samples",
    "target_aligned_hybrid_moment_curves",
    "target_loading_rotation",
    "canonical_feature_interpolation_error",
    "combine_spectral_rules",
    "coherent_reconstruction",
    "core_enriched_index_set",
    "endpoint_conditioned_factor",
    "deterministic_piecewise_affine_defect_envelope",
    "fbm_covariance",
    "first_chaos_history",
    "gaussian_moments",
    "greedy_downward_closed_index_set",
    "history_sensitivity_rotation",
    "multi_anchor_conditioned_factor",
    "nested_lawson_operator_indicators",
    "joint_defect_indicator",
    "operator_geometry_tail_pair_ranking",
    "operator_geometry_pair_fingerprint",
    "operator_geometry_triad_index_set",
    "path_kl_factor",
    "pivoted_cholesky_factor",
    "physical_lawson_defect_samples",
    "pulled_back_affine_spectral_drift",
    "quadratic_tail_core_enriched_index_set",
    "recover_lawson_coefficients",
    "recover_lawson_trajectory",
    "residual_coherent_cross_moment",
    "rotate_factor",
    "shifted_polynomial_coefficients",
    "stable_affine_rational_lawson_field",
    "transformed_candidate_indicators",
    "weighted_total_degree_index_set",
]
