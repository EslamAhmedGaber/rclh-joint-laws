"""Stable centered spectral amplitudes retained in conditional eigenbases.

The validated centered-Filon projector assembles a physical Hermite
coefficient vector at every spectral node.  The objects in this module keep
each conditional core group's linear and quadratic tail moments in the
eigenbasis in which its quadratic form is diagonal.  Linear and quadratic
rotations can then be moved through the Filon sum and performed once per
panel, rather than once per spectral node.

Only compact node data are cached:

``core_D``
    The already centered core part of the logarithmic amplitude.
``delta_gram``
    The change in the weighted linear-core Gram matrix.
``delta_mass``
    The change in conditional characteristic mass.

The potentially large tail-by-tail quadratic moment is formed only while a
panel contraction is being evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centered_filon_projection import CoreModulatedCenteredFilonProjection


def _log1p_minus_identity(values: np.ndarray) -> np.ndarray:
    """Return ``log1p(z) - z`` without losing its quadratic term."""

    z = np.asarray(values, dtype=complex)
    output = np.empty_like(z)
    small = np.abs(z) < 1.0e-4
    if np.any(~small):
        output[~small] = np.log1p(z[~small]) - z[~small]
    if np.any(small):
        selected = z[small]
        # -z**2/2 + z**3/3 - ...; twelve terms make the truncation far
        # smaller than double precision at the branch threshold.
        term = selected * selected
        total = -0.5 * term
        for power in range(3, 13):
            term = term * selected
            total = total + ((-1.0) ** (power + 1) / power) * term
        output[small] = total
    return output


def _expm1_minus_identity(values: np.ndarray) -> np.ndarray:
    """Return ``expm1(z) - z`` without cancellation near the origin."""

    z = np.asarray(values, dtype=complex)
    output = np.empty_like(z)
    small = np.abs(z) < 1.0e-4
    if np.any(~small):
        output[~small] = np.expm1(z[~small]) - z[~small]
    if np.any(small):
        selected = z[small]
        term = 0.5 * selected * selected
        total = term.copy()
        for power in range(3, 13):
            term = term * selected / power
            total = total + term
        output[small] = total
    return output


@dataclass(frozen=True, slots=True)
class CompactCenteredGroupBatch:
    """Compact stable node data for one conditional core group."""

    core_d: np.ndarray
    delta_gram: np.ndarray
    delta_mass: np.ndarray


@dataclass(frozen=True, slots=True)
class NativeGroupBatch:
    """A group contribution before rotation to physical tail coordinates."""

    core: np.ndarray
    linear_eigen: np.ndarray
    second_eigen: np.ndarray


@dataclass(frozen=True, slots=True)
class NativeGroupReference:
    """The groupwise ``U_g`` and ``R_g`` centered reference blocks."""

    unit: NativeGroupBatch
    linear: NativeGroupBatch


@dataclass(frozen=True, slots=True)
class _PreparedCenteredGroup:
    eta_c: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rotated_linear_map: np.ndarray
    core_matrix: np.ndarray
    linear_core_matrix: np.ndarray
    core_weights: np.ndarray
    centered_core_values: np.ndarray
    weighted_linear_gram: np.ndarray
    group_mass: float
    unit_core: np.ndarray
    reference_core: np.ndarray
    reference_linear_eigen: np.ndarray
    reference_second_eigen: np.ndarray


class StableCenteredEigenAmplitude:
    """Prepare and evaluate a centered transform amplitude group by group.

    Parameters
    ----------
    projection:
        A validated scalar centered-Filon projection.  It is used only for
        its already checked support and core-quadrature construction.
    state:
        Real retained Hermite coefficient vector at the current nonlinear
        right-hand-side evaluation.
    width:
        Positive transform width ``a``.
    """

    def __init__(
        self,
        projection: CoreModulatedCenteredFilonProjection,
        state: np.ndarray,
        width: float,
    ) -> None:
        if not isinstance(projection, CoreModulatedCenteredFilonProjection):
            raise TypeError(
                "projection must be CoreModulatedCenteredFilonProjection"
            )
        raw_state = np.asarray(state)
        if np.iscomplexobj(raw_state):
            raise TypeError("state must be real-valued")
        coefficients = np.asarray(raw_state, dtype=float)
        if coefficients.shape != (projection.basis.size,):
            raise ValueError(
                "state shape must equal the retained Hermite basis size"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("state must be finite")
        scale = float(width)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("width must be finite and positive")

        base = projection._base
        structure = base._structure
        (
            core_values,
            linear_values,
            symmetric_zero,
            symmetric_one,
        ) = base._quadratic_matrices(coefficients)
        core_polynomial = base._core_matrix @ core_values
        mean = float(coefficients[0])
        centered = coefficients.copy()
        centered[0] = 0.0
        deviation = float(np.linalg.norm(centered))
        gamma = deviation / scale
        tail_dimension = symmetric_zero.shape[0]
        linear_core_size = structure._linear_core_basis.size

        groups: list[_PreparedCenteredGroup] = []
        for modulation_value, group_indices in zip(
            base._modulation_values,
            base._core_groups,
        ):
            eta_c = float(modulation_value)
            symmetric = symmetric_zero + eta_c * symmetric_one
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            rotated_linear_map = linear_values.T @ eigenvectors
            core_matrix = base._core_matrix[group_indices]
            linear_core_matrix = base._linear_core_matrix[group_indices]
            core_weights = base._core_weights[group_indices]
            centered_core_values = (
                core_polynomial[group_indices] - mean
            )
            weighted_linear_gram = (
                linear_core_matrix.T
                @ (core_weights[:, None] * linear_core_matrix)
            )
            group_mass = float(np.sum(core_weights))
            unit_core = core_matrix.T @ core_weights
            reference_core = (
                core_matrix.T
                @ (
                    core_weights
                    * (
                        gamma
                        + 1.0j * centered_core_values / scale
                    )
                )
            )
            reference_linear_eigen = (
                1.0j
                / scale
                * rotated_linear_map.T
                @ weighted_linear_gram
            )
            reference_second_eigen = np.diag(
                2.0j * group_mass * eigenvalues / scale
            )
            groups.append(
                _PreparedCenteredGroup(
                    eta_c=eta_c,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    rotated_linear_map=rotated_linear_map,
                    core_matrix=core_matrix,
                    linear_core_matrix=linear_core_matrix,
                    core_weights=core_weights,
                    centered_core_values=centered_core_values,
                    weighted_linear_gram=weighted_linear_gram,
                    group_mass=group_mass,
                    unit_core=unit_core,
                    reference_core=reference_core,
                    reference_linear_eigen=reference_linear_eigen,
                    reference_second_eigen=reference_second_eigen,
                )
            )

        self._projection = projection
        self._structure = structure
        self._state = coefficients
        self._width = scale
        self._mean = mean
        self._deviation = deviation
        self._gamma = gamma
        self._groups = tuple(groups)
        self._tail_dimension = tail_dimension
        self._linear_core_size = linear_core_size
        self._core_size = structure._core_basis.size

    @property
    def basis_size(self) -> int:
        return self._projection.basis.size

    @property
    def group_count(self) -> int:
        return len(self._groups)

    @property
    def tail_dimension(self) -> int:
        return self._tail_dimension

    @property
    def linear_core_size(self) -> int:
        return self._linear_core_size

    @property
    def core_size(self) -> int:
        return self._core_size

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def width(self) -> float:
        return self._width

    @property
    def groups(self) -> tuple[_PreparedCenteredGroup, ...]:
        """Prepared groups, exposed read-only for the fused panel engine."""

        return self._groups

    def for_width(self, width: float) -> StableCenteredEigenAmplitude:
        """Reuse the state-dependent eigensystems for another transform width."""

        scale = float(width)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("width must be finite and positive")
        gamma = self._deviation / scale
        groups = []
        for group in self._groups:
            reference_core = (
                group.core_matrix.T
                @ (
                    group.core_weights
                    * (
                        gamma
                        + 1.0j
                        * group.centered_core_values
                        / scale
                    )
                )
            )
            reference_linear_eigen = (
                1.0j
                / scale
                * group.rotated_linear_map.T
                @ group.weighted_linear_gram
            )
            reference_second_eigen = np.diag(
                2.0j
                * group.group_mass
                * group.eigenvalues
                / scale
            )
            groups.append(
                _PreparedCenteredGroup(
                    eta_c=group.eta_c,
                    eigenvalues=group.eigenvalues,
                    eigenvectors=group.eigenvectors,
                    rotated_linear_map=group.rotated_linear_map,
                    core_matrix=group.core_matrix,
                    linear_core_matrix=group.linear_core_matrix,
                    core_weights=group.core_weights,
                    centered_core_values=group.centered_core_values,
                    weighted_linear_gram=group.weighted_linear_gram,
                    group_mass=group.group_mass,
                    unit_core=group.unit_core,
                    reference_core=reference_core,
                    reference_linear_eigen=reference_linear_eigen,
                    reference_second_eigen=reference_second_eigen,
                )
            )
        reused = object.__new__(StableCenteredEigenAmplitude)
        reused._projection = self._projection
        reused._structure = self._structure
        reused._state = self._state
        reused._width = scale
        reused._mean = self._mean
        reused._deviation = self._deviation
        reused._gamma = gamma
        reused._groups = tuple(groups)
        reused._tail_dimension = self._tail_dimension
        reused._linear_core_size = self._linear_core_size
        reused._core_size = self._core_size
        return reused

    def reference_blocks(self) -> tuple[NativeGroupReference, ...]:
        """Return the exact groupwise ``U_g`` and ``R_g`` decomposition."""

        references: list[NativeGroupReference] = []
        tail = self.tail_dimension
        linear = self.linear_core_size
        for group in self._groups:
            zeros_linear = np.zeros((tail, linear), dtype=complex)
            zeros_second = np.zeros((tail, tail), dtype=complex)
            references.append(
                NativeGroupReference(
                    unit=NativeGroupBatch(
                        core=group.unit_core[None, :].astype(complex),
                        linear_eigen=zeros_linear[None, :, :],
                        second_eigen=zeros_second[None, :, :],
                    ),
                    linear=NativeGroupBatch(
                        core=group.reference_core[None, :],
                        linear_eigen=(
                            group.reference_linear_eigen[None, :, :]
                        ),
                        second_eigen=(
                            group.reference_second_eigen[None, :, :]
                        ),
                    ),
                )
            )
        return tuple(references)

    @staticmethod
    def _validate_locations(locations: np.ndarray) -> np.ndarray:
        raw = np.asarray(locations)
        if np.iscomplexobj(raw):
            raise TypeError("locations must be real-valued")
        values = np.asarray(raw, dtype=float)
        if values.ndim != 1:
            raise ValueError("locations must be one-dimensional")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("locations must be finite and nonnegative")
        return values

    def compact_group_many(
        self,
        group_index: int,
        locations: np.ndarray,
    ) -> CompactCenteredGroupBatch:
        """Evaluate stable compact log-amplitude data for one group."""

        values = self._validate_locations(locations)
        index = int(group_index)
        if not 0 <= index < self.group_count:
            raise IndexError("group_index is out of range")
        group = self._groups[index]
        count = values.size
        core_d = np.zeros((count, self.core_size), dtype=complex)
        delta_gram = np.zeros(
            (count, self.linear_core_size, self.linear_core_size),
            dtype=complex,
        )
        delta_mass = np.zeros(count, dtype=complex)
        positive = values > 0.0
        if not np.any(positive):
            return CompactCenteredGroupBatch(
                core_d=core_d,
                delta_gram=delta_gram,
                delta_mass=delta_mass,
            )

        selected = values[positive]
        frequency = selected / self.width
        eigenvalues = group.eigenvalues
        denominator_offset = (
            -2.0j * frequency[:, None] * eigenvalues[None, :]
        )
        inverse = 1.0 / (1.0 + denominator_offset)
        determinant_remainder = np.sum(
            _log1p_minus_identity(denominator_offset),
            axis=1,
        )

        node_linear_map = (
            group.linear_core_matrix @ group.rotated_linear_map
        )
        quadratic = np.einsum(
            "nt,ft,nt->fn",
            node_linear_map,
            inverse,
            node_linear_map,
            optimize=True,
        )
        delta = (
            -0.5 * determinant_remainder[:, None]
            - 0.5 * frequency[:, None] ** 2 * quadratic
        )
        centered_slope = (
            self.gamma
            + 1.0j * group.centered_core_values / self.width
        )
        shifted_log = (
            selected[:, None] * centered_slope[None, :] + delta
        )
        stable_core_bracket = (
            _expm1_minus_identity(shifted_log) + delta
        )
        decay = np.exp(-self.gamma * selected)
        core_d[positive] = (
            decay[:, None]
            / selected[:, None]
            * (
                (group.core_weights[None, :] * stable_core_bracket)
                @ group.core_matrix
            )
        )

        centered_log = (
            1.0j
            * frequency[:, None]
            * group.centered_core_values[None, :]
            + delta
        )
        delta_weights = (
            group.core_weights[None, :] * np.expm1(centered_log)
        )
        delta_gram[positive] = np.einsum(
            "na,fn,nb->fab",
            group.linear_core_matrix,
            delta_weights,
            group.linear_core_matrix,
            optimize=True,
        )
        delta_mass[positive] = np.sum(delta_weights, axis=1)
        return CompactCenteredGroupBatch(
            core_d=core_d,
            delta_gram=delta_gram,
            delta_mass=delta_mass,
        )

    def compact_many(
        self,
        locations: np.ndarray,
    ) -> tuple[CompactCenteredGroupBatch, ...]:
        """Evaluate compact data for every conditional group."""

        values = self._validate_locations(locations)
        return tuple(
            self.compact_group_many(index, values)
            for index in range(self.group_count)
        )

    def native_group_many(
        self,
        group_index: int,
        locations: np.ndarray,
        *,
        rational: bool = False,
        compact: CompactCenteredGroupBatch | None = None,
    ) -> NativeGroupBatch:
        """Expand one compact group to native log or rational amplitudes.

        This method is primarily an oracle and debugging surface.  The fused
        Filon implementation contracts the compact quantities directly and
        never stores a frequency-by-tail-by-tail tensor.
        """

        values = self._validate_locations(locations)
        index = int(group_index)
        if not 0 <= index < self.group_count:
            raise IndexError("group_index is out of range")
        group = self._groups[index]
        data = (
            self.compact_group_many(index, values)
            if compact is None
            else compact
        )
        expected_shapes = (
            (values.size, self.core_size),
            (
                values.size,
                self.linear_core_size,
                self.linear_core_size,
            ),
            (values.size,),
        )
        if (
            data.core_d.shape != expected_shapes[0]
            or data.delta_gram.shape != expected_shapes[1]
            or data.delta_mass.shape != expected_shapes[2]
        ):
            raise ValueError("compact batch shape does not match locations")

        frequency = values / self.width
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
            - np.expm1(-self.gamma * values)[:, None]
        )
        b_transpose = group.rotated_linear_map.T
        linear = (
            1.0j
            / self.width
            * (
                inverse[:, :, None]
                * np.einsum(
                    "ta,fab->ftb",
                    b_transpose,
                    data.delta_gram,
                    optimize=True,
                )
                + delta_inverse[:, :, None]
                * (
                    b_transpose
                    @ group.weighted_linear_gram
                )[None, :, :]
            )
        )

        diagonal = (
            2.0j
            / self.width
            * group.eigenvalues[None, :]
            * (
                inverse * data.delta_mass[:, None]
                + delta_inverse * group.group_mass
            )
        )
        second = np.zeros(
            (values.size, self.tail_dimension, self.tail_dimension),
            dtype=complex,
        )
        diagonal_indices = np.arange(self.tail_dimension)
        second[:, diagonal_indices, diagonal_indices] = diagonal
        t_bar = (
            1.0j
            / self.width
            * group.rotated_linear_map[None, :, :]
            * inverse[:, None, :]
        )
        full_gram = (
            group.weighted_linear_gram[None, :, :]
            + data.delta_gram
        )
        second += (
            values[:, None, None]
            * np.einsum(
                "fat,fab,fbe->fte",
                t_bar,
                full_gram,
                t_bar,
                optimize=True,
            )
        )
        core = data.core_d.copy()
        if rational:
            core *= values[:, None]
            linear *= values[:, None, None]
            second *= values[:, None, None]
        # Make the exact removable value explicit even if a platform happens
        # to produce a signed zero in an intermediate expression.
        zero = values == 0.0
        core[zero] = 0.0
        linear[zero] = 0.0
        second[zero] = 0.0
        return NativeGroupBatch(
            core=core,
            linear_eigen=linear,
            second_eigen=second,
        )

    def assemble_native(
        self,
        batches: tuple[NativeGroupBatch, ...],
    ) -> np.ndarray:
        """Rotate and scatter a groupwise native batch to retained support."""

        if len(batches) != self.group_count:
            raise ValueError("one native batch is required for every group")
        count: int | None = None
        result: np.ndarray | None = None
        structure = self._structure
        projected_second: np.ndarray | None = None
        projected_modulated: np.ndarray | None = None

        for group, batch in zip(self._groups, batches):
            if batch.core.ndim != 2:
                raise ValueError("native core batch must be two-dimensional")
            current_count = batch.core.shape[0]
            if count is None:
                count = current_count
                result = np.zeros(
                    (count, self.basis_size), dtype=complex
                )
                projected_second = np.zeros(
                    (count, self.tail_dimension, self.tail_dimension),
                    dtype=complex,
                )
                projected_modulated = np.zeros_like(projected_second)
            if (
                current_count != count
                or batch.core.shape != (count, self.core_size)
                or batch.linear_eigen.shape
                != (
                    count,
                    self.tail_dimension,
                    self.linear_core_size,
                )
                or batch.second_eigen.shape
                != (
                    count,
                    self.tail_dimension,
                    self.tail_dimension,
                )
            ):
                raise ValueError("native group batch shapes are inconsistent")
            assert result is not None
            assert projected_second is not None
            assert projected_modulated is not None
            result[:, structure._core_positions] += batch.core
            result[:, structure._linear_positions] += np.einsum(
                "ij,fjk->fik",
                group.eigenvectors,
                batch.linear_eigen,
                optimize=True,
            )
            physical_second = np.einsum(
                "ij,fjk,lk->fil",
                group.eigenvectors,
                batch.second_eigen,
                group.eigenvectors,
                optimize=True,
            )
            projected_second += physical_second
            projected_modulated += group.eta_c * physical_second

        if count is None:
            raise RuntimeError("at least one conditional group is required")
        assert result is not None
        assert projected_second is not None
        assert projected_modulated is not None
        result[:, structure._diagonal_positions] = (
            np.diagonal(projected_second, axis1=1, axis2=2)
            / np.sqrt(2.0)
        )
        result[:, structure._pair_positions] = projected_second[
            :, structure._pair_left, structure._pair_right
        ]
        if structure._modulated_positions.size:
            modulated = projected_modulated[
                :,
                structure._modulated_left,
                structure._modulated_right,
            ].copy()
            modulated[:, structure._modulated_diagonal] /= np.sqrt(2.0)
            result[:, structure._modulated_positions] = modulated
        return result

    def physical_many(
        self,
        locations: np.ndarray,
        *,
        rational: bool = False,
    ) -> np.ndarray:
        """Oracle path: assemble the groupwise centered amplitude per node."""

        values = self._validate_locations(locations)
        compact = self.compact_many(values)
        native = tuple(
            self.native_group_many(
                index,
                values,
                rational=rational,
                compact=compact[index],
            )
            for index in range(self.group_count)
        )
        return self.assemble_native(native)

    def characteristic_many(self, frequencies: np.ndarray) -> np.ndarray:
        """Assemble physical Hermite-weighted characteristic moments.

        This path is intended for the finite cosine atoms of a drift.  Filon
        nodes use :meth:`compact_many` and panel-level fusion instead.
        """

        values = self._validate_locations(frequencies)
        native_by_group: list[NativeGroupBatch] = []
        for group in self._groups:
            inverse = 1.0 / (
                1.0
                - 2.0j
                * values[:, None]
                * group.eigenvalues[None, :]
            )
            denominator_offset = (
                -2.0j
                * values[:, None]
                * group.eigenvalues[None, :]
            )
            determinant_remainder = np.sum(
                _log1p_minus_identity(denominator_offset),
                axis=1,
            )
            node_linear_map = (
                group.linear_core_matrix @ group.rotated_linear_map
            )
            quadratic = np.einsum(
                "nt,ft,nt->fn",
                node_linear_map,
                inverse,
                node_linear_map,
                optimize=True,
            )
            delta = (
                -0.5 * determinant_remainder[:, None]
                - 0.5 * values[:, None] ** 2 * quadratic
            )
            centered_log = (
                1.0j
                * values[:, None]
                * group.centered_core_values[None, :]
                + delta
            )
            characteristic_weights = (
                group.core_weights[None, :] * np.exp(centered_log)
            )
            gram = np.einsum(
                "na,fn,nb->fab",
                group.linear_core_matrix,
                characteristic_weights,
                group.linear_core_matrix,
                optimize=True,
            )
            core = characteristic_weights @ group.core_matrix
            tilted = (
                1.0j
                * values[:, None, None]
                * group.rotated_linear_map[None, :, :]
                * inverse[:, None, :]
            )
            linear = np.einsum(
                "fda,fab->fdb",
                np.swapaxes(tilted, 1, 2),
                gram,
                optimize=True,
            )
            second = np.einsum(
                "fad,fab,fbe->fde",
                tilted,
                gram,
                tilted,
                optimize=True,
            )
            diagonal = (
                np.sum(characteristic_weights, axis=1)[:, None]
                * (inverse - 1.0)
            )
            diagonal_indices = np.arange(self.tail_dimension)
            second[:, diagonal_indices, diagonal_indices] += diagonal
            native_by_group.append(
                NativeGroupBatch(
                    core=core,
                    linear_eigen=linear,
                    second_eigen=second,
                )
            )
        assembled = self.assemble_native(tuple(native_by_group))
        assembled *= np.exp(1.0j * values * self.mean)[:, None]
        zero = values == 0.0
        if np.any(zero):
            assembled[zero] = 0.0
            assembled[zero, 0] = 1.0
        return assembled

    def assembled_reference(self) -> tuple[np.ndarray, np.ndarray]:
        """Return physical sums of ``U_g`` and ``R_g`` for audit tests."""

        references = self.reference_blocks()
        unit = self.assemble_native(
            tuple(reference.unit for reference in references)
        )[0]
        linear = self.assemble_native(
            tuple(reference.linear for reference in references)
        )[0]
        return unit, linear


__all__ = [
    "CompactCenteredGroupBatch",
    "NativeGroupBatch",
    "NativeGroupReference",
    "StableCenteredEigenAmplitude",
]
