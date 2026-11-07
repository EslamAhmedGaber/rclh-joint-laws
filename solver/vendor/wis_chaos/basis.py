"""Normalized probabilists' Hermite basis and deterministic quadrature."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import factorial
from typing import Iterable

import numpy as np
from scipy.special import ndtri, roots_laguerre
from scipy.stats import qmc


def _compositions(total: int, dimension: int):
    if dimension == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _compositions(total - first, dimension - 1):
            yield (first,) + tail


@dataclass(frozen=True)
class HermiteBasis:
    dimension: int
    degree: int
    index_set: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.degree < 0:
            raise ValueError("degree must be nonnegative")
        if self.index_set is None:
            return

        canonical = tuple(
            sorted(
                {
                    tuple(int(value) for value in alpha)
                    for alpha in self.index_set
                },
                key=lambda alpha: (sum(alpha), alpha),
            )
        )
        if not canonical:
            raise ValueError("index_set must be nonempty")
        if any(len(alpha) != self.dimension for alpha in canonical):
            raise ValueError("every multi-index must match dimension")
        if any(value < 0 for alpha in canonical for value in alpha):
            raise ValueError("multi-indices must be nonnegative")
        if any(sum(alpha) > self.degree for alpha in canonical):
            raise ValueError("multi-index exceeds declared degree")
        zero = (0,) * self.dimension
        if zero not in canonical:
            raise ValueError("index_set must contain the zero multi-index")

        available = set(canonical)
        for alpha in canonical:
            for coordinate, value in enumerate(alpha):
                if value == 0:
                    continue
                predecessor = list(alpha)
                predecessor[coordinate] -= 1
                if tuple(predecessor) not in available:
                    raise ValueError(
                        "index_set must be downward closed"
                    )
        object.__setattr__(self, "index_set", canonical)

    @cached_property
    def indices(self) -> np.ndarray:
        if self.index_set is not None:
            return np.asarray(self.index_set, dtype=np.int16)
        rows = [
            alpha
            for total in range(self.degree + 1)
            for alpha in _compositions(total, self.dimension)
        ]
        return np.asarray(rows, dtype=np.int16)

    @cached_property
    def lookup(self) -> dict[tuple[int, ...], int]:
        return {
            tuple(int(value) for value in alpha): position
            for position, alpha in enumerate(self.indices)
        }

    @cached_property
    def total_degrees(self) -> np.ndarray:
        return np.sum(self.indices, axis=1)

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])

    @cached_property
    def sqrt_factorials(self) -> np.ndarray:
        values = np.ones(self.size, dtype=float)
        for row, alpha in enumerate(self.indices):
            for value in alpha:
                values[row] *= factorial(int(value))
        return np.sqrt(values)

    def evaluate(self, nodes: np.ndarray) -> np.ndarray:
        points = np.asarray(nodes, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("nodes must have shape (n_nodes, dimension)")

        matrix = np.ones((points.shape[0], self.size), dtype=float)
        for coordinate in range(self.dimension):
            orders = self.indices[:, coordinate]
            active = np.flatnonzero(orders)
            if not active.size:
                continue
            maximum_order = int(np.max(orders[active]))
            table = np.empty(
                (points.shape[0], maximum_order + 1),
                dtype=float,
            )
            table[:, 0] = 1.0
            if maximum_order:
                table[:, 1] = points[:, coordinate]
            for order in range(1, maximum_order):
                table[:, order + 1] = (
                    points[:, coordinate] * table[:, order]
                    - np.sqrt(order) * table[:, order - 1]
                ) / np.sqrt(order + 1)
            matrix[:, active] *= table[:, orders[active]]
        return matrix

    def exact_wick_exponential_coefficients(self, loading: np.ndarray) -> np.ndarray:
        vector = np.asarray(loading, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError("loading has wrong dimension")
        powers = np.ones(self.size, dtype=float)
        for coordinate in range(self.dimension):
            powers *= vector[coordinate] ** self.indices[:, coordinate]
        return powers / self.sqrt_factorials

    @cached_property
    def first_raising_edges(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Target-centric edges for the normalized creation operator.

        A downward-closed set contains ``beta-e_i`` whenever ``beta_i>0``.
        Enumerating those predecessors visits only the actual first-raising
        edges, rather than scanning all ``dimension * degree * size``
        source/coordinate/order combinations.  The final array stores the
        first edge offset whose target total degree is at least ``r`` for
        every ``r``.  Those offsets let the exponential recurrence skip
        degree layers that are identically zero.

        Storage is ``O(E)`` where ``E`` is the number of first-raising edges;
        no coordinate-by-coefficient table is constructed.
        """

        sources: list[int] = []
        targets: list[int] = []
        coordinates: list[int] = []
        factors: list[float] = []
        target_degrees: list[int] = []
        for target_position, beta_array in enumerate(self.indices):
            target_degree = int(np.sum(beta_array))
            for coordinate in np.flatnonzero(beta_array):
                source = beta_array.copy()
                source[coordinate] -= 1
                sources.append(
                    self.lookup[tuple(int(value) for value in source)]
                )
                targets.append(target_position)
                coordinates.append(int(coordinate))
                factors.append(np.sqrt(float(beta_array[coordinate])))
                target_degrees.append(target_degree)

        source_array = np.asarray(sources, dtype=np.int64)
        target_array = np.asarray(targets, dtype=np.int64)
        coordinate_array = np.asarray(coordinates, dtype=np.int64)
        factor_array = np.asarray(factors, dtype=float)
        degree_array = np.asarray(target_degrees, dtype=np.int16)
        maximum_degree = int(np.max(self.total_degrees, initial=0))
        degree_offsets = np.asarray(
            [
                np.searchsorted(degree_array, order, side="left")
                for order in range(1, maximum_degree + 1)
            ],
            dtype=np.int64,
        )
        return (
            source_array,
            target_array,
            coordinate_array,
            factor_array,
            degree_offsets,
        )

    @cached_property
    def raising_transitions(
        self,
    ) -> tuple[tuple[tuple[int, np.ndarray, np.ndarray, np.ndarray], ...], ...]:
        """Coordinate transitions retained for compatibility.

        Construction is target-centric, so its work is proportional to the
        number of transitions that actually exist.  The main exponential
        path below needs only :attr:`first_raising_edges`.
        """

        transitions: list[list[list[list[float] | list[int]]]] = [
            [
                [[], [], []]
                for _ in range(self.degree)
            ]
            for _ in range(self.dimension)
        ]
        for target_position, beta_array in enumerate(self.indices):
            for coordinate in np.flatnonzero(beta_array):
                source = beta_array.copy()
                factor_value = 1.0
                target_order = int(beta_array[coordinate])
                for raise_count in range(1, target_order + 1):
                    source[coordinate] -= 1
                    factor_value *= (
                        np.sqrt(target_order - raise_count + 1.0)
                        / raise_count
                    )
                    bucket = transitions[int(coordinate)][raise_count - 1]
                    bucket[0].append(
                        self.lookup[tuple(int(value) for value in source)]
                    )
                    bucket[1].append(target_position)
                    bucket[2].append(factor_value)

        all_coordinates = []
        for coordinate_buckets in transitions:
            coordinate_transitions = []
            for raise_count, (sources, targets, factors) in enumerate(
                coordinate_buckets,
                start=1,
            ):
                if not sources:
                    continue
                coordinate_transitions.append(
                    (
                        raise_count,
                        np.asarray(sources, dtype=np.int64),
                        np.asarray(targets, dtype=np.int64),
                        np.asarray(factors, dtype=float),
                    )
                )
            all_coordinates.append(tuple(coordinate_transitions))
        return tuple(all_coordinates)

    def apply_wick_exponential(
        self, coefficients: np.ndarray, loading: np.ndarray
    ) -> np.ndarray:
        """Apply ``exp(sum loading_k C_k)`` exactly on the finite basis.

        Projected creations commute on every downward-closed set: whenever a
        two-step target is present, downward closure guarantees both
        intermediate indices.  Hence the coordinate product equals the
        finite Taylor polynomial in the combined creation operator ``A``.
        The recurrence below forms ``A**r state / r!`` from first-raising
        edges only.  For fixed maximum chaos degree ``P`` its work is
        ``O(P E)`` (less after degree slicing) and its storage is ``O(E+K)``;
        it avoids the former ``O(dimension * K)`` copies per application.
        """

        state = np.asarray(coefficients, dtype=float)
        vector = np.asarray(loading, dtype=float)
        if state.shape != (self.size,):
            raise ValueError("coefficient vector has wrong size")
        if vector.shape != (self.dimension,):
            raise ValueError("loading has wrong dimension")

        if not np.any(vector):
            return state.copy()
        (
            sources,
            targets,
            coordinates,
            factors,
            degree_offsets,
        ) = self.first_raising_edges
        if not sources.size:
            return state.copy()

        edge_weights = factors * vector[coordinates]
        result = state.copy()
        term = state.copy()
        for order, start in enumerate(degree_offsets, start=1):
            active = slice(int(start), None)
            term = np.bincount(
                targets[active],
                weights=(
                    edge_weights[active] * term[sources[active]]
                ),
                minlength=self.size,
            ) / order
            result += term
        return result

    def apply_creation(
        self, coefficients: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Apply sum_k direction_k C_k in normalized Hermite coordinates."""

        state = np.asarray(coefficients, dtype=float)
        vector = np.asarray(direction, dtype=float)
        if state.shape != (self.size,):
            raise ValueError("coefficient vector has wrong size")
        if vector.shape != (self.dimension,):
            raise ValueError("direction has wrong dimension")
        (
            sources,
            targets,
            coordinates,
            factors,
            _,
        ) = self.first_raising_edges
        if not sources.size:
            return np.zeros_like(state)
        return np.bincount(
            targets,
            weights=factors * vector[coordinates] * state[sources],
            minlength=self.size,
        )

    def wick_multiply(
        self, left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        """Truncated Wick product in the normalized multi-index basis."""

        lhs = np.asarray(left, dtype=float)
        rhs = np.asarray(right, dtype=float)
        if lhs.shape != (self.size,) or rhs.shape != (self.size,):
            raise ValueError("coefficient vectors have wrong size")
        result = np.zeros(self.size, dtype=float)
        nonzero_left = np.flatnonzero(lhs)
        nonzero_right = np.flatnonzero(rhs)
        for left_position in nonzero_left:
            alpha = self.indices[left_position].astype(int)
            for right_position in nonzero_right:
                gamma = alpha + self.indices[right_position]
                if int(np.sum(gamma)) > self.degree:
                    continue
                target = self.lookup.get(
                    tuple(int(value) for value in gamma)
                )
                if target is None:
                    continue
                factor_value = (
                    self.sqrt_factorials[target]
                    / (
                        self.sqrt_factorials[left_position]
                        * self.sqrt_factorials[right_position]
                    )
                )
                result[target] += (
                    lhs[left_position] * rhs[right_position] * factor_value
                )
        return result


def core_enriched_index_set(
    dimension: int,
    core_modes: int,
    core_degree: int,
    *,
    cross_degree: int = 0,
    squared_tail_modes: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Build a downward-closed sparse chaos set.

    The first ``core_modes`` coordinates receive a full total-degree basis.
    Every remaining coordinate is represented linearly and may interact with
    core polynomials up to ``cross_degree``.  Optionally, the first
    ``squared_tail_modes`` tail coordinates also receive a univariate square.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not 1 <= core_modes <= dimension:
        raise ValueError("core_modes must lie in [1, dimension]")
    if core_degree < 1:
        raise ValueError("core_degree must be positive")
    if not 0 <= cross_degree < core_degree:
        raise ValueError("cross_degree must lie in [0, core_degree)")
    tail_modes = dimension - core_modes
    if not 0 <= squared_tail_modes <= tail_modes:
        raise ValueError(
            "squared_tail_modes must not exceed the tail dimension"
        )

    indices: set[tuple[int, ...]] = set()
    for total in range(core_degree + 1):
        for core_alpha in _compositions(total, core_modes):
            indices.add(core_alpha + (0,) * tail_modes)

    for tail_offset in range(tail_modes):
        tail_coordinate = core_modes + tail_offset
        for total in range(cross_degree + 1):
            for core_alpha in _compositions(total, core_modes):
                alpha = list(core_alpha + (0,) * tail_modes)
                alpha[tail_coordinate] = 1
                indices.add(tuple(alpha))
        if tail_offset < squared_tail_modes:
            alpha = [0] * dimension
            alpha[tail_coordinate] = 2
            indices.add(tuple(alpha))

    return tuple(sorted(indices, key=lambda alpha: (sum(alpha), alpha)))


def quadratic_tail_core_enriched_index_set(
    dimension: int,
    core_modes: int,
    core_degree: int,
    *,
    cross_degree: int = 0,
    pairwise_tail_modes: int | None = None,
    tail_pair_core_modes: int = 0,
    tail_pair_core_degree: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Build a core-enriched set with an explicit quadratic tail layer.

    The core and core--tail terms are the same as in
    :func:`core_enriched_index_set`.  In addition, every quadratic product
    ``xi_i xi_j`` among the first ``pairwise_tail_modes`` tail coordinates is
    included, with ``i == j`` allowed.  Optionally, these tail pairs are
    multiplied by core polynomials in the first ``tail_pair_core_modes``
    coordinates up to ``tail_pair_core_degree``.  The latter degree may not
    exceed ``cross_degree``; this condition keeps every tail predecessor in
    the set.  The resulting set is downward closed, so projected creation
    operators retain their exact finite-dimensional semigroup algebra.
    """

    indices = set(
        core_enriched_index_set(
            dimension,
            core_modes,
            core_degree,
            cross_degree=cross_degree,
        )
    )
    tail_modes = dimension - core_modes
    selected_tail_modes = (
        tail_modes
        if pairwise_tail_modes is None
        else int(pairwise_tail_modes)
    )
    if not 0 <= selected_tail_modes <= tail_modes:
        raise ValueError(
            "pairwise_tail_modes must not exceed the tail dimension"
        )
    selected_pair_core_modes = int(tail_pair_core_modes)
    selected_pair_core_degree = int(tail_pair_core_degree)
    if not 0 <= selected_pair_core_modes <= core_modes:
        raise ValueError(
            "tail_pair_core_modes must not exceed core_modes"
        )
    if not 0 <= selected_pair_core_degree <= cross_degree:
        raise ValueError(
            "tail_pair_core_degree must lie in [0, cross_degree]"
        )
    if selected_pair_core_degree and not selected_pair_core_modes:
        raise ValueError(
            "positive tail_pair_core_degree needs tail_pair_core_modes"
        )

    for left_offset in range(selected_tail_modes):
        left = core_modes + left_offset
        for right_offset in range(left_offset, selected_tail_modes):
            right = core_modes + right_offset
            for total in range(selected_pair_core_degree + 1):
                for selected_core_alpha in _compositions(
                    total,
                    max(selected_pair_core_modes, 1),
                ):
                    alpha = [0] * dimension
                    if selected_pair_core_modes:
                        alpha[:selected_pair_core_modes] = (
                            selected_core_alpha
                        )
                    alpha[left] += 1
                    alpha[right] += 1
                    indices.add(tuple(alpha))

    return tuple(sorted(indices, key=lambda alpha: (sum(alpha), alpha)))


def weighted_total_degree_index_set(
    weights: Iterable[int],
    budget: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate ``sum_k weights[k] * alpha[k] <= budget``."""

    integer_weights = tuple(int(value) for value in weights)
    if not integer_weights:
        raise ValueError("weights must be nonempty")
    if any(value < 1 for value in integer_weights):
        raise ValueError("weights must be positive integers")
    if budget < 0:
        raise ValueError("budget must be nonnegative")

    rows: list[tuple[int, ...]] = []
    current = [0] * len(integer_weights)

    def recurse(coordinate: int, remaining: int) -> None:
        if coordinate == len(integer_weights):
            rows.append(tuple(current))
            return
        weight = integer_weights[coordinate]
        for order in range(remaining // weight + 1):
            current[coordinate] = order
            recurse(coordinate + 1, remaining - order * weight)
        current[coordinate] = 0

    recurse(0, int(budget))
    return tuple(sorted(set(rows), key=lambda alpha: (sum(alpha), alpha)))


@dataclass(frozen=True)
class TensorGaussHermite:
    """Tensor Gauss--Hermite rule for the standard Gaussian measure."""

    dimension: int
    order: int
    max_nodes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.order < 1:
            raise ValueError("dimension and order must be positive")
        if self.order**self.dimension > self.max_nodes:
            raise MemoryError(
                f"tensor rule would have {self.order**self.dimension:,} nodes"
            )

    @cached_property
    def nodes_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        nodes_1d, weights_1d = np.polynomial.hermite_e.hermegauss(self.order)
        weights_1d = weights_1d / np.sqrt(2.0 * np.pi)
        node_mesh = np.meshgrid(
            *([nodes_1d] * self.dimension), indexing="ij"
        )
        weight_mesh = np.meshgrid(
            *([weights_1d] * self.dimension), indexing="ij"
        )
        nodes = np.stack([axis.reshape(-1) for axis in node_mesh], axis=1)
        weights = np.ones(nodes.shape[0], dtype=float)
        for axis in weight_mesh:
            weights *= axis.reshape(-1)
        return nodes, weights

    def basis_matrix(self, basis: HermiteBasis) -> np.ndarray:
        if basis.dimension != self.dimension:
            raise ValueError("basis and quadrature dimensions differ")
        nodes, _ = self.nodes_and_weights
        return basis.evaluate(nodes)


@dataclass(frozen=True)
class SobolHermiteLeastSquares:
    """Frozen Sobol Gaussian design for dimension-scalable Hermite projection."""

    basis: HermiteBasis
    node_count: int
    seed: int = 1729

    def __post_init__(self) -> None:
        if self.node_count < 2 * self.basis.size:
            raise ValueError(
                "least-squares design needs at least twice as many nodes "
                "as coefficients"
            )
        if self.node_count & (self.node_count - 1):
            raise ValueError("Sobol node_count must be a power of two")

    @cached_property
    def nodes(self) -> np.ndarray:
        exponent = int(np.log2(self.node_count))
        engine = qmc.Sobol(
            d=self.basis.dimension, scramble=True, seed=self.seed
        )
        uniforms = engine.random_base2(exponent)
        tiny = np.finfo(float).eps
        return ndtri(np.clip(uniforms, tiny, 1.0 - tiny))

    @cached_property
    def basis_matrix(self) -> np.ndarray:
        return self.basis.evaluate(self.nodes)

    @cached_property
    def gram(self) -> np.ndarray:
        return (self.basis_matrix.T @ self.basis_matrix) / self.node_count

    @cached_property
    def projection_operator(self) -> np.ndarray:
        right = self.basis_matrix.T / self.node_count
        return np.linalg.solve(self.gram, right)

    @property
    def gram_identity_error(self) -> float:
        return float(
            np.linalg.norm(
                self.gram - np.eye(self.basis.size), ord=2
            )
        )

    @property
    def gram_condition_number(self) -> float:
        return float(np.linalg.cond(self.gram))


@dataclass(frozen=True)
class MatrixFreeSobolHermiteProjection:
    """Blockwise Sobol Hermite forward/adjoint without a dense design.

    The class deliberately implements the Gaussian projection

    ``Q^{-1} Phi.T f``

    rather than an empirical least-squares inverse.  For nonlinear dynamics
    it should be paired with an exact coefficient-space control variate for
    every known affine component; otherwise the sampled Gram error is
    repeatedly applied to that component and may destabilize time evolution.
    """

    basis: HermiteBasis
    node_count: int
    seed: int = 1729
    block_size: int = 256

    def __post_init__(self) -> None:
        if self.node_count < 1:
            raise ValueError("node_count must be positive")
        if self.node_count & (self.node_count - 1):
            raise ValueError("Sobol node_count must be a power of two")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

    @cached_property
    def nodes(self) -> np.ndarray:
        exponent = int(np.log2(self.node_count))
        engine = qmc.Sobol(
            d=self.basis.dimension,
            scramble=True,
            seed=self.seed,
        )
        uniforms = engine.random_base2(exponent)
        tiny = np.finfo(float).eps
        return ndtri(np.clip(uniforms, tiny, 1.0 - tiny))

    def _blocks(self):
        for start in range(0, self.node_count, self.block_size):
            stop = min(start + self.block_size, self.node_count)
            yield slice(start, stop)

    def apply_basis(self, coefficients: np.ndarray) -> np.ndarray:
        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        values = np.empty(self.node_count, dtype=float)
        for block in self._blocks():
            matrix = self.basis.evaluate(self.nodes[block])
            values[block] = matrix @ state
        return values

    def apply_adjoint(self, values: np.ndarray) -> np.ndarray:
        samples = np.asarray(values, dtype=float)
        if samples.shape != (self.node_count,):
            raise ValueError("sample vector has wrong size")
        result = np.zeros(self.basis.size, dtype=float)
        for block in self._blocks():
            matrix = self.basis.evaluate(self.nodes[block])
            result += matrix.T @ samples[block]
        return result / self.node_count

    def project_with_linear_control(
        self,
        time: float,
        coefficients: np.ndarray,
        function,
        linear_rate: float,
        affine_intercept: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Project a residual after applying an affine part exactly."""

        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        rate = float(linear_rate)
        intercept = float(affine_intercept)
        nonlinear_coefficients = np.zeros(
            self.basis.size,
            dtype=float,
        )
        nonlinear_norm_squared = 0.0
        for block in self._blocks():
            matrix = self.basis.evaluate(self.nodes[block])
            values = matrix @ state
            nonlinear_values = (
                function(float(time), values) - rate * values - intercept
            )
            nonlinear_coefficients += (
                matrix.T @ nonlinear_values
            )
            nonlinear_norm_squared += float(
                np.dot(nonlinear_values, nonlinear_values)
            )
        nonlinear_coefficients /= self.node_count
        nonlinear_norm_squared /= self.node_count
        # This is a Parseval diagnostic for the direct quadrature projection,
        # not a least-squares residual.  Its exactness improves as the sampled
        # Gram approaches the identity.
        residual = float(
            np.sqrt(
                max(
                    nonlinear_norm_squared
                    - float(
                        np.dot(
                            nonlinear_coefficients,
                            nonlinear_coefficients,
                        )
                    ),
                    0.0,
                )
            )
        )
        exact_affine = rate * state
        exact_affine[0] += intercept
        return exact_affine + nonlinear_coefficients, residual

    def gram_probe_error(
        self,
        *,
        probes: int = 3,
        seed: int = 9817,
    ) -> float:
        """Estimate ``||Phi.T Phi / Q - I||`` on frozen random probes."""

        if probes < 1:
            raise ValueError("probes must be positive")
        rng = np.random.default_rng(int(seed))
        maximum = 0.0
        for _ in range(int(probes)):
            vector = rng.choice(
                (-1.0, 1.0),
                size=self.basis.size,
            )
            image = self.apply_adjoint(
                self.apply_basis(vector)
            )
            maximum = max(
                maximum,
                float(
                    np.linalg.norm(image - vector)
                    / np.linalg.norm(vector)
                ),
            )
        return maximum


@dataclass(frozen=True)
class StructuredQuadraticTailSobolProjection:
    """Matrix-free forward/adjoint for a core plus quadratic-tail set.

    The supported multi-index structure is

    * arbitrary downward-closed core-only polynomials;
    * every tail-linear term times one common core polynomial set; and
    * every core-free quadratic tail pair, together with an arbitrary sparse
      selection of core-modulated quadratic tail pairs.

    The modulated layer uses a fixed ``Q x R`` feature cache, where ``R`` is
    the number of selected pairs.  It costs ``O(Q R)`` memory and work but is
    built only once, rather than once per nonlinear right-hand-side call.
    The cache remains separate from the much larger ``Q x K`` Hermite design;
    all contractions are algebraically identical to its forward and adjoint
    products.
    """

    basis: HermiteBasis
    core_dimensions: int
    node_count: int
    seed: int = 1729
    block_size: int = 4096

    def __post_init__(self) -> None:
        if not 1 <= self.core_dimensions < self.basis.dimension:
            raise ValueError(
                "core_dimensions must lie in [1, basis.dimension)"
            )
        if self.node_count < 1:
            raise ValueError("node_count must be positive")
        if self.node_count & (self.node_count - 1):
            raise ValueError("Sobol node_count must be a power of two")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

        core = int(self.core_dimensions)
        tail = self.basis.dimension - core
        zero_tail = (0,) * tail
        core_only: dict[tuple[int, ...], int] = {}
        linear_entries: dict[
            tuple[int, tuple[int, ...]], int
        ] = {}
        diagonal = np.full(tail, -1, dtype=np.int64)
        off_diagonal: dict[tuple[int, int], int] = {}
        modulated_entries: list[
            tuple[int, tuple[int, ...], int, int, bool]
        ] = []

        for position, alpha_array in enumerate(self.basis.indices):
            alpha = tuple(int(value) for value in alpha_array)
            core_alpha = alpha[:core]
            tail_alpha = alpha[core:]
            tail_degree = sum(tail_alpha)
            if tail_degree == 0:
                core_only[core_alpha] = position
                continue
            nonzero = [
                index
                for index, value in enumerate(tail_alpha)
                if value
            ]
            if tail_degree == 1:
                if len(nonzero) != 1 or tail_alpha[nonzero[0]] != 1:
                    raise ValueError("invalid tail-linear multi-index")
                linear_entries[(nonzero[0], core_alpha)] = position
                continue
            if tail_degree == 2:
                is_diagonal = (
                    len(nonzero) == 1
                    and tail_alpha[nonzero[0]] == 2
                )
                is_off_diagonal = (
                    len(nonzero) == 2
                    and all(tail_alpha[index] == 1 for index in nonzero)
                )
                if is_diagonal:
                    left = right = nonzero[0]
                elif is_off_diagonal:
                    left, right = nonzero
                else:
                    raise ValueError("invalid quadratic tail multi-index")
                if any(core_alpha):
                    modulated_entries.append(
                        (
                            position,
                            core_alpha,
                            left,
                            right,
                            is_diagonal,
                        )
                    )
                elif is_diagonal:
                    diagonal[left] = position
                else:
                    off_diagonal[(left, right)] = position
                continue
            raise ValueError(
                "basis is not a core/linear-tail/quadratic-tail set"
            )

        if np.any(diagonal < 0):
            raise ValueError("quadratic tail diagonal is incomplete")
        expected_pairs = tail * (tail - 1) // 2
        if len(off_diagonal) != expected_pairs:
            raise ValueError("quadratic tail off-diagonal is incomplete")

        core_patterns = tuple(
            sorted(core_only, key=lambda alpha: (sum(alpha), alpha))
        )
        linear_patterns = tuple(
            sorted(
                {alpha for _, alpha in linear_entries},
                key=lambda alpha: (sum(alpha), alpha),
            )
        )
        if not linear_patterns:
            raise ValueError("tail-linear layer is missing")
        core_basis = HermiteBasis(
            core,
            max(sum(alpha) for alpha in core_patterns),
            index_set=core_patterns,
        )
        linear_core_basis = HermiteBasis(
            core,
            max(sum(alpha) for alpha in linear_patterns),
            index_set=linear_patterns,
        )
        core_positions = np.asarray(
            [core_only[tuple(alpha)] for alpha in core_basis.index_set],
            dtype=np.int64,
        )
        linear_positions = np.empty(
            (tail, linear_core_basis.size),
            dtype=np.int64,
        )
        for tail_coordinate in range(tail):
            for column, alpha in enumerate(linear_core_basis.index_set):
                key = (tail_coordinate, tuple(alpha))
                if key not in linear_entries:
                    raise ValueError("tail-linear core layer is incomplete")
                linear_positions[tail_coordinate, column] = (
                    linear_entries[key]
                )
        pair_left = []
        pair_right = []
        pair_positions = []
        for (left, right), position in sorted(off_diagonal.items()):
            pair_left.append(left)
            pair_right.append(right)
            pair_positions.append(position)

        core_columns = {
            tuple(alpha): column
            for column, alpha in enumerate(core_basis.index_set)
        }
        try:
            linear_core_columns = np.asarray(
                [
                    core_columns[tuple(alpha)]
                    for alpha in linear_core_basis.index_set
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "tail-linear core pattern lacks its core predecessor"
            ) from error
        modulated_positions = []
        modulated_core_columns = []
        modulated_left = []
        modulated_right = []
        modulated_diagonal = []
        for (
            position,
            core_alpha,
            left,
            right,
            is_diagonal,
        ) in modulated_entries:
            if core_alpha not in core_columns:
                raise ValueError(
                    "modulated tail pair lacks its core predecessor"
                )
            modulated_positions.append(position)
            modulated_core_columns.append(core_columns[core_alpha])
            modulated_left.append(left)
            modulated_right.append(right)
            modulated_diagonal.append(is_diagonal)

        object.__setattr__(self, "_core_basis", core_basis)
        object.__setattr__(self, "_linear_core_basis", linear_core_basis)
        object.__setattr__(
            self,
            "_linear_core_columns",
            linear_core_columns,
        )
        object.__setattr__(self, "_core_positions", core_positions)
        object.__setattr__(self, "_linear_positions", linear_positions)
        object.__setattr__(self, "_diagonal_positions", diagonal)
        object.__setattr__(
            self,
            "_pair_left",
            np.asarray(pair_left, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_pair_right",
            np.asarray(pair_right, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_pair_positions",
            np.asarray(pair_positions, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_modulated_positions",
            np.asarray(modulated_positions, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_modulated_core_columns",
            np.asarray(modulated_core_columns, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_modulated_left",
            np.asarray(modulated_left, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_modulated_right",
            np.asarray(modulated_right, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_modulated_diagonal",
            np.asarray(modulated_diagonal, dtype=bool),
        )

    @cached_property
    def nodes(self) -> np.ndarray:
        exponent = int(np.log2(self.node_count))
        engine = qmc.Sobol(
            d=self.basis.dimension,
            scramble=True,
            seed=self.seed,
        )
        uniforms = engine.random_base2(exponent)
        tiny = np.finfo(float).eps
        return ndtri(np.clip(uniforms, tiny, 1.0 - tiny))

    @cached_property
    def _core_design_matrix(self) -> np.ndarray:
        """Fixed ``Q x K_core`` design, cached once for all RHS calls.

        The structured solver never stores the full ``Q x K`` chaos design.
        This cache costs ``O(Q K_core)`` memory; ``K_core`` is independent of
        the tail dimension for the L3/P6/X2 family.
        """

        return self._core_basis.evaluate(
            self.nodes[:, : self.core_dimensions]
        )

    @cached_property
    def _linear_core_design_matrix(self) -> np.ndarray:
        """Tail-linear core columns extracted from the fixed core design."""

        return self._core_design_matrix[:, self._linear_core_columns]

    @cached_property
    def _modulated_design_matrix(self) -> np.ndarray:
        """Fixed selected-pair features with ``O(Q R)`` storage.

        The selected support, Sobol nodes, and core design do not change
        during coefficient integration.  Caching these columns eliminates
        repeated advanced indexing and elementwise construction at every RK
        stage while retaining the exact structured feature values.
        """

        if not self._modulated_positions.size:
            return np.empty((self.node_count, 0), dtype=float)
        return self._modulated_features(
            self._core_design_matrix,
            self.nodes[:, self.core_dimensions :],
        )

    def _blocks(self):
        for start in range(0, self.node_count, self.block_size):
            stop = min(start + self.block_size, self.node_count)
            yield slice(start, stop)

    def _structured_coefficients(
        self,
        coefficients: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        core_values = state[self._core_positions]
        linear_values = state[self._linear_positions]
        diagonal_values = state[self._diagonal_positions]
        tail = self.basis.dimension - self.core_dimensions
        symmetric = np.zeros((tail, tail), dtype=float)
        off_values = state[self._pair_positions]
        symmetric[self._pair_left, self._pair_right] = 0.5 * off_values
        symmetric[self._pair_right, self._pair_left] = 0.5 * off_values
        np.fill_diagonal(symmetric, diagonal_values / np.sqrt(2.0))
        modulated_values = state[self._modulated_positions]
        return (
            core_values,
            linear_values,
            diagonal_values,
            symmetric,
            modulated_values,
        )

    def _modulated_features(
        self,
        core_matrix: np.ndarray,
        tail_nodes: np.ndarray,
    ) -> np.ndarray:
        count = self._modulated_positions.size
        if count == 0:
            return np.empty((tail_nodes.shape[0], 0), dtype=float)
        left = tail_nodes[:, self._modulated_left]
        right = tail_nodes[:, self._modulated_right]
        tail_features = left * right
        if np.any(self._modulated_diagonal):
            tail_features[:, self._modulated_diagonal] -= 1.0
            tail_features[:, self._modulated_diagonal] /= np.sqrt(2.0)
        return (
            core_matrix[:, self._modulated_core_columns]
            * tail_features
        )

    def _forward_block(
        self,
        points: np.ndarray,
        core_values: np.ndarray,
        linear_values: np.ndarray,
        symmetric: np.ndarray,
        modulated_values: np.ndarray,
        *,
        core_matrix: np.ndarray | None = None,
        linear_core_matrix: np.ndarray | None = None,
        modulated_features: np.ndarray | None = None,
    ) -> np.ndarray:
        tail_nodes = points[:, self.core_dimensions :]
        if core_matrix is None:
            core_matrix = self._core_basis.evaluate(
                points[:, : self.core_dimensions]
            )
        if linear_core_matrix is None:
            linear_core_matrix = core_matrix[:, self._linear_core_columns]
        values = core_matrix @ core_values
        values += np.sum(
            tail_nodes * (linear_core_matrix @ linear_values.T),
            axis=1,
        )
        values += np.sum((tail_nodes @ symmetric) * tail_nodes, axis=1)
        values -= float(np.trace(symmetric))
        if modulated_values.size:
            if modulated_features is None:
                modulated_features = self._modulated_features(
                    core_matrix,
                    tail_nodes,
                )
            values += (
                modulated_features @ modulated_values
            )
        return values

    def apply_basis(self, coefficients: np.ndarray) -> np.ndarray:
        core_values, linear_values, _, symmetric, modulated_values = (
            self._structured_coefficients(coefficients)
        )
        values = np.empty(self.node_count, dtype=float)
        nodes = self.nodes
        core_design = self._core_design_matrix
        linear_core_design = self._linear_core_design_matrix
        modulated_design = (
            self._modulated_design_matrix
            if self._modulated_positions.size
            else None
        )
        for block in self._blocks():
            values[block] = self._forward_block(
                nodes[block],
                core_values,
                linear_values,
                symmetric,
                modulated_values,
                core_matrix=core_design[block],
                linear_core_matrix=linear_core_design[block],
                modulated_features=(
                    None
                    if modulated_design is None
                    else modulated_design[block]
                ),
            )
        return values

    def apply_adjoint(self, values: np.ndarray) -> np.ndarray:
        samples = np.asarray(values, dtype=float)
        if samples.shape != (self.node_count,):
            raise ValueError("sample vector has wrong size")
        result = np.zeros(self.basis.size, dtype=float)
        nodes = self.nodes
        core_design = self._core_design_matrix
        linear_core_design = self._linear_core_design_matrix
        modulated_design = (
            self._modulated_design_matrix
            if self._modulated_positions.size
            else None
        )
        for block in self._blocks():
            points = nodes[block]
            weights = samples[block]
            tail_nodes = points[:, self.core_dimensions :]
            core_matrix = core_design[block]
            linear_core_matrix = linear_core_design[block]
            result[self._core_positions] += core_matrix.T @ weights
            linear = tail_nodes.T @ (
                weights[:, None] * linear_core_matrix
            )
            result[self._linear_positions] += linear
            weighted_tail = weights[:, None] * tail_nodes
            second = tail_nodes.T @ weighted_tail
            mean_weight = float(np.sum(weights))
            result[self._diagonal_positions] += (
                np.diag(second) - mean_weight
            ) / np.sqrt(2.0)
            result[self._pair_positions] += second[
                self._pair_left,
                self._pair_right,
            ]
            if self._modulated_positions.size:
                assert modulated_design is not None
                modulated_features = modulated_design[block]
                result[self._modulated_positions] += (
                    modulated_features.T @ weights
                )
        return result / self.node_count

    def project_with_linear_control(
        self,
        time: float,
        coefficients: np.ndarray,
        function,
        linear_rate: float,
        affine_intercept: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        state = np.asarray(coefficients, dtype=float)
        core_values, linear_values, _, symmetric, modulated_values = (
            self._structured_coefficients(state)
        )
        nonlinear_coefficients = np.zeros(self.basis.size, dtype=float)
        nonlinear_norm_squared = 0.0
        rate = float(linear_rate)
        intercept = float(affine_intercept)
        nodes = self.nodes
        core_design = self._core_design_matrix
        linear_core_design = self._linear_core_design_matrix
        modulated_design = (
            self._modulated_design_matrix
            if self._modulated_positions.size
            else None
        )
        for block in self._blocks():
            points = nodes[block]
            tail_nodes = points[:, self.core_dimensions :]
            core_matrix = core_design[block]
            linear_core_matrix = linear_core_design[block]
            modulated_features = None
            if self._modulated_positions.size:
                assert modulated_design is not None
                modulated_features = modulated_design[block]
            values = self._forward_block(
                points,
                core_values,
                linear_values,
                symmetric,
                modulated_values,
                core_matrix=core_matrix,
                linear_core_matrix=linear_core_matrix,
                modulated_features=modulated_features,
            )
            nonlinear_values = (
                function(float(time), values) - rate * values - intercept
            )
            nonlinear_norm_squared += float(
                np.dot(nonlinear_values, nonlinear_values)
            )
            nonlinear_coefficients[self._core_positions] += (
                core_matrix.T @ nonlinear_values
            )
            linear = tail_nodes.T @ (
                nonlinear_values[:, None] * linear_core_matrix
            )
            nonlinear_coefficients[self._linear_positions] += linear
            weighted_tail = nonlinear_values[:, None] * tail_nodes
            second = tail_nodes.T @ weighted_tail
            mean_weight = float(np.sum(nonlinear_values))
            nonlinear_coefficients[self._diagonal_positions] += (
                np.diag(second) - mean_weight
            ) / np.sqrt(2.0)
            nonlinear_coefficients[self._pair_positions] += second[
                self._pair_left,
                self._pair_right,
            ]
            if self._modulated_positions.size:
                assert modulated_features is not None
                nonlinear_coefficients[self._modulated_positions] += (
                    modulated_features.T @ nonlinear_values
                )
        nonlinear_coefficients /= self.node_count
        nonlinear_norm_squared /= self.node_count
        residual = float(
            np.sqrt(
                max(
                    nonlinear_norm_squared
                    - float(
                        np.dot(
                            nonlinear_coefficients,
                            nonlinear_coefficients,
                        )
                    ),
                    0.0,
                )
            )
        )
        exact_affine = rate * state
        exact_affine[0] += intercept
        return exact_affine + nonlinear_coefficients, residual

    def gram_probe_error(
        self,
        *,
        probes: int = 3,
        seed: int = 9817,
    ) -> float:
        if probes < 1:
            raise ValueError("probes must be positive")
        rng = np.random.default_rng(int(seed))
        maximum = 0.0
        for _ in range(int(probes)):
            vector = rng.choice((-1.0, 1.0), size=self.basis.size)
            image = self.apply_adjoint(self.apply_basis(vector))
            maximum = max(
                maximum,
                float(
                    np.linalg.norm(image - vector)
                    / np.linalg.norm(vector)
                ),
            )
        return maximum


@dataclass(frozen=True)
class StructuredRationalLaguerreProjection:
    """Deterministic projection for ``scale*x/(1+x*x)``.

    The coefficient state must have the same core/linear-tail/quadratic-tail
    structure as :class:`StructuredQuadraticTailSobolProjection`.  Conditional
    on the core variables, the represented random variable is a quadratic
    polynomial of a standard Gaussian tail.  The identity

    ``x/(1+x*x) = integral_0^infinity exp(-u) sin(u*x) du``

    and the characteristic function of a non-central Gaussian quadratic form
    then integrate the full tail analytically.  Tensor Gauss--Hermite is used
    only in the low-dimensional core and Gauss--Laguerre only in ``u``.

    This backend is specialized to the rational residual.  It deliberately
    reports no closure-residual norm; callers should not reinterpret the
    quadrature-order comparison as a chaos-truncation certificate.
    """

    basis: HermiteBasis
    core_dimensions: int
    core_order: int = 9
    laguerre_order: int = 24
    max_core_nodes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.core_order < 1:
            raise ValueError("core_order must be positive")
        if self.laguerre_order < 1:
            raise ValueError("laguerre_order must be positive")
        structure = StructuredQuadraticTailSobolProjection(
            self.basis,
            int(self.core_dimensions),
            node_count=1,
            seed=0,
            block_size=1,
        )
        if structure._modulated_positions.size:
            raise ValueError(
                "rational Laguerre projection does not support "
                "core-modulated quadratic tail pairs"
            )
        core_rule = TensorGaussHermite(
            int(self.core_dimensions),
            int(self.core_order),
            max_nodes=int(self.max_core_nodes),
        )
        core_nodes, core_weights = core_rule.nodes_and_weights
        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "_core_nodes", core_nodes)
        object.__setattr__(self, "_core_weights", core_weights)
        object.__setattr__(
            self,
            "_core_matrix",
            structure._core_basis.evaluate(core_nodes),
        )
        object.__setattr__(
            self,
            "_linear_core_matrix",
            structure._linear_core_basis.evaluate(core_nodes),
        )
        laguerre_nodes, laguerre_weights = roots_laguerre(
            int(self.laguerre_order)
        )
        if not (
            np.all(np.isfinite(laguerre_nodes))
            and np.all(np.isfinite(laguerre_weights))
            and float(np.sum(laguerre_weights)) > 0.0
        ):
            raise FloatingPointError(
                "Gauss--Laguerre construction returned nonfinite data"
            )
        object.__setattr__(self, "_laguerre_nodes", laguerre_nodes)
        object.__setattr__(self, "_laguerre_weights", laguerre_weights)

    @property
    def core_node_count(self) -> int:
        return int(self._core_nodes.shape[0])

    def project_rational_residual(
        self,
        coefficients: np.ndarray,
        scale: float = 4.0,
    ) -> np.ndarray:
        """Project ``scale*X/(1+X**2)`` onto the structured basis."""

        state = np.asarray(coefficients, dtype=float)
        if state.shape != (self.basis.size,):
            raise ValueError("coefficient vector has wrong size")
        structure = self._structure
        core_values, linear_values, _, symmetric, _ = (
            structure._structured_coefficients(state)
        )
        core_polynomial = self._core_matrix @ core_values

        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        rotated_linear_map = linear_values.T @ eigenvectors
        trace = float(np.trace(symmetric))
        tail_dimension = symmetric.shape[0]

        projected_core = np.zeros(
            structure._core_basis.size,
            dtype=float,
        )
        projected_linear_eigen = np.zeros(
            (tail_dimension, structure._linear_core_basis.size),
            dtype=float,
        )
        projected_second_eigen = np.zeros(
            (tail_dimension, tail_dimension),
            dtype=float,
        )
        core_weights = self._core_weights
        multiplier = float(scale)

        for frequency, weight in zip(
            self._laguerre_nodes,
            self._laguerre_weights,
        ):
            denominator = 1.0 - 2.0j * frequency * eigenvalues
            inverse_denominator = 1.0 / denominator
            log_determinant = np.sum(np.log(denominator))
            quadratic_kernel = (
                rotated_linear_map * inverse_denominator
            ) @ rotated_linear_map.T
            quadratic = np.sum(
                (self._linear_core_matrix @ quadratic_kernel)
                * self._linear_core_matrix,
                axis=1,
            )
            characteristic = np.exp(
                1.0j * frequency * (core_polynomial - trace)
                - 0.5 * log_determinant
                - 0.5 * frequency * frequency * quadratic
            )
            tilted_mean_map = (
                1.0j
                * frequency
                * rotated_linear_map
                * inverse_denominator
            )

            weighted_scale = multiplier * float(weight)
            core_characteristic_weights = core_weights * characteristic
            core_characteristic_gram = (
                self._linear_core_matrix.T
                @ (
                    core_characteristic_weights[:, None]
                    * self._linear_core_matrix
                )
            )
            projected_core += weighted_scale * np.imag(
                self._core_matrix.T @ core_characteristic_weights
            )
            projected_linear_eigen += weighted_scale * np.imag(
                tilted_mean_map.T @ core_characteristic_gram
            )
            second_characteristic_eigen = (
                np.diag(
                    np.sum(core_characteristic_weights)
                    * (inverse_denominator - 1.0)
                )
                + tilted_mean_map.T
                @ core_characteristic_gram
                @ tilted_mean_map
            )
            projected_second_eigen += weighted_scale * np.imag(
                second_characteristic_eigen
            )

        projected_linear = eigenvectors @ projected_linear_eigen
        projected_second = (
            eigenvectors
            @ projected_second_eigen
            @ eigenvectors.T
        )
        result = np.zeros(self.basis.size, dtype=float)
        result[structure._core_positions] = projected_core
        result[structure._linear_positions] = projected_linear
        result[structure._diagonal_positions] = (
            np.diag(projected_second) / np.sqrt(2.0)
        )
        result[structure._pair_positions] = projected_second[
            structure._pair_left,
            structure._pair_right,
        ]
        return result

    def project_with_linear_control(
        self,
        coefficients: np.ndarray,
        linear_rate: float,
        rational_scale: float = 4.0,
        affine_intercept: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Apply the affine part exactly and project the rational residual."""

        state = np.asarray(coefficients, dtype=float)
        projected = (
            float(linear_rate) * state
            + self.project_rational_residual(state, rational_scale)
        )
        projected[0] += float(affine_intercept)
        return projected, float("nan")


@dataclass(frozen=True)
class HybridCoreGaussHermiteSobol:
    """Tensor Gauss--Hermite in core directions and Sobol in the tail."""

    basis: HermiteBasis
    core_dimensions: int
    core_order: int
    tail_node_count: int
    seed: int = 1729
    max_nodes: int = 2_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.core_dimensions < self.basis.dimension:
            raise ValueError(
                "core_dimensions must lie in [1, dimension)"
            )
        if self.core_order < 1:
            raise ValueError("core_order must be positive")
        if self.tail_node_count < 1:
            raise ValueError("tail_node_count must be positive")
        if self.tail_node_count & (self.tail_node_count - 1):
            raise ValueError(
                "tail_node_count must be a power of two"
            )
        tail_patterns = {
            tuple(int(value) for value in alpha[self.core_dimensions :])
            for alpha in self.basis.indices
        }
        if self.tail_node_count < len(tail_patterns):
            raise ValueError(
                "hybrid rule has fewer distinct tail nodes than tail "
                "basis patterns and is necessarily rank deficient"
            )
        core_node_count = self.core_order**self.core_dimensions
        core_patterns = {
            tuple(int(value) for value in alpha[: self.core_dimensions])
            for alpha in self.basis.indices
        }
        if core_node_count < len(core_patterns):
            raise ValueError(
                "hybrid rule has fewer core nodes than core basis "
                "patterns and is necessarily rank deficient"
            )
        total = (
            core_node_count * self.tail_node_count
        )
        if total > self.max_nodes:
            raise MemoryError(
                f"hybrid rule would have {total:,} nodes"
            )

    @cached_property
    def nodes_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        core_rule = TensorGaussHermite(
            self.core_dimensions,
            self.core_order,
            max_nodes=self.max_nodes,
        )
        core_nodes, core_weights = core_rule.nodes_and_weights
        tail_dimension = self.basis.dimension - self.core_dimensions
        exponent = int(np.log2(self.tail_node_count))
        engine = qmc.Sobol(
            d=tail_dimension,
            scramble=True,
            seed=self.seed,
        )
        uniforms = engine.random_base2(exponent)
        tiny = np.finfo(float).eps
        tail_nodes = ndtri(
            np.clip(uniforms, tiny, 1.0 - tiny)
        )
        nodes = np.column_stack(
            (
                np.repeat(
                    core_nodes,
                    self.tail_node_count,
                    axis=0,
                ),
                np.tile(tail_nodes, (core_nodes.shape[0], 1)),
            )
        )
        weights = (
            np.repeat(core_weights, self.tail_node_count)
            / self.tail_node_count
        )
        return nodes, weights

    @cached_property
    def nodes(self) -> np.ndarray:
        return self.nodes_and_weights[0]

    @cached_property
    def weights(self) -> np.ndarray:
        return self.nodes_and_weights[1]

    @cached_property
    def basis_matrix(self) -> np.ndarray:
        return self.basis.evaluate(self.nodes)

    @cached_property
    def gram(self) -> np.ndarray:
        return self.basis_matrix.T @ (
            self.weights[:, None] * self.basis_matrix
        )

    @cached_property
    def projection_operator(self) -> np.ndarray:
        right = self.basis_matrix.T * self.weights
        return np.linalg.solve(self.gram, right)

    @property
    def gram_identity_error(self) -> float:
        return float(
            np.linalg.norm(
                self.gram - np.eye(self.basis.size),
                ord=2,
            )
        )

    @property
    def gram_condition_number(self) -> float:
        return float(np.linalg.cond(self.gram))

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])


def gaussian_moments(coefficients: np.ndarray) -> tuple[float, float, float]:
    """Return mean, variance, and second moment in an orthonormal chaos basis."""

    values = np.asarray(coefficients, dtype=float)
    mean = float(values[0])
    variance = float(np.dot(values[1:], values[1:]))
    return mean, variance, variance + mean * mean
