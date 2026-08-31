import math
from abc import ABC, abstractmethod
from typing import cast, final, override

import numba
import numpy as np
import scipy

from .interface import MatrixInterface
from .types import Matrix, Vector


class IterativePreconditioner(ABC):
    """Abstract base class for preconditioning methods which can be
    iteratively refined/improved over time.

    It is expected, although not mandatory, that the preconditioner becomes "better"
    as it is updated/improved (i.e. over time, it reduces the problem's conditioning number
    even more than it initially did).
    """

    @abstractmethod
    def apply(self, rhs: Vector) -> Vector:
        """Applies the current iteration of the preconditioner to the given vector.

        Returns the vector obtained by applying the preconditioner.
        """

    @abstractmethod
    def update_inner(self) -> None:
        """Perform one inner iteration/step of the preconditioner update process, which usually involves performing one more step of a decomposition or approximation.

        This method should be relatively "fast" and called repeatedly.
        """

    @abstractmethod
    def update_outer(self) -> None:
        """Updates the internal structures used by the preconditioner to approximate the matrix inverse.

        For some preconditioners, updating the internal approximation/decomposition of the system matrix doesn't provide enough information to be able to apply the preconditioner to a vector. Hence, this method must also be called before trying to apply it (after a few inner updates).
        """

    def compute_to_rank(self, target_rank: int) -> None:
        """Computes this preconditioner up to the target rank.

        Equivalent to performing inner updates until the current inner rank equals the target rank,
        then performing one outer update. Actual implementation could be more efficient.
        """
        if target_rank < 0:
            raise ValueError("Target rank cannot be negative")

        if target_rank > self.max_rank:
            raise ValueError(
                "Target rank cannot be bigger than configured maximum rank"
            )

        while self.current_inner_rank < self.max_rank:
            self.update_inner()

        self.update_outer()

    def compute_full(self) -> None:
        """Computes the "full" variant of this preconditioner.

        Equivalent to performing inner updates until the level of approximation of this preconditioner reaches the maximum configured value and then running an outer update, but the actual implementation could be more efficient.
        """
        self.compute_to_rank(self.max_rank)

    @property
    @abstractmethod
    def current_inner_rank(self) -> int: ...

    @property
    @abstractmethod
    def current_outer_rank(self) -> int: ...

    @property
    @abstractmethod
    def max_rank(self) -> int: ...


class PivotedCholeskyPreconditioner(IterativePreconditioner, ABC):
    "Iterative preconditioner based on the pivoted partial Cholesky decomposition."

    _dimension: int
    _max_rank: int

    _current_inner_rank: int
    _current_outer_rank: int

    _matrix: MatrixInterface
    _matrix_diagonal: Vector

    _allocation_size: int
    _current_allocated_capacity: int

    _indices: list[int]
    _pivots: list[float]
    _preconditioner_upper: Matrix

    _regularization_factor: float
    _capacitance_cholesky: Matrix

    def __init__(
        self,
        matrix: MatrixInterface,
        max_rank: int,
        regularization_factor: float | None,
        allocation_size: int = 100,
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError("`matrix` must be a matrix")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")

        dimension = matrix.shape[0]
        self._dimension = dimension

        if max_rank < 0:
            raise ValueError("max rank must be non-negative")

        if max_rank > dimension:
            raise ValueError("max rank must be less than or equal to matrix dimension")

        self._max_rank = max_rank
        self._current_inner_rank = 0
        self._current_outer_rank = 0

        self._matrix = matrix
        self._matrix_diagonal = matrix.diagonal().copy()
        self._indices = []
        self._pivots = []

        self._allocation_size = allocation_size
        self._current_allocated_capacity = min(allocation_size, max_rank)

        self._preconditioner_upper = np.empty(
            (self._current_allocated_capacity, dimension), dtype=matrix.dtype
        )

        if regularization_factor is None:
            regularization_factor = 1e-10

        self._regularization_factor = regularization_factor

        self._capacitance_cholesky = np.empty(
            (self._current_allocated_capacity, max_rank), dtype=matrix.dtype
        )

    @override
    def apply(self, rhs: Vector) -> Vector:
        """Computes (U^T U + lambda*I)^-1 * v
        using the identity z = (1/reg) * (r - U^T * (U U^T + reg*I)^-1 * U r)
        """
        if self._current_outer_rank == 0:
            # logger.warning(
            #     "The capacitance matrix hasn't been computed yet. Using identity matrix as preconditioner."
            # )
            return rhs

        rank = self._current_outer_rank
        U = self._preconditioner_upper[:rank, :]

        b = U @ rhs
        b = b.reshape(rank, 1)

        # Small system solve (stable)
        w = scipy.linalg.cho_solve((self._capacitance_cholesky[:rank, :rank], True), b)

        w = w.reshape(-1, 1)

        Ut_w = U.T @ w

        result = rhs - Ut_w.flatten()
        if self._regularization_factor != 0:
            result /= self._regularization_factor

        return result

    def _ensure_capacity(self) -> None:
        if self._current_inner_rank == self._current_allocated_capacity:
            self._current_allocated_capacity = min(
                self._current_allocated_capacity + self._allocation_size, self.max_rank
            )

            new_preconditioner_upper = np.empty(
                (self._current_allocated_capacity, self._dimension),
                dtype=self._preconditioner_upper.dtype,
            )

            new_preconditioner_upper[: self.current_inner_rank, :] = (
                self._preconditioner_upper[: self.current_inner_rank]
            )

            self._preconditioner_upper = new_preconditioner_upper

            new_capacitance_cholesky = np.empty(
                (self._current_allocated_capacity, self._current_allocated_capacity),
                dtype=self._capacitance_cholesky.dtype,
            )

            new_capacitance_cholesky[
                : self._current_inner_rank, : self._current_inner_rank
            ] = self._capacitance_cholesky[
                : self._current_inner_rank, : self._current_inner_rank
            ]

            self._capacitance_cholesky = new_capacitance_cholesky

    @override
    def update_outer(self) -> None:
        current_rank = self._current_inner_rank
        previous_rank = self._current_outer_rank
        if current_rank == previous_rank:
            # Nothing to do
            return

        # Extract the sub-array representing the valid part of the Cholesky decomposition
        U = self._preconditioner_upper[:current_rank, :]

        # Use the previous capacitance matrix (more precisely, its Cholesky factor)
        # as an intermediary buffer to store the result of the matrix multiplication.
        buffer = self._capacitance_cholesky[:current_rank, :current_rank]

        # Compute `U U^T + mu*I`
        # TODO: find a way to also do this in-place, as an option
        capacitance = np.matmul(U, U.mT, out=buffer)
        if self._regularization_factor != 0:
            _add_regularization_inplace(capacitance, self._regularization_factor)

        capacitance_cholesky, _ = scipy.linalg.cho_factor(
            capacitance, lower=True, overwrite_a=True, check_finite=False
        )

        # Copy new values
        self._capacitance_cholesky[:current_rank, :current_rank] = capacitance_cholesky
        self._current_outer_rank = current_rank

    @property
    @override
    def current_inner_rank(self) -> int:
        return self._current_inner_rank

    @property
    @override
    def current_outer_rank(self) -> int:
        return self._current_outer_rank

    @property
    @override
    def max_rank(self) -> int:
        return self._max_rank


@numba.njit
def _add_regularization_inplace(matrix: Matrix, factor: float) -> None:
    for i in range(len(matrix)):
        matrix[i, i] += factor


@final
class GreedilyPivotedCholeskyPreconditioner(PivotedCholeskyPreconditioner):
    @override
    def update_inner(self) -> None:
        if self._current_inner_rank == self._max_rank:
            raise ValueError(
                "Cannot update preconditioner anymore, maximum rank has been reached"
            )

        self._ensure_capacity()

        pivot_index = int(np.argmax(self._matrix_diagonal).item())
        self._indices.append(pivot_index)

        pivot = cast(float, self._matrix_diagonal[pivot_index])
        self._pivots.append(pivot)

        # At this point, the pivot'th row of the matrix might be computed
        row = self._matrix[pivot_index]
        self._preconditioner_upper[self._current_inner_rank, :] = (
            row
            - self._preconditioner_upper[: self._current_inner_rank, [pivot_index]].mT
            @ self._preconditioner_upper[: self._current_inner_rank, :]
        ) / math.sqrt(cast(float, self._matrix_diagonal[pivot_index]))
        self._matrix_diagonal -= (
            self._preconditioner_upper[self._current_inner_rank, :] ** 2
        )
        self._matrix_diagonal = self._matrix_diagonal.clip(0, None)

        self._current_inner_rank += 1


@final
class UniformlyRandomPivotedCholeskyPreconditioner(PivotedCholeskyPreconditioner):
    _generator: np.random.Generator
    _available_indices: set[int]

    def __init__(
        self,
        generator: np.random.Generator,
        matrix: MatrixInterface,
        max_rank: int,
        regularization_factor: float | None,
    ) -> None:
        super().__init__(matrix, max_rank, regularization_factor)

        self._generator = generator

        dimension = matrix.shape[0]
        self._available_indices = set(range(dimension))

    @override
    def update_inner(self) -> None:
        self._ensure_capacity()

        available_indices = list(self._available_indices)
        pivot = self._generator.choice(available_indices, size=1).item()
        self._indices.append(pivot)
        self._available_indices.remove(pivot)

        # At this point, the pivot'th row of the matrix might be computed
        row = self._matrix[pivot]
        self._preconditioner_upper[self._current_inner_rank, :] = (
            row
            - self._preconditioner_upper[: self._current_inner_rank, [pivot]].mT
            @ self._preconditioner_upper[: self._current_inner_rank, :]
        ) / math.sqrt(cast(float, self._matrix_diagonal[pivot]))
        self._matrix_diagonal -= (
            self._preconditioner_upper[self._current_inner_rank, :] ** 2
        )
        self._matrix_diagonal = self._matrix_diagonal.clip(0, None)

        self._current_inner_rank += 1
