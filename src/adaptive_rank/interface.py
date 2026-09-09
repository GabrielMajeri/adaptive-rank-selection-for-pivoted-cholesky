from typing import Protocol, override

import numpy as np

from .types import Matrix, Vector


class MatrixInterface(Protocol):
    "Abstract interface for dense linear systems."

    @property
    def dimension(self) -> int: ...

    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def dtype(self) -> np.dtype: ...

    def diagonal(self) -> Vector: ...

    def __matmul__(self, rhs: Vector) -> Vector: ...
    def __getitem__(self, row_index: int) -> Vector: ...


class NumPyArrayAdapter(MatrixInterface):
    "Implements the abstract matrix interface for a 2D NumPy array."

    matrix: Matrix

    def __init__(self, matrix: Matrix) -> None:
        if matrix.ndim != 2:
            raise ValueError("`matrix` must be a matrix (2D array)")

        self.matrix = matrix

    @property
    @override
    def dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    @override
    def ndim(self) -> int:
        return self.matrix.ndim

    @property
    @override
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    @override
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    @override
    def diagonal(self) -> Vector:
        return np.diagonal(self.matrix)

    @override
    def __matmul__(self, rhs: Vector) -> Vector:
        return self.matrix @ rhs

    @override
    def __getitem__(self, row_index: int) -> Vector:
        return self.matrix[row_index, :]
