from collections.abc import Callable
from typing import Protocol, override

import numpy as np
from pykeops.numpy import LazyTensor

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


class LazyTensorKernelAdapter(MatrixInterface):
    """Implements the abstract matrix interface for a PyKeOps LazyTensor,
    representing a kernel matrix (in particular, the diagonal must be 1).
    """

    tensor: LazyTensor
    _dtype: np.dtype
    regularization_factor: float
    backend: str
    _get_row: Callable[[int], Vector] | None

    def __init__(
        self,
        tensor: LazyTensor,
        dtype: np.dtype,
        regularization_factor: float,
        backend: str | None = None,
        get_row: Callable[[int], Vector] | None = None,
    ) -> None:
        self.tensor = tensor
        self._dtype = dtype
        self.regularization_factor = regularization_factor
        self.backend = backend or "auto"
        self._get_row = get_row

    @property
    @override
    def dimension(self) -> int:
        return self.tensor.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.tensor.shape)

    @property
    def shape(self) -> tuple:
        return self.tensor.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __matmul__(self, rhs: Vector) -> Vector:
        return (
            self.tensor.__matmul__(rhs, backend=self.backend)
            + self.regularization_factor * rhs
        )

    def diagonal(self) -> Vector:
        return np.ones(self.tensor.shape[0], dtype=self.tensor.dtype)

    def __getitem__(self, row_index: int) -> Vector:
        if self._get_row is None:
            raise NotImplementedError(
                "Row access is not implemented for this LazyTensorKernelAdapter."
            )

        return self._get_row(row_index)
