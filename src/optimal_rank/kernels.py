import math

import numpy as np
from pykeops.numpy import LazyTensor

from .types import Array


def rbf_kernel(X: Array, Y: Array, bandwidth: float | None = None) -> Array:
    """Evaluates the radial basis function (RBF) kernel on two sets of vectors,
    computing the pairwise inner products (the vectors are assumed to all have
    the same number of coordinates).

    The RBF kernel is also known as an exponential kernel, since it's given by
        K(x, y) = exp(- ||x - y||^2 / (2 * \\sigma^2))
    where N is the dimension of the vectors.

    See also https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    """
    if bandwidth is None:
        bandwidth = math.sqrt(X.shape[-1])

    X_squared = np.sum(np.square(X), axis=-1)[:, np.newaxis]
    Y_squared = np.sum(np.square(Y), axis=-1)[np.newaxis, :]

    # Note that ||X - Y||^2 = <X - Y, X - Y> = X^2 + Y^2 - 2 <X, Y>
    result = X_squared + Y_squared - 2 * X @ Y.T

    # Clip it to positive entries (due to numerical errors, some entries could be negative)
    result = result.clip(0.0, None)

    # Formula for RBF kernel
    return np.exp(-result / (2 * bandwidth**2))


def rbf_kernel_keops(X: Array, Y: Array) -> LazyTensor:
    """Constructs a new PyKeOps `LazyTensor` which computes the evaluation of the
    radial basis function (exponential) kernel on two sets of vectors.

    See the documentation for the `rbf_kernel` function for more details.
    """

    X_i = LazyTensor(X[:, np.newaxis, :])
    Y_j = LazyTensor(Y[np.newaxis, :, :])

    # Formula for RBF kernel
    D_ij = ((X_i - Y_j) ** 2).sum(dim=2)

    # Matrix of K(x_i, x_j)'s
    K_ij = (-D_ij / X.shape[-1]).exp()

    return K_ij
