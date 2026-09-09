import numpy as np
import scipy
from conftest import (  # pyright: ignore[reportImplicitRelativeImport]
    generate_random_spd_matrix,
)

from adaptive_rank.interface import NumPyArrayAdapter
from adaptive_rank.preconditioner import (
    GreedilyPivotedCholeskyPreconditioner,
    UniformlyRandomPivotedCholeskyPreconditioner,
)


def test_greedily_pivoted_cholesky_can_compute_full_decomposition() -> None:
    rng = np.random.default_rng(7)

    dimension = 10
    tolerance = 1e-10

    A = generate_random_spd_matrix(rng, dimension)

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        matrix=NumPyArrayAdapter(A), max_rank=dimension, regularization_factor=0
    )

    for _ in range(dimension):
        preconditioner.update_inner()

    # The preconditioner should now have the full Cholesky decomposition
    L = preconditioner._preconditioner_upper[:dimension, :].T  # pyright: ignore[reportPrivateUsage]

    reconstructed_A = L @ L.T

    assert np.allclose(A, reconstructed_A, atol=tolerance), (
        "The full Cholesky decomposition was not computed correctly"
    )


def test_greedily_pivoted_cholesky_decomposition_matches_lapack() -> None:
    rng = np.random.default_rng(7)

    dimension = 4
    A = generate_random_spd_matrix(rng, dimension)

    tolerance = 1e-5

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        matrix=NumPyArrayAdapter(A), max_rank=dimension, regularization_factor=0
    )
    preconditioner.compute_full()

    U = preconditioner._preconditioner_upper  # pyright: ignore[reportPrivateUsage]

    U_lapack, piv, _rank, _info = scipy.linalg.lapack.dpstrf(
        A, tol=tolerance, lower=False
    )

    # LAPACK returns junk values in the lower half
    U_lapack = np.triu(U_lapack)

    P = np.zeros((dimension, dimension), dtype=np.float64)
    for k in range(len(piv)):
        P[piv[k] - 1, k] = 1

    assert U.shape == U_lapack.shape
    assert np.allclose(U, U_lapack @ P.mT, atol=tolerance, rtol=tolerance)
    assert np.allclose(
        U.mT @ U, P @ U_lapack.mT @ U_lapack @ P.mT, rtol=tolerance, atol=tolerance
    )


def test_greedily_pivoted_cholesky_decomposition_gives_correct_factors() -> None:
    rng = np.random.default_rng(7)

    dimension = 32

    A = generate_random_spd_matrix(rng, dimension)

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        matrix=NumPyArrayAdapter(A), max_rank=dimension, regularization_factor=1e-3
    )

    rank = 28
    for _ in range(rank):
        preconditioner.update_inner()

    U = preconditioner._preconditioner_upper[:rank, :]  # pyright: ignore[reportPrivateUsage]

    x = rng.normal(size=(dimension,))
    x_norm = np.linalg.norm(x).item()

    correct_result = A @ x
    lra_result = U.T @ (U @ x)

    assert np.linalg.norm(lra_result - correct_result).item() / x_norm < 1


def test_uniformly_random_pivoted_cholesky_can_compute_full_decomposition() -> None:
    rng = np.random.default_rng(7)

    dimension = 10
    tolerance = 1e-10

    A = generate_random_spd_matrix(rng, dimension)

    preconditioner = UniformlyRandomPivotedCholeskyPreconditioner(
        generator=rng,
        matrix=NumPyArrayAdapter(A),
        max_rank=dimension,
        regularization_factor=None,
    )

    for _ in range(dimension):
        preconditioner.update_inner()

    # The preconditioner should now have the full Cholesky decomposition
    L = preconditioner._preconditioner_upper[:dimension, :].T  # pyright: ignore[reportPrivateUsage]

    reconstructed_A = L @ L.T

    assert np.allclose(A, reconstructed_A, atol=tolerance), (
        "The full Cholesky decomposition was not computed correctly"
    )
