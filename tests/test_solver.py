import numpy as np
from conftest import (  # pyright: ignore[reportImplicitRelativeImport]
    generate_random_spd_matrix,
)

from adaptive_rank.interface import NumPyArrayAdapter
from adaptive_rank.preconditioner import (
    GreedilyPivotedCholeskyPreconditioner,
    IdentityPreconditioner,
)
from adaptive_rank.solver import (
    PreconditionedConjugateGradientSolver,
)
from adaptive_rank.types import Matrix, Vector


def generate_random_spd_system(
    rng: np.random.Generator, dimension: int, regularization_factor: float = 1e-3
) -> tuple[Matrix, Vector]:
    A = generate_random_spd_matrix(rng, dimension, regularization_factor)
    b = rng.normal(size=(dimension,))
    return A, b


def test_can_solve_spd_system_using_cg_with_no_preconditioner() -> None:
    rng = np.random.default_rng(1234)

    dimension = 16
    A, b = generate_random_spd_system(rng, dimension)

    tolerance = 1e-5

    solver = PreconditionedConjugateGradientSolver(
        coefficients=NumPyArrayAdapter(A),
        biases=b,
        preconditioner=IdentityPreconditioner(),
        tolerance=tolerance,
    )

    result, iterations = solver.solve()

    assert iterations < 2 * dimension
    assert np.linalg.vector_norm(A @ result - b) < tolerance


def test_can_solve_spd_system_using_cg_with_pivoted_cholesky_preconditioner_at_partial_rank() -> (
    None
):
    rng = np.random.default_rng(42)

    dimension = 64
    A, b = generate_random_spd_system(rng, dimension)

    A_adapted = NumPyArrayAdapter(A)

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        A_adapted, dimension, regularization_factor=1e-3
    )

    tolerance = 1e-5

    solver = PreconditionedConjugateGradientSolver(
        coefficients=A_adapted,
        biases=b,
        preconditioner=preconditioner,
        tolerance=tolerance,
    )

    result_without_prec, iterations_without_prec = solver.solve()
    assert np.linalg.vector_norm(A @ result_without_prec - b).item() <= tolerance

    assert iterations_without_prec < 1000

    for _ in range(32):
        preconditioner.update_inner()
    preconditioner.update_outer()

    solver.restart(np.ones_like(solver.solution))

    result_with_prec, iterations_with_prec = solver.solve()

    assert iterations_with_prec < iterations_without_prec
    assert np.linalg.vector_norm(A @ result_with_prec - b).item() <= tolerance


def test_can_solve_spd_system_using_cg_with_pivoted_cholesky_preconditioner_at_full_rank() -> (
    None
):
    rng = np.random.default_rng(1234)

    dimension = 16
    A, b = generate_random_spd_system(rng, dimension)

    A_adapted = NumPyArrayAdapter(A)

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        A_adapted, dimension, regularization_factor=1e-5
    )
    preconditioner.compute_full()

    tolerance = 1e-5

    solver = PreconditionedConjugateGradientSolver(
        coefficients=A_adapted,
        biases=b,
        preconditioner=preconditioner,
        tolerance=tolerance,
    )

    result, iterations = solver.solve()

    assert iterations <= 3
    assert np.linalg.vector_norm(A @ result - b) < tolerance
