import numpy as np

from ..interface import NumPyArrayAdapter
from ..preconditioner import GreedilyPivotedCholeskyPreconditioner
from ..solver import PreconditionedConjugateGradientSolver


def warm_up_code(dimension: int, tolerance: float = 1e-3) -> None:
    """Warms-up the preconditioned conjugate gradient code by
    constructing a dense linear system (matrix of ones) and
    a small preconditioner, then performing one step of the PCG method.
    """
    A = np.ones((dimension, dimension), dtype=np.float64)
    A_adapted = NumPyArrayAdapter(A)

    b = np.ones(dimension, dtype=np.float64)

    # Warm-up preconditioner computation code
    preconditioner = GreedilyPivotedCholeskyPreconditioner(A_adapted, 1, 1e-3)
    preconditioner.compute_full()

    # Warm-up solver code
    solver = PreconditionedConjugateGradientSolver(
        A_adapted, b, preconditioner, tolerance
    )

    solver.step()
